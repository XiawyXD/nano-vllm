"""CPU-only small-vocabulary correctness/distribution tests; no model or CUDA.

Statistical gates fixed before execution: N=100000, max cell error <=0.01.
They compare distributions, never text equality under the same random seed.
Run directly with Python to avoid importing the GPU engine package initializer.
"""
import importlib.util
import itertools
import json
from fractions import Fraction
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch


def load_algorithm():
    name = '_standalone_speculative_sampling'
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2]/'nanovllm/engine/speculative_sampling.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


ss = load_algorithm()
METRICS = {}
N = 100000
MAX_CELL_ERROR = 0.01


def load_wrapper_fixture():
    """Load only pure Context/Proposer and the wrapper; bypass package __init__."""
    root = Path(__file__).resolve().parents[2]
    for name in ('nanovllm','nanovllm.engine','nanovllm.utils','nanovllm.layers'):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
    sys.modules['nanovllm.engine.speculative_sampling'] = ss
    def load(name, relative):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name,root/relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        return sys.modules[name]
    context = load('nanovllm.utils.context','nanovllm/utils/context.py')
    load('nanovllm.layers.quant_graph','nanovllm/layers/quant_graph.py')
    proposer = load('_standalone_draft_proposer','nanovllm/engine/proposer.py')
    wrapper = load('_standalone_stochastic_draft','nanovllm/engine/stochastic_draft.py')
    owner = SimpleNamespace(device=torch.device('cpu'),capacity=16,chunk_size=2,request_id=None,
        block_table=torch.arange(4,dtype=torch.int32).unsqueeze(0),workspace_bytes=0,
        proposer=proposer.DraftProposer(lambda *args:0,16,2))
    class FakeModel:
        def __init__(self):
            self.lm_head = SimpleNamespace(weight=torch.zeros((3,1)))
            self.table = torch.tensor([[0.,1.,2.],[2.,0.,1.],[1.,2.,0.]])
            self.calls = []
            self.fail_call = None
            self.nan_logits = False
        def __call__(self, ids, positions):
            self.calls.append(dict(ids=ids.tolist(),positions=positions.tolist(),
                                   published=len(owner.proposer.cached_tokens)))
            if len(self.calls) == self.fail_call:
                raise RuntimeError('injected partial forward failure')
            current = context.get_context()
            assert current.is_prefill and torch.equal(current.slot_mapping,positions.int())
            assert current.cu_seqlens_k.tolist() == [0,int(positions[-1])+1]
            return self.table[ids]
        def compute_logits(self, hidden):
            return torch.full_like(hidden[-1:],float('nan')) if self.nan_logits else hidden[-1:]
    owner.model = FakeModel()
    return wrapper.StochasticDraft(owner,use_graph=False),owner,context


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.rng = ss.make_generators(20261023, device='cpu')

    def correct(self, ids, q, p, u, v=.37):
        return ss.speculative_correction(torch.tensor(ids, dtype=torch.int64),
            torch.tensor(q, dtype=torch.float64), torch.tensor(p, dtype=torch.float64),
            generators=self.rng, acceptance_uniforms=torch.tensor(u, dtype=torch.float64),
            sample_uniforms=torch.tensor(v, dtype=torch.float64))

    def test_full_accept_bonus_and_zero_q_other_tokens(self):
        r = self.correct([0, 1], [[1, 0], [0, 1]], [[1, 0], [0, 1], [0, 1]], [0, .999])
        self.assertEqual(r.to_host(), ([0, 1, 1], 2))

    def test_first_rejection_p_zero_even_uniform_zero(self):
        r = self.correct([0, 0], [[1, 0, 0], [1, 0, 0]],
                         [[0, .25, .75], [1, 0, 0], [1, 0, 0]], [0, 0], .5)
        self.assertEqual(r.to_host(), ([2], 0))
        self.assertEqual(r.packet.tolist()[3:], [-1, -1])

    def test_middle_rejection_and_discarded_tail(self):
        r = self.correct([0, 1, 0], [[1, 0], [0, 1], [1, 0]],
                         [[1, 0], [1, 0], [1, 0], [1, 0]], [.2, .3, .4])
        self.assertEqual(r.to_host(), ([0, 0], 1))

    def test_last_rejection(self):
        r = self.correct([0, 1], [[1, 0], [0, 1]], [[1, 0], [1, 0], [0, 1]], [.1, .9])
        self.assertEqual(r.to_host(), ([0, 0], 1))

    def test_empty_draft_bonus(self):
        r = ss.speculative_correction(torch.empty(0, dtype=torch.int64),
            torch.empty((0, 2), dtype=torch.float64), torch.tensor([[0., 1.]], dtype=torch.float64),
            generators=self.rng)
        self.assertEqual(r.to_host(), ([1], 0))

    def test_tiny_and_subnormal_residual_no_epsilon_fallback(self):
        for mass in (1e-15, 1e-100, 1e-310):
            # A rare but positive-q candidate: p and q differ only in tiny tails.
            r = self.correct([2], [[1.-mass, 0, mass]], [[1.-mass, mass, 0], [1, 0, 0]], [0])
            self.assertEqual(r.to_host(), ([1], 0))

    def test_zero_residual_is_error_not_target_fallback(self):
        r = self.correct([0], [[.5, .5]], [[0, 0], [1, 0]], [.3])
        self.assertTrue(int(r.packet[0]) & ss.ZERO_REJECTION_RESIDUAL)
        self.assertTrue(int(r.packet[0]) & ss.INVALID_PROBABILITIES)
        self.assertEqual(r.packet.tolist()[1:], [-1, -1, -1])
        with self.assertRaises(ValueError):
            r.to_host()

    def test_invalid_candidate_or_zero_support_never_commits(self):
        for candidate in (-1, 2, 1):
            r = self.correct([candidate], [[1, 0]], [[1, 0], [1, 0]], [.2])
            self.assertTrue(int(r.packet[0]) & ss.INVALID_CANDIDATE)
            with self.assertRaises(ValueError):
                r.to_host()

    def test_bad_probability_and_uniform_data(self):
        for p in ([float('nan'), 1], [float('inf'), 0], [-.1, 1.1], [.1, .1]):
            r = self.correct([0], [[1, 0]], [p, [1, 0]], [.1])
            self.assertTrue(int(r.packet[0]) & ss.INVALID_PROBABILITIES)
        for u in (-.1, 1., float('nan')):
            r = self.correct([0], [[1, 0]], [[1, 0], [1, 0]], [u])
            self.assertTrue(int(r.packet[0]) & ss.INVALID_UNIFORM)

    def test_categorical_zero_bins_boundaries(self):
        probs = torch.tensor([[0., .3, 0., .7, 0.]], dtype=torch.float64).expand(3, -1)
        uniform = torch.tensor([0., .3, torch.nextafter(torch.tensor(1.,dtype=torch.float64),
                                                     torch.tensor(0.,dtype=torch.float64))],dtype=torch.float64)
        out = ss.sample_from_probs(probs, generator=self.rng.draft, uniforms=uniform)
        self.assertEqual(out.tolist(), [1, 3, 3])

    def test_positive_temperature_only_and_no_filters(self):
        logits = torch.tensor([0., 1., -float('inf')])
        actual = ss.temperature_probabilities(logits, .7)
        weights = torch.softmax(logits.float()/.7, -1).double()
        expected = weights / weights.sum(-1, keepdim=True)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertTrue(bool(ss._normalize(actual)[1].all()))
        self.assertTrue(torch.equal(actual, expected))
        for temperature in (0, 1e-11, 1e-10, -1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                ss.temperature_probabilities(logits, temperature)
        for option in ({'top_k':1}, {'top_p':.9}, {'repetition_penalty':1.1}):
            with self.assertRaises(ValueError):
                ss.temperature_probabilities(logits, 1., **option)
        for invalid in ([float('nan'),0], [float('inf'),0], [-float('inf'),-float('inf')]):
            proposal = ss.sample_draft(torch.tensor(invalid), 1., generators=self.rng)
            self.assertEqual(int(proposal.ids), -1)
        proposal = ss.sample_draft(torch.tensor([1e20, 0.],dtype=torch.float64), 1.1e-10, generators=self.rng)
        self.assertEqual(int(proposal.ids), 0)

    def test_generators_are_independent_and_global_rng_untouched(self):
        state = torch.random.get_rng_state().clone()
        accept_state = self.rng.acceptance.get_state().clone()
        target_state = self.rng.target.get_state().clone()
        ss.sample_draft(torch.zeros((64, 3)), 1., generators=self.rng)
        self.assertTrue(torch.equal(accept_state, self.rng.acceptance.get_state()))
        self.assertTrue(torch.equal(target_state, self.rng.target.get_state()))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        with self.assertRaises(ValueError):
            ss.SamplingGenerators(self.rng.draft,self.rng.draft,self.rng.target).validate(torch.device('cpu'))
        same = torch.Generator().manual_seed(self.rng.draft.initial_seed())
        with self.assertRaises(ValueError):
            ss.SamplingGenerators(self.rng.draft,same,self.rng.target).validate(torch.device('cpu'))

    def test_exhaustive_rational_small_vocabulary(self):
        distributions = [tuple(Fraction(x,4) for x in triple)
                         for triple in itertools.product(range(5), repeat=3) if sum(triple)==4]
        branches = 0
        for p in distributions:
            for q in distributions:
                output = [Fraction(0)]*3
                residual = [max(a-b,0) for a,b in zip(p,q)]
                mass = sum(residual)
                for token in range(3):
                    if q[token] == 0:
                        continue
                    alpha = min(Fraction(1),p[token]/q[token])
                    if alpha:
                        r = self.correct([token], [list(map(float,q))],
                            [list(map(float,p)), [1,0,0]], [float(alpha/2)])
                        ids, accepted = r.to_host()
                        self.assertEqual(accepted,1)
                        output[ids[0]] += q[token]*alpha
                        branches += 1
                    if alpha < 1:
                        cumulative = Fraction(0)
                        for destination, weight in enumerate(residual):
                            if not weight:
                                continue
                            probability = weight/mass
                            v = float(cumulative+probability/2)
                            r = self.correct([token], [list(map(float,q))],
                                [list(map(float,p)),[1,0,0]], [float((alpha+1)/2)],v)
                            ids, accepted = r.to_host()
                            self.assertEqual((ids,accepted),([destination],0))
                            output[ids[0]] += q[token]*(1-alpha)*probability
                            cumulative += probability
                            branches += 1
                self.assertEqual(tuple(output),p)
        METRICS['exhaustive'] = dict(distribution_pairs=len(distributions)**2, checked_branches=branches,
                                     reconstructed_law='exact Fraction equality to p')

    def test_one_token_empirical_target_law(self):
        p = torch.tensor([.07,.31,.62],dtype=torch.float64)
        errors = []
        for q in ([1.,0.,0.], [.2,.7,.1], [.07,.31,.62]):
            q = torch.tensor(q,dtype=torch.float64).expand(N,1,3)
            candidates = ss.sample_from_probs(q, generator=self.rng.draft)
            result = ss.speculative_correction(candidates,q,p.expand(N,2,3),generators=self.rng)
            self.assertEqual(int(result.packet[:,0].sum()),0)
            frequency = torch.bincount(result.packet[:,2],minlength=3).double()/N
            error = float((frequency-p).abs().max())
            self.assertLessEqual(error,MAX_CELL_ERROR)
            errors.append(error)
        # Negative control: using p as the rejection distribution is biased.
        q = torch.tensor([.2,.7,.1],dtype=torch.float64)
        wrong = torch.minimum(p,q)+(1-torch.minimum(p,q).sum())*p
        self.assertGreater(float((wrong-p).abs().max()),.05)
        METRICS['one_token'] = dict(samples_per_case=N,max_cell_errors=errors,threshold=MAX_CELL_ERROR)

    def test_two_token_history_dependent_joint_law(self):
        p0 = torch.tensor([.55,.35,.10],dtype=torch.float64)
        pt = torch.tensor([[.1,.3,.6],[.7,.2,.1],[.2,.5,.3]],dtype=torch.float64)
        q0 = torch.tensor([.1,.3,.6],dtype=torch.float64)
        qt = torch.tensor([[.6,.3,.1],[.2,.1,.7],[.1,.6,.3]],dtype=torch.float64)
        x0 = ss.sample_from_probs(q0.expand(N,3),generator=self.rng.draft)
        x1 = ss.sample_from_probs(qt[x0],generator=self.rng.draft)
        q = torch.stack((q0.expand(N,3),qt[x0]),1)
        p = torch.stack((p0.expand(N,3),pt[x0],pt[x1]),1)
        result = ss.speculative_correction(torch.stack((x0,x1),1),q,p,generators=self.rng)
        self.assertEqual(int(result.packet[:,0].sum()),0)
        first = result.packet[:,2]
        # If the first candidate was rejected, finish token 2 by ordinary target
        # sampling at the corrected history (not at the rejected draft token).
        continuation = ss.sample_from_probs(pt[first],generator=self.rng.target)
        second = torch.where(result.packet[:,1]==0,continuation,result.packet[:,3])
        empirical = torch.bincount(first*3+second,minlength=9).reshape(3,3).double()/N
        expected = p0[:,None]*pt
        error = float((empirical-expected).abs().max())
        self.assertLessEqual(error,MAX_CELL_ERROR)
        METRICS['history_dependent_joint'] = dict(samples=N,max_cell_error=error,threshold=MAX_CELL_ERROR,
            accepted_histogram=torch.bincount(result.packet[:,1],minlength=3).tolist())

    def test_all_accept_bonus_empirical_law(self):
        q = torch.tensor([.25,.25,.5],dtype=torch.float64).expand(N,2,3)
        candidates = ss.sample_from_probs(q,generator=self.rng.draft)
        bonus = torch.tensor([.2,.6,.2],dtype=torch.float64)
        p = torch.cat((q,bonus.expand(N,1,3)),1)
        result = ss.speculative_correction(candidates,q,p,generators=self.rng)
        self.assertTrue(bool((result.packet[:,1] == 2).all()))
        frequency = torch.bincount(result.packet[:,4],minlength=3).double()/N
        error = float((frequency-bonus).abs().max())
        self.assertLessEqual(error,MAX_CELL_ERROR)
        METRICS['bonus'] = dict(samples=N,max_cell_error=error,threshold=MAX_CELL_ERROR)

    def test_wrapper_real_q_sampling_and_atomic_publish(self):
        wrapper,owner,context = load_wrapper_fixture()
        history = [0,1,2,0,1]
        proposal = wrapper.propose(history,3,.8,123,'request-a')
        self.assertEqual(proposal.token_ids,proposal.ids.tolist())
        self.assertEqual(proposal.probabilities.shape,(3,3))
        previous = [history[-1],*proposal.token_ids[:-1]]
        for i,token in enumerate(previous):
            weights = torch.softmax(owner.model.table[token]/.8,-1).double()
            expected = weights / weights.sum(-1, keepdim=True)
            self.assertEqual(proposal.probabilities.dtype, torch.float64)
            self.assertTrue(bool(ss._normalize(proposal.probabilities)[1].all()))
            self.assertTrue(torch.equal(proposal.probabilities[i], expected))
        self.assertEqual(owner.proposer.cached_tokens,history+proposal.token_ids[:-1])
        self.assertTrue(all(call['published']==0 for call in owner.model.calls))
        self.assertEqual(wrapper.stats()['candidate_host_transfers'],1)
        self.assertIsNone(context.get_context().slot_mapping)
        saved = proposal.probabilities.clone()
        generators = wrapper.generators
        committed = history+proposal.token_ids+[2]
        wrapper.synchronize(committed,'request-a')
        previous_calls = len(owner.model.calls)
        wrapper.propose(committed,1,.8,123,'request-a')
        self.assertEqual(owner.model.calls[previous_calls]['ids'],[proposal.token_ids[-1],2])
        self.assertIs(wrapper.generators,generators)
        self.assertTrue(torch.equal(proposal.probabilities,saved))
        wrapper.clear()
        self.assertEqual(owner.proposer.cached_tokens,[])
        self.assertIsNone(wrapper.generators)
        self.assertIsNone(owner.request_id)

    def test_wrapper_rejection_resynchronizes_owner_cache(self):
        wrapper,owner,_ = load_wrapper_fixture()
        history = [0,1,2]
        proposal = wrapper.propose(history,3,1.,99,8)
        committed = history+[proposal.token_ids[0],(proposal.token_ids[1]+1)%3]
        wrapper.synchronize(committed,8)
        self.assertEqual(owner.proposer.cached_tokens,committed[:-1])
        previous_calls = len(owner.model.calls)
        wrapper.propose(committed,1,1.,99,8)
        self.assertEqual(owner.model.calls[previous_calls]['ids'],committed[-1:])
        wrapper.close()
        self.assertIsNone(wrapper.generators)
        with self.assertRaises(RuntimeError):
            wrapper.propose(history,1,1.,99,8)

    def test_wrapper_failure_clears_rng_and_logical_cache(self):
        for invalid_logits in (False,True):
            wrapper,owner,context = load_wrapper_fixture()
            owner.model.nan_logits = invalid_logits
            owner.model.fail_call = None if invalid_logits else 2
            with self.assertRaises((ValueError,RuntimeError)):
                wrapper.propose([0,1,2],3,1.,99,8)
            self.assertEqual(owner.proposer.cached_tokens,[])
            self.assertIsNone(owner.request_id)
            self.assertIsNone(wrapper.generators)
            self.assertIsNone(context.get_context().slot_mapping)

    def test_wrapper_request_change_and_temperature_boundary(self):
        wrapper,owner,_ = load_wrapper_fixture()
        wrapper.propose([0,1,2],2,1.,123,8)
        previous_rng = wrapper.generators
        wrapper.propose([1,0],1,1.,124,9)
        self.assertIsNot(wrapper.generators,previous_rng)
        self.assertEqual(owner.request_id,9)
        with self.assertRaises(ValueError):
            wrapper.propose([1,0],1,1e-10,124,9)
        self.assertIsNone(wrapper.generators)
        self.assertEqual(owner.proposer.cached_tokens,[])


def main():
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SamplingTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        device='cpu',torch_version=torch.__version__,metrics=METRICS,
        statistical_scope='Fixed N and absolute cell thresholds; distribution checks, not seeded-text equality.'),indent=2))
    return 0 if result.wasSuccessful() else 2


if __name__ == '__main__':
    raise SystemExit(main())
