"""Independent sampling/real DraftRunner regression, never an engine speed claim.

No CUDA is touched without --gpu, the task owner's lease, and the existing guard.
--cpu-self-test exercises the probability fixtures without importing the engine.
Outputs are exclusive-create JSON; all vocabulary/KV comparisons remain on device.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
N, THRESHOLD = 100000, 0.01


def load_core():
    name = '_gpu_regression_sampling'
    spec = importlib.util.spec_from_file_location(name, ROOT/'nanovllm/engine/speculative_sampling.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def probability_checks(torch, ss, device):
    rng = ss.make_generators(20261023, device=device)
    def tensor(value, dtype=torch.float64):
        return torch.tensor(value, dtype=dtype, device=device)
    def correct(ids, q, p, u, v=.37):
        return ss.speculative_correction(tensor(ids, torch.int64), tensor(q), tensor(p),
            generators=rng, acceptance_uniforms=tensor(u), sample_uniforms=tensor(v))
    cases = [
        ('all_accept_bonus', [0,1], [[1,0],[0,1]], [[1,0],[0,1],[0,1]], [0,.999], ([0,1,1],2)),
        ('first_reject_zero_p_u0', [0,0], [[1,0],[1,0]], [[0,1],[1,0],[1,0]], [0,0], ([1],0)),
        ('middle_reject', [0,1,0], [[1,0],[0,1],[1,0]], [[1,0],[1,0],[1,0],[1,0]], [.2,.3,.4], ([0,0],1)),
    ]
    for name, ids, q, p, u, expected in cases:
        require(correct(ids,q,p,u).to_host() == expected, name)
    for mass in (1e-15,1e-100,1e-310):
        require(correct([2],[[1-mass,0,mass]],[[1-mass,mass,0],[1,0,0]],[0]).to_host()==([1],0),
                f'tiny residual {mass}: never replace residual with p')
    invalid = correct([0],[[.5,.5]],[[0,0],[1,0]],[.3])
    require(int(invalid.packet[0]) & ss.ZERO_REJECTION_RESIDUAL, 'zero residual missing error')
    try:
        invalid.to_host()
    except ValueError:
        pass
    else:
        raise AssertionError('invalid zero residual committed')
    for token in (-1,1,2):
        bad = correct([token],[[1,0]],[[1,0],[1,0]],[.2])
        require(int(bad.packet[0]) & ss.INVALID_CANDIDATE, 'invalid candidate not rejected')
    bins = tensor([[0,.3,0,.7,0]]).expand(3,-1)
    last = torch.nextafter(tensor(1.),tensor(0.))
    selected = ss.sample_from_probs(bins,generator=rng.draft,uniforms=torch.stack((tensor(0.),tensor(.3),last)))
    require(selected.cpu().tolist()==[1,3,3], 'zero CDF bin selected')
    errors = []
    target = tensor([.07,.31,.62])
    for weights in ([1,0,0],[.2,.7,.1],[.07,.31,.62]):
        q = tensor(weights).expand(N,1,3)
        ids = ss.sample_from_probs(q,generator=rng.draft)
        result = ss.speculative_correction(ids,q,target.expand(N,2,3),generators=rng)
        require(bool((result.packet[:,0]==0).all()), 'invalid distribution trial')
        empirical = torch.bincount(result.packet[:,2],minlength=3).double()/N
        errors.append(float((empirical-target).abs().max()))
    p0, q0 = tensor([.55,.35,.10]), tensor([.1,.3,.6])
    pt = tensor([[.1,.3,.6],[.7,.2,.1],[.2,.5,.3]])
    qt = tensor([[.6,.3,.1],[.2,.1,.7],[.1,.6,.3]])
    x0 = ss.sample_from_probs(q0.expand(N,3),generator=rng.draft)
    x1 = ss.sample_from_probs(qt[x0],generator=rng.draft)
    q = torch.stack((q0.expand(N,3),qt[x0]),1)
    p = torch.stack((p0.expand(N,3),pt[x0],pt[x1]),1)
    result = ss.speculative_correction(torch.stack((x0,x1),1),q,p,generators=rng)
    require(bool((result.packet[:,0]==0).all()), 'invalid joint trial')
    first = result.packet[:,2]
    continuation = ss.sample_from_probs(pt[first],generator=rng.target)
    second = torch.where(result.packet[:,1]==0,continuation,result.packet[:,3])
    empirical = torch.bincount(first*3+second,minlength=9).reshape(3,3).double()/N
    joint = float((empirical-p0[:,None]*pt).abs().max())
    q = tensor([.25,.25,.5]).expand(N,2,3)
    ids = ss.sample_from_probs(q,generator=rng.draft)
    bonus = tensor([.2,.6,.2])
    result = ss.speculative_correction(ids,q,torch.cat((q,bonus.expand(N,1,3)),1),generators=rng)
    require(bool((result.packet[:,0]==0).all() & (result.packet[:,1]==2).all()), 'all acceptance failed')
    bonus_error = float((torch.bincount(result.packet[:,4],minlength=3).double()/N-bonus).abs().max())
    require(max(*errors,joint,bonus_error)<=THRESHOLD, 'fixed distribution threshold exceeded')
    return dict(status='PASS',samples_per_case=N,threshold=THRESHOLD,one_token_errors=errors,
                joint_error=joint,bonus_error=bonus_error,edge_cases=[x[0] for x in cases],
                residual_masses=[1e-15,1e-100,1e-310],zero_residual='ERROR_NO_FALLBACK')


def tensor_diff(torch, a, b):
    require(a.shape==b.shape, 'comparison shape mismatch')
    return dict(equal=bool(torch.equal(a,b)),max_abs=float((a.double()-b.double()).abs().max()),
                different=int((a!=b).sum()),elements=a.numel())


def timing(torch, fn, repeats=12):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    wall, gpu = [], []
    for _ in range(repeats):
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        result = fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter()-started)*1000)
        gpu.append(begin.elapsed_time(end))
        del result
    return dict(repeats=repeats,wall_median_ms=statistics.median(wall),gpu_median_ms=statistics.median(gpu),
                wall_samples_ms=wall,gpu_samples_ms=gpu)


def wrapper_checks(torch, ss, args, guard, report):
    import torch.distributed as dist
    from nanovllm.engine.draft_runner import DraftRunner
    from nanovllm.engine.stochastic_draft import StochasticDraft
    from nanovllm.utils.context import get_context
    owner = wrapper = None
    with tempfile.TemporaryDirectory(prefix='stochastic-regression-') as directory:
        guard('before_nccl_model')
        dist.init_process_group('nccl',init_method='file://'+directory+'/store',rank=0,world_size=1)
        try:
            owner = DraftRunner(SimpleNamespace(draft_model=args.model,max_model_len=1280,
                kvcache_block_size=256,max_num_batched_tokens=256,draft_execution='eager',
                enforce_eager=False,draft_graph_max_tokens=8,enable_stochastic_speculation=False))
            if args.draft_numerical_backend=='consistent':
                # Explicit test startup choice, before probability graph capture/requests.
                owner.model.set_numerical_backend('consistent')
            wrapper = StochasticDraft(owner,use_graph=True,max_tokens=8)
            require(sorted(wrapper.graphs)==[1,2], 'missing Q1/Q2 logits graph')
            report['model'] = dict(path=str(Path(args.model).resolve()),vocabulary=wrapper.vocab_size,
                dtype=str(owner.kv.dtype),numerical_backend=getattr(owner.model,'numerical_backend','unknown'),
                config_sha256=hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest())
            rows = report['wrapper_cases'] = []
            graphs = wrapper.graphs
            def valid_kv(length):
                # [K/V,layer,page,slot,head,dim] -> logical token prefix only.
                return owner.kv.flatten(2,3)[:,:,:length]
            def clean():
                wrapper.clear()
                owner.kv.zero_()
            def pair(history, count, name, seed=8123):
                """Exact same logical/physical initial state and same RNG draws."""
                initial = owner.kv.clone()
                cached = list(owner.proposer.cached_tokens)
                request = wrapper.request_id
                if request is None:
                    request = name
                rng = wrapper.generators
                states = None if rng is None else [g.get_state() for g in (rng.draft,rng.acceptance,rng.target)]
                before = wrapper.counters['candidate_host_transfers']
                wrapper.graphs = graphs
                actual = wrapper.propose(history,count,.8,seed,request)
                end_cache = list(owner.proposer.cached_tokens)
                graph_kv = valid_kv(len(end_cache)).clone()
                graph_states = [g.get_state() for g in (wrapper.generators.draft,wrapper.generators.acceptance,wrapper.generators.target)]
                owner.kv.copy_(initial)
                owner.proposer.cached_tokens[:] = cached
                if rng is None:
                    wrapper.clear()
                else:
                    for g,s in zip((rng.draft,rng.acceptance,rng.target),states):
                        g.set_state(s)
                wrapper.graphs = {}
                try:
                    reference = wrapper.propose(history,count,.8,seed,request)
                finally:
                    wrapper.graphs = graphs
                qdiff = tensor_diff(torch,actual.probabilities,reference.probabilities)
                kvdiff = tensor_diff(torch,graph_kv,valid_kv(len(end_cache)))
                ids_equal = actual.token_ids == reference.token_ids
                states_equal = all(torch.equal(a,g.get_state()) for a,g in zip(graph_states,
                    (wrapper.generators.draft,wrapper.generators.acceptance,wrapper.generators.target)))
                row = dict(name=name,history=len(history),cached_before=len(cached),K=count,
                           q=qdiff,kv=kvdiff,ids_equal=ids_equal,rng_states_equal=states_equal,
                           cache_equal=owner.proposer.cached_tokens==end_cache,
                           candidate_transfers=wrapper.counters['candidate_host_transfers']-before)
                rows.append(row)
                require(qdiff['equal'] and kvdiff['equal'] and ids_equal and states_equal and row['cache_equal'],
                        'eager/graph exact parity failed: '+name)
                require(row['candidate_transfers']==2, 'not one small D2H per proposal')
                require(get_context().slot_mapping is None, 'context leaked')
                return reference
            for length in (1,127,255,256,511,1019):
                guard('prefix_'+str(length))
                for k in (1,4,8):
                    clean()
                    history = [7+(i*31)%1000 for i in range(length)]
                    proposal = pair(history,k,f'prefix{length}_K{k}')
                    if k!=4:
                        continue
                    # Exercise all three commit outcomes; target rows are controlled
                    # fixtures, NOT claims about a real target-model distribution.
                    for accepted in (0,2,4):
                        clean()
                        proposal = wrapper.propose(history,k,.8,8123,f'branch{length}_{accepted}')
                        p = torch.cat((proposal.probabilities,proposal.probabilities[-1:]),0).clone()
                        if accepted<k:
                            p[accepted].zero_()
                            replacement = (proposal.token_ids[accepted]+1)%wrapper.vocab_size
                            p[accepted,replacement]=1.
                        result = ss.speculative_correction(proposal.ids,proposal.probabilities,p,
                            generators=wrapper.generators,acceptance_uniforms=torch.zeros(k,device=owner.device,dtype=torch.float64))
                        tokens, actual_accepted = result.to_host()
                        require(actual_accepted==accepted, 'controlled commit branch failed')
                        committed = history+tokens
                        wrapper.synchronize(committed,wrapper.request_id)
                        expected_suffix = 2 if accepted==k else 1
                        require(len(committed)-len(owner.proposer.cached_tokens)==expected_suffix, 'wrong rollback suffix')
                        # Finite poison only unreachable stale KV. Compare poison vs
                        # zeros using identical cached prefix/query partition (not
                        # a cold prefill with different Torch GEMM shapes).
                        boundary=len(owner.proposer.cached_tokens)
                        saved_kv=owner.kv.clone()
                        valid_kv(owner.capacity)[:,:,boundary:].fill_(.125)
                        poison=pair(committed,4,f'prefix{length}_accepted{accepted}')
                        saved_q=poison.probabilities.clone()
                        owner.kv.copy_(saved_kv)
                        valid_kv(owner.capacity)[:,:,boundary:].zero_()
                        owner.proposer.cached_tokens[:]=committed[:boundary]
                        wrapper.generators=ss.make_generators(8123,device=owner.device)
                        # q[0] is independent of new RNG draws; later q conditions
                        # on candidates, so only compare the common-history row.
                        fresh=wrapper.propose(committed,1,.8,8123,wrapper.request_id)
                        diff=tensor_diff(torch,saved_q[0],fresh.probabilities[0])
                        require(diff['equal'], 'stale/rejected future KV affected current q')
                        rows[-1]['stale_tail_mask']=diff
            guard('cancellation_exception')
            clean()
            wrapper.propose([7,8,9],4,.8,8123,'cancel')
            wrapper.clear()
            require(wrapper.generators is None and not owner.proposer.cached_tokens and owner.request_id is None,
                    'cancel leaked logical state/RNG')
            original = wrapper._logits
            calls = 0
            def fail_after_write(ids,start):
                nonlocal calls
                value=original(ids,start)
                calls+=1
                if calls==2:
                    raise RuntimeError('injected after real GPU KV write')
                return value
            wrapper._logits=fail_after_write
            try:
                try:
                    wrapper.propose([12,13,14],4,.8,8123,'fail')
                except RuntimeError as error:
                    require(str(error)=='injected after real GPU KV write','unexpected injection error')
                else:
                    raise AssertionError('injected error swallowed')
            finally:
                wrapper._logits=original
            require(wrapper.generators is None and not owner.proposer.cached_tokens and owner.request_id is None,
                    'exception leaked logical state/RNG')
            require(get_context().slot_mapping is None,'exception leaked Context')
            recovered=wrapper.propose([23,24,25],4,.8,8123,'recovery')
            q_recovered=recovered.probabilities.clone()
            clean()
            fresh=wrapper.propose([23,24,25],4,.8,8123,'fresh')
            require(torch.equal(q_recovered,fresh.probabilities),'new request sees stale failed-request KV')
            report['lifecycle']=dict(cancel='PASS',partial_forward_exception='PASS',new_request='PASS')
            guard('real_vocab_microbenchmark')
            vocab=wrapper.vocab_size
            rng=ss.make_generators(987,device=owner.device)
            logits=original(torch.tensor([7],device=owner.device),0).clone()
            perf=report['performance']=dict(vocabulary=vocab,scope='microbenchmark, NO target model/end-to-end speed claim',cases=[])
            perf['ordinary_single_sample']=timing(torch,lambda: torch.multinomial(torch.softmax(logits.float()/.8,-1),1,generator=rng.target).cpu().tolist())
            for k in (1,4,8):
                q=ss.temperature_probabilities(logits,.8).expand(k,-1).contiguous()
                ids=ss.sample_from_probs(q,generator=rng.draft)
                p=torch.cat((q,q[-1:]),0)
                measure=timing(torch,lambda: ss.speculative_correction(ids,q,p,generators=rng).to_host())
                perf['cases'].append(dict(K=k,correction=measure,
                    correction_over_single_sample=measure['wall_median_ms']/perf['ordinary_single_sample']['wall_median_ms']))
            report['wrapper_stats']=wrapper.stats()
        finally:
            if wrapper is not None:
                wrapper.close()
            if owner is not None:
                owner.close()
            if dist.is_initialized():
                dist.destroy_process_group()
            report['resources_after_close']=dict(rng_cleared=wrapper is None or wrapper.generators is None,
                owner_cache_cleared=owner is None or not owner.proposer.cached_tokens,
                context_cleared=get_context().slot_mapping is None,distributed_cleared=not dist.is_initialized())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--gpu',action='store_true')
    mode.add_argument('--cpu-self-test',action='store_true')
    parser.add_argument('--model',help='optional real Qwen3 draft directory; core-only if omitted')
    parser.add_argument('--draft-numerical-backend',choices=('default','consistent'),default='default')
    parser.add_argument('--output',required=True,help='new JSON file, never overwritten')
    args=parser.parse_args()
    with Path(args.output).open('x',encoding='utf-8') as output:
        report=dict(status='RUNNING',gpu_tests_run=False,guards=[],fixed_N=N,fixed_threshold=THRESHOLD)
        def guard(stage):
            from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
            require(bool(os.environ.get('NANOVLLM_BENCH_GPU_INDEX')), 'explicit approved physical GPU index required')
            require(bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'explicit approved complete GPU UUID required')
            snapshot=ensure_gpu1_available()
            selected=dict(index=snapshot['gpu']['index'],uuid=snapshot['gpu']['uuid'])
            require(report.get('selected_gpu',selected)==selected,'selected GPU changed during test')
            report['selected_gpu']=selected
            report['guards'].append(dict(stage=stage,**snapshot))
        try:
            if args.gpu:
                guard('before_cuda_initialization')
            elif args.model:
                parser.error('--model requires --gpu')
            import torch
            torch.set_num_threads(1)
            ss=load_core()
            device=torch.device('cuda:0' if args.gpu else 'cpu')
            report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'nanovllm').rglob('*.py'))}
            report['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            report['torch']=torch.__version__
            report['probability']=probability_checks(torch,ss,device)
            if args.gpu:
                report['gpu_tests_run']=True
                report['gpu_name']=torch.cuda.get_device_name(device)
                report['gpu_uuid']=report['selected_gpu']['uuid']
            if args.model:
                with torch.inference_mode():
                    wrapper_checks(torch,ss,args,guard,report)
            report['status']='PASS'
        except BaseException:
            report['status']='FAIL'
            report['traceback']=traceback.format_exc()
        finally:
            json.dump(report,output,indent=2)
            output.write('\n')
        print(json.dumps({key:report[key] for key in ('status','gpu_tests_run','probability','traceback') if key in report},indent=2))
        return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
