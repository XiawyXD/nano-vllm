"""CPU contracts for mixed host KV and native draft synchronization.

Real CPU KV values + delayed-copy tickets exercise ownership; they do not
validate CUDA ordering, BF16 accuracy, GPU performance or production recovery.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tempfile
import random
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.regression.check_planned_engine import make_engine, FakePlanRunner
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.proposer import DraftProposer, validate_tokenizers
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.tiered_cache import HostPrefixCache
from nanovllm.sampling_params import SamplingParams
from nanovllm.server import _sampling_params, _openai_params, RequestManager


class DelayedHostCache(HostPrefixCache):
    """Mock async completion separately from consumer stream dependencies."""
    asynchronous = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inflight = {}
        self.waited = set()
        self.ready = set()

    def available_slots(self):
        return 4 - len(self.inflight)

    def load(self, key, bid):
        super().load(key, bid)
        self.inflight[bid] = self.pages[key]

    def wait_for_blocks(self, block_ids):
        self.waited.update(set(block_ids) & self.inflight.keys())

    def reap(self, blocking=False):
        ready = set(self.inflight) if blocking else self.ready & self.inflight.keys()
        for bid in ready:
            del self.inflight[bid]
        self.ready.difference_update(ready)
        return ready


class PhysicalPlanRunner(FakePlanRunner):
    def __init__(self, kv, host):
        super().__init__(kv.shape[3])
        self.kv = kv
        self.host = host

    def call(self, method, plan, sequences):
        for item in plan.items:
            if getattr(self.host, "asynchronous", False):
                consumed = set(item.block_table) & self.host.inflight.keys()
                assert consumed <= self.host.waited, "forward consumed an unfenced host copy"
            for position in range(item.start_pos):
                bid, offset = item.block_table[position // self.block_size], position % self.block_size
                self.cache[bid, offset] = int(self.kv[0, 0, bid, offset, 0, 0])
        try:
            return super().call(method, plan, sequences)
        finally:
            for item in plan.items:
                for offset, token in enumerate(item.input_token_ids, item.start_pos):
                    bid = item.block_table[offset // self.block_size]
                    self.kv[:, :, bid, offset % self.block_size].fill_(token)


def cached_engine(async_copy=False):
    engine = make_engine(blocks=6, block_size=2, budget=2, chunk=2, policy="off")
    kv = torch.zeros(2, 2, 6, 2, 1, 2)
    cls = DelayedHostCache if async_copy else HostPrefixCache
    cache = cls(kv, 4096, "mixed-test", min_restore_tokens=2)
    engine.scheduler.block_manager.host_cache = cache
    engine.model_runner = PhysicalPlanRunner(kv, cache)
    return engine, kv, cache


def seed_and_evict(manager, kv, tokens=(1, 2, 3, 4, 9)):
    seq = Sequence(list(tokens))
    manager.allocate(seq, 0)
    for index, token in enumerate(tokens):
        kv[:, :, seq.block_table[index // manager.block_size], index % manager.block_size].fill_(token)
    seq.num_scheduled_tokens = len(tokens)
    manager.hash_blocks(seq)
    manager.deallocate(seq)
    # All physical pages are reused, so any later prefix must come from CPU.
    eviction = Sequence(list(range(100, 100 + len(manager.blocks) * manager.block_size)))
    manager.allocate(eviction, 0)
    manager.flush_host_stores()
    kv.fill_(-100)
    manager.deallocate(eviction)


def assert_drained(test, manager):
    test.assertFalse(manager.used_block_ids)
    test.assertFalse(manager._reservations)
    test.assertFalse(manager._host_load_leases)
    test.assertFalse(manager._pending_host_stores)
    test.assertEqual(set(manager.free_block_ids), set(range(len(manager.blocks))))
    test.assertEqual(len(manager.free_block_ids), len(manager.blocks))
    test.assertTrue(all(b.ref_count == 0 for b in manager.blocks))


class MixedCacheTests(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_mixed_restore_real_values_reduces_prefill(self):
        for asynchronous in (False, True):
            engine, kv, cache = cached_engine(asynchronous)
            manager = engine.scheduler.block_manager
            seed_and_evict(manager, kv)
            seq = engine.add_request([1, 2, 3, 4, 7], SamplingParams(temperature=0, max_tokens=3))
            while not engine.is_finished():
                engine.step()
            self.assertEqual(engine.model_runner.plans[0].items[0].start_pos, 4)
            self.assertEqual(engine.scheduler.total_prefill_tokens, 1)
            self.assertEqual(seq.completion_token_ids, [2, 3, 1])
            self.assertEqual(cache.counters["restore_submitted_tokens"], 4)
            manager.reap_host_loads(blocking=True)
            assert_drained(self, manager)

    def test_progressive_restore_when_full_prompt_cannot_fit_now(self):
        engine, kv, cache = cached_engine()
        manager = engine.scheduler.block_manager
        seed_and_evict(manager, kv)
        holder = Sequence([80] * 8)
        manager.allocate(holder, 0)  # Only two free pages remain.
        seq = Sequence([1, 2, 3, 4, 7, 8, 9, 10, 11])
        self.assertEqual(manager.restore_host_prefix(seq), 0)
        self.assertEqual(manager.restore_host_prefix(seq, progressive=True), 1)
        match = manager.probe_prefix(seq)
        self.assertEqual(match.cached_tokens, 2)
        reservation = manager.reserve_to(seq, 4, match)
        self.assertEqual(reservation.start_pos, 2)
        manager.rollback(reservation)
        manager.deallocate(holder)
        assert_drained(self, manager)

    def test_cancel_keeps_copy_lease_until_completion(self):
        engine, kv, cache = cached_engine(True)
        manager = engine.scheduler.block_manager
        seed_and_evict(manager, kv)
        seq = engine.add_request([1, 2, 3, 4, 7], SamplingParams(temperature=0, max_tokens=4))
        engine.step()
        pending = set(cache.inflight)
        self.assertTrue(pending)
        manager.deallocate(seq)
        self.assertTrue(pending <= manager.used_block_ids)
        self.assertTrue(all(manager.blocks[bid].ref_count == 1 for bid in pending))
        self.assertTrue(engine.cancel_request(seq.seq_id))
        assert_drained(self, manager)

    def test_failed_forward_rolls_back_but_does_not_free_copy_destination(self):
        engine, kv, cache = cached_engine(True)
        manager = engine.scheduler.block_manager
        seed_and_evict(manager, kv)
        seq = engine.add_request([1, 2, 3, 4, 7], SamplingParams(temperature=0, max_tokens=4))
        engine.model_runner.fail_after_write = True
        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
            engine.step()
        self.assertFalse(seq.block_table)
        self.assertFalse(manager._reservations)
        self.assertTrue(manager._host_load_leases)
        engine.cancel_request(seq.seq_id)
        assert_drained(self, manager)

    def test_failed_restore_does_not_publish_partial_page(self):
        engine, kv, cache = cached_engine()
        manager = engine.scheduler.block_manager
        seed_and_evict(manager, kv)
        with patch.object(cache, "load", side_effect=RuntimeError("copy failed")):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                manager.restore_host_prefix(Sequence([1, 2, 3, 4, 7]), progressive=True)
        self.assertFalse(manager.hash_to_block_id)
        assert_drained(self, manager)

    def test_pending_unrelated_copies_cannot_block_all_prefix_admission(self):
        engine, kv, cache = cached_engine(True)
        manager = engine.scheduler.block_manager
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])
        manager.allocate(seq, 0)
        for position, token in enumerate(seq.token_ids):
            kv[:, :, seq.block_table[position // 2], position % 2].fill_(token)
        seq.num_scheduled_tokens = len(seq)
        manager.hash_blocks(seq)
        manager.deallocate(seq)
        # The two non-prefix pages hold unrelated pending transfers. Four free
        # GPU-hit pages plus a new query page cannot fit until those leases end.
        for _ in range(2):
            bid = manager._allocate_block(offload=False)
            manager._host_load_leases.add(bid)
            cache.inflight[bid] = torch.zeros_like(kv[:, :, bid])
        request = engine.add_request(seq.token_ids, SamplingParams(temperature=0, max_tokens=1))
        engine.step()
        self.assertTrue(request.is_finished)
        self.assertEqual(cache.counters["mixed_progress_waits"], 1)
        assert_drained(self, manager)

    def test_shared_prefix_pressure_cancel_and_fair_progress(self):
        for asynchronous in (False, True):
            engine, kv, cache = cached_engine(asynchronous)
            manager = engine.scheduler.block_manager
            seed_and_evict(manager, kv)
            params = SamplingParams(temperature=0, max_tokens=3)
            seqs = [engine.add_request([1, 2, 3, 4, 5 + i], params) for i in range(3)]
            for step in range(200):
                if engine.is_finished():
                    break
                engine.step()
                if step == 1:
                    engine.cancel_request(seqs[-1].seq_id)
                # Outstanding copies can finish independently between rounds.
                if asynchronous:
                    cache.ready.update(cache.inflight)
            self.assertTrue(engine.is_finished(), "mixed host cache failed to make bounded progress")
            manager.reap_host_loads(blocking=True)
            assert_drained(self, manager)

    def test_planner_defers_d2h_until_before_kv_overwrite(self):
        engine, kv, cache = cached_engine()
        manager = engine.scheduler.block_manager
        seed = Sequence(list(range(1, 13)))
        manager.allocate(seed, 0)
        for position, token in enumerate(seed.token_ids):
            kv[:, :, seed.block_table[position // 2], position % 2].fill_(token)
        seed.num_scheduled_tokens = len(seed)
        manager.hash_blocks(seed)
        old_key = manager.blocks[seed.block_table[0]].host_key
        manager.deallocate(seed)
        planning = False
        original_schedule = engine.scheduler.schedule_plan
        original_put = cache.put

        def schedule(*args, **kwargs):
            nonlocal planning
            planning = True
            try:
                return original_schedule(*args, **kwargs)
            finally:
                planning = False

        def put(*args):
            self.assertFalse(planning, "CPU planner issued a D2H copy")
            return original_put(*args)

        engine.scheduler.schedule_plan = schedule
        cache.put = put
        request = engine.add_request([50, 51, 52], SamplingParams(temperature=0, max_tokens=1))
        while not engine.is_finished():
            engine.step()
        self.assertTrue(request.is_finished)
        self.assertEqual(cache.pages[old_key][0, 0, :, 0, 0].tolist(), [1., 2.])
        assert_drained(self, manager)


class HistoryForward:
    """Numerical stand-in whose predictions depend on the entire cached history."""
    def __init__(self):
        self.physical = []
        self.calls = []
        self.fail = False

    def __call__(self, tokens, start):
        if self.fail:
            raise RuntimeError("draft write failed")
        self.physical[start:start + len(tokens)] = tokens
        self.calls.append((start, tuple(tokens)))
        return sum(self.physical[:start + len(tokens)]) % 17


class CPUDraftRunner:
    def __init__(self, rule=lambda token: token % 3 + 1):
        self.request_id = None
        self.proposer = DraftProposer(lambda tokens, start: rule(tokens[-1]), 256, 8)

    def propose(self, request_id, tokens, limit):
        if request_id != self.request_id:
            self.proposer.clear()
            self.request_id = request_id
        return self.proposer.propose(tokens, limit)

    def synchronize(self, request_id, tokens):
        if request_id == self.request_id:
            self.proposer.synchronize(tokens)

    def forget(self, request_id):
        if request_id == self.request_id:
            self.proposer.clear()
            self.request_id = None


def draft_engine(rule=lambda token: token % 3 + 1):
    engine = make_engine(blocks=16, block_size=2, budget=8, chunk=4, seqs=1)
    engine.config.draft_model = "cpu-fake"
    draft = CPUDraftRunner(rule)
    engine.model_runner.draft_propose = draft.propose
    engine.model_runner.draft_synchronize = draft.synchronize
    engine.model_runner.draft_forget = draft.forget
    return engine, draft


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_legacy_and_explicit_methods(self):
        self.assertFalse(SamplingParams().use_speculative)
        self.assertTrue(SamplingParams(speculative_ngram_size=2).use_speculative)
        self.assertFalse(SamplingParams(speculative_ngram_size=2, speculative_method="off").use_speculative)
        self.assertTrue(SamplingParams(speculative_method="draft_model").use_speculative)
        for kwargs in (dict(speculative_method="bad"), dict(speculative_method="ngram"),
                       dict(speculative_method="draft_model", speculative_ngram_size=1)):
            with self.assertRaises(ValueError):
                SamplingParams(**kwargs)

    def test_rejection_sync_preserves_only_true_prefix(self):
        for accepted in (0, 1, 2, 3, 4):
            forward = HistoryForward()
            proposer = DraftProposer(forward, 64, 2)
            original = [2, 5, 7]
            draft = proposer.propose(original, 4)
            committed = original + draft[:accepted] + [99]
            proposer.synchronize(committed)
            self.assertEqual(proposer.cached_tokens, committed[:len(proposer.cached_tokens)])
            next_draft = proposer.propose(committed, 3)
            fresh = DraftProposer(HistoryForward(), 64, 2).propose(committed, 3)
            self.assertEqual(next_draft, fresh)
            self.assertEqual(original, [2, 5, 7])

    def test_draft_capacity_falls_back_without_overwrite(self):
        proposer = DraftProposer(HistoryForward(), 3, 2)
        self.assertEqual(len(proposer.propose([1, 2, 3], 8)), 1)
        self.assertEqual(proposer.propose([1, 2, 3, 4], 1), [])
        self.assertFalse(proposer.cached_tokens)
        self.assertEqual(proposer.counters["capacity_fallbacks"], 1)

    def test_failed_draft_invalidates_logical_kv(self):
        forward = HistoryForward()
        proposer = DraftProposer(forward, 32, 2)
        proposer.propose([2, 3, 5], 3)
        forward.fail = True
        with self.assertRaisesRegex(RuntimeError, "draft write failed"):
            proposer.propose([2, 3, 5, 99], 3)
        self.assertFalse(proposer.cached_tokens)

    def test_end_to_end_all_partial_and_zero_acceptance(self):
        for rule in (lambda token: token % 3 + 1, lambda token: 3, lambda token: 99):
            engine, draft = draft_engine(rule)
            seq = engine.add_request([1, 2, 3], SamplingParams(temperature=0, max_tokens=9,
                speculative_method="draft_model", speculative_max_draft_tokens=4))
            while not engine.is_finished():
                engine.step()
                self.assertEqual(draft.proposer.cached_tokens,
                                 seq.token_ids[:len(draft.proposer.cached_tokens)])
            self.assertEqual(seq.completion_token_ids, [1, 2, 3, 1, 2, 3, 1, 2, 3])
            self.assertTrue(any(plan.has_verification for plan in engine.model_runner.plans))
            self.assertFalse(draft.proposer.cached_tokens)
            assert_drained(self, engine.scheduler.block_manager)

    def test_stopping_and_length_one(self):
        for kwargs in (dict(max_tokens=1), dict(max_tokens=9, stop="23")):
            engine, draft = draft_engine()
            result = engine.generate([[1, 2, 3]], SamplingParams(temperature=0,
                speculative_method="draft_model", **kwargs), use_tqdm=False)
            expected = [1] if kwargs["max_tokens"] == 1 else [1, 2, 3]
            self.assertEqual(result[0]["token_ids"], expected)
            self.assertFalse(draft.proposer.cached_tokens)
            assert_drained(self, engine.scheduler.block_manager)

    def test_single_request_and_sequential_isolation(self):
        engine, draft = draft_engine()
        params = SamplingParams(temperature=0, max_tokens=4, speculative_method="draft_model")
        with self.assertRaisesRegex(ValueError, "exactly one prompt"):
            engine.generate([[1], [2]], params, use_tqdm=False)
        self.assertTrue(engine.is_finished())
        seq = engine.add_request([1], params)
        with self.assertRaisesRegex(ValueError, "one unfinished"):
            engine.add_request([2], params)
        engine.step()
        engine.step()
        engine.cancel_request(seq.seq_id)
        self.assertFalse(draft.proposer.cached_tokens)
        output = engine.generate([[3]], params, use_tqdm=False)
        self.assertEqual(output[0]["token_ids"], [1, 2, 3, 1])
        assert_drained(self, engine.scheduler.block_manager)

    def test_forward_failure_and_generator_close_clear_draft(self):
        engine, draft = draft_engine()
        params = SamplingParams(temperature=0, max_tokens=12, speculative_method="draft_model")
        seq = engine.add_request([1], params)
        engine.step()
        engine.model_runner.fail_after_write = True
        with self.assertRaises(RuntimeError):
            engine.step()
        self.assertFalse(draft.proposer.cached_tokens)
        engine.cancel_request(seq.seq_id)
        assert_drained(self, engine.scheduler.block_manager)
        engine.model_runner.fail_after_write = False
        generator = engine.generate_stream([2], params)
        next(generator)
        next(generator)
        generator.close()
        self.assertFalse(draft.proposer.cached_tokens)
        assert_drained(self, engine.scheduler.block_manager)

    def test_greedy_missing_config_and_http_guards(self):
        engine = make_engine()
        with self.assertRaisesRegex(ValueError, "requires Config.draft_model"):
            engine.add_request([1], SamplingParams(temperature=0, speculative_method="draft_model"))
        for kwargs in (dict(temperature=1), dict(temperature=0, top_k=1),
                       dict(temperature=0, top_p=.9), dict(temperature=0, repetition_penalty=1.1)):
            with self.assertRaises(ValueError):
                engine.add_request([1], SamplingParams(speculative_method="draft_model", **kwargs))
        for parser in (_sampling_params, _openai_params):
            with self.assertRaisesRegex(ValueError, "Python API"):
                parser(dict(speculative_method="draft_model"))
        engine, _ = draft_engine()
        with self.assertRaisesRegex(ValueError, "Python-only"):
            RequestManager(engine)

    def test_tokenizer_mapping_and_special_tokens(self):
        def tokenizer(vocab=None, eos=2):
            return SimpleNamespace(get_vocab=lambda: vocab or {"a": 1, "b": 2},
                                   eos_token_id=eos, bos_token_id=None, pad_token_id=0,
                                   all_special_ids=[0, eos])
        validate_tokenizers(tokenizer(), tokenizer())
        for other in (tokenizer({"a": 2, "b": 1}), tokenizer(eos=3)):
            with self.assertRaises(ValueError):
                validate_tokenizers(tokenizer(), other)

    def test_config_boundaries_without_gpu(self):
        from transformers import Qwen3Config
        with tempfile.TemporaryDirectory() as directory:
            Qwen3Config().save_pretrained(directory)
            base = dict(draft_model=directory, batching_mode="mixed", speculative_policy="fixed", max_num_seqs=1)
            Config(directory, **base)
            Config(directory, batching_mode="mixed", host_kv_cache_bytes=1024)
            for override in (dict(max_num_seqs=2), dict(tensor_parallel_size=2),
                             dict(batching_mode="separate"), dict(speculative_policy="adaptive"),
                             dict(host_kv_cache_bytes=1024)):
                with self.assertRaises(ValueError):
                    Config(directory, **(base | override))

    def test_native_forward_metadata_and_exception_context_cleanup(self):
        from nanovllm.engine.draft_runner import DraftRunner
        from nanovllm.utils.context import get_context

        class TinyModel:
            def __init__(self):
                self.kv = torch.zeros(32, dtype=torch.long)
                self.fail = False

            def __call__(self, ids, positions):
                context = get_context()
                assert context.is_prefill
                assert context.max_seqlen_q == len(ids)
                assert context.max_seqlen_k == int(positions[-1]) + 1
                assert torch.equal(context.slot_mapping, positions.to(torch.int32))
                if self.fail:
                    raise RuntimeError("native forward failed")
                self.kv[positions] = ids
                return int(self.kv[:context.max_seqlen_k].sum()) % 17

            def compute_logits(self, value):
                return torch.nn.functional.one_hot(torch.tensor([value]), 17).float()

        runner = DraftRunner.__new__(DraftRunner)
        runner.capacity, runner.chunk_size, runner.request_id = 32, 3, None
        runner.counters = Counter()
        runner.device = torch.device("cpu")
        runner.block_table = torch.arange(16, dtype=torch.int32).unsqueeze(0)
        runner.model = TinyModel()
        runner.proposer = DraftProposer(runner._forward, 32, 3)
        draft = runner.propose(1, [2, 5, 7], 4)
        expected = DraftProposer(HistoryForward(), 32, 3).propose([2, 5, 7], 4)
        self.assertEqual(draft, expected)
        self.assertIsNone(get_context().slot_mapping)
        runner.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "native forward failed"):
            runner.propose(1, [2, 5, 7, 99], 4)
        self.assertFalse(runner.proposer.cached_tokens)
        self.assertIsNone(get_context().slot_mapping)
        with patch("torch.distributed.destroy_process_group", side_effect=AssertionError("borrowed group")):
            runner.close()

    def test_randomized_rejection_matches_fresh_draft(self):
        rng = random.Random(23)
        forward = HistoryForward()
        proposer = DraftProposer(forward, 2048, 7)
        history = [2, 5, 7]
        for _ in range(100):
            length = rng.randint(1, 8)
            draft = proposer.propose(history, length)
            fresh = DraftProposer(HistoryForward(), 2048, 7).propose(history, length)
            self.assertEqual(draft, fresh)
            accepted = rng.randint(0, length)
            history += draft[:accepted] + [rng.randint(0, 16)]
            proposer.synchronize(history)
            self.assertEqual(proposer.cached_tokens, history[:len(proposer.cached_tokens)])

    def test_native_proposals_run_before_cpu_reservations(self):
        engine, draft = draft_engine()
        original = engine.model_runner.draft_propose

        def propose(*args):
            self.assertFalse(engine.scheduler._active_items)
            self.assertFalse(engine.scheduler.block_manager._reservations)
            return original(*args)

        engine.model_runner.draft_propose = propose
        result = engine.generate([[1]], SamplingParams(temperature=0, max_tokens=8,
            speculative_method="draft_model"), use_tqdm=False)
        self.assertEqual(result[0]["token_ids"], [2, 3, 1, 2, 3, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
