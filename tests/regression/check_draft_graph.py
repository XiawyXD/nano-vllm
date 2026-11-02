"""CPU state contracts; optional guarded real-GPU eager/graph comparison.

CPU: python -m tests.regression.check_draft_graph
GPU: python -m tests.regression.check_draft_graph --gpu --model DRAFT --output NEW_DIR
GPU mode must only be started after the task owner grants the GPU lease.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.proposer import DraftProposer


class HistoryForward:
    def __init__(self):
        self.physical = []

    def __call__(self, tokens, start):
        self.physical[start:start + len(tokens)] = tokens
        return sum(self.physical[:start + len(tokens)]) % 17


class Rollout:
    def __init__(self, forward, chunk):
        self.forward, self.chunk = forward, chunk

    def __call__(self, suffix, start, limit):
        for offset in range(0, len(suffix), self.chunk):
            prediction = self.forward(suffix[offset:offset + self.chunk], start + offset)
        result = [prediction]
        for i in range(limit - 1):
            result.append(self.forward([result[-1]], start + len(suffix) + i))
        return result


class GraphTests(unittest.TestCase):
    def test_execution_selection_and_enforce_eager(self):
        from nanovllm.engine.draft_runner import DraftRunner
        for requested, enforce, effective in (("eager", False, "eager"),
                ("graph", False, "graph"), ("graph", True, "eager")):
            config = SimpleNamespace(draft_model="unused", max_model_len=32,
                kvcache_block_size=16, max_num_batched_tokens=8,
                draft_execution=requested, enforce_eager=enforce)
            runner = DraftRunner.__new__(DraftRunner)
            with patch("torch.distributed.is_initialized", return_value=True), \
                    patch("torch.distributed.get_world_size", return_value=1), \
                    patch("nanovllm.engine.draft_runner.AutoConfig.from_pretrained",
                          side_effect=RuntimeError("stop before GPU")):
                with self.assertRaisesRegex(RuntimeError, "stop before GPU"):
                    runner.__init__(config)
            self.assertEqual(runner.execution, effective)
            self.assertIsNone(runner.graph)

    def test_random_rejection_and_chunked_suffix(self):
        for chunk in (1, 2, 7):
            forward = HistoryForward()
            proposer = DraftProposer(forward, 2048, chunk, Rollout(forward, chunk))
            rng, history = random.Random(23), [2, 5, 7]
            for _ in range(100):
                k = rng.choice((1, 2, 4, 8))
                result = proposer.propose(history, k)
                self.assertEqual(result, DraftProposer(HistoryForward(), 2048, chunk).propose(history, k))
                history += result[:rng.randint(0, k)] + [rng.randrange(17)]
                proposer.synchronize(history)
                self.assertEqual(proposer.cached_tokens, history[:len(proposer.cached_tokens)])

    def test_capacity_and_zero_limit(self):
        forward = HistoryForward()
        proposer = DraftProposer(forward, 3, 2, Rollout(forward, 2))
        self.assertEqual(proposer.propose([], 0), [])
        self.assertEqual(len(proposer.propose([1, 2, 3], 8)), 1)
        self.assertEqual(proposer.cached_tokens, [1, 2, 3])
        self.assertEqual(proposer.propose([1, 2, 3, 4], 1), [])
        self.assertEqual(proposer.cached_tokens, [])

    def test_invalid_and_failed_rollout_clear_logical_state(self):
        def fail(*args):
            raise RuntimeError("partial graph write")
        for result in ([], [1], [1, -1], [1, True], [1, 2.0], (1, 2), None):
            proposer = DraftProposer(HistoryForward(), 32, 2, lambda *args: result)
            proposer.cached_tokens = [1]
            with self.assertRaises(ValueError):
                proposer.propose([1, 2], 2)
            self.assertEqual(proposer.cached_tokens, [])
        proposer.rollout = fail
        proposer.cached_tokens = [1]
        with self.assertRaisesRegex(RuntimeError, "partial graph write"):
            proposer.propose([1, 2], 2)
        self.assertEqual(proposer.cached_tokens, [])

    def make_runner(self):
        from nanovllm.engine.draft_runner import DraftRunner
        from nanovllm.engine.draft_graph import DraftTokenGraph, DraftSuffixGraph
        from nanovllm.utils.context import get_context

        class TinyModel:
            def __init__(self):
                self.kv = torch.zeros(32, dtype=torch.long)
                self.fail = False

            def __call__(self, ids, positions):
                context = get_context()
                if self.fail:
                    raise RuntimeError("graph failed")
                assert context.is_prefill
                assert torch.equal(context.slot_mapping, positions.to(torch.int32))
                end = int(context.cu_seqlens_k[-1])
                assert end == int(positions[-1]) + 1
                assert end <= context.max_seqlen_k
                self.kv[positions] = ids
                return self.kv[:end].sum() % 17

            def compute_logits(self, value):
                return torch.nn.functional.one_hot(value.reshape(1), 17).float()

        runner = DraftRunner.__new__(DraftRunner)
        runner.capacity, runner.chunk_size, runner.request_id = 32, 3, None
        runner.counters = Counter()
        runner.device = torch.device("cpu")
        runner.block_table = torch.arange(16, dtype=torch.int32).unsqueeze(0)
        runner.model = TinyModel()
        runner.graph_max_tokens = 8
        graph = DraftTokenGraph.__new__(DraftTokenGraph)
        graph.model, graph.block_table = runner.model, runner.block_table
        graph.capacity, graph.max_tokens = 32, 8
        graph.state = torch.zeros(3, dtype=torch.int64)
        graph.host_state = torch.zeros_like(graph.state)
        graph.ids, graph.positions, graph.index = graph.state[:1], graph.state[1:2], graph.state[2:3]
        graph.slot = torch.zeros(1, dtype=torch.int32)
        graph.cu_q = torch.tensor([0, 1], dtype=torch.int32)
        graph.cu_k = torch.tensor([0, 1], dtype=torch.int32)
        graph.outputs = torch.empty(8, dtype=torch.int64)
        graph.graph = SimpleNamespace(replay=graph._step)
        runner.graph = graph
        suffix = DraftSuffixGraph.__new__(DraftSuffixGraph)
        suffix.model, suffix.block_table = runner.model, runner.block_table
        suffix.capacity, suffix.width = 32, 2
        suffix.state = torch.zeros(3, dtype=torch.int64)
        suffix.host_state = torch.zeros_like(suffix.state)
        suffix.ids, suffix.start = suffix.state[:2], suffix.state[2:]
        suffix.offsets = torch.arange(2, dtype=torch.int64)
        suffix.positions = torch.empty_like(suffix.offsets)
        suffix.slot = torch.empty(2, dtype=torch.int32)
        suffix.cu_q = torch.tensor([0, 2], dtype=torch.int32)
        suffix.cu_k = torch.tensor([0, 2], dtype=torch.int32)
        suffix.prediction = torch.empty(1, dtype=torch.int64)
        suffix.graph = SimpleNamespace(replay=suffix._step)
        runner.suffix_graphs = {2: suffix}
        runner.proposer = DraftProposer(runner._forward, 32, 3, runner._rollout)
        return runner

    def make_prefill_runner(self, tails=False):
        from nanovllm.engine.draft_graph import DraftSuffixGraph
        runner = self.make_runner()
        runner.prefill_graphs = {}
        runner.prefill_replay_stream = None
        specs = [(3,3,False), (3,3,True), (3,6,False), (3,6,True)]
        if tails: specs += [(1,4,True),(1,7,True)]
        for width, end, predict in specs:
            graph = DraftSuffixGraph.__new__(DraftSuffixGraph)
            graph.model, graph.block_table = runner.model, runner.block_table
            graph.capacity, graph.width, graph.max_key_length = 32, width, end
            graph.needs_prediction = predict
            graph.state = torch.zeros(width+1, dtype=torch.int64)
            graph.host_state = torch.zeros_like(graph.state)
            graph.ids, graph.start = graph.state[:width], graph.state[width:]
            graph.offsets = torch.arange(width, dtype=torch.int64)
            graph.positions = torch.empty_like(graph.offsets)
            graph.slot = torch.empty(width, dtype=torch.int32)
            graph.cu_q = torch.tensor([0,width], dtype=torch.int32)
            graph.cu_k = torch.tensor([0,width], dtype=torch.int32)
            graph.prediction = torch.empty(1, dtype=torch.int64)
            graph.graph = SimpleNamespace(replay=graph._step)
            runner.prefill_graphs[width,end,predict] = graph
        return runner

    def test_prefill_graph_exact_shape_and_kv_only(self):
        for k in (1,2,4,8):
            runner = self.make_prefill_runner()
            history = [2,5,7,3,9,1,4]
            with patch.object(runner.model, "compute_logits", wraps=runner.model.compute_logits) as head:
                result = runner.propose(1, history, k)
            self.assertEqual(result, DraftProposer(HistoryForward(),32,3).propose(history,k))
            self.assertEqual(head.call_count,k)
            self.assertEqual(runner.counters["prefill_graph_replays"],2)
            self.assertEqual(runner.counters["eager_suffix_forward_calls"],1)
            self.assertEqual(runner.counters["skipped_prefix_prediction_calls"],2)
            self.assertEqual(runner.counters["host_result_syncs"],1)
            self.assertTrue(torch.equal(runner.model.kv[:len(runner.proposer.cached_tokens)],
                                        torch.tensor(runner.proposer.cached_tokens)))

    def test_prefill_prediction_graph_and_k_bound(self):
        runner = self.make_prefill_runner()
        history = [2,5,7,3,9,1]
        self.assertEqual(runner.propose(1,history,4),DraftProposer(HistoryForward(),32,3).propose(history,4))
        self.assertEqual(runner.counters["prefill_graph_replays"],2)
        self.assertEqual(runner.counters["eager_suffix_forward_calls"],0)
        graph=runner.prefill_graphs[3,3,True]
        with self.assertRaises(ValueError):graph.run([1,2,3],1)

    def test_prefill_graph_failure_and_close_fence(self):
        from unittest.mock import Mock
        runner = self.make_prefill_runner()
        runner.prefill_graphs[3,6,True].graph.replay=Mock(side_effect=RuntimeError("failed second prefix block"))
        with self.assertRaisesRegex(RuntimeError,"failed second prefix"):
            runner.propose(1,[2,5,7,3,9,1],4)
        self.assertEqual(runner.proposer.cached_tokens,[])
        stream=runner.prefill_replay_stream=Mock()
        runner.close()
        stream.synchronize.assert_called_once_with()
        self.assertEqual(runner.prefill_graphs,{})

    def test_prefill_tail_exact_k_and_no_eager(self):
        for n in (4,7):
            for k in (1,2,4,8):
                runner = self.make_prefill_runner(tails=True)
                history = [2,5,7,3,9,1,4][:n]
                actual = runner.propose(1,history,k)
                expected = DraftProposer(HistoryForward(),32,3).propose(history,k)
                self.assertEqual(actual,expected)
                self.assertEqual(runner.counters["eager_suffix_forward_calls"],0)
                self.assertEqual(runner.counters["prefill_tail_graph_replays"],1)
                self.assertEqual(runner.counters["host_result_syncs"],1)
                self.assertTrue(torch.equal(runner.model.kv[:len(runner.proposer.cached_tokens)],torch.tensor(runner.proposer.cached_tokens)))
                with self.assertRaises(ValueError):runner.prefill_graphs[1,n,True].run([1],n)

    def test_prefill_tail_default_and_invalid_combinations(self):
        from nanovllm.config import Config
        from nanovllm.engine.draft_runner import DraftRunner
        self.assertIs(Config.__dataclass_fields__["draft_prefill_tail_graph"].default,False)
        for flag,prefill,stochastic in [(1,True,False),(True,False,False),(True,True,True)]:
            cfg=SimpleNamespace(max_model_len=32,kvcache_block_size=16,max_num_batched_tokens=8,
                draft_prefill_tail_graph=flag,draft_prefill_graph=prefill,enable_stochastic_speculation=stochastic)
            with patch("torch.distributed.is_initialized",return_value=True),patch("torch.distributed.get_world_size",return_value=1):
                with self.assertRaisesRegex(ValueError,"draft_prefill_tail_graph"):DraftRunner(cfg)

    def test_prefill_option_is_default_off(self):
        from nanovllm.config import Config
        self.assertIs(Config.__dataclass_fields__["draft_prefill_graph"].default,False)

    def test_cold_prefix_only_final_chunk_computes_prediction(self):
        for k in (1, 2, 4, 8):
            runner = self.make_runner()
            history = [2, 5, 7, 3, 9, 1, 4]
            with patch.object(runner.model, "compute_logits", wraps=runner.model.compute_logits) as head:
                proposal = runner.propose(1, history, k)
            expected = DraftProposer(HistoryForward(), 32, 3).propose(history, k)
            self.assertEqual(proposal, expected)
            self.assertEqual(head.call_count, k)  # final cold chunk + K-1 Q1 steps
            self.assertEqual(runner.counters["skipped_prefix_prediction_calls"], 2)
            self.assertEqual(runner.counters["forwarded_tokens"], len(history) + k - 1)
            self.assertEqual(runner.counters["host_result_syncs"], 1)
            self.assertTrue(torch.equal(runner.model.kv[:len(runner.proposer.cached_tokens)],
                                        torch.tensor(runner.proposer.cached_tokens)))

    def test_full_acceptance_bonus_uses_q2_without_eager(self):
        for k in (1, 2, 4, 8):
            runner = self.make_runner()
            history = [2, 5, 7]
            proposal = runner.propose(1, history, k)
            for _ in range(2):
                history += proposal + [11]
                runner.synchronize(1, history)
                self.assertEqual(len(history) - len(runner.proposer.cached_tokens), 2)
                before = runner.counters.copy()
                proposal = runner.propose(1, history, k)
                self.assertEqual(proposal, DraftProposer(HistoryForward(), 32, 3).propose(history, k))
                self.assertEqual(runner.counters["eager_forward_calls"], before["eager_forward_calls"])
                self.assertEqual(runner.counters["graph_prefill_replays"] - before["graph_prefill_replays"], 1)
                self.assertEqual(runner.counters["host_result_syncs"] - before["host_result_syncs"], 1)

    def test_suffix_graph_bounds_capacity_and_failure(self):
        from nanovllm.utils.context import get_context
        runner = self.make_runner()
        suffix = runner.suffix_graphs[2]
        for tokens, start in (([1], 0), ([1, 2], -1), ([1, 2], 31)):
            with self.assertRaises(ValueError):
                suffix.run(tokens, start)
        # The first proposal writes 29 history tokens and one proposed input.
        history = [2] * 29
        proposal = runner.propose(1, history, 2)
        history += proposal + [7]
        runner.synchronize(1, history)
        self.assertEqual(len(history) - len(runner.proposer.cached_tokens), 2)
        self.assertEqual(runner.propose(1, history, 8), DraftProposer(HistoryForward(), 32, 3).propose(history, 8))
        runner.forget(1)
        runner.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "graph failed"):
            runner.propose(2, [2, 3], 4)
        self.assertEqual(runner.proposer.cached_tokens, [])
        self.assertIsNone(get_context().slot_mapping)

    def test_graph_metadata_reset_rejection_and_one_host_result(self):
        from nanovllm.utils.context import get_context
        for k in (1, 2, 4, 8):
            runner = self.make_runner()
            for history in ([2], [2, 5, 7, 3], [2, 5, 99], [2, 5, 99, 4]):
                before = runner.counters["host_result_syncs"]
                expected = DraftProposer(HistoryForward(), 32, 3).propose(list(history), k)
                self.assertEqual(runner.propose(1, list(history), k), expected)
                self.assertEqual(runner.counters["host_result_syncs"] - before, 1)
                self.assertIsNone(get_context().slot_mapping)

    def test_graph_generic_limit_fallback(self):
        runner = self.make_runner()
        self.assertEqual(runner.propose(1, [2], 9), DraftProposer(HistoryForward(), 32, 3).propose([2], 9))
        self.assertEqual(runner.counters["graph_limit_fallbacks"], 1)
        self.assertEqual(runner.counters["graph_replays"], 0)

    def test_graph_request_change_forget_and_failure(self):
        from nanovllm.utils.context import get_context
        runner = self.make_runner()
        runner.propose(1, [2, 5], 4)
        self.assertEqual(runner.propose(2, [7], 4), DraftProposer(HistoryForward(), 32, 3).propose([7], 4))
        runner.forget(1)
        self.assertTrue(runner.proposer.cached_tokens)
        runner.forget(2)
        self.assertEqual(runner.proposer.cached_tokens, [])
        runner.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "graph failed"):
            runner.propose(3, [3], 2)
        self.assertEqual(runner.proposer.cached_tokens, [])
        self.assertIsNone(get_context().slot_mapping)
        runner.close()
        self.assertIsNone(runner.graph)

    def test_graph_bounds_before_replay(self):
        graph = self.make_runner().graph
        for start, count in ((-1, 1), (0, 0), (0, 9), (32, 1), (31, 2)):
            with self.assertRaises(ValueError):
                graph.run(1, start, count)

    def test_graph_last_capacity_slot(self):
        runner = self.make_runner()
        history = [2] * 32
        self.assertEqual(runner.propose(1, history, 8), DraftProposer(HistoryForward(), 32, 3).propose(history, 8))
        self.assertEqual(runner.propose(1, history + [1], 1), [])

    def test_rollout_engine_eos_stop_and_length_cleanup(self):
        from tests.regression.check_cache_draft import draft_engine, assert_drained
        from nanovllm.sampling_params import SamplingParams
        from nanovllm.engine.sequence import Sequence
        old = Sequence.block_size
        try:
            for eos, stop, maximum, expected in ((2, None, 9, [1, 2]),
                    (-1, "23", 9, [1, 2, 3]), (-1, None, 1, [1])):
                engine, runner = draft_engine()
                engine.config.eos = eos
                engine.scheduler.eos = eos
                runner.proposer.rollout = Rollout(runner.proposer.forward, runner.proposer.chunk_size)
                result = engine.generate([[1, 2, 3]], SamplingParams(temperature=0,
                    max_tokens=maximum, stop=stop, speculative_method="draft_model"), use_tqdm=False)
                self.assertEqual(result[0]["token_ids"], expected)
                self.assertEqual(runner.proposer.cached_tokens, [])
                assert_drained(self, engine.scheduler.block_manager)
        finally:
            Sequence.block_size = old


def gpu_check(args):
    # Guard precedes any CUDA initialization, model load, or distributed setup.
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="NOT_RUN", async_metadata=args.async_metadata,
        scope="draft-only eager versus graph; no target numerical or end-to-end speed claim")
    runner = None
    initialized = False
    try:
        report["gpu"] = ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
        import tempfile
        import time
        import torch.distributed as dist
        from nanovllm.engine.draft_runner import DraftRunner
        from nanovllm.utils.context import get_context
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", init_method="file://" + directory + "/store", rank=0, world_size=1)
            initialized = True
            cases = []
            # Includes 256-page boundaries, chunked prefill, capacity, all/partial/zero acceptance.
            for k in (1, 2, 4, 8):
                for n in (1, 7, 255, 256, 257, 264, 510, 512):
                    cases.append((k, [2 + i % 97 for i in range(n)]))
            results = {}
            for execution in ("eager", "graph"):
                ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
                runner = DraftRunner(SimpleNamespace(draft_model=args.model, max_model_len=512,
                    kvcache_block_size=256, max_num_batched_tokens=128, enforce_eager=False,
                    draft_execution=execution, draft_graph_max_tokens=8,
                    draft_async_metadata=args.async_metadata and execution=="graph",
                    draft_dtype=args.draft_dtype,
                    draft_quantization="awq" if args.draft_quantization=="awq" else None,
                    draft_quant_backend="marlin" if args.draft_quantization=="awq" else "triton",
                    marlin_library=args.marlin_library))
                assert runner.async_metadata is (args.async_metadata and execution=="graph")
                outputs = []
                for index, (k, original) in enumerate(cases):
                    history = original.copy()
                    rounds = []
                    for accepted in (0, k // 2, k):
                        proposal = runner.propose(index, history, k)
                        rounds.append(proposal)
                        history += proposal[:accepted] + [41]
                        runner.synchronize(index, history)
                        assert runner.proposer.cached_tokens == history[:len(runner.proposer.cached_tokens)]
                        assert get_context().slot_mapping is None
                    runner.forget(index)
                    assert not runner.proposer.cached_tokens
                    outputs.append(rounds)
                timings = []
                for k in (1, 2, 4, 8):
                    for repeat in range(8):
                        runner.forget(runner.request_id)
                        # Warm the actual prompt first, then measure a reusable prefix.
                        history = [2 + i % 97 for i in range(264)]
                        runner.propose(10000, history, k)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        for _ in range(20):
                            runner.propose(10000, history, k)
                        torch.cuda.synchronize()
                        timings.append(dict(k=k, repeat=repeat, ms_per_proposal=(time.perf_counter()-started)*1000/20))
                bonus_timings, bonus_outputs = [], []
                for k in (1, 2, 4, 8):
                    for repeat in range(5):
                        runner.forget(runner.request_id)
                        history = [2 + i % 97 for i in range(248)]
                        proposal = runner.propose(20000, history, k)
                        # Simulate committing every candidate plus the target's
                        # bonus. Every measured round must then consume Q2.
                        history += proposal + [41]
                        runner.synchronize(20000, history)
                        before = runner.counters.copy()
                        round_outputs = []
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        for _ in range(20):
                            assert len(history) - len(runner.proposer.cached_tokens) == 2
                            proposal = runner.propose(20000, history, k)
                            round_outputs.append(proposal)
                            history += proposal + [41]
                            runner.synchronize(20000, history)
                        torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter() - started) * 1000 / 20
                        delta = {key: value - before[key] for key, value in runner.counters.items()}
                        if execution == "graph":
                            assert delta.get("eager_forward_calls", 0) == 0
                            assert delta["graph_prefill_replays"] == 20
                            assert delta["host_result_syncs"] == 20
                        bonus_timings.append(dict(k=k, repeat=repeat, ms_per_round=elapsed_ms,
                            counter_delta=delta, initial_history_length=248,
                            final_history_length=len(history), rounds=20))
                        bonus_outputs.append(dict(k=k, repeat=repeat, proposals=round_outputs))
                results[execution] = dict(outputs=outputs, timings=timings,
                    full_accept_bonus_timings=bonus_timings, full_accept_bonus_outputs=bonus_outputs,
                    stats=runner.stats())
                runner.close()
                runner = None
                torch.cuda.empty_cache()
            equal = results["eager"]["outputs"] == results["graph"]["outputs"]
            bonus_equal = results["eager"]["full_accept_bonus_outputs"] == results["graph"]["full_accept_bonus_outputs"]
            report.update(status="PASS_SCOPED" if equal and bonus_equal else "REVIEW_REQUIRED",
                          equal=equal, full_accept_bonus_equal=bonus_equal, results=results)
    except Exception as error:
        report.update(status="NOT_RUN_GPU_UNAVAILABLE" if type(error).__name__ == "GPUUnavailable" else "FAIL",
                      error=f"{type(error).__name__}: {error}")
        import traceback
        report["traceback"] = traceback.format_exc()
    finally:
        if runner is not None:
            runner.close()
        if initialized:
            torch.distributed.destroy_process_group()
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0 if report["status"] == "PASS_SCOPED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--async-metadata", action="store_true",
                        help="Graph arm only: explicit pinned/event H2D; eager reference always OFF")
    parser.add_argument("--draft-dtype", choices=("checkpoint","float16"), default="checkpoint")
    parser.add_argument("--draft-quantization", choices=("none","awq"), default="none")
    parser.add_argument("--marlin-library")
    parser.add_argument("--output", type=Path)
    args, rest = parser.parse_known_args()
    if args.async_metadata and not args.gpu:
        parser.error("--async-metadata is a GPU graph test option; CPU tests do not exercise CUDA H2D")
    if args.gpu and args.draft_quantization=="awq" and not args.marlin_library:
        parser.error("AWQ graph test requires --marlin-library")
    if args.gpu:
        if not args.model or not args.output:
            parser.error("--gpu requires --model and a new --output directory")
        raise SystemExit(gpu_check(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    unittest.main(argv=[__file__] + rest)
