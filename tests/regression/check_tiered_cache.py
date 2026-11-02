"""CPU 数值页模拟 + 可选真实 CUDA 复制验收，不加载模型。

默认 unittest 不分配 GPU；--gpu 在 GPU guard 允许后用 CUDA tensor 跑同一套生命周期测试。
模型生成一致性仍需独立验收，不能用本测试代替。
"""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.tiered_cache import HostPrefixCache
from nanovllm.engine.packed_graph import PackedGraphs
from benchmarks.slo_goodput import evaluate

DEVICE = "cpu"


class TieredTests(unittest.TestCase):
    def setUp(self):
        self.old_size = Sequence.block_size
        Sequence.block_size = 2
        self.kv = torch.zeros(2, 2, 5, 2, 1, 2, device=DEVICE)
        self.manager = BlockManager(5, 2)
        self.cache = HostPrefixCache(self.kv, 4096, "test-model-config",
                                     min_restore_tokens=2)
        self.manager.host_cache = self.cache

    def tearDown(self):
        Sequence.block_size = self.old_size

    def seq(self, tokens):
        return Sequence(tokens)

    def commit(self, tokens, value):
        seq = self.seq(tokens)
        self.manager.allocate(seq, 0)
        for index, bid in enumerate(seq.block_table):
            self.kv[:, :, bid].fill_(value + index)
        seq.num_scheduled_tokens = len(tokens)
        self.manager.hash_blocks(seq)
        self.manager.deallocate(seq)
        return seq

    def evict(self):
        other = self.seq(list(range(100, 110)))
        self.manager.allocate(other, 0)
        self.kv.fill_(-5)
        self.manager.deallocate(other)

    def invariant(self):
        self.assertFalse(self.manager.used_block_ids)
        self.assertEqual(len(set(self.manager.free_block_ids)), 5)
        self.assertTrue(all(b.ref_count == 0 for b in self.manager.blocks))
        self.assertLessEqual(self.cache.stats()["resident_bytes"], self.cache.max_bytes)

    def test_eviction_restore_exact_pages(self):
        self.commit([1, 2, 3, 4, 5], 20)
        self.evict()
        seq = self.seq([1, 2, 3, 4, 9])
        self.assertEqual(self.manager.restore_host_prefix(seq), 2)
        self.invariant()
        hits = self.manager.can_allocate(seq)
        self.assertEqual(hits, 2)
        self.manager.allocate(seq, hits)
        for i, bid in enumerate(seq.block_table[:2]):
            self.assertTrue(torch.equal(self.kv[:, :, bid], torch.full_like(self.kv[:, :, bid], 20+i)))
        self.manager.deallocate(seq)
        self.invariant()

    def test_different_parent_never_hits(self):
        self.commit([1, 2, 3, 4, 5], 20)
        self.evict()
        self.assertEqual(self.manager.restore_host_prefix(self.seq([8, 9, 3, 4, 5])), 0)
        self.assertNotEqual(self.cache.key(None, [1, 2]),
                            HostPrefixCache(self.kv, 4096, "other").key(None, [1, 2]))
        self.invariant()

    def test_lru_capacity_and_copy_ownership(self):
        self.cache.capacity = 1
        key = self.cache.key(None, [1, 2])
        self.kv[:, :, 0].fill_(7)
        self.cache.put(key, 0)
        self.kv[:, :, 0].fill_(8)
        self.cache.load(key, 1)
        self.assertTrue((self.kv[:, :, 1] == 7).all())
        self.cache.put(self.cache.key(None, [3, 4]), 0)
        self.assertFalse(self.cache.contains(key))
        self.assertEqual(len(self.cache.pages), 1)

    def test_no_partial_or_uncommitted_backup(self):
        seq = self.seq([1, 2, 3])
        self.manager.allocate(seq, 0)
        self.manager.deallocate(seq)
        self.evict()
        self.assertFalse(self.cache.pages)
        self.invariant()

    def test_restore_failure_does_not_publish_or_leak(self):
        self.commit([1, 2, 3, 4, 5], 20)
        self.evict()
        with patch.object(self.cache, "load", side_effect=RuntimeError("copy failed")):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                self.manager.restore_host_prefix(self.seq([1, 2, 3, 4, 9]))
        self.assertFalse(self.manager.hash_to_block_id)
        self.invariant()

    def test_skip_when_capacity_insufficient(self):
        self.commit([1, 2, 3, 4, 5], 20)
        self.evict()
        seq = self.seq(list(range(20, 28)))
        self.manager.allocate(seq, 0)
        self.assertEqual(self.manager.restore_host_prefix(self.seq([1, 2, 3, 4, 5])), 0)
        self.manager.deallocate(seq)
        self.invariant()

    def test_budget_and_minimum_restore(self):
        self.commit([1, 2, 3, 4, 5], 20)
        self.evict()
        self.cache.min_restore_tokens = 10
        seq = self.seq([1, 2, 3, 4, 5])
        self.assertEqual(self.manager.restore_host_prefix(seq), 0)
        self.cache.min_restore_tokens = 2
        self.cache.max_restore_blocks = 1
        self.assertEqual(self.manager.restore_host_prefix(seq), 1)
        self.invariant()

    def test_restore_pins_later_gpu_hits_before_allocating(self):
        self.commit([1, 2, 3, 4, 5], 20)
        # 只将第一页移到 CPU，后面的 GPU 命中页仍存在。
        bid = self.manager.hash_to_block_id[self.manager.compute_hash([1, 2])]
        self.cache.put(self.manager.blocks[bid].host_key, bid)
        self.manager._remove_prefix_index(bid)
        # 把其它空白页占住，使回载路径真正面临受限的分配。
        seq = self.seq([1, 2, 3, 4, 8])
        self.assertEqual(self.manager.restore_host_prefix(seq), 1)
        self.assertEqual(self.manager.can_allocate(seq), 2)
        self.invariant()

    def test_scheduler_restore_then_cancel(self):
        from nanovllm.engine.scheduler import Scheduler
        config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=16,
            prefill_chunk_size=None, max_prefill_steps=0, eos=-1,
            kvcache_block_size=2, num_kvcache_blocks=5, enable_prefix_caching=True)
        scheduler = Scheduler(config)
        scheduler.block_manager = self.manager
        self.commit([1, 2, 3, 4, 5], 20)
        self.evict()
        seq = self.seq([1, 2, 3, 4, 9])
        scheduler.add(seq)
        scheduled, prefill = scheduler.schedule()
        self.assertEqual(scheduled, [seq])
        self.assertTrue(prefill)
        self.assertEqual(seq.num_cached_tokens, 4)
        self.assertEqual(seq.num_scheduled_tokens, 1)
        self.assertTrue(scheduler.cancel(seq))
        self.invariant()

    def test_invalid_config_rejected_before_model_load(self):
        import tempfile
        from nanovllm.config import Config
        with tempfile.TemporaryDirectory() as directory:
            for kwargs in (dict(host_kv_cache_bytes=-1),
                           dict(max_kvcache_blocks=0), dict(max_kvcache_blocks=True),
                           dict(max_kvcache_blocks=-1), dict(max_kvcache_blocks=1.5),
                           dict(host_kv_cache_bytes=1024, tensor_parallel_size=2),
                           dict(host_kv_cache_bytes=1024, enable_prefix_caching=False)):
                with self.assertRaises(ValueError):
                    Config(directory, **kwargs)


class PerformanceTests(unittest.TestCase):
    def test_frozen_graph_still_replays_known_shape(self):
        from nanovllm.engine.packed_graph import metadata_signature, clone_metadata
        from nanovllm.utils.context import Context, reset_context
        from unittest.mock import Mock
        graphs = PackedGraphs(allow_capture=False)
        context = Context()
        ids, pos = torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)
        graph, output = Mock(), torch.ones(2, 4)
        key = (tuple(ids.shape), ("test",), metadata_signature(context))
        graphs.entries[key] = (graph, ids.clone(), pos.clone(), clone_metadata(context), output)
        try:
            self.assertIs(graphs.run(None, ids, pos, context, ("test",)), output)
            graph.replay.assert_called_once()
            self.assertFalse(graphs.captured_last)
        finally:
            reset_context()

    def test_frozen_graph_unknown_shape_no_cuda_call(self):
        graphs = PackedGraphs(allow_capture=False)
        context = SimpleNamespace(batch_kind="DV_PACKED")
        # SimpleNamespace 不能作为 graph key；真实 Context 是 dataclass。
        from nanovllm.utils.context import Context
        context = Context()
        with patch("torch.cuda.CUDAGraph", side_effect=AssertionError("must not capture")):
            self.assertIsNone(graphs.run(None, torch.zeros(2, dtype=torch.long),
                                        torch.zeros(2, dtype=torch.long), context, ("test",)))
        self.assertEqual(graphs.stats()["fallback_capture_disabled"], 1)

    def test_goodput_includes_failed_missing_cancelled(self):
        rows = [
            dict(output_tokens=3, ttft_ms=10, tpot_ms=2, finish_reason="length"),
            dict(output_tokens=3, ttft_ms=100, tpot_ms=2, finish_reason="length"),
            dict(output_tokens=3, ttft_ms=10, tpot_ms=2, finish_reason="cancelled"),
            dict(output_tokens=3, ttft_ms=10, tpot_ms=2, finish_reason="length", failed=True),
            dict(output_tokens=1, ttft_ms=10, tpot_ms=None, finish_reason="stop"),
        ]
        result = evaluate(dict(seconds=2, requests=rows, missing_requests=1), 50, 5)
        self.assertEqual(result["slo_met_requests"], 2)
        self.assertEqual(result["total_requests"], 6)
        self.assertEqual(result["goodput_requests_s"], 1)
        self.assertEqual(result["goodput_tokens_s"], 2)
        rows[0]["tpot_ms"] = float("nan")
        self.assertEqual(evaluate(dict(seconds=2, requests=rows), 50, 5)["slo_met_requests"], 1)
        with self.assertRaises(ValueError):
            evaluate(dict(seconds=0, requests=[]), 50, 5)


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        ensure_gpu1_available()
        sys.argv.remove("--gpu")
        DEVICE = "cuda"
    unittest.main()
