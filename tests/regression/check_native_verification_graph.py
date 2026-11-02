"""CPU metadata regression tests; GPU capture/numerics remain separate gates."""
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from transformers import Qwen3Config

from nanovllm.config import Config
from nanovllm.engine.native_verification_graph import NativeVerificationGraphs
from nanovllm.utils.context import Context, AttentionGroup


class NativeGraphTests(unittest.TestCase):
    def test_packed_upload_rejection_tail_and_stream_ownership(self):
        graphs = NativeVerificationGraphs(packed_metadata=True)
        record = NativeVerificationGraphs.make_packet(3,4,"cpu")
        graphs.metadata_packets[3] = record
        event = SimpleNamespace(query=lambda: True, record=lambda stream: None, synchronize=lambda: None)
        with patch('torch.cuda.current_stream',return_value=SimpleNamespace(cuda_stream=123)), \
             patch('torch.cuda.Event',return_value=event):
            self.assertTrue(graphs.prepare_cpu(3,[11,22,33],[255,256,257],[2047,512,513],[7,2],258))
            self.assertEqual(record['device_views'][-1].tolist(),[[7,2,0,0]])
            self.assertTrue(graphs.prepare_cpu(3,[9,8,7],[9,10,11],[1289,1290,1291],[5],12))
            self.assertEqual(record['device_views'][-1].tolist(),[[5,0,0,0]])
            self.assertEqual(record['device_views'][3].tolist(),[0,12])
            self.assertFalse(graphs.prepare_cpu(2,[1,2],[0,1],[0,1],[0],2))
            with self.assertRaises(ValueError): graphs.prepare_cpu(3,[1,2,3],[0,1,2],[0,1,2],[0]*5,3)
        with patch('torch.cuda.current_stream',return_value=SimpleNamespace(cuda_stream=456)):
            with self.assertRaises(RuntimeError): graphs.prepare_cpu(3,[1,2,3],[0,1,2],[0,1,2],[0],3)
        self.assertEqual(graphs.counters['metadata_packet_h2d_copies'],2)

    def test_packet_views_preserve_mixed_dtype_and_no_overlap(self):
        for query in (2,3,5,9):
            packet = NativeVerificationGraphs.make_packet(query,4,"cpu")
            ids,pos,slots,length,pages = packet["host_views"]
            ids.fill_(90001)
            pos.copy_(torch.arange(255,255+query))
            slots.fill_(2048)
            length.copy_(torch.tensor([0,255+query],dtype=torch.int32))
            pages.copy_(torch.tensor([[7,2,0,0]],dtype=torch.int32))
            packet["packet"].copy_(packet["host"])
            for a,b in zip(packet["host_views"],packet["device_views"]):
                self.assertTrue(torch.equal(a,b))
            self.assertEqual(ids.tolist(),[90001]*query)
            self.assertEqual(slots.tolist(),[2048]*query)
            self.assertEqual(packet["host"].numel(),20*query+8+16)

    def test_long_history_fixture_never_skips_unprocessed_prefix(self):
        from tests.regression.check_draft_model import common_history_logits

        class Manager:
            def __init__(self):
                self._reservations = {}

            def reserve_to(self, seq, end):
                self._reservations[seq.seq_id] = seq
                return seq

            def commit(self, seq, end):
                seq.num_cached_tokens = end
                self._reservations.pop(seq.seq_id)

            def deallocate(self, seq):
                pass

        class Runner:
            def __init__(self):
                self.valid = set()

            def run_speculative(self, seqs):
                seq = seqs[0]
                assert len(seq) - seq.num_cached_tokens <= 256
                assert all(p in self.valid for p in range(seq.num_cached_tokens))
                positions = list(range(seq.num_cached_tokens, len(seq)))
                self.valid.update(positions)
                return torch.tensor(positions, dtype=torch.float32).reshape(-1, 1)

            def prepare_decode(self, seqs):
                seq = seqs[0]
                assert len(seq) - seq.num_cached_tokens == 1, "skipped unprocessed prefix"
                return seqs, None

            def run_model(self, seqs, positions, is_prefill):
                return self.run_speculative(seqs)

        for prefix in (255, 511, 1019):
            for chunk in (1, 4, 8):
                runner = Runner()
                engine = SimpleNamespace(config=SimpleNamespace(max_num_batched_tokens=256),
                    scheduler=SimpleNamespace(block_manager=Manager()), model_runner=runner)
                result = common_history_logits(engine, [2] * (prefix + 17), prefix, chunk)
                self.assertEqual(result.flatten().tolist(), list(range(prefix - 1, prefix + 17)))
                self.assertEqual(runner.valid, set(range(prefix + 17)))

    def setup_graph(self):
        graphs = NativeVerificationGraphs()
        static = Context(is_prefill=True, cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32), max_seqlen_q=3,
            max_seqlen_k=1024, slot_mapping=torch.zeros(3, dtype=torch.int32),
            block_tables=torch.zeros((1, 4), dtype=torch.int32), return_all_logits=True)
        calls = []
        graph = SimpleNamespace(replay=lambda: calls.append(static.cu_seqlens_k.clone()))
        entry = (graph, torch.zeros(3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64),
                 static, torch.ones(3, 7))
        graphs.entries[3] = entry
        return graphs, static, calls, entry

    def test_replay_updates_slots_lengths_and_clears_page_tail(self):
        graphs, static, calls, entry = self.setup_graph()
        for positions, table, slots in (([255, 256, 257], [[7, 2]], [2047, 512, 513]),
                                       ([9, 10, 11], [[5]], [1289, 1290, 1291])):
            context = Context(is_prefill=True, cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, positions[-1] + 1], dtype=torch.int32),
                block_tables=torch.tensor(table, dtype=torch.int32),
                slot_mapping=torch.tensor(slots, dtype=torch.int32), return_all_logits=True)
            ids = torch.tensor([11, 22, 33])
            result = graphs.run(ids, torch.tensor(positions), context)
            self.assertIs(result, entry[-1])
            self.assertEqual(entry[1].tolist(), ids.tolist())
            self.assertEqual(entry[2].tolist(), positions)
            self.assertEqual(static.slot_mapping.tolist(), slots)
            self.assertEqual(static.block_tables[0].tolist(), table[0] + [0] * (4-len(table[0])))
            self.assertEqual(calls[-1].tolist(), [0, positions[-1]+1])

    def test_packed_metadata_and_unsupported_shapes(self):
        graphs, static, calls, _ = self.setup_graph()
        group = AttentionGroup("varlen", 0, 3, torch.tensor([[3, 1]], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32), torch.tensor([0, 259], dtype=torch.int32), 3, 384)
        context = Context(slot_mapping=torch.tensor([511, 256, 257], dtype=torch.int32), attention_groups=(group,))
        self.assertIsNotNone(graphs.run(torch.tensor([1, 2, 3]), torch.tensor([256, 257, 258]), context))
        self.assertEqual(static.cu_seqlens_k.tolist(), [0, 259])
        self.assertIsNone(graphs.run(torch.tensor([1, 2]), torch.tensor([256, 257]), context))
        group.kind = "varlen_current"
        self.assertIsNone(graphs.run(torch.tensor([1, 2, 3]), torch.tensor([256, 257, 258]), context))
        self.assertEqual(len(calls), 1)

    def test_config_is_explicit_and_eager_override_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            Qwen3Config().save_pretrained(directory)
            base = dict(draft_model=directory, batching_mode="mixed", speculative_policy="fixed", max_num_seqs=1)
            config = Config(directory, **base, draft_execution="graph", draft_verify_graph=True, enforce_eager=True)
            self.assertEqual(config.runtime_summary()["effective_draft_execution"], "eager")
            self.assertFalse(config.runtime_summary()["effective_draft_verify_graph"])
            for override in (dict(draft_execution="bad"), dict(draft_graph_max_tokens=True),
                             dict(draft_graph_max_tokens=17), dict(draft_verify_graph=1)):
                with self.assertRaises(ValueError):
                    Config(directory, **base, **override)
            with self.assertRaisesRegex(ValueError, "require draft_model"):
                Config(directory, draft_verify_graph=True)

    def test_consistent_backend_scope_and_effective_config(self):
        with tempfile.TemporaryDirectory() as directory:
            Qwen3Config(dtype="bfloat16", attention_bias=False).save_pretrained(directory)
            config = Config(directory, max_num_seqs=1, numerical_backend="consistent")
            self.assertEqual(config.linear_backend, "triton_stable")
            self.assertEqual(config.rmsnorm_backend, "triton")
            self.assertEqual(config.runtime_summary()["numerical_backend"], "consistent")
            for override in (dict(max_num_seqs=2), dict(tensor_parallel_size=2), dict(mlp_backend="auto"),
                             dict(decode_attention_backend="triton")):
                with self.assertRaises(ValueError):
                    Config(directory, **(dict(max_num_seqs=1, numerical_backend="consistent") | override))


if __name__ == "__main__":
    unittest.main()
