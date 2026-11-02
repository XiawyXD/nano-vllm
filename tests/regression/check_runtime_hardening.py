"""CPU 回归：缓存淘汰、graph 分桶、上下文清理和采样输入不可变性。

无需模型权重或 GPU；图重放使用 fake graph，只检验运行时选择与静态 buffer。
真实 CUDA Graph 数值正确性仍由 check_verification_graph 等 GPU 实验覆盖。
运行：python -m tests.regression.check_runtime_hardening
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.model_runner import ModelRunner, decode_graph_batch_sizes
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context, set_context


def check_lru():
    old_size = Sequence.block_size
    Sequence.block_size = 2
    try:
        manager = BlockManager(4, 2)
        original = Sequence([1, 2, 3, 4])
        manager.allocate(original, manager.can_allocate(original))
        original.num_scheduled_tokens = 4
        manager.hash_blocks(original)
        first, second = original.block_table
        first_hash = manager.blocks[first].hash
        second_hash = manager.blocks[second].hash
        manager.deallocate(original)

        for token in (8, 9):
            seq = Sequence([token])
            manager.allocate(seq, manager.can_allocate(seq))
        assert manager.prefix_cache_evictions == 0
        assert manager.hash_to_block_id[first_hash] == first
        assert manager.hash_to_block_id[second_hash] == second

        # 访问 first 后，second 成为最久未用的空闲缓存；空白块已耗尽。
        assert manager.can_allocate(Sequence([1, 2, 7])) == 1
        extra = Sequence([10])
        manager.allocate(extra, manager.can_allocate(extra))
        assert extra.block_table == [second]
        assert manager.prefix_cache_evictions == 1
        assert first_hash in manager.hash_to_block_id
        assert second_hash not in manager.hash_to_block_id

        # 最后一个 free prefix 可以淘汰；已引用的 block 永远不被驱逐。
        last = Sequence([11])
        manager.allocate(last, manager.can_allocate(last))
        assert len(manager.used_block_ids) == 4
        try:
            manager._allocate_block()
        except RuntimeError as exc:
            assert "no free KV cache blocks" in str(exc)
        else:
            raise AssertionError("allocation from an empty free list must fail clearly")
        assert len(manager.used_block_ids) == 4
    finally:
        Sequence.block_size = old_size


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, ids, positions):
        self.calls += 1
        return ids.float().unsqueeze(1)

    def compute_logits(self, hidden):
        return hidden + 1


class FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def make_fake_runner(maximum=10):
    runner = ModelRunner.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.model = FakeModel()
    runner.graph_bs = decode_graph_batch_sizes(maximum)
    runner.graphs = {size: FakeGraph() for size in runner.graph_bs}
    runner.graph_vars = {
        "input_ids": torch.zeros(maximum, dtype=torch.long),
        "positions": torch.zeros(maximum, dtype=torch.long),
        "slot_mapping": torch.zeros(maximum, dtype=torch.int32),
        "context_lens": torch.zeros(maximum, dtype=torch.int32),
        "block_tables": torch.zeros(maximum, 2, dtype=torch.int32),
        "outputs": torch.full((maximum, 1), 99.0),
    }
    return runner


def set_fake_decode_context(batch, columns=1):
    set_context(
        False,
        slot_mapping=torch.arange(batch, dtype=torch.int32),
        context_lens=torch.ones(batch, dtype=torch.int32),
        block_tables=torch.zeros(batch, columns, dtype=torch.int32),
    )


def check_graph_buckets_and_fallback():
    for maximum in range(1, 513):
        buckets = decode_graph_batch_sizes(maximum)
        assert buckets[-1] == maximum
        assert all(1 <= size <= maximum for size in buckets)
        assert all(any(size >= batch for size in buckets) for batch in range(1, maximum + 1))
    assert decode_graph_batch_sizes(1024)[-1] == 512

    try:
        runner = make_fake_runner(10)
        ids = torch.arange(9)
        set_fake_decode_context(9)
        output = runner.run_model(ids, ids, False)
        assert runner.graphs[10].replays == 1
        assert output.shape == (9, 1) and torch.all(output == 100)
        assert runner.graph_vars["slot_mapping"][9] == -1
        assert runner.graph_vars["context_lens"][9] == 0

        runner.graphs.clear()
        result = runner.run_model(ids, ids, False)
        assert torch.equal(result[:, 0], ids + 1) and runner.model.calls == 1

        runner = make_fake_runner(10)
        set_fake_decode_context(9, columns=3)
        result = runner.run_model(ids, ids, False)
        assert torch.equal(result[:, 0], ids + 1) and runner.model.calls == 1
        assert all(graph.replays == 0 for graph in runner.graphs.values())
    finally:
        reset_context()


def check_context_cleanup():
    for stage in ("prepare", "forward", "sample"):
        runner = ModelRunner.__new__(ModelRunner)
        runner.rank = 0

        def prepare(seqs):
            set_context(True, slot_mapping=torch.tensor([7]))
            if stage == "prepare":
                raise RuntimeError(stage)
            return torch.tensor([1]), torch.tensor([0])

        def forward(*args):
            if stage == "forward":
                raise RuntimeError(stage)
            return torch.ones(1, 3)

        def sample(*args):
            raise RuntimeError(stage)

        runner.prepare_prefill = prepare
        runner.run_model = forward
        runner.sampler = sample
        try:
            runner.run([object()], True)
        except RuntimeError as exc:
            assert str(exc) == stage
        else:
            raise AssertionError("injected failure did not propagate")
        assert get_context().slot_mapping is None
        assert not get_context().is_prefill

    runner = ModelRunner.__new__(ModelRunner)
    def fail_spec_prepare(*args, **kwargs):
        set_context(True, slot_mapping=torch.tensor([9]), return_all_logits=True)
        raise RuntimeError("speculative preparation failed")
    runner.prepare_prefill = fail_spec_prepare
    try:
        runner.run_speculative([object()])
    except RuntimeError as exc:
        assert str(exc) == "speculative preparation failed"
    else:
        raise AssertionError("speculative preparation failure did not propagate")
    assert get_context().slot_mapping is None
    assert not get_context().return_all_logits


def request(**overrides):
    values = dict(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                  token_ids=[1, 2], seed=123, num_completion_tokens=0)
    values.update(overrides)
    return SimpleNamespace(**values)


def check_sampler():
    logits = torch.tensor([[1.0, 3.0, 3.0, 0.0], [2.0, 1.0, 0.0, -1.0]])
    seqs = [request(), request()]
    with patch("nanovllm.layers.sampler.fused_greedy_argmax",
               side_effect=AssertionError("default sampler must not use Triton")):
        assert Sampler().backend == "torch"
        assert torch.equal(Sampler()(logits, seqs), torch.tensor([1, 0]))
    with patch("nanovllm.layers.sampler.fused_greedy_argmax",
               return_value=torch.tensor([1, 0])) as fused:
        assert torch.equal(Sampler("triton")(logits, seqs), torch.tensor([1, 0]))
        fused.assert_called_once()

    # float32 原地 penalty/temperature/top-k/top-p 不能改动 logits 的底层存储。
    for settings in (
        dict(repetition_penalty=2.0),
        dict(temperature=0.7),
        dict(temperature=0.7, top_k=2, top_p=0.8, repetition_penalty=1.2),
    ):
        original = logits.clone()
        sampled = Sampler()(logits, [request(**settings), request(**settings)])
        assert torch.equal(logits, original)
        assert torch.equal(sampled, Sampler()(logits, [request(**settings), request(**settings)]))
    try:
        Sampler("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown sampler backend accepted")

    # 独立、可手算的参考输入：penalty 后最大项变化；top-k/top-p
    # 只留下一个候选，不依赖第二次运行相同实现作为正确性 oracle。
    cases = (
        (torch.tensor([[1.0, 3.0, 2.5, -1.0]]),
         request(token_ids=[1], repetition_penalty=2.0), 2),
        (torch.tensor([[0.0, 1.0, 3.0, 2.0]]),
         request(temperature=1.0, top_k=1), 2),
        (torch.tensor([[0.1, 0.2, 0.7]]).log(),
         request(temperature=1.0, top_p=0.6), 2),
    )
    for independent_logits, seq, expected in cases:
        reference_input = independent_logits.clone()
        assert Sampler()(independent_logits, [seq]).item() == expected
        assert torch.equal(independent_logits, reference_input)


def check_sampling_validation():
    assert SamplingParams(stop=[]).stop == []
    assert SamplingParams(stop="END").stop == ["END"]
    assert SamplingParams(seed=-1).seed == -1
    invalid = (
        {"max_tokens": 1.5}, {"max_tokens": True}, {"max_tokens": 0},
        {"top_k": 1.5}, {"top_k": -1}, {"top_k": True},
        {"speculative_ngram_size": 1.5}, {"speculative_ngram_size": -1},
        {"speculative_max_draft_tokens": 1.5}, {"speculative_max_draft_tokens": 0},
        {"seed": 1.5}, {"seed": True}, {"seed": 2**64 - 1, "max_tokens": 2},
        {"temperature": float("nan")}, {"temperature": float("inf")},
        {"temperature": -1.0}, {"top_p": float("nan")}, {"top_p": 0.0},
        {"top_p": 1.1}, {"repetition_penalty": float("inf")},
        {"repetition_penalty": 0}, {"ignore_eos": 1},
        {"stop": ""}, {"stop": [""]}, {"stop": [None]}, {"stop": 123},
    )
    for values in invalid:
        try:
            SamplingParams(**values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid sampling parameters accepted: {values}")


def check_config():
    with patch("nanovllm.config.os.path.isdir", return_value=True), patch(
        "nanovllm.config.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(max_position_embeddings=2048),
    ):
        assert Config("fake").sampler_backend == "torch"
        assert Config("fake", sampler_backend="triton").sampler_backend == "triton"
        cases = (
            {"max_num_batched_tokens": 0}, {"max_num_seqs": -1},
            {"max_model_len": 0}, {"kvcache_block_size": 0},
            {"kvcache_block_size": 255}, {"tensor_parallel_size": 9},
            {"gpu_memory_utilization": 0}, {"gpu_memory_utilization": float("nan")},
            {"prefill_chunk_size": 0}, {"max_prefill_steps": -1},
            {"sampler_backend": "unknown"}, {"decode_attention_block_n": 3},
            {"decode_attention_num_warps": 3}, {"kv_cache_dtype": "fp8"},
        )
        for options in cases:
            try:
                Config("fake", **options)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid config accepted: {options}")


def main():
    checks = {
        "lru_preserves_cache_until_needed": check_lru,
        "graph_bucket_coverage_and_fallback": check_graph_buckets_and_fallback,
        "context_cleanup_on_failure": check_context_cleanup,
        "sampler_backend_and_input_immutability": check_sampler,
        "sampling_parameter_validation": check_sampling_validation,
        "config_validation": check_config,
    }
    for check in checks.values():
        check()
    print(json.dumps({"status": "PASS", "device": "cpu",
                      "checks": {name: True for name in checks}}, indent=2))


if __name__ == "__main__":
    main()
