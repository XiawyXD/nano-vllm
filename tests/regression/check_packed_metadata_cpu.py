"""纯 CPU 检查 packed 元数据、Graph key/copy 与异常 Context 清理。

按文件加载真实实现，使用 numpy tensor 替身，不导入真实 torch、不初始化 GPU。
它检查 CPU 协议而非 CUDA Graph 录制/重放或 BF16 数值；真实验证仍由
check_packed_runtime 完成。
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    """仅提供被测 metadata helper 使用的确定性 CPU tensor 接口。"""
    def __init__(self, values, dtype="int64"):
        self.array = np.array(values, dtype=dtype)
        self.dtype = dtype

    @property
    def shape(self):
        return self.array.shape

    def stride(self):
        return tuple(value // self.array.itemsize for value in self.array.strides)

    def clone(self):
        return Tensor(self.array.copy(), self.dtype)

    def copy_(self, other):
        np.copyto(self.array, other.array)

    def numel(self):
        return self.array.size

    def tolist(self):
        return self.array.tolist()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def value_of(value):
    """转成可比较快照，确保嵌套 AttentionGroup 每个字段均被检查。"""
    if isinstance(value, Tensor):
        return value.dtype, value.shape, value.tolist()
    if is_dataclass(value):
        return tuple((field.name, value_of(getattr(value, field.name))) for field in fields(value))
    if isinstance(value, tuple):
        return tuple(value_of(item) for item in value)
    return value


def expect(error, action):
    try:
        action()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


class Pool:
    """只检查 prepare_plan 的命名字段、容量和 begin/end，不模拟异步 H2D。"""
    grew = False

    def __init__(self):
        self.active = False
        self.values = {}

    def begin(self):
        assert not self.active
        self.active = True
        self.values = {}

    def tensor(self, name, values, dtype, *, shape=None, max_elements):
        assert self.active
        result = Tensor(values, dtype)
        if shape is not None:
            result.array = result.array.reshape(shape)
        assert result.numel() <= max_elements
        self.values[name] = result
        return result

    def end(self):
        self.active = False


class DecodeSeq:
    """Pure-Decode fast path 的最小真实协议替身。"""

    def __init__(self, request_id, last_token, block_table, length):
        self.seq_id = request_id
        self.last_token = last_token
        self.block_table = list(block_table)
        self.length = length

    def __len__(self):
        return self.length

    @property
    def last_block_num_tokens(self):
        return self.length - ((self.length - 1) // 4) * 4


class Model:
    """输出由本轮 token/position 决定，用来检查 logits 行归属，不模拟 Attention。"""
    def __init__(self, context):
        self.context = context
        self.fail = False

    def __call__(self, ids, positions):
        if self.fail:
            raise RuntimeError("injected model failure")
        return Tensor((ids.array * 100 + positions.array).reshape(-1, 1))

    def compute_logits(self, hidden):
        indices = self.context.get_context().logits_indices
        return hidden if indices is None else Tensor(hidden.array[indices.array])


def make_runner(context, backend="split"):
    runner = SimpleNamespace(config=SimpleNamespace(max_num_batched_tokens=32,
        max_num_seqs=4, max_model_len=32, num_kvcache_blocks=32,
        mixed_attention_backend=backend, enable_mixed_graph=False),
        block_size=4, metadata_buffers=Pool(), rank=0, enforce_eager=True)
    runner.model = Model(context)
    runner.sampler = lambda logits, owners: Tensor(logits.array[:, 0])

    def ordinary(ids, positions, prefill):
        assert not prefill and context.get_context().attention_groups is None
        runner.last_model_path = "decode_eager"
        return runner.model.compute_logits(runner.model(ids, positions))
    runner.run_model = ordinary

    def prepare_decode(seqs):
        ids = Tensor([seq.last_token for seq in seqs])
        positions = Tensor([len(seq) - 1 for seq in seqs])
        slots = Tensor([seq.block_table[-1] * 4 + seq.last_block_num_tokens - 1
                        for seq in seqs], "int32")
        lengths = Tensor([len(seq) for seq in seqs], "int32")
        columns = max(len(seq.block_table) for seq in seqs)
        tables = Tensor([seq.block_table + [-1] * (columns - len(seq.block_table))
                         for seq in seqs], "int32")
        context.set_context(False, slot_mapping=slots,
                            context_lens=lengths, block_tables=tables)
        return ids, positions

    runner.prepare_decode = prepare_decode
    return runner


def check_graph_metadata(context, graph):
    def state(shift):
        t = lambda values: Tensor(np.array(values) + shift, "int32")
        return context.Context(is_prefill=True, cu_seqlens_q=t([0, 2, 3]),
            cu_seqlens_k=t([0, 5, 13]), max_seqlen_q=2, max_seqlen_k=128,
            slot_mapping=t([3, 8, 19]), context_lens=t([5, 8]),
            block_tables=t([[0, 2], [3, 4]]), prefill_sequence_ids=t([0, 0, 1]),
            prefill_positions=t([3, 4, 7]), logits_indices=t([1, 2]),
            attention_groups=(context.AttentionGroup("varlen", 0, 2,
                t([[0, 2]]), t([0, 2]), t([0, 5]), 2, 128),
                context.AttentionGroup("decode", 2, 3, t([[3, 4]]),
                    context_lens=t([8]))), batch_kind="PD_PACKED")
    a, b = state(0), state(9)
    signature = graph.metadata_signature(a)
    assert signature == graph.metadata_signature(b), "tensor values must not create graph keys"
    static = graph.clone_metadata(a)
    assert value_of(static) == value_of(a)
    assert static.slot_mapping is not a.slot_mapping
    assert static.attention_groups[0].block_tables is not a.attention_groups[0].block_tables
    for source in (b, a, b, a):
        graph.copy_metadata(static, source)
        assert value_of(static) == value_of(source), "at least one graph field remained stale"
        assert graph.metadata_signature(static) == signature
    assert graph.metadata_signature(replace(a, max_seqlen_k=256)) != signature
    assert graph.metadata_signature(replace(a, slot_mapping=Tensor([1, 2]))) != signature
    expect(ValueError, lambda: graph.copy_metadata(static, replace(a, max_seqlen_k=256)))
    expect(ValueError, lambda: graph.copy_metadata(static, replace(a, slot_mapping=Tensor([1, 2]))))
    # 未达到捕获门槛不访问 fake CUDA；只验证 seen/cache/query 上限控制逻辑。
    cache = graph.PackedGraphs(max_entries=4, min_observations=1000)
    for size in range(1, 70):
        assert cache.run(None, Tensor([0] * size), None, a, ()) is None
    assert len(cache.seen) == 64 and not cache.entries
    assert cache.run(None, Tensor([0] * 129), None, replace(a, batch_kind="DV_PACKED"), ()) is None
    assert cache.counters["fallback_query_limit"] == 1


def check_packing_and_cleanup(context, packed, execution):
    I, P, R = execution.ExecutionItem, execution.ExecutionPlan, execution.ExecutionRole
    p = I(10, R.PREFILL, (11, 12), 3, (0, 2), (), (1,))
    d = I(20, R.DECODE, (13,), 7, (3, 4), (), (0,))
    v = I(30, R.VERIFY, (21, 22, 23), 3, (5, 6), (22, 23), (0, 1, 2))
    sequences = [SimpleNamespace(seq_id=i) for i in (10, 20, 30)]
    for backend in ("split", "varlen"):
        runner = make_runner(context, backend)
        plan = P((p, d))
        prepared = packed.prepare_plan(runner, plan)
        metadata = context.get_context()
        assert prepared.row_request_ids == (10, 20)
        assert prepared.input_ids.tolist() == [11, 12, 13]
        assert prepared.positions.tolist() == [3, 4, 7]
        assert metadata.slot_mapping.tolist() == [3, 8, 19]
        assert metadata.logits_indices.tolist() == [1, 2]
        assert metadata.attention_groups[0].kind == "varlen", \
            "partial/prefix-aware Prefill must keep reading paged KV history"
        assert not runner.metadata_buffers.active
        context.reset_context()
        assert packed.run_plan(runner, plan, sequences) == {10: [1204], 20: [1307]}
        assert packed.run_plan(runner, P((replace(p, logit_positions=()), d)), sequences) == {10: [], 20: [1307]}
        assert packed.run_plan(runner, P((replace(p, logit_positions=()),)), sequences) == {10: []}
        assert packed.run_plan(runner, P((v, d)), sequences) == {30: [2103, 2204, 2305], 20: [1307]}
        # 纯 D 无论 mixed backend 配置如何，都回到普通 Decode/Graph 路径；
        # varlen 仅用于真正的 Prefill/Verify + Decode 混合轮次。
        if backend == "varlen":
            packed.prepare_plan(runner, P((d,)))
            pure_decode_context = context.get_context()
            assert pure_decode_context.batch_kind == "D_NORMAL"
            assert pure_decode_context.attention_groups is None
            assert pure_decode_context.context_lens.tolist() == [8]
            context.reset_context()
        decode_sequences = [DecodeSeq(20, 13, (3, 4), 8)]
        assert packed.run_plan(runner, P((d,)), decode_sequences) == {20: [1307]}
        assert context.get_context().slot_mapping is None
        runner.model.fail = True
        expect(RuntimeError, lambda: packed.run_plan(runner, plan, sequences))
        assert context.get_context().slot_mapping is None
        assert not runner.metadata_buffers.active
        runner.model.fail = False
        runner.sampler = lambda *_: (_ for _ in ()).throw(RuntimeError("injected sampler failure"))
        expect(RuntimeError, lambda: packed.run_plan(runner, plan, sequences))
        assert context.get_context().slot_mapping is None
        malformed = P((replace(d, block_table=(3,)),))
        expect(ValueError, lambda: packed.run_plan(runner, malformed, decode_sequences))
        assert context.get_context().slot_mapping is None and not runner.metadata_buffers.active
        for bad_page in (-1, runner.config.num_kvcache_blocks, True):
            # query位于第2页且slot合法；第1页历史非法也必须在任何GPU执行前拒绝。
            malformed = P((replace(d, block_table=(bad_page, 4)),))
            expect(ValueError, lambda: packed.run_plan(runner, malformed, decode_sequences))
            assert context.get_context().slot_mapping is None and not runner.metadata_buffers.active

    # Fresh Prefill starts at logical position zero, so current K/V is its whole
    # attention history and can be read contiguously without a page-table lookup.
    runner = make_runner(context, "split")
    fresh = I(40, R.PREFILL, (31, 32), 0, (0,), (), (1,))
    packed.prepare_plan(runner, P((fresh,)))
    metadata = context.get_context()
    assert metadata.attention_groups[0].kind == "varlen_current"
    assert metadata.attention_groups[0].cu_seqlens_q.tolist() == [0, 2]
    assert metadata.attention_groups[0].cu_seqlens_k.tolist() == [0, 2]
    assert metadata.attention_groups[0].max_seqlen_k == 2
    context.reset_context()


def main():
    fake = SimpleNamespace(Tensor=Tensor, int64="int64", int32="int32",
                           inference_mode=lambda: (lambda function: function))
    names = ("nanovllm.utils.context", "_packed_graph_cpu", "_packed_runner_cpu", "_execution_cpu")
    with patch.dict(sys.modules, {"torch": fake}):
        try:
            context = load(names[0], "nanovllm/utils/context.py")
            graph = load(names[1], "nanovllm/engine/packed_graph.py")
            packed = load(names[2], "nanovllm/engine/packed_runner.py")
            execution = load(names[3], "nanovllm/engine/execution_plan.py")
            check_graph_metadata(context, graph)
            check_packing_and_cleanup(context, packed, execution)
        finally:
            for name in names:
                sys.modules.pop(name, None)
    print(json.dumps(dict(status="PASS", device="CPU fake tensors", checks=[
        "all_context_fields_A_B_A", "shape_and_scalar_key_isolation", "bounded_seen_and_query_limit",
        "PD_DV_and_partial_prefill_rows", "pure_decode_split_and_varlen_dispatch",
        "fresh_prefill_current_kv_path_and_partial_prefill_paged_path",
        "invalid_historical_pages_rejected_before_attention",
        "prepare_model_sampler_exception_context_cleanup"],
        scope="No real torch/CUDA, Graph capture, attention arithmetic or model quality claim"), indent=2))


if __name__ == "__main__":
    main()
