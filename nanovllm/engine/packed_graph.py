"""P+D / D+V 的有界、精确形状 CUDA Graph。

不创建虚拟请求，不把普通 decode bucket 用在多 query 上。图仅录制 transformer；
LM head 和采样仍消费明确的真实输出行。缓存满直接 eager，不进行运行期 autotune。
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import fields, is_dataclass

import torch

from nanovllm.utils.context import get_context, set_context


def clone_metadata(value):
    """递归复制 Context/AttentionGroup 中 tensor，分离每张图的静态存储。"""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if is_dataclass(value):
        return type(value)(**{f.name: clone_metadata(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, tuple):
        return tuple(clone_metadata(item) for item in value)
    return value


def copy_metadata(destination, source):
    """同 shape 重放前更新每个 tensor；标量变化必须已由 graph key 隔离。"""
    if isinstance(source, torch.Tensor):
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError("graph metadata shape/dtype changed")
        destination.copy_(source)
    elif is_dataclass(source):
        for field in fields(source):
            copy_metadata(getattr(destination, field.name), getattr(source, field.name))
    elif isinstance(source, tuple):
        if len(destination) != len(source):
            raise ValueError("graph metadata group count changed")
        for dst, src in zip(destination, source):
            copy_metadata(dst, src)
    elif destination != source:
        raise ValueError("graph scalar changed without a new key")


def metadata_signature(value):
    """tensor 值可变，shape/dtype 固定；CPU 标量作为编译/分路常量进入 key。"""
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), tuple(value.stride()))
    if is_dataclass(value):
        return tuple((f.name, metadata_signature(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, tuple):
        return tuple(metadata_signature(item) for item in value)
    return value


class PackedGraphs:
    """最多四张图；seen 只保存 64 个形状的出现次数，不随请求增长。"""

    def __init__(self, max_entries=4, min_observations=3, allow_capture=True):
        self.max_entries = max_entries
        self.min_observations = min_observations
        self.entries = {}
        self.seen = OrderedDict()
        self.counters = Counter()
        self.captured_last = False
        self.allow_capture = allow_capture

    def fallback(self, reason):
        self.counters["fallbacks"] += 1
        self.counters[f"fallback_{reason}"] += 1
        return None

    def run(self, model, input_ids, positions, context, backend_signature):
        """返回静态 hidden 输出或 None；调用方必须在下次 replay 前消费输出。"""
        self.captured_last = False
        cap = 128 if context.batch_kind == "DV_PACKED" else 512
        if input_ids.numel() > cap:
            return self.fallback("query_limit")
        key = (tuple(input_ids.shape), backend_signature, metadata_signature(context))
        entry = self.entries.get(key)
        if entry is None:
            if not self.allow_capture:
                return self.fallback("capture_disabled")
            if len(self.entries) >= self.max_entries:
                return self.fallback("cache_full")
            self.seen[key] = self.seen.get(key, 0) + 1
            self.seen.move_to_end(key)
            while len(self.seen) > 64:
                self.seen.popitem(last=False)
            if self.seen[key] < self.min_observations:
                return self.fallback("shape_observations")
            static = clone_metadata(context)
            ids, pos = input_ids.clone(), positions.clone()
            set_context(**{f.name: getattr(static, f.name) for f in fields(static)})
            try:
                for _ in range(2):
                    model(ids, pos)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                from nanovllm.layers.quant_graph import capture_model_graph
                with capture_model_graph(model, graph):
                    output = model(ids, pos)
                entry = (graph, ids, pos, static, output)
                self.entries[key] = entry
                self.captured_last = True
                self.counters["captures"] += 1
            except BaseException:
                # 不把 CUDA 错误当普通 fallback。引擎必须终止，外层 finally 清 Context。
                raise
        else:
            graph, ids, pos, static, output = entry
            ids.copy_(input_ids)
            pos.copy_(positions)
            copy_metadata(static, context)
            set_context(**{f.name: getattr(static, f.name) for f in fields(static)})
        graph, _, _, _, output = entry
        graph.replay()
        self.counters["replays"] += 1
        return output

    def stats(self):
        return dict(self.counters, entries=len(self.entries), seen_shapes=len(self.seen),
                    allow_capture=self.allow_capture,
                    max_entries=self.max_entries)
