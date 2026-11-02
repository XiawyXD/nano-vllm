"""单卡、多 token 验证的 CUDA Graph 缓存（实验性、默认关闭）。

只捕获 transformer，不改变 draft 接受/回滚逻辑。每个 graph 拥有自己的
静态输入和输出；重放前更新数据，不能把上轮的位置、长度和物理页表复用。
"""
from dataclasses import fields
import torch
from nanovllm.utils.context import Context, set_context


class VerificationGraphs:
    def __init__(self, max_entries=4):
        self.max_entries = max_entries
        self.entries = {}
        self.stats = dict(captures=0, replays=0, fallbacks=0)

    def run(self, model, input_ids, positions, context):
        # 不padding请求/query：保留真实的varlen边界，避免虚拟请求写坏KV。
        # 仅将FlashAttention的最大K长度上界向128取整；真实长度仍来自cu_seqlens_k。
        max_k = ((context.max_seqlen_k + 127) // 128) * 128
        key = (input_ids.numel(), context.cu_seqlens_q.numel(),
               context.max_seqlen_q, max_k, tuple(context.block_tables.shape))
        entry = self.entries.get(key)
        if entry is None:
            # 有界缓存：遇到新形状且已满，回退eager，不在请求期间反复驱逐/捕获。
            if len(self.entries) >= self.max_entries or input_ids.numel() > 128:
                self.stats["fallbacks"] += 1
                return None
            static = {}
            for field in fields(Context):
                value = getattr(context, field.name)
                static[field.name] = value.clone() if isinstance(value, torch.Tensor) else value
            static["max_seqlen_k"] = max_k
            ids, pos = input_ids.clone(), positions.clone()
            set_context(**static)
            # 当前输入是真实验证批次。重复forward只重写同一批新KV槽位，
            # 不修改历史token或调度状态。先预热，避免首次编译发生在capture内。
            for _ in range(2):
                model(ids, pos)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            from nanovllm.layers.quant_graph import capture_model_graph
            with capture_model_graph(model, graph):
                output = model(ids, pos)
            entry = (graph, ids, pos, static, output)
            self.entries[key] = entry
            self.stats["captures"] += 1
        else:
            graph, ids, pos, static, output = entry
            ids.copy_(input_ids)
            pos.copy_(positions)
            # 数据不同而shape相同是graph复用的关键：每轮都要更新所有tensor。
            for field in fields(Context):
                value = getattr(context, field.name)
                if isinstance(value, torch.Tensor):
                    static[field.name].copy_(value)
            set_context(**static)
        graph, _, _, _, output = entry
        graph.replay()
        self.stats["replays"] += 1
        # output属于静态buffer，必须在下一次replay之前消费；调用方立即算logits。
        return output
