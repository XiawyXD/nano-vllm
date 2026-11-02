"""在模型层和 Attention 之间传递本轮运行时元数据。

模型的 `forward` 接口只接收 input_ids 和 positions，但 Attention 还需要
知道这是 prefill 还是 decode、KV Cache 的 slot、变长序列边界等信息。
这里用一个进程内的 Context 暂存这些数据，避免把大量参数逐层传递。
"""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class AttentionGroup:
    """packed token 数组中的连续子区间，以及它独立的因果/页表元数据。"""
    kind: str
    start: int
    end: int
    block_tables: torch.Tensor
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    context_lens: torch.Tensor | None = None


@dataclass(slots=True)
class Context:
    # True 表示本轮处理 prompt；False 表示增量 decode。
    is_prefill: bool = False
    # FlashAttention varlen 使用的 query 累积长度数组。
    cu_seqlens_q: torch.Tensor | None = None
    # FlashAttention varlen 使用的 key 累积长度数组。
    cu_seqlens_k: torch.Tensor | None = None
    # 本批次 query/key 中最长序列长度。
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 每个输入 token 对应的 KV Cache 物理 slot。
    slot_mapping: torch.Tensor | None = None
    # decode 时每条序列在 cache 中已有的长度。
    context_lens: torch.Tensor | None = None
    # 每条序列的逻辑 block 到物理 block 映射。
    block_tables: torch.Tensor | None = None
    # FP8 prefill dequant flat 输出中每个 token 的序列 id 和相对位置。
    prefill_sequence_ids: torch.Tensor | None = None
    prefill_positions: torch.Tensor | None = None
    # speculative verification 需要保留每个 query 位置的 logits。
    return_all_logits: bool = False
    # 非 None 时，模型只为这些 packed 行计算 logits；空 tensor 表示没有采样行。
    logits_indices: torch.Tensor | None = None
    # 新执行计划使用连续的 P/V 子区间和 D 子区间；旧默认路径保持 None。
    attention_groups: tuple[AttentionGroup, ...] | None = None
    batch_kind: str = "legacy"
    # CPU 原始页表/长度供每轮一次 plan；wrapper 供各层复用，不存 KV 本身。
    flashinfer_metadata: tuple | None = None
    flashinfer_wrapper: object | None = None


# 当前进程/当前模型调用所使用的 Context。
_CONTEXT = Context()


def get_context():
    """返回当前 Context，Attention 等模块从这里读取运行时信息。"""
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    prefill_sequence_ids=None,
    prefill_positions=None,
    return_all_logits=False,
    logits_indices=None,
    attention_groups=None,
    batch_kind="legacy",
    flashinfer_metadata=None,
    flashinfer_wrapper=None,
):
    """在一次 forward 前写入本轮的变长序列和 KV Cache 元数据。"""
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        prefill_sequence_ids,
        prefill_positions,
        return_all_logits,
        logits_indices,
        attention_groups,
        batch_kind,
        flashinfer_metadata,
        flashinfer_wrapper,
    )


def reset_context():
    """forward 结束后清空 Context，避免下一轮误用旧的 batch 信息。"""
    global _CONTEXT
    _CONTEXT = Context()
