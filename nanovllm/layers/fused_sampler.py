"""面向 greedy decode 的轻量 fused sampler。

普通请求如果使用 temperature=0、top_k=0、top_p=1 且没有重复惩罚，
本质上只需要对每一行 logits 做 argmax。这个文件用一个 Triton kernel
把“按块扫描词表 + 取最大值 + 写出 token id”融合起来，避免 Python 逐行
循环和中间的 full-vocab softmax；复杂采样参数仍回退到参考 Sampler。
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU/未安装 Triton 时的兼容路径
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_greedy_argmax_kernel(
        logits_ptr,
        out_ptr,
        num_rows,
        vocab_size,
        stride_row,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_mask = row < num_rows
        best_value = tl.full((), -float("inf"), tl.float32)
        best_index = tl.zeros((), tl.int32)
        # 一个 program 处理一行，但分块扫描大词表，避免一次性分配
        # 151k 规模的超大寄存器向量。
        for start in tl.range(0, vocab_size, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            mask = row_mask & (offsets < vocab_size)
            values = tl.load(
                logits_ptr + row * stride_row + offsets,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            local_index = tl.argmax(values, axis=0)
            local_value = tl.max(values, axis=0)
            global_index = start + local_index
            better = (local_value > best_value) | (
                (local_value == best_value) & (global_index < best_index)
            )
            best_value = tl.where(better, local_value, best_value)
            best_index = tl.where(better, global_index, best_index)
        tl.store(out_ptr + row, best_index, mask=row_mask)


def fused_greedy_argmax(logits: torch.Tensor) -> torch.Tensor:
    """返回 ``[batch]`` 的 greedy token id，优先走 Triton kernel。"""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [batch, vocab], got {tuple(logits.shape)}")
    if not logits.is_cuda or triton is None:
        return logits.argmax(dim=-1)
    logits = logits.contiguous()
    output = torch.empty((logits.size(0),), device=logits.device, dtype=torch.int32)
    # 1024 在 Qwen 词表上可以兼顾 occupancy 和寄存器压力。
    _fused_greedy_argmax_kernel[(logits.size(0),)](
        logits,
        output,
        logits.size(0),
        logits.size(1),
        logits.stride(0),
        BLOCK=1024,
        num_warps=4,
    )
    return output.to(torch.long)

