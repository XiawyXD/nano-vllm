"""可读的 Triton Paged Decode Attention 和 FP8 cache dequant kernel。

默认生产路径仍然是 FlashAttention。这里的 kernel 用于学习、扫参和
FP8 KV Cache 实验：decode 直接按 block table 读取分页 cache，prefill
则可以把 FP8 cache dequant 成模型 dtype 后交给 FlashAttention varlen。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# 离线 benchmark 会尝试这些候选；engine 默认仍使用 block_n=64。
ATTENTION_CONFIGS = (
    (32, 2),
    (32, 4),
    (64, 4),
    (64, 8),
    (128, 4),
    (128, 8),
    (256, 8),
)


@triton.jit
def _paged_decode_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    out_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_d,
    k_stride_block,
    k_stride_token,
    k_stride_head,
    k_stride_d,
    v_stride_block,
    v_stride_token,
    v_stride_head,
    v_stride_d,
    k_scale_stride_block,
    k_scale_stride_token,
    v_scale_stride_block,
    v_scale_stride_token,
    block_table_stride_b,
    block_table_stride_block,
    out_stride_b,
    out_stride_h,
    out_stride_d,
    scale,
    CACHE_BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    USE_SCALES: tl.constexpr,
):
    """一个 program 负责一个 (batch, query_head) 的在线 softmax。"""
    batch_id = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_group_size = NUM_QUERY_HEADS // NUM_KV_HEADS
    kv_head = query_head // kv_group_size

    offs_d = tl.arange(0, HEAD_DIM)
    d_mask = offs_d < ACTUAL_HEAD_DIM
    q_ptrs = q_ptr + batch_id * q_stride_b + query_head * q_stride_h + offs_d * q_stride_d
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    context_len = tl.load(context_lens_ptr + batch_id)
    max_score = tl.full((), -float("inf"), dtype=tl.float32)
    normalizer = tl.zeros((), dtype=tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for start in range(0, MAX_BLOCKS * CACHE_BLOCK_SIZE, BLOCK_N):
        positions = start + tl.arange(0, BLOCK_N)
        valid = positions < context_len
        logical_block = positions // CACHE_BLOCK_SIZE
        offset_in_block = positions % CACHE_BLOCK_SIZE
        physical_block = tl.load(
            block_tables_ptr
            + batch_id * block_table_stride_b
            + logical_block * block_table_stride_block,
        )

        k_ptrs = (
            k_cache_ptr
            + physical_block[:, None] * k_stride_block
            + offset_in_block[:, None] * k_stride_token
            + kv_head * k_stride_head
            + offs_d[None, :] * k_stride_d
        )
        v_ptrs = (
            v_cache_ptr
            + physical_block[:, None] * v_stride_block
            + offset_in_block[:, None] * v_stride_token
            + kv_head * v_stride_head
            + offs_d[None, :] * v_stride_d
        )
        element_mask = valid[:, None] & d_mask[None, :]
        keys = tl.load(k_ptrs, mask=element_mask, other=0.0).to(tl.float32)
        values = tl.load(v_ptrs, mask=element_mask, other=0.0).to(tl.float32)
        if USE_SCALES:
            scale_offsets = (
                physical_block * k_scale_stride_block
                + offset_in_block * k_scale_stride_token
            )
            key_scales = tl.load(
                k_scale_ptr + scale_offsets, mask=valid, other=1.0
            ).to(tl.float32)
            value_scales = tl.load(
                v_scale_ptr + scale_offsets, mask=valid, other=1.0
            ).to(tl.float32)
            keys *= key_scales[:, None]
            values *= value_scales[:, None]

        scores = tl.sum(keys * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, -float("inf"))
        chunk_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_score, chunk_max)
        old_scale = tl.exp(max_score - new_max)
        weights = tl.exp(scores - new_max)
        normalizer = normalizer * old_scale + tl.sum(weights, axis=0)
        accumulator = accumulator * old_scale + tl.sum(weights[:, None] * values, axis=0)
        max_score = new_max

    output = accumulator / normalizer
    out_ptrs = out_ptr + batch_id * out_stride_b + query_head * out_stride_h + offs_d * out_stride_d
    tl.store(out_ptrs, output, mask=d_mask)


def _default_num_warps(head_dim: int) -> int:
    return 8 if triton.next_power_of_2(head_dim) >= 128 else 4


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    scale: float,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """运行分页 decode attention，q 的形状是 [batch, query_heads, head_dim]。

    block_n 和 num_warps 只控制 Triton launch 配置。engine 可以固定它们，
    benchmark 则会枚举候选配置；不在 CUDA Graph capture 中动态 autotune。
    """
    if q.ndim != 3:
        raise ValueError(f"q must have shape [batch, heads, head_dim], got {q.shape}")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache must have the same shape")
    batch, num_query_heads, head_dim = q.shape
    num_blocks, cache_block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if head_dim != cache_head_dim:
        raise ValueError("query and cache head dimensions must match")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    if context_lens.shape != (batch,):
        raise ValueError("context_lens must have shape [batch]")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch:
        raise ValueError("block_tables must have shape [batch, max_blocks]")
    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("paged_decode_attention requires CUDA tensors")
    if block_n is None:
        block_n = 64
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError("block_n must be a positive power of two")
    if num_warps is None:
        num_warps = _default_num_warps(head_dim)
    if num_warps not in {1, 2, 4, 8, 16}:
        raise ValueError("num_warps must be one of 1, 2, 4, 8, 16")

    use_scales = k_cache.dtype == torch.float8_e4m3fn
    if use_scales:
        if k_scale_cache is None or v_scale_cache is None:
            raise ValueError("FP8 cache requires K/V scale caches")
        if k_scale_cache.shape != (num_blocks, cache_block_size):
            raise ValueError("K scale cache must have shape [num_blocks, block_size]")
        if v_scale_cache.shape != k_scale_cache.shape:
            raise ValueError("K/V scale cache shapes must match")
    else:
        # 保持 kernel 签名统一；USE_SCALES=False 时这些指针不会被读取。
        k_scale_cache = torch.empty((1, 1), device=q.device, dtype=torch.float16)
        v_scale_cache = torch.empty((1, 1), device=q.device, dtype=torch.float16)

    output = torch.empty_like(q)
    head_dim_padded = triton.next_power_of_2(head_dim)
    grid = (batch, num_query_heads)
    _paged_decode_attention_kernel[grid](
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_tables,
        context_lens,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        v_scale_cache.stride(0),
        v_scale_cache.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        scale,
        CACHE_BLOCK_SIZE=cache_block_size,
        MAX_BLOCKS=block_tables.shape[1],
        BLOCK_N=block_n,
        HEAD_DIM=head_dim_padded,
        ACTUAL_HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=num_kv_heads,
        USE_SCALES=use_scales,
        num_warps=num_warps,
    )
    return output


@triton.jit
def _dequantize_paged_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    sequence_ids_ptr,
    positions_ptr,
    out_k_ptr,
    out_v_ptr,
    k_stride_block,
    k_stride_token,
    k_stride_head,
    k_stride_d,
    v_stride_block,
    v_stride_token,
    v_stride_head,
    v_stride_d,
    scale_stride_block,
    scale_stride_token,
    block_table_stride_b,
    block_table_stride_block,
    out_stride_token,
    out_stride_head,
    out_stride_d,
    CACHE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
):
    """把分页 FP8 cache 的一个 (token, kv_head) 解量化到连续输出。"""
    token_id = tl.program_id(0)
    kv_head = tl.program_id(1)
    position = tl.load(positions_ptr + token_id)
    sequence_id = tl.load(sequence_ids_ptr + token_id)
    logical_block = position // CACHE_BLOCK_SIZE
    offset_in_block = position % CACHE_BLOCK_SIZE
    physical_block = tl.load(
        block_tables_ptr
        + sequence_id * block_table_stride_b
        + logical_block * block_table_stride_block
    )
    offs_d = tl.arange(0, HEAD_DIM)
    mask = offs_d < ACTUAL_HEAD_DIM
    k_ptrs = (
        k_cache_ptr
        + physical_block * k_stride_block
        + offset_in_block * k_stride_token
        + kv_head * k_stride_head
        + offs_d * k_stride_d
    )
    v_ptrs = (
        v_cache_ptr
        + physical_block * v_stride_block
        + offset_in_block * v_stride_token
        + kv_head * v_stride_head
        + offs_d * v_stride_d
    )
    scale_offset = physical_block * scale_stride_block + offset_in_block * scale_stride_token
    k_scale = tl.load(k_scale_ptr + scale_offset).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + scale_offset).to(tl.float32)
    k = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32) * k_scale
    v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32) * v_scale
    out_k_ptrs = out_k_ptr + token_id * out_stride_token + kv_head * out_stride_head + offs_d * out_stride_d
    out_v_ptrs = out_v_ptr + token_id * out_stride_token + kv_head * out_stride_head + offs_d * out_stride_d
    tl.store(out_k_ptrs, k, mask=mask)
    tl.store(out_v_ptrs, v, mask=mask)


def dequantize_paged_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_ids: torch.Tensor,
    positions: torch.Tensor,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 FP8 paged cache 解量化成 [total_k, num_kv_heads, head_dim]。"""
    if k_cache.dtype != torch.float8_e4m3fn or v_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("dequantize_paged_kvcache expects FP8 K/V cache")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache shapes must match")
    if sequence_ids.shape != positions.shape or sequence_ids.ndim != 1:
        raise ValueError("sequence_ids and positions must be one-dimensional and aligned")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [batch, max_blocks]")
    if not all(t.is_cuda for t in (k_cache, v_cache, k_scale_cache, v_scale_cache, block_tables, sequence_ids, positions)):
        raise ValueError("dequantize_paged_kvcache requires CUDA tensors")
    num_blocks, cache_block_size, num_kv_heads, head_dim = k_cache.shape
    if k_scale_cache.shape != (num_blocks, cache_block_size):
        raise ValueError("K scale cache must have shape [num_blocks, block_size]")
    if v_scale_cache.shape != k_scale_cache.shape:
        raise ValueError("K/V scale cache shapes must match")
    total_k = sequence_ids.numel()
    out_k = torch.empty((total_k, num_kv_heads, head_dim), device=k_cache.device, dtype=output_dtype)
    out_v = torch.empty_like(out_k)
    head_dim_padded = triton.next_power_of_2(head_dim)
    _dequantize_paged_kvcache_kernel[(total_k, num_kv_heads)](
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_tables,
        sequence_ids,
        positions,
        out_k,
        out_v,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        CACHE_BLOCK_SIZE=cache_block_size,
        HEAD_DIM=head_dim_padded,
        ACTUAL_HEAD_DIM=head_dim,
        num_warps=4 if head_dim_padded >= 128 else 2,
    )
    return out_k, out_v


def benchmark_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    scale: float,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
    candidates: tuple[tuple[int, int], ...] = ATTENTION_CONFIGS,
    warmup: int = 20,
    repeats: int = 50,
) -> list[dict[str, float | int]]:
    """离线扫参；返回按 kernel latency 排序的配置列表。"""
    rows = []
    for block_n, num_warps in candidates:
        latency_ms = _timed(
            lambda: paged_decode_attention(
                q,
                k_cache,
                v_cache,
                context_lens,
                block_tables,
                scale,
                k_scale_cache,
                v_scale_cache,
                block_n,
                num_warps,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        rows.append({"block_n": block_n, "num_warps": num_warps, "latency_ms": latency_ms})
    return sorted(rows, key=lambda row: row["latency_ms"])


def _timed(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats
