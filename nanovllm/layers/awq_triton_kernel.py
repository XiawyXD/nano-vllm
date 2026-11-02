"""Packed AWQ GEMM for FP16 activations, no full-weight dequantization fallback.

The AWQ packing and tile-local dequantization contract follows the Apache-2.0
vLLM reference at v0.17.1/vllm/model_executor/layers/quantization/awq_triton.py.
This independent adapter uses direct nibble addressing, FP32 accumulation and
a FP32 split-K scratch/reduction. It is not a claim of identical vLLM arithmetic
or Marlin-level performance. Scratch depends on M*N, never the full K*N weight.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _awq_tile_gemm(X, W, Z, S, P, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, GROUP: tl.constexpr, SPLIT: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    nt = (N + BN - 1) // BN
    rows = (tile // nt) * BM + tl.arange(0, BM)
    cols = (tile % nt) * BN + tl.arange(0, BN)
    # Logical channels 0..7 reside in packed nibble slots 0,4,1,5,2,6,3,7.
    lane = cols % 8
    shifts = ((lane // 2) + (lane % 2) * 4) * 4
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range((K + BK * SPLIT - 1) // (BK * SPLIT)):
        base = (step * SPLIT + part) * BK
        kk = base + tl.arange(0, BK)
        a = tl.load(X + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        word = tl.load(W + kk[:, None] * (N // 8) + cols[None, :] // 8,
                       (kk[:, None] < K) & (cols[None, :] < N), other=0)
        # BK divides GROUP, so all K rows in this tile share scale/zero rows.
        group = base // GROUP
        zw = tl.load(Z + group * (N // 8) + cols // 8,
                     (group < K // GROUP) & (cols < N), other=0)
        scale = tl.load(S + group * N + cols,
                        (group < K // GROUP) & (cols < N), other=0).to(tl.float32)
        q = (word >> shifts[None, :]) & 15
        zero = (zw >> shifts) & 15
        b = ((q.to(tl.float32) - zero[None, :].to(tl.float32)) * scale[None, :]).to(tl.float16)
        acc += tl.dot(a, b)
    tl.store(P + part * M * N + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _awq_reduce(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, SPLIT)
    partial = tl.load(P + parts[:, None] * SIZE + offsets[None, :], offsets[None, :] < SIZE, other=0)
    tl.store(Y + offsets, tl.sum(partial, axis=0).to(tl.float16), offsets < SIZE)


def awq_packed_gemm(x, qweight, qzeros, scales, group_size=128):
    if x.device.type != "cuda" or x.dtype != torch.float16 or x.ndim < 2 or not x.is_contiguous():
        raise ValueError("AWQ GEMM requires contiguous CUDA FP16 input with rank >= 2")
    if any(t.device != x.device or not t.is_contiguous() for t in (qweight, qzeros, scales)):
        raise ValueError("AWQ tensors must be contiguous and on the input CUDA device")
    if qweight.ndim != 2 or scales.ndim != 2 or qzeros.ndim != 2:
        raise ValueError("AWQ tensors must be matrices")
    k, n = x.shape[-1], qweight.shape[1] * 8
    m = x.numel() // k if k else 0
    if (group_size != 128 or k <= 0 or k % 128 or n <= 0
            or qweight.shape != (k, n // 8) or qzeros.shape != (k // 128, n // 8)
            or scales.shape != (k // 128, n)
            or qweight.dtype != torch.int32 or qzeros.dtype != torch.int32
            or scales.dtype not in (torch.float16, torch.bfloat16)):
        raise ValueError("invalid AWQ GEMM dimensions, dtypes, or group size")
    if m == 0:
        return torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    # Conservative deterministic launch choice; no benchmarking/autotuning on
    # request paths. Revalidate/tune on SM89 only after acquiring the GPU lease.
    split = 4 if m <= 16 else 1
    bm, bn, bk = (16 if m <= 16 else 32), 64, 32
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    grid = (((m + bm - 1) // bm) * ((n + bn - 1) // bn), split)
    _awq_tile_gemm[grid](x, qweight, qzeros, scales, partial, m, n, k, group_size,
                          split, bm, bn, bk, num_warps=4)
    _awq_reduce[((m * n + 255) // 256,)](partial, output, m * n, split, 256, num_warps=4)
    return output
