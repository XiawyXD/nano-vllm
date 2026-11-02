"""稳定的 CUDA RoPE：单次 launch 处理 Q/K，不依赖 Dynamo 编译缓存。

只支持本项目的 full-head、split-half RoPE，不支持 interleaved/partial RoPE。
Q/K 为 [tokens, heads, head_dim]，cache 为 FP32 [positions, 1, head_dim]。
输入可以是不连续视图；输出是新分配的连续张量，不修改输入。
CPU reference 仍由 rotary_embedding.apply_rotary_emb 提供。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    P, Q, K, CACHE, OUT_Q, OUT_K,
    PS: tl.constexpr,
    QT: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
    KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
    CT: tl.constexpr, CD: tl.constexpr,
    HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    MAX_POSITION: tl.constexpr, BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    half: tl.constexpr = D // 2
    position = tl.load(P + token * PS).to(tl.int64)
    valid_position = (position >= 0) & (position < MAX_POSITION)
    # 始终开启 device assert；mask 同时保证错误 position 也不会越界读。
    # 非法 position 是调用错误，CUDA 异步 assert 后应结束该进程。
    tl.device_assert(valid_position, "RoPE position must be in [0, cache.shape[0])")
    mask = (col < half) & valid_position
    cos = tl.load(CACHE + position * CT + col * CD, mask, other=0.0)
    sin = tl.load(CACHE + position * CT + (col + half) * CD, mask, other=0.0)
    if head < HQ:
        src = Q + token * QT + head * QH
        x1 = tl.load(src + col * QD, mask, other=0.0).to(tl.float32)
        x2 = tl.load(src + (col + half) * QD, mask, other=0.0).to(tl.float32)
        dst = OUT_Q + (token * HQ + head) * D
    else:
        src = K + token * KT + (head - HQ) * KH
        x1 = tl.load(src + col * KD, mask, other=0.0).to(tl.float32)
        x2 = tl.load(src + (col + half) * KD, mask, other=0.0).to(tl.float32)
        dst = OUT_K + (token * HK + head - HQ) * D
    # 必须分别舍入四次 FP32 乘法，再做加/减，最后转换回输入 dtype。
    # 不允许编译器把 a*b +/- c*d 收缩成 FMA，否则不同执行路径会差几个 ULP。
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    tl.store(dst + col, y1, col < half)
    tl.store(dst + col + half, y2, col < half)


def _validate(positions, query, key, cos_sin_cache):
    tensors = (positions, query, key, cos_sin_cache)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("fused_rotary requires four torch.Tensor inputs")
    if any(x.layout != torch.strided for x in tensors):
        raise ValueError("fused_rotary requires strided tensors")
    # 惰性负号/共轭标记不体现在 stride 或裸 storage 中，不能静默忽略。
    if any(x.is_neg() or x.is_conj() for x in tensors):
        raise ValueError("fused_rotary does not accept unresolved negative/conjugate views")
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("Q/K must have shape [tokens, heads, head_dim]")
    n, hq, d = query.shape
    if key.shape[0] != n or key.shape[2] != d or hq < 1 or key.shape[1] < 1:
        raise ValueError("Q/K require matching token/head dimensions and positive head counts")
    if d < 2 or d > 512 or d % 2:
        raise ValueError("full-head RoPE requires an even head_dim in [2, 512]")
    if positions.ndim != 1 or positions.shape[0] != n:
        raise ValueError("positions must have shape [tokens]")
    if cos_sin_cache.ndim != 3 or cos_sin_cache.shape[1:] != (1, d) or cos_sin_cache.shape[0] < 1:
        raise ValueError("cos_sin_cache must have shape [max_position > 0, 1, head_dim]")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32) or key.dtype != query.dtype:
        raise TypeError("Q/K must have the same FP16, BF16 or FP32 dtype")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("positions must be INT32 or INT64")
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError("cos_sin_cache must be FP32, matching the existing RoPE reference")
    if any(x.device != query.device for x in tensors) or not query.is_cuda:
        raise ValueError("all fused_rotary inputs must be on the same CUDA device")
    if any(any(s < 0 for s in x.stride()) for x in tensors):
        raise ValueError("negative strides are unsupported")
    if any(x.requires_grad for x in (query, key, cos_sin_cache)):
        raise ValueError("fused_rotary is inference-only; autograd inputs are unsupported")


def _launch(positions, query, key, cache, out_q, out_k):
    """内部入口；连续输出由调用方保证。测试用带前后哨兵的输出检查写边界。"""
    n, hq, d = query.shape
    if n == 0:
        return
    with torch.cuda.device(query.device):
        _rotary_kernel[(n, hq + key.shape[1])](
            positions, query, key, cache, out_q, out_k,
            positions.stride(0), *query.stride(), *key.stride(),
            cache.stride(0), cache.stride(2), hq, key.shape[1], d,
            cache.shape[0], BLOCK=triton.next_power_of_2(d // 2),
            num_warps=4, enable_fp_fusion=False, debug=True,
        )


def fused_rotary(positions, query, key, cos_sin_cache):
    """返回新 Q/K；不读取 GPU 数据到 CPU，可用于 CUDA Graph capture/replay。

    形状、dtype、设备错误在 launch 前报错；position 值必须非负且小于
    cache 长度，越界触发 GPU assert（异步报告），而非 clamp/静默替换。
    不调整全局 Dynamo 设置，不提供有损近似或容差回退。
    """
    _validate(positions, query, key, cos_sin_cache)
    out_q = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    out_k = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    _launch(positions, query, key, cos_sin_cache, out_q, out_k)
    return out_q, out_k
