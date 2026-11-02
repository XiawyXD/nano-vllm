"""Correctness, memory and latency experiment for the FP8 KV cache path.

This benchmark compares the experimental FP8+Triton path against the same
paged attention kernel with BF16 cache. FlashAttention remains the production
reference for the normal model-dtype cache.
"""

import argparse

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.attention_triton import paged_decode_attention


def timed(fn, warmup=30, repeats=100):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    max_blocks = (args.context + args.block_size - 1) // args.block_size
    num_tokens = args.batch * max_blocks * args.block_size
    num_blocks = args.batch * max_blocks
    key = torch.randn(num_tokens, args.kv_heads, args.head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    slots = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    fp8_k = torch.empty(num_blocks, args.block_size, args.kv_heads, args.head_dim,
                        device="cuda", dtype=torch.float8_e4m3fn)
    fp8_v = torch.empty_like(fp8_k)
    k_scales = torch.empty(num_blocks, args.block_size, device="cuda", dtype=torch.float16)
    v_scales = torch.empty_like(k_scales)
    model_k = torch.empty(num_blocks, args.block_size, args.kv_heads, args.head_dim,
                          device="cuda", dtype=dtype)
    model_v = torch.empty_like(model_k)

    store_kvcache(key, value, fp8_k, fp8_v, slots, k_scales, v_scales)
    model_k.view(num_tokens, args.kv_heads, args.head_dim).copy_(key)
    model_v.view(num_tokens, args.kv_heads, args.head_dim).copy_(value)
    torch.cuda.synchronize()

    q = torch.randn(args.batch, args.heads, args.head_dim, device="cuda", dtype=dtype)
    block_tables = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(args.batch, max_blocks)
    context_lens = torch.tensor(
        [max(1, args.context - (i % 3) * args.block_size // 2) for i in range(args.batch)],
        device="cuda", dtype=torch.int32,
    )
    scale = args.head_dim ** -0.5
    reference = flash_attn_with_kvcache(
        q.unsqueeze(1), model_k, model_v, cache_seqlens=context_lens,
        block_table=block_tables, softmax_scale=scale, causal=True,
    ).squeeze(1)
    output = paged_decode_attention(
        q, fp8_k, fp8_v, context_lens, block_tables, scale, k_scales, v_scales
    )
    torch.cuda.synchronize()
    max_error = (reference.float() - output.float()).abs().max().item()
    print("max_abs_error:", max_error)
    print("model_dtype_cache_bytes:", model_k.numel() * model_k.element_size() * 2)
    print("fp8_cache_bytes:", fp8_k.numel() * fp8_k.element_size() * 2 +
          k_scales.numel() * k_scales.element_size() * 2)
    torch.testing.assert_close(output.float(), reference.float(), atol=0.12, rtol=0.04)
    model_ms = timed(lambda: paged_decode_attention(
        q, model_k, model_v, context_lens, block_tables, scale
    ))
    fp8_ms = timed(lambda: paged_decode_attention(
        q, fp8_k, fp8_v, context_lens, block_tables, scale, k_scales, v_scales
    ))
    print(f"model_dtype_triton_ms: {model_ms:.4f}")
    print(f"fp8_triton_ms: {fp8_ms:.4f}")
    print(f"fp8_speedup_vs_model_dtype_triton: {model_ms / fp8_ms:.3f}x")


if __name__ == "__main__":
    main()
