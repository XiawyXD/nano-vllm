"""Correctness, microbenchmark, and offline autotune sweep for paged Decode Attention.

The sweep is intentionally offline: it selects a launch configuration for a
fixed workload, while the production engine keeps an explicit static choice so
CUDA Graph capture remains safe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.attention_triton import (
    ATTENTION_CONFIGS,
    benchmark_paged_decode_attention,
    paged_decode_attention,
)


def timed(fn, warmup: int, repeats: int) -> float:
    """Measure average CUDA time after warmup iterations."""
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


def make_inputs(args: argparse.Namespace):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    max_blocks = (args.context + args.block_size - 1) // args.block_size
    q = torch.randn(args.batch, args.heads, args.head_dim, device="cuda", dtype=dtype)
    num_blocks = args.batch * max_blocks
    k_cache = torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).view(args.batch, max_blocks)
    context_lens = torch.tensor(
        [
            max(1, args.context - (i % 3) * args.block_size // 2)
            for i in range(args.batch)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    return q, k_cache, v_cache, context_lens, block_tables, args.head_dim ** -0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be >= 0 and --repeats must be > 0")
    torch.manual_seed(args.seed)
    q, k_cache, v_cache, context_lens, block_tables, scale = make_inputs(args)

    reference = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context_lens,
        block_table=block_tables,
        softmax_scale=scale,
        causal=True,
    ).squeeze(1)
    output = paged_decode_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        scale,
        block_n=args.block_n,
        num_warps=args.num_warps,
    )
    torch.cuda.synchronize()
    max_abs_error = (reference.float() - output.float()).abs().max().item()
    atol = 0.02 if args.dtype == "float16" else 0.08
    torch.testing.assert_close(output.float(), reference.float(), atol=atol, rtol=0.02)

    flash_ms = timed(
        lambda: flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=scale,
            causal=True,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    triton_ms = timed(
        lambda: paged_decode_attention(
            q,
            k_cache,
            v_cache,
            context_lens,
            block_tables,
            scale,
            block_n=args.block_n,
            num_warps=args.num_warps,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    result = {
        "config": {
            "batch": args.batch,
            "heads": args.heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "context": args.context,
            "dtype": args.dtype,
            "seed": args.seed,
            "selected_block_n": args.block_n or 64,
            "selected_num_warps": args.num_warps,
        },
        "correctness": {
            "max_abs_error": max_abs_error,
            "atol": atol,
        },
        "baseline": {
            "flash_ms": flash_ms,
            "triton_ms": triton_ms,
            "speedup": flash_ms / triton_ms,
        },
    }

    if args.sweep:
        result["sweep"] = benchmark_paged_decode_attention(
            q,
            k_cache,
            v_cache,
            context_lens,
            block_tables,
            scale,
            candidates=ATTENTION_CONFIGS,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        result["best"] = result["sweep"][0]

    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
