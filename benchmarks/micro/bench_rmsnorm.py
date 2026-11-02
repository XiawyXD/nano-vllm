"""Benchmark the optional Triton fused Add-RMSNorm operator."""

import argparse

import torch

from nanovllm.layers.fused_rmsnorm import fused_add_rmsnorm


def reference(x, residual, weight, eps):
    summed = x.float() + residual.float()
    output = summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + eps)
    return output.to(x.dtype) * weight, summed.to(x.dtype)


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
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    x = torch.randn(args.tokens, args.hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype)
    eps = 1e-6

    ref_y, ref_residual = reference(x, residual, weight, eps)
    fused_y, fused_residual = fused_add_rmsnorm(x, residual, weight, eps)
    torch.cuda.synchronize()
    print("max_output_error:", (ref_y.float() - fused_y.float()).abs().max().item())
    print("max_residual_error:", (ref_residual.float() - fused_residual.float()).abs().max().item())

    ref_ms = timed(lambda: reference(x, residual, weight, eps))
    fused_ms = timed(lambda: fused_add_rmsnorm(x, residual, weight, eps))
    print(f"reference_ms: {ref_ms:.4f}")
    print(f"fused_triton_ms: {fused_ms:.4f}")
    print(f"speedup: {ref_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
