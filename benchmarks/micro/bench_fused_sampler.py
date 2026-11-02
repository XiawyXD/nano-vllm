"""Triton fused greedy sampler 的正确性和简单速度检查。"""

import time

import torch

from nanovllm.layers.fused_sampler import fused_greedy_argmax


def main():
    if not torch.cuda.is_available():
        print({"status": "SKIP", "reason": "CUDA is unavailable"})
        return
    device = "cuda"
    batch, vocab = 8, 151936
    logits = torch.randn(batch, vocab, device=device, dtype=torch.float16)
    expected = logits.argmax(dim=-1)
    actual = fused_greedy_argmax(logits)
    torch.testing.assert_close(actual, expected)
    # 预热 Triton 编译后再计时，避免把一次性编译时间混进结果。
    for _ in range(10):
        fused_greedy_argmax(logits)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        fused_greedy_argmax(logits)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - start) * 1000 / 50
    start = time.perf_counter()
    for _ in range(50):
        logits.argmax(dim=-1)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) * 1000 / 50
    print({
        "status": "PASS",
        "batch": batch,
        "vocab": vocab,
        "max_abs_id_diff": int((actual - expected).abs().max().item()),
        "fused_ms": fused_ms,
        "torch_argmax_ms": torch_ms,
        "speedup_vs_torch_argmax": torch_ms / fused_ms,
    })


if __name__ == "__main__":
    main()
