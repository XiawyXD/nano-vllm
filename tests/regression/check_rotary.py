"""独立 RoPE 回归：CPU 参数拒绝 + GPU eager/reference/graph 逐位相等。

示例（从项目根目录运行，输出必须是新文件）：
  CUDA_VISIBLE_DEVICES='' python tests/regression/check_rotary.py --cpu --output /tmp/rope-cpu.json
  CUDA_VISIBLE_DEVICES=<GPU2完整UUID> NANOVLLM_BENCH_GPU_INDEX=2 \
    python tests/regression/check_rotary.py --output /tmp/rope-gpu.json
  同样环境下也可使用 compute-sanitizer --tool memcheck --error-exitcode 99 \
    python tests/regression/check_rotary.py --output /tmp/rope-memcheck.json

没有 atol/rtol：检查原始位模式（包括正负零），不是近似相等。
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact(a, b, name):
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(f"{name}: shape/dtype mismatch")
    integer = torch.int32 if a.dtype == torch.float32 else torch.int16
    different = a.contiguous().view(integer) != b.contiguous().view(integer)
    count = int(different.sum().item())
    if count:
        raise AssertionError(f"{name}: {count} bitwise-different elements")


def reference(rope, p, q, k, cache):
    cos, sin = cache[p].chunk(2, dim=-1)
    return rope.apply_rotary_emb(q, cos, sin), rope.apply_rotary_emb(k, cos, sin)


def cpu_tests(fused, rope):
    import torch
    p = torch.tensor([0, 1], dtype=torch.int64)
    q = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8)
    k = q[:, :1]
    cache = torch.cat((torch.ones(2, 1, 4), torch.zeros(2, 1, 4)), -1)
    expected = reference(rope, p, q, k, cache)
    exact(expected[0], q, "CPU identity Q")
    exact(expected[1], k, "CPU identity K")
    cases = [
        ("cpu_device", (p, q, k, cache), ValueError, "CUDA"),
        ("not_tensor", (None, q, k, cache), TypeError, "torch.Tensor"),
        ("q_rank", (p, q[0], k, cache), ValueError, "shape"),
        ("key_tokens", (p, q, k[:1], cache), ValueError, "matching"),
        ("key_dim", (p, q, k[:, :, :6], cache), ValueError, "matching"),
        ("zero_heads", (p, q[:, :0], k, cache), ValueError, "head counts"),
        ("odd_dim", (p, q[:, :, :7], k[:, :, :7], cache[:, :, :7]), ValueError, "even"),
        ("large_dim", (p, torch.empty(2, 1, 514), torch.empty(2, 1, 514), torch.empty(2, 1, 514)), ValueError, "head_dim"),
        ("position_shape", (p[:, None], q, k, cache), ValueError, "positions"),
        ("position_count", (p[:1], q, k, cache), ValueError, "positions"),
        ("cache_rank", (p, q, k, cache[:, 0]), ValueError, "cache"),
        ("cache_empty", (p, q, k, cache[:0]), ValueError, "cache"),
        ("q_dtype", (p, q.double(), k.double(), cache), TypeError, "FP16"),
        ("mixed_dtype", (p, q.half(), k, cache), TypeError, "same"),
        ("position_dtype", (p.float(), q, k, cache), TypeError, "INT32"),
        ("cache_dtype", (p, q, k, cache.half()), TypeError, "FP32"),
        ("negative_q_view", (p, torch._neg_view(q), k, cache), ValueError, "unresolved"),
        ("negative_k_view", (p, q, torch._neg_view(k), cache), ValueError, "unresolved"),
        ("negative_position_view", (torch._neg_view(p), q, k, cache), ValueError, "unresolved"),
        ("negative_cache_view", (p, q, k, torch._neg_view(cache)), ValueError, "unresolved"),
    ]
    passed = []
    for name, inputs, error, message in cases:
        try:
            fused.fused_rotary(*inputs)
        except error as exc:
            if message not in str(exc):
                raise AssertionError(f"{name}: unexpected error {exc}") from exc
        else:
            raise AssertionError(f"{name}: did not reject invalid input")
        passed.append(name)
    return dict(status="PASS", rejected=passed, reference_identity=2)


def strided_copy(x, enabled):
    import torch
    if not enabled:
        backing = x.clone()
        return backing, backing
    # 每一维均跳步，前后都有 padding；调用方保存整个 backing 以检查输入未修改。
    shape = tuple(2 * s + 3 for s in x.shape)
    backing = torch.full(shape, 17, dtype=x.dtype, device=x.device)
    view = backing[tuple(slice(1, 1 + 2 * s, 2) for s in x.shape)]
    view.copy_(x)
    return view, backing


def gpu_case(fused, rope, n, dtype, d, noncontig, extreme, graph, guard):
    import torch
    guard()
    hq, hk, capacity = 5, 2, 1025
    q = torch.randn(n, hq, d, device="cuda", dtype=dtype)
    k = torch.randn(n, hk, d, device="cuda", dtype=dtype)
    angle = torch.randn(capacity, 1, d // 2, device="cuda", dtype=torch.float32)
    cache = torch.cat((angle.cos(), angle.sin()), -1)
    # 既包含边界又打乱/重复 position，不能用 token 索引替代 position。
    p = ((torch.arange(n, device="cuda", dtype=torch.int64) * 37 + 255) % capacity)
    if n:
        p[0] = capacity - 1
    if extreme:
        info = torch.finfo(dtype)
        values = torch.tensor([0., -0., info.tiny, -info.tiny, info.max / 8,
                               -info.max / 8, 1., -1.], device="cuda", dtype=dtype)
        for x in (q, k):
            x.copy_(values.repeat((x.numel() + 7) // 8)[:x.numel()].reshape_as(x))
            # 相近两半配相同 cos/sin，暴露“乘法分别舍入”与 FMA 收缩的差别。
            x[..., d // 2:].copy_(torch.nextafter(x[..., :d // 2], torch.full_like(x[..., :d // 2], float("inf"))))
        cache[..., d // 2:] = cache[..., :d // 2]
    inputs, backings = [], []
    for x in (p, q, k, cache):
        view, backing = strided_copy(x, noncontig)
        inputs.append(view)
        backings.append(backing)
    p, q, k, cache = inputs
    saved = [x.clone() for x in backings]
    expected = reference(rope, p, q, k, cache)
    result = fused.fused_rotary(p, q, k, cache)
    for label, actual, ref in zip(("Q", "K"), result, expected):
        exact(actual, ref, label)
        if actual.numel() and actual.data_ptr() in (q.data_ptr(), k.data_ptr()):
            raise AssertionError("output aliases an input")
    # 检查内部 kernel 的输出写入边界：同一 kernel、相同 launch 配置，额外输出哨兵。
    protected, outputs = [], []
    for x in (q, k):
        backing = torch.full((x.numel() + 66,), 23, dtype=dtype, device="cuda")
        protected.append(backing)
        outputs.append(backing[33:33 + x.numel()].view(x.shape))
    fused._launch(p, q, k, cache, *outputs)
    for actual, ref, backing in zip(outputs, expected, protected):
        exact(actual, ref, "sentinel output")
        if not bool(torch.all(backing[:33] == 23) & torch.all(backing[-33:] == 23)):
            raise AssertionError("kernel overwrote output canary")
    replay_count = 0
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused.fused_rotary(p, q, k, cache)
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            captured = fused.fused_rotary(p, q, k, cache)
        for _ in range(5):
            g.replay()
            for actual, ref in zip(captured, expected):
                exact(actual, ref, "graph/reference")
            replay_count += 1
        # 同一 graph 更换 position 值，证明不是捕获了上一次 reference/位置。
        p.copy_((p + 1) % capacity)
        updated = reference(rope, p, q, k, cache)
        position_snapshot = backings[0].clone()
        g.replay()
        for actual, ref in zip(captured, updated):
            exact(actual, ref, "graph changed positions")
        if not torch.equal(backings[0], position_snapshot):
            raise AssertionError("graph mutated positions")
        # 恢复测试主动修改的 position，然后检查其他 backing 和 padding。
        backings[0].copy_(saved[0])
        replay_count += 1
    for i, (actual, original) in enumerate(zip(backings, saved)):
        if i == 0:
            if not torch.equal(actual, original):
                raise AssertionError("positions input mutated")
        else:
            exact(actual, original, "input backing")
    torch.cuda.synchronize()
    return dict(tokens=n, dtype=str(dtype), head_dim=d, heads=[hq, hk],
                noncontiguous=noncontig, extreme=extreme, graph_replays=replay_count,
                status="PASS", equality="bitwise", output_canaries="unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 先占用唯一输出路径，禁止覆盖此前结果。
    with args.output.open("x", encoding="utf-8") as file:
        file.write("{}\n")
    report = dict(status="RUNNING", argv=sys.argv, pid=os.getpid(), start=time.time(),
                  files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__).resolve(), ROOT / "nanovllm/layers/fused_rotary.py",
                                   ROOT / "nanovllm/layers/rotary_embedding.py")})
    try:
        import torch
        import triton
        report["environment"] = dict(torch=torch.__version__, triton=triton.__version__,
                                     cuda=torch.version.cuda, visible=os.getenv("CUDA_VISIBLE_DEVICES"))
        fused = load_file("_rotary_under_test", "nanovllm/layers/fused_rotary.py")
        rope = load_file("_rotary_reference", "nanovllm/layers/rotary_embedding.py")
        report["cpu"] = cpu_tests(fused, rope)
        if not args.cpu:
            from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
            def guard():
                return ensure_gpu1_available(evidence_path=str(args.output) + ".guard.jsonl")
            report["gpu"] = guard()
            report["device"] = torch.cuda.get_device_name()
            torch.manual_seed(20261024)
            rows = report["cases"] = []
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for n in (1, 2, 4, 8, 255, 256, 257, 507, 512):
                    for noncontig in (False, True):
                        rows.append(gpu_case(fused, rope, n, dtype, 128, noncontig, False, True, guard))
                for d in (2, 6, 48, 64, 80, 96, 256, 512):
                    rows.append(gpu_case(fused, rope, 4, dtype, d, True, True, True, guard))
                rows.append(gpu_case(fused, rope, 0, dtype, 128, False, False, False, guard))
            # INT32 positions、广播 stride=0，以及设备/梯度错误的 GPU 参数校验。
            guard()
            q = torch.randn(2, 1, 128, device="cuda").expand(2, 5, 128)
            k = q[:, :2]
            p = torch.tensor([0, 1024], device="cuda", dtype=torch.int32)
            cache = torch.randn(1025, 1, 128, device="cuda")
            for a, b in zip(fused.fused_rotary(p, q, k, cache), reference(rope, p, q, k, cache)):
                exact(a, b, "broadcast/int32")
            for name, inputs in (("mixed_device", (p.cpu(), q, k, cache)),
                                 ("autograd", (p, q.detach().requires_grad_(), k, cache))):
                try:
                    fused.fused_rotary(*inputs)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"did not reject {name}")
            report["gpu_extra"] = ["broadcast_int32_exact", "mixed_device_rejected", "autograd_rejected"]
            guard()
        report["status"] = "PASS"
    except BaseException:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    report["end"] = time.time()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], output=str(args.output),
                          gpu_cases=len(report.get("cases", [])),
                          error=report.get("traceback")), indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
