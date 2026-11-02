"""Lease-gated synthetic packed AWQ GPU checks; never a model-quality verdict.

Freeze-only imports no CUDA code. All thresholds are constants, no relaxation
CLI. Dense dequantized tensors here are TEST REFERENCES, not model runtime.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


GPU_UUIDS = {1: "GPU-01bfd1ad-d71e-726e-4265-b3869b42b5d2",
             2: "GPU-fae44001-5277-3d60-bd3a-0e634ed0e6a8"}
LIMITS = dict(atol=0.02, rtol=0.005, normalized_l2=0.0015,
              graph_eager_exact=True, replay_memory_growth_bytes=0,
              finite_required=True)
SEED = 20261023


def cases(suite):
    edges = [(1, 128, 8), (3, 384, 72), (17, 256, 128)]
    if suite == "smoke":
        return edges + [(1, 4096, 6144), (8, 4096, 24576), (32, 12288, 4096)]
    return edges + [(m, k, n) for k, n in ((4096, 6144), (4096, 24576),
                                          (12288, 4096), (4096, 4096))
                    for m in (1, 2, 4, 8, 16, 32, 128, 256)]


def freeze(suite):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "nanovllm/layers/awq_packed.py",
             root / "nanovllm/layers/awq_triton_kernel.py"]
    return dict(schema=1, seed=SEED, suite=suite, cases=cases(suite),
                thresholds=LIMITS, repeats=5, warmup=10, iterations=30,
                graph_changed_inputs=10, graph_memory_replays=100,
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths},
                primary_reference="same packed payload dequantized FP16; FP32 matmul, TF32 off",
                bf16_role="performance comparison only; not AWQ activation dtype",
                performance_gate="none: report raw trials and speedup; no promised speedup",
                quality_manifest_sha256="f17787f84fe49b2376e4da32c953f6fccf89a8485078840c32c943b0d5e320f0",
                model_quality_test=False)


def timing(torch, fn, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return dict(cuda_ms=start.elapsed_time(end) / iterations,
                wall_ms=(time.perf_counter() - wall_start) * 1000 / iterations)


def check_error(torch, output, reference):
    error = (output.float() - reference).abs()
    finite = bool(torch.isfinite(output).all() and torch.isfinite(reference).all())
    normalized = float(error.norm() / reference.norm().clamp_min(1e-12))
    elementwise = bool((error <= LIMITS["atol"] + LIMITS["rtol"] * reference.abs()).all())
    return dict(finite=finite, max_abs=float(error.max()), normalized_l2=normalized,
                elementwise_pass=elementwise,
                passed=finite and elementwise and normalized <= LIMITS["normalized_l2"])


def run_case(torch, shape, case_id, plan):
    from nanovllm.layers.awq_packed import AWQPackedLinear, pack_awq_int4_cpu, reference_weight_cpu
    m, k, n = shape
    generator = torch.Generator(device="cpu").manual_seed(SEED + case_id)
    q = torch.randint(0, 16, (k, n), generator=generator, dtype=torch.int32)
    z = torch.randint(0, 16, (k // 128, n), generator=generator, dtype=torch.int32)
    s = (torch.rand((k // 128, n), generator=generator) * .04 + .002).half()
    payload = dict(qweight=pack_awq_int4_cpu(q), qzeros=pack_awq_int4_cpu(z), scales=s)
    del q, z, s
    dense_cpu = reference_weight_cpu(**payload)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()
    module = AWQPackedLinear(k, n, device="cuda")
    for name, value in payload.items():
        parameter = getattr(module, name)
        parameter.weight_loader(parameter, value)
        if not torch.equal(parameter.cpu(), value):
            raise AssertionError("GPU checkpoint load changed tensor data")
    module.assert_loaded()
    packed_allocation = torch.cuda.memory_allocated() - base_allocated
    del payload
    x_cpu = torch.randn((m, k), generator=generator).half()
    x = x_cpu.cuda()
    # Isolated packed runtime allocation: before dense test references exist.
    for _ in range(plan["warmup"]):
        module(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = module(x)
    torch.cuda.synchronize()
    transient_peak = torch.cuda.max_memory_allocated() - before
    packed_resident = torch.cuda.memory_allocated() - base_allocated
    dense_fp16 = dense_cpu.cuda().T.contiguous()
    dense_bf16 = dense_cpu.to(torch.bfloat16).cuda().T.contiguous()
    x_bf16 = x.to(torch.bfloat16)
    reference = torch.mm(x.float(), dense_fp16.T.float())
    accuracy = check_error(torch, result, reference)
    del dense_cpu, result
    # Compile and allocator warmup on a separate stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = module(x)
    graph_output_address = graph_output.data_ptr()
    graph_exact, changed_checks = True, []
    for change in range(plan["graph_changed_inputs"]):
        # A zero input explicitly checks exact zero, not just relative errors.
        fresh = (torch.zeros_like(x_cpu) if change == 0 else
                 torch.randn((m, k), generator=generator).half()).cuda()
        x.copy_(fresh)
        graph.replay()
        eager = module(x)
        expected = torch.mm(x.float(), dense_fp16.T.float())
        test = check_error(torch, graph_output, expected)
        exact = torch.equal(graph_output, eager)
        if change == 0:
            exact = exact and bool((graph_output == 0).all())
        graph_exact = graph_exact and exact and graph_output.data_ptr() == graph_output_address
        changed_checks.append(test | dict(exact_graph_eager=exact))
        del fresh, eager, expected
    torch.cuda.synchronize()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for _ in range(plan["graph_memory_replays"]):
        graph.replay()
    torch.cuda.synchronize()
    allocation_growth = torch.cuda.memory_allocated() - allocated
    reserved_growth = torch.cuda.memory_reserved() - reserved
    # Keep both dense dtypes as explicit diagnostic/performance baselines.
    x.copy_(x_cpu)
    x_bf16.copy_(x.to(torch.bfloat16))
    methods = dict(packed_eager=lambda: module(x), packed_graph=graph.replay,
                   dense_fp16=lambda: torch.nn.functional.linear(x, dense_fp16),
                   dense_bf16=lambda: torch.nn.functional.linear(x_bf16, dense_bf16))
    # Dense graph comparisons prevent attributing Python/launch savings solely
    # to weight quantization. Keep output and graph objects alive throughout.
    dense_graphs, dense_outputs = {}, {}
    for name in ("dense_fp16", "dense_bf16"):
        with torch.cuda.stream(stream):
            for _ in range(3):
                methods[name]()
        torch.cuda.current_stream().wait_stream(stream)
        dg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(dg, stream=stream):
            dense_outputs[name] = methods[name]()
        dense_graphs[name] = dg
        methods[name + "_graph"] = dg.replay
    for fn in methods.values():
        for _ in range(plan["warmup"]):
            fn()
    trials = {name: [] for name in methods}
    names = list(methods)
    for trial in range(plan["repeats"]):
        # Rotate ordering rather than benchmark one backend always first.
        for name in names[trial % len(names):] + names[:trial % len(names)]:
            trials[name].append(timing(torch, methods[name], plan["iterations"]))
    medians = {name: statistics.median(t["cuda_ms"] for t in runs) for name, runs in trials.items()}
    passed = (accuracy["passed"] and graph_exact and all(c["passed"] for c in changed_checks)
              and allocation_growth == 0 and reserved_growth == 0)
    return dict(shape=shape, passed=passed, accuracy=accuracy, changed_inputs=changed_checks,
                exact_graph_eager=graph_exact, allocation_growth=allocation_growth,
                reserved_growth=reserved_growth, packed_parameter_allocation_bytes=packed_allocation,
                packed_runtime_peak_extra_bytes=transient_peak, packed_runtime_resident_bytes=packed_resident,
                memory_scope="packed measurements precede dense references; graph growth excludes checks",
                module_stats=module.stats(), trials=trials, median_cuda_ms=medians,
                fp16_eager_over_packed_eager=medians["dense_fp16"] / medians["packed_eager"],
                fp16_graph_over_packed_graph=medians["dense_fp16_graph"] / medians["packed_graph"],
                bf16_eager_over_packed_eager=medians["dense_bf16"] / medians["packed_eager"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "qwen"), default="smoke")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--frozen-plan", type=Path)
    parser.add_argument("--authorize-gpu-window", action="store_true")
    parser.add_argument("--gpu-index", type=int, choices=tuple(GPU_UUIDS), default=2)
    args = parser.parse_args()
    plan = freeze(args.suite)
    if args.frozen_plan and json.loads(args.frozen_plan.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError("frozen plan/source hashes changed; create separately versioned evidence, never overwrite")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.freeze_only:
        print(json.dumps(dict(status="FROZEN_NOT_RUN", cases=len(plan["cases"]), output=str(args.output))))
        return
    if not args.authorize_gpu_window:
        raise ValueError("explicit allocated GPU window required")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUIDS[args.gpu_index]
            or os.environ.get("NANOVLLM_BENCH_GPU_INDEX") != str(args.gpu_index)):
        raise ValueError("physical index and UUID must match the explicitly allocated GPU window")
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    ensure_gpu1_available(evidence_path=args.output / "guard.jsonl")
    import torch
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this validation plan is frozen for SM89")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    evidence = dict(torch=torch.__version__, cuda=torch.version.cuda,
                    physical_gpu_index=args.gpu_index, gpu_uuid=GPU_UUIDS[args.gpu_index],
                    device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                    thresholds=LIMITS, cases=[])
    with torch.inference_mode():
        for case_id, shape in enumerate(plan["cases"]):
            ensure_gpu1_available(evidence_path=args.output / "guard.jsonl")
            record = run_case(torch, shape, case_id, plan)
            with (args.output / "cases.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            evidence["cases"].append(record)
            print(json.dumps(dict(case=case_id, shape=shape, passed=record["passed"])), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    evidence["passed"] = all(c["passed"] for c in evidence["cases"])
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if not evidence["passed"]:
        raise SystemExit("FAIL: fixed thresholds violated; inspect retained evidence")


if __name__ == "__main__":
    main()
