"""单卡Tensor Core线性层/SwiGLU：首调用与五轮交错稳态基准，失败不放宽公差。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from benchmarks.tensorcore_utils import (compiled_activation, compare, make_inputs,
    output_path, provenance, reference, save_report, shapes, validate_cli)


def first_call(fn):
    """CUDA同步包围第一次调用；记录含JIT/编译/分配的wall时间，不冒充纯kernel延迟。"""
    torch.cuda.synchronize()
    began = time.perf_counter()
    output = fn()
    torch.cuda.synchronize()
    return output, (time.perf_counter()-began)*1000


def event_time(fn, iterations):
    """用CUDA events测一组调用的设备经过时间，除以调用次数；包含该路径launch间隔。"""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)/iterations


def calibrate(fn, target_ms, max_iterations):
    """有限预热后估算迭代数，使每轮接近--ms；max-iters防止超快case无限放大循环。"""
    for _ in range(5):
        fn()
    measured = max(event_time(fn, 5), 1e-6)
    iterations = max(1, min(max_iterations, math.ceil(target_ms/measured)))
    return iterations, measured


def measure_pair(functions, args, calls_per_iteration=1):
    """每轮交替baseline/custom顺序，用5轮结果的median和范围抑制顺序与时钟偏差。"""
    calibrated = {name: calibrate(fn, args.ms, args.max_iters) for name, fn in functions.items()}
    samples = {name: [] for name in functions}
    orders = []
    names = list(functions)
    for repeat in range(5):
        order = names if repeat % 2 == 0 else names[::-1]
        orders.append(order)
        for name in order:
            samples[name].append(event_time(functions[name], calibrated[name][0])/calls_per_iteration)
    result = {name: dict(iterations_per_round=calibrated[name][0],
                        calibration_ms=calibrated[name][1], rounds_ms=values,
                        median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values))
              for name, values in samples.items()}
    return dict(backends=result, round_orders=orders, calls_per_iteration=calls_per_iteration,
                speedup_baseline_over_custom=result['baseline']['median_ms']/result['custom']['median_ms'])


def graph_measure_pair(functions, args):
    """每个后端独立捕获固定10次调用；重放计时除以10，减少CPU逐次enqueue间隙。"""
    captures, outputs, capture_ms = {}, {}, {}
    for name, fn in functions.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        began = time.perf_counter()
        values = []
        with torch.cuda.graph(graph):
            for _ in range(10):
                values.append(fn())
        torch.cuda.synchronize()
        captures[name], outputs[name] = graph, values
        capture_ms[name] = (time.perf_counter()-began)*1000
    for graph in captures.values():
        graph.replay()
    torch.cuda.synchronize()
    correctness = compare(outputs['custom'][-1], outputs['baseline'][-1], args.atol, args.rtol)
    if not correctness['allclose']:
        return dict(capture_wall_ms=capture_ms, output_comparison=correctness,
                    timing=None, note='graph output failed preset tolerance; no valid speedup')
    measured = measure_pair({name: graph.replay for name, graph in captures.items()}, args,
                            calls_per_iteration=10)
    return dict(capture_wall_ms=capture_ms, output_comparison=correctness, timing=measured,
                note='Each graph replay contains 10 operator calls; reported round/median/min/max are per operator call. Capture wall time is excluded.')


def main():
    """依次创建一个形状的张量，先检验再稳态测量；产物完整记录形状和后端。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--ms', type=float, default=10.0, help='target CUDA milliseconds per backend per round')
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--graph-timing', action='store_true', help='also time independent CUDA graphs containing 10 calls each')
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--atol', type=float, default=0.02)
    parser.add_argument('--rtol', type=float, default=0.02)
    args = parser.parse_args()
    validate_cli(args)
    if not math.isfinite(args.ms) or not 0 < args.ms <= 100 or not 1 <= args.max_iters <= 10000:
        parser.error('--ms must be in (0,100] and --max-iters in [1,10000]')
    target = output_path(args.output)
    report = dict(status='RUNNING', args=vars(args), provenance=provenance(), cases=[],
        rounds=5, input_distribution='x~N(0,1), weight~N(0,1/sqrt(K)), then BF16 arithmetic',
        timing_note='first_call_wall_ms includes compilation/initialization and may reuse on-disk compiler cache; CUDA-event steady times are five interleaved rounds. timing is the compatibility alias of eager_timing. graph_timing is separate, not full-model speedup.')
    began = time.perf_counter()
    try:
        from nanovllm.layers.tensorcore_mlp import bf16_linear, fused_swiglu
        activation = compiled_activation()
        with torch.inference_mode():
            for index, (name, m, k, intermediate) in enumerate(shapes(args.quick)):
                for kind, input_dim, output_dim, operation in (
                    ('gate_up_linear', k, 2*intermediate, bf16_linear),
                    ('down_linear', intermediate, k, bf16_linear),
                    ('fused_swiglu', k, 2*intermediate, fused_swiglu),
                ):
                    x, weight = make_inputs(m, input_dim, output_dim, args.seed+index)
                    baseline = ((lambda: activation(F.linear(x, weight))) if kind == 'fused_swiglu'
                                else (lambda: F.linear(x, weight)))
                    custom = lambda: operation(x, weight)
                    functions = dict(baseline=baseline, custom=custom)
                    first_order = ('baseline', 'custom') if index % 2 == 0 else ('custom', 'baseline')
                    initial, cold = {}, {}
                    for backend in first_order:
                        initial[backend], cold[backend] = first_call(functions[backend])
                    fp32 = reference(kind, x, weight)
                    numerical = dict(against_baseline=compare(initial['custom'], initial['baseline'], args.atol, args.rtol),
                                     against_fp32=compare(initial['custom'], fp32, args.atol, args.rtol),
                                     baseline_against_fp32=compare(initial['baseline'], fp32, args.atol, args.rtol))
                    valid = numerical['against_baseline']['allclose'] and numerical['against_fp32']['allclose']
                    row = dict(case=name, operation=kind, x_shape=list(x.shape), weight_shape=list(weight.shape),
                               output_dtype=str(initial['custom'].dtype), first_call_order=first_order,
                               first_call_wall_ms=cold, numerical=numerical, numerical_passed=valid)
                    # 数值失败仍保留原结果，但不把错误实现的计时作为有效加速比。
                    row['timing'] = measure_pair(functions, args) if valid else None
                    row['eager_timing'] = row['timing']
                    row['graph_timing'] = graph_measure_pair(functions, args) if valid and args.graph_timing else None
                    if row['graph_timing'] is not None and not row['graph_timing']['output_comparison']['allclose']:
                        row['numerical_passed'] = False
                    if not valid:
                        row['timing_skipped_reason'] = 'predeclared numerical tolerance failed'
                    report['cases'].append(row)
                    save_report(target, report)
                    print(json.dumps(dict(case=name, operation=kind, numerical_passed=valid,
                        speedup=(row['timing'] or {}).get('speedup_baseline_over_custom'))), flush=True)
        report['status'] = 'PASS' if all(row['numerical_passed'] for row in report['cases']) else 'REVIEW_REQUIRED'
    except Exception as exc:
        report['status'], report['error'] = 'FAIL', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-began
        save_report(target, report)
    if report['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
