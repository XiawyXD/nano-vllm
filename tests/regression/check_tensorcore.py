"""单卡BF16 Tensor Core算子的正确性、输入契约和CUDA Graph回归。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from benchmarks.tensorcore_utils import (compiled_activation, compare, make_inputs,
    output_path, provenance, reference, save_report, shapes, validate_cli)


def check_values(args, linear, fused, activation):
    """对真实/odd尺寸同时比较生产式BF16基线和独立FP32参考，保留所有数值失败。"""
    results = []
    for index, (name, m, k, intermediate) in enumerate(shapes(args.quick)):
        for kind, input_dim, output_dim, operation in (
            ('gate_up_linear', k, 2*intermediate, linear),
            ('down_linear', intermediate, k, linear),
            ('fused_swiglu', k, 2*intermediate, fused),
        ):
            x, weight = make_inputs(m, input_dim, output_dim, args.seed + index)
            original_x, original_weight = x.clone(), weight.clone()
            expected = F.linear(x, weight)
            if kind == 'fused_swiglu':
                expected = activation(expected)
            fp32 = reference(kind, x, weight)
            actual = operation(x, weight)
            repeated = operation(x, weight)
            torch.cuda.synchronize()
            row = dict(case=name, operation=kind, shape=list(x.shape), weight_shape=list(weight.shape),
                output_shape=list(actual.shape), dtype=str(actual.dtype),
                against_baseline=compare(actual, expected, args.atol, args.rtol),
                against_fp32=compare(actual, fp32, args.atol, args.rtol),
                baseline_against_fp32=compare(expected, fp32, args.atol, args.rtol),
                inputs_unchanged=bool(torch.equal(x, original_x) and torch.equal(weight, original_weight)),
                repeat_identical=bool(torch.equal(actual, repeated)),
                output_contract=(actual.dtype == torch.bfloat16 and actual.device == x.device and actual.is_contiguous()))
            row['passed'] = (row['against_baseline']['allclose'] and row['against_fp32']['allclose']
                             and row['inputs_unchanged'] and row['repeat_identical'] and row['output_contract'])
            results.append(row)
    return results


def check_contract(linear, fused):
    """检查M=0、NaN传播和明确拒绝的输入；不初始化或占用第二张GPU。"""
    results = []
    for name, operation, rows in (('linear', linear, 32), ('swiglu', fused, 64)):
        empty_x, weight = make_inputs(0, 64, rows, 42)
        result = operation(empty_x, weight)
        expected_columns = rows if name == 'linear' else rows//2
        assert result.shape == (0, expected_columns) and result.dtype == torch.bfloat16
        assert result.device == weight.device
        x, weight = make_inputs(3, 64, rows, 43)
        x[0, 0] = float('nan')
        result = operation(x, weight)
        assert torch.isnan(result[0]).all() and torch.isfinite(result[1:]).all()
        results.append(dict(operation=name, empty_rows=True, nan_propagation=True))
        x, weight = make_inputs(3, 64, rows, 44)
        invalid = [
            ('cpu_input', x.cpu(), weight), ('cpu_both', x.cpu(), weight.cpu()),
            ('fp32_input', x.float(), weight), ('fp32_weight', x, weight.float()),
            ('input_rank', x[0], weight), ('weight_rank', x, weight[0]),
            ('input_noncontiguous', x[:, ::2], weight[:, :32].contiguous()),
            ('weight_noncontiguous', x[:, :32].contiguous(), weight[:, ::2]),
            ('k_mismatch', x, weight[:, :-1].contiguous()),
            ('k_zero', x[:, :0], weight[:, :0]), ('n_zero', x, weight[:0]),
        ]
        if name == 'swiglu':
            invalid.append(('odd_gate_up_rows', x, weight[:-1]))
        for label, invalid_x, invalid_weight in invalid:
            try:
                operation(invalid_x, invalid_weight)
            except (ValueError, TypeError) as exc:
                results.append(dict(operation=name, case=label, rejected=True, exception=type(exc).__name__))
            else:
                raise AssertionError(f'{name} accepted unsupported case: {label}')
    return results


def check_graph(args, linear, fused, activation):
    """图捕获后更新静态输入并重放两次，检验kernel读取新值且可用于推理图。"""
    results = []
    for index, (name, operation) in enumerate((('linear', linear), ('fused_swiglu', fused))):
        x, weight = make_inputs(8, 129, 194, args.seed + 200 + index)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                operation(x, weight)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation(x, weight)
        cases = []
        for replay in range(2):
            new_x, _ = make_inputs(8, 129, 194, args.seed + 300 + replay)
            x.copy_(new_x)
            graph.replay()
            expected = F.linear(x, weight)
            if name == 'fused_swiglu':
                expected = activation(expected)
            torch.cuda.synchronize()
            comparison = compare(output, expected, args.atol, args.rtol)
            cases.append(dict(replay=replay, comparison=comparison))
        results.append(dict(operation=name, passed=all(row['comparison']['allclose'] for row in cases), replays=cases))
    return results


def main():
    """执行有界单卡检查；数值超阈记录REVIEW_REQUIRED，执行异常记录FAIL并退出非零。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--atol', type=float, default=0.02)
    parser.add_argument('--rtol', type=float, default=0.02)
    args = parser.parse_args()
    validate_cli(args)
    target = output_path(args.output)
    report = dict(status='RUNNING', args=vars(args), provenance=provenance(), checks={},
        precision_note='Candidate BF16 output and independent FP32 arithmetic share exact BF16 inputs; tolerances never adapt to results.')
    began = time.perf_counter()
    try:
        from nanovllm.layers.tensorcore_mlp import bf16_linear, fused_swiglu
        activation = compiled_activation()
        with torch.inference_mode():
            report['checks']['values'] = check_values(args, bf16_linear, fused_swiglu, activation)
            report['checks']['contract'] = check_contract(bf16_linear, fused_swiglu)
            report['checks']['cuda_graph'] = check_graph(args, bf16_linear, fused_swiglu, activation)
        failures = [row for key in ('values', 'cuda_graph') for row in report['checks'][key] if not row['passed']]
        report['status'] = 'REVIEW_REQUIRED' if failures else 'PASS'
        report['numerical_failure_count'] = len(failures)
    except Exception as exc:
        report['status'], report['error'] = 'FAIL', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds_not_a_benchmark'] = time.perf_counter()-began
        save_report(target, report)
        print(json.dumps(dict(status=report['status'], output=str(target))), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
