"""比较两个独立进程生成的 Tensor Core 模型报告，不重新运行 GPU。"""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.tensorcore_utils import compare, output_path, save_report


def compare_reports(baseline_path: Path, candidate_path: Path, atol: float, rtol: float):
    """先核对固定输入，再对完整词表 logits 和最终 token 分别给出判断。"""
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if baseline['args']['model'] != candidate['args']['model']:
        raise ValueError('model checkpoints differ')
    if baseline['prompts'] != candidate['prompts']:
        raise ValueError('prompt text/token IDs differ')
    a = torch.load(baseline_path.with_suffix('.pt'), map_location='cpu', weights_only=True)
    b = torch.load(candidate_path.with_suffix('.pt'), map_location='cpu', weights_only=True)
    metadata = ('input_ids', 'positions', 'cu_seqlens_q', 'cu_seqlens_k')
    for key in metadata:
        if key not in a or key not in b or not torch.equal(a[key], b[key]):
            raise ValueError(f'first prefill metadata differs: {key}')
    reference, actual = a['first_prefill_logits'], b['first_prefill_logits']
    numerical = compare(actual, reference, atol, rtol)
    top1_equal = bool(torch.equal(actual.argmax(-1), reference.argmax(-1)))
    all_forwards = []
    forward_status = 'NOT_CAPTURED'
    if 'forward_records' in a and 'forward_records' in b:
        forward_status = 'PASS'
        if len(a['forward_records']) != len(b['forward_records']):
            forward_status = 'REVIEW_REQUIRED'
        for index, (left, right) in enumerate(zip(a['forward_records'], b['forward_records'])):
            same_input = left['is_prefill'] == right['is_prefill']
            for key in ('input_ids', 'positions', 'cu_seqlens_q', 'cu_seqlens_k', 'context_lens'):
                same_input = same_input and ((key not in left and key not in right) or
                    (key in left and key in right and torch.equal(left[key], right[key])))
            if not same_input:
                all_forwards.append(dict(step=index, fixed_input_equal=False,
                                         note='History/input diverged; do not treat later logits as same-input comparisons.'))
                forward_status = 'REVIEW_REQUIRED'
                break
            metrics = compare(right['logits'], left['logits'], atol, rtol)
            ids_equal = bool(torch.equal(left['logits'].argmax(-1), right['logits'].argmax(-1)))
            all_forwards.append(dict(step=index, is_prefill=left['is_prefill'],
                                     fixed_input_equal=True, logits=metrics, argmax_equal=ids_equal))
            if not metrics['allclose'] or not ids_equal:
                forward_status = 'REVIEW_REQUIRED'
    tokens_a = [row['token_ids'] for row in baseline['outputs']]
    tokens_b = [row['token_ids'] for row in candidate['outputs']]
    differences = []
    for request_id, (left, right) in enumerate(zip(tokens_a, tokens_b)):
        if left != right:
            differences.append(dict(request_index=request_id, baseline=left, candidate=right))
    finite_unit = (baseline['layer0_unit']['passed'] and candidate['layer0_unit']['passed'])
    passed = (finite_unit and numerical['allclose'] and top1_equal and tokens_a == tokens_b
              and forward_status != 'REVIEW_REQUIRED')
    return dict(
        status='PASS' if passed else 'REVIEW_REQUIRED',
        baseline=str(baseline_path), candidate=str(candidate_path),
        fixed_prefill_metadata_equal=True, first_prefill_logits=numerical,
        first_prefill_argmax_equal=top1_equal,
        baseline_top2_margin=(reference.float().topk(2, -1).values.diff(dim=-1).neg().flatten().tolist()),
        candidate_top2_margin=(actual.float().topk(2, -1).values.diff(dim=-1).neg().flatten().tolist()),
        layer0_units_passed=finite_unit, generated_token_ids_equal=tokens_a == tokens_b,
        all_forward_status=forward_status, fixed_history_forward_comparisons=all_forwards,
        request_count=len(tokens_a), tokens_per_request=[len(row) for row in tokens_a],
        token_differences=differences,
        limitation='Small fixed smoke only. Does not resolve prior speculative path divergence or prove all prompts identical.',
    )


def main():
    """公差在比较前固定；越界保留原始证据并返回非零，不修改token或tie-break。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--atol', type=float, default=0.0625)
    parser.add_argument('--rtol', type=float, default=0.01)
    args = parser.parse_args()
    import math
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error('tolerances must be finite and nonnegative')
    target = output_path(args.output)
    report = compare_reports(args.baseline, args.candidate, args.atol, args.rtol)
    save_report(target, report)
    print(json.dumps(report, ensure_ascii=False))
    if report['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
