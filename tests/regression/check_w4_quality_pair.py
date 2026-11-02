"""严格区分测量完成和质量通过；不改变已冻结的 W4 验收标准。

只有相同输入、同精度基线、相同推理实现且全部有限的报告才能配对。
本工具不做性能推荐，也不以 argmax token 相等代替有损量化质量验收。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

FROZEN_MANIFEST = 'f17787f84fe49b2376e4da32c953f6fccf89a8485078840c32c943b0d5e320f0'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def validate_report(report, manifest):
    if report.get('status') != 'PASS_MEASUREMENT':
        raise ValueError('measurement incomplete or nonfinite')
    if report.get('quality_manifest_sha256') != FROZEN_MANIFEST or report.get('split') != 'validation':
        raise ValueError('wrong frozen validation protocol')
    if report.get('dtype') != 'float16' or report.get('tokenizer_mapping_and_special_ids_equal') is not True:
        raise ValueError('FP16 / tokenizer contract not met')
    expected = manifest['workloads']['validation']
    if len(report['rows']) != len(expected):
        raise ValueError('missing or extra rows')
    total, count = 0.0, 0
    for row, fixed in zip(report['rows'], expected):
        if row['row'] != fixed['row'] or row['tokens'] != fixed['tokens']:
            raise ValueError('validation token IDs or order changed')
        losses = row['nll']
        labels = len(fixed['tokens'])-1
        if len(losses) != labels or len(row['predicted_tokens']) != labels or row['finite_positions'] != labels:
            raise ValueError('incomplete per-position evidence')
        if not all(math.isfinite(x) for x in losses):
            raise ValueError('nonfinite NLL')
        total += sum(losses)
        count += labels
    mean = total/count
    if report['label_tokens'] != count or report['finite_positions'] != count:
        raise ValueError('incorrect aggregate counts')
    # This is report-consistency validation, not a new model-error tolerance.
    if mean != report['mean_teacher_forced_nll']:
        raise ValueError('aggregate NLL does not equal archived per-token evidence')
    return mean

def compare(reference, candidate, manifest):
    body = {k:v for k,v in manifest.items() if k != 'sha256'}
    if manifest.get('sha256') != FROZEN_MANIFEST or digest(body) != FROZEN_MANIFEST:
        raise ValueError('frozen manifest identity mismatch')
    if reference.get('quantization') is not None:
        raise ValueError('baseline must be unquantized')
    if candidate.get('quantization') != 'awq':
        raise ValueError('candidate must execute packed AWQ, not dense reference')
    modules=candidate.get('packed_modules',{})
    backend={'marlin':'awq_marlin','triton':'awq_packed_triton'}.get(candidate.get('quant_backend'))
    if not backend or not modules or any(m.get('backend')!=backend or m.get('dense_weight_bytes')!=0 or not m.get('ready') for m in modules.values()):
        raise ValueError('missing real packed execution evidence')
    baseline_sources = {k:v for k,v in reference['source_sha256'].items() if k.startswith('nanovllm/')}
    quant_sources = {k:v for k,v in candidate['source_sha256'].items() if k.startswith('nanovllm/')}
    if not baseline_sources or baseline_sources != quant_sources:
        raise ValueError('runtime source differs between paired reports')
    base = validate_report(reference, manifest)
    quant = validate_report(candidate, manifest)
    threshold = manifest['thresholds']['max_validation_mean_nll_increase_nats']
    delta = quant-base
    return dict(status='PASS_QUALITY' if delta <= threshold else 'FAIL_QUALITY',
        baseline_nll=base, candidate_nll=quant, delta_nll=delta, max_delta=threshold,
        quality_manifest_sha256=FROZEN_MANIFEST, label_tokens=reference['label_tokens'],
        candidate_model=candidate['model'], baseline_model=reference['model'],
        backend=candidate['quant_backend'], scope='Frozen English WikiText subset only; no speed or default-promotion claim')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=compare(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()),json.loads(a.manifest.read_text()))
    result['reports']={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name,path in (('reference',a.reference),('candidate',a.candidate))}
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
    raise SystemExit(0 if result['status']=='PASS_QUALITY' else 1)

if __name__=='__main__':main()
