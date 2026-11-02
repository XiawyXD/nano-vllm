"""CPU 回归入口：独立子进程，明确隐藏 GPU，不改变原断言。"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CORE = [
    'engine-hardening', 'runtime-hardening', 'service-hardening', 'execution-plan',
    'cache-draft', 'speculative-sampling', 'quant-graph', 'gpu-guard', 'harness',
]
CHECKS = {
    'engine-hardening': 'check_engine_hardening',
    'runtime-hardening': 'check_runtime_hardening',
    'service-hardening': 'check_service_hardening',
    'execution-plan': 'check_execution_plan',
    'cache-draft': 'check_cache_draft',
    'speculative-sampling': 'check_speculative_sampling',
    'quant-graph': 'check_quant_graph_cpu',
    'gpu-guard': 'check_gpu_guard',
    'harness': 'check_runtime_v2_harness',
    'adaptive': 'check_adaptive_speculation',
    'packed-metadata': 'check_packed_metadata_cpu',
    'planned-engine': 'check_planned_engine',
    'metadata': 'check_metadata_buffers',
    'mlp-dispatch': 'check_mlp_dispatch',
    'tp-runtime': 'check_tp_runtime',
    'awq-packed': 'check_awq_packed',
    'weight-formats': 'check_weight_formats',
    'marlin-layout': 'check_marlin_layout_cpu',
    'tiered-cache': 'check_tiered_cache',
    'engine-utils': 'check_runtime_v2_engine_utils',
    'flashinfer-reference': 'check_flashinfer_sampler',
    'evaluation-harness': 'check_runtime_v2_evaluation',
    'http-sse': 'check_http_sse',
    'openai-api': 'check_openai_api',
    'ordinary-scheduler': 'test_unified_scheduler',
    'cache-cost': 'check_cache_cost',
    'cache-idle': 'check_cache_idle',
    'cache-joint': 'check_joint_cache',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('action', choices=['list', 'run'])
    parser.add_argument('name', nargs='?', choices=['core', 'extended', *CHECKS])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'list':
        print('core: 9 suites; extended: all listed CPU suites. GPU/model checks are explicit modules.')
        for name, module in CHECKS.items():
            print(f'{name:24} tests.regression.{module}')
        print('Explicit CPU inputs: tests.regression.check_weight_shards (--model, --output)')
        print('Explicit CPU inputs: tests.regression.check_marlin_sidecar_cpu (--bundle, --directory, --output)')
        return 0
    if not args.name:
        parser.error('run requires core, extended, or a listed check name')
    selected = CORE if args.name == 'core' else list(CHECKS) if args.name == 'extended' else [args.name]
    output = args.output or ROOT/'artifacts/layout_checks'/str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    rows = []
    for name in selected:
        command = [sys.executable, '-B', '-m', 'tests.regression.' + CHECKS[name]]
        started = time.perf_counter()
        timed_out = False
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=180)
            code, content = result.returncode, result.stdout
        except subprocess.TimeoutExpired as exc:
            code, content, timed_out = 124, exc.stdout or b'', True
        (output/(name+'.log')).write_bytes(content)
        counts = [int(n) for n in re.findall(rb'Ran (\d+) tests?', content)]
        row = dict(name=name, command=command, exit_code=code, timed_out=timed_out,
                   seconds=time.perf_counter()-started, unittest_count=sum(counts))
        rows.append(row)
        print(f'{name}: {"PASS" if code == 0 else "FAIL"}', flush=True)
    report = dict(status='PASS' if all(r['exit_code']==0 for r in rows) else 'FAIL',
                  scope='CPU only; no new GPU correctness or performance claim',
                  suites=len(rows), checks=rows)
    (output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], suites=len(rows), output=str(output))))
    return 0 if report['status']=='PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
