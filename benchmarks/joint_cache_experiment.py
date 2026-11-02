"""Bounded, opt-in joint-residency experiment. No runtime changes or auto-promotion.

Deploy this file unchanged into BOTH R/{baseline,candidate}/benchmarks/. R defaults
to /home/hrxu/nano-vllm-archive/cache-joint-20261028-001. All server subprocesses use
/home/hrxu/nano-vllm/.venv/bin/python. Preparation, fitting, freezing and analysis
are CPU-only; only worker/suite admit and use a GPU. No profiler is installed in
the timed path. The runtime owns validation of host_kv_joint_residency_path.

Server usage (choose fresh output names on every invocation):
  P=/home/hrxu/nano-vllm/.venv/bin/python
  R=/home/hrxu/nano-vllm-archive/cache-joint-20261028-001
  B=$R/candidate/benchmarks/joint_cache_experiment.py
  $P -B "$B" prepare --output "$R/inputs.json"
  $P -B "$B" fit --output "$R/calibration-joint.json"
  $P -B "$B" suite --split dev --output "$R/dev-001"
  $P -B "$B" analyze --index "$R/dev-001/index.json" --output "$R/dev-analysis-001.json"
  $P -B "$B" freeze --dev-analysis "$R/dev-analysis-001.json" --output "$R/freeze-001.json"
  $P -B "$B" suite --split final --freeze "$R/freeze-001.json" --output "$R/final-001"
  $P -B "$B" analyze --index "$R/final-001/index.json" --output "$R/final-analysis-001.json"
  $P -B "$B" suite --split final --freeze "$R/freeze-001.json" --normal-capacity --output "$R/normal-001"
  $P -B "$B" analyze --index "$R/normal-001/index.json" --output "$R/normal-analysis-001.json"
Standalone worker is diagnostic only, e.g. worker --variant joint --split dev
--round 0 --output "$R/worker-dev-001". A formal worker requires --freeze.
DEV runs one three-worker screen. Each final suite runs all three arms, all five
cases, five fresh-worker rounds (30 workers for bounded plus normal capacity).
Fit has one TRAIN-only formula, no grid or parameter overrides. Failed DEV may
still be frozen/run formally, but is permanently marked diagnostic-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import struct
import subprocess
import sys
import time
import traceback


SERVER_ROOT = Path('/home/hrxu/nano-vllm-archive/cache-joint-20261028-001')
SERVER_PYTHON = Path('/home/hrxu/nano-vllm/.venv/bin/python')
VARIANTS = ('gpu_only', 'legacy_async', 'joint')
BRANCHES = ('baseline', 'candidate')
CASE_JOBS = dict(phase_shift=32, first_cpu_reuse=16, mixed_hot_cold=24,
                 unique=12, shared_branches=16)
SEEDS = dict(train=2026102811, dev=2026102812, final=2026102813)
PAGE_TOKENS, PAGE_BYTES = 256, 28 * 2**20
GPU_PAGES, CPU_PAGES, MAX_TOKENS, ROUNDS = 12, 8, 8, 5
TRIAL_TIMEOUT, WORKER_TIMEOUT = 180, 3600
SAMPLING = dict(temperature=0, ignore_eos=True, speculative_method='off')
HELPERS = ('cache_runtime_experiment.py', 'runtime_v2_performance.py',
           'runtime_v2_gpu_guard.py', 'slo_goodput.py')
GATES = dict(median_paired_speed_min=1.03, all_five_faster=True,
             ttft_p95_ratio_max=1.10, event_gap_p95_ratio_max=1.10,
             greedy_exact=True)
ABSOLUTE_SLO = dict(ttft_ms=1000, tpot_ms=30)
NUMERICS_LABEL = dict(
    status='REVIEW_REQUIRED_KNOWN_LOGITS_ISSUE',
    note='Existing cross-shape logits issue is a label only. This harness checks '
         'paired exact output tokens; it neither measures nor claims logits/KV parity.',
    original_logits_atol=.01, original_logits_rtol=.01,
    original_valid_kv_bitwise=True, logits_measured=False, kv_bitwise_measured=False)
RECIPE = dict(candidate_count=1, scan_rule='clip(4 * TRAIN p90 complete-prefix pages, 8, 64)',
              quantile='nearest rank, one observation per TRAIN request',
              complete_prefix_pages='floor((prompt_tokens - 1) / 256)',
              history_capacity=4096, parent_depth='runtime-only, derived from request prompt',
              cost_model='original measured legacy document, embedded unchanged',
              final_used_for_tuning=False)

# These are separately authored documents/structures, not the old passage pool
# with split tags or nonces. Randomness supplies substantive measurements,
# schedules and record contents inside each authored structure.
TEMPLATES = {
    'train': dict(
        title='河岸采样站的离线交接簿 / Estuary field notebook',
        opening='整理潮汐采样的交接记录，按站点解释异常与复核步骤。',
        zh='采样瓶先用站点原水冲洗，浊度超出本批中位数时保留平行样；潮位变化不等于仪器漂移。',
        en='The estuary team separates instrument drift from tidal variation. '
           'Keep duplicate readings and document which observation justifies a repeat visit.',
        code='def reconcile_samples(readings, threshold):\n'
             '    valid = [r for r in readings if r["salinity"] <= threshold]\n'
             '    return {r["station"]: round(r["salinity"], 2) for r in valid}\n',
        log='INFO sample_collected station={site} bottle={row} depth_cm={value} duplicates={count}',
        fields=('salinity', 'turbidity', 'tide_minutes'),
        task='Write a bilingual handover: name the uncertain reading, the next sampling action, and the evidence.'),
    'dev': dict(
        title='天文台的供电轮换计划 / Observatory maintenance ledger',
        opening='核对观测夜的电池交接、遮光程序与设备维护，解释哪些工单必须延期。',
        zh='遮光罩闭合后才能切换供电回路，低温引起的电压变化须与实际放电区分；维护记录保留原始读数。',
        en='The observatory schedules battery rotation around dark exposures. '
           'A low temperature reading alone cannot justify replacing a healthy battery.',
        code='def allocate_exposures(windows, reserve_minutes):\n'
             '    plan = []\n'
             '    for window in sorted(windows, key=lambda w: w["start"]):\n'
             '        if window["minutes"] > reserve_minutes:\n'
             '            plan.append((window["camera"], window["start"]))\n'
             '    return plan\n',
        log='NOTICE shutter_interlock camera={site} exposure={row} battery_mv={value} queue={count}',
        fields=('battery_voltage', 'dome_temperature', 'exposure_seconds'),
        task='Draft a shift memo with a maintenance order, an exposure exception, and a reason for each decision.'),
    'final': dict(
        title='博物馆的入库与湿度巡检 / Collection accession dossier',
        opening='根据器物入库、包装和展柜湿度记录拟定轮值方案，区分测量偏差与藏品状态变化。',
        zh='纸质标签与电子档案逐件核对，包装袋内外湿度分别记录；展柜开合后的短暂变化应复测而非直接归因于藏品。',
        en='The collection team records packaging and cabinet humidity independently. '
           'Accession evidence must connect each object to a reversible storage action.',
        code='class AccessionQueue:\n'
             '    def __init__(self):\n'
             '        self.pending = {}\n'
             '    def register(self, object_id, cabinet, inspected):\n'
             '        self.pending[object_id] = (cabinet, bool(inspected))\n'
             '    def ready(self):\n'
             '        return sorted(k for k, (_, ok) in self.pending.items() if ok)\n',
        log='AUDIT cabinet_check cabinet={site} accession={row} humidity_tenths={value} trays={count}',
        fields=('cabinet_humidity', 'packing_age_days', 'inspection_minutes'),
        task='Prepare an accession checklist with storage constraints, a follow-up measurement, and an unresolved question.'),
}


def require(condition, message):
    # Do not let python -O remove evidence/binding checks.
    if not condition:
        raise RuntimeError(message)


def absolute(path):
    return Path(path).expanduser().resolve()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def branch_for(variant):
    require(variant in VARIANTS, 'unknown arm')
    return 'candidate' if variant == 'joint' else 'baseline'


def helpers(root, branch=None):
    """Resolve imports by absolute branch, never by the coordinator's cwd."""
    cached = sys.modules.get('benchmarks.cache_runtime_experiment')
    if branch is None:
        branch = Path(cached.__file__).resolve().parents[1].name if cached else 'baseline'
    directory = absolute(root / branch)
    require(directory.is_dir(), f'missing branch checkout: {directory}')
    if not sys.path or sys.path[0] != str(directory):
        sys.path.insert(0, str(directory))
    modules = []
    for name in ('cache_runtime_experiment', 'runtime_v2_performance', 'slo_goodput'):
        module = importlib.import_module('benchmarks.' + name)
        require(absolute(module.__file__) == directory / 'benchmarks' / (name + '.py'),
                f'wrong branch helper import: {module.__file__}')
        modules.append(module)
    return tuple(modules)


def old_paths(root):
    return [root.parent / name / 'inputs.json' for name in
            ('cache-policy-v2-20261028-001', 'cache-runtime-20261028-001')]


def all_cases(data):
    return [case for cases in data['splits'].values() for case in cases]


def prefix_hashes(cases):
    result = dict(full_prompt=set(), first256=set(), prefix_chain=set())
    for case in cases:
        for job in case['jobs']:
            ids = job['prompt_token_ids']
            require(len(ids) >= PAGE_TOKENS, 'audit requires at least one full page')
            result['full_prompt'].add(digest(ids))
            result['first256'].add(digest(ids[:PAGE_TOKENS]))
            parent = bytes(32)
            for end in range(PAGE_TOKENS, len(ids), PAGE_TOKENS):
                page = ids[end - PAGE_TOKENS:end]
                # Namespace-free, deterministic audit hash of the complete chain.
                parent = hashlib.sha256(parent + struct.pack('<256q', *page)).digest()
                result['prefix_chain'].add(parent.hex())
    return result


def audit_inputs(splits, priors):
    groups = {name: prefix_hashes(cases) for name, cases in splits.items()}
    groups.update({name: prefix_hashes(all_cases(data)) for name, data in priors.items()})
    pairs = [('train', 'dev'), ('train', 'final'), ('dev', 'final')]
    pairs += [(split, name) for split in SEEDS for name in priors]
    comparisons = {left + '/' + right: {
        kind: len(groups[left][kind] & groups[right][kind]) for kind in groups[left]}
        for left, right in pairs}
    require(not any(count for row in comparisons.values() for count in row.values()),
            'new/old or cross-split prompt/prefix-chain overlap')
    return dict(status='PASS_ZERO_CROSS_SPLIT_AND_OLDER_OVERLAP',
                comparisons=comparisons,
                cardinalities={name: {k: len(v) for k, v in group.items()}
                               for name, group in groups.items()},
                older_splits={name: sorted(data['splits']) for name, data in priors.items()},
                chain_definition='SHA256(parent_digest || little-endian signed-int64 page tokens), '
                                 '256-token complete pages, all depths; no runtime namespace',
                intentional_within_split_reuse=True)


def build_split(tokenizer, split, seed):
    rng, template, documents = random.Random(seed), TEMPLATES[split], {}

    def doc(length=1025):
        serial = len(documents)
        site = rng.choice(('delta', 'north', 'west', 'central', 'outer'))
        values = [rng.randint(11, 940) for _ in range(6)]
        text = (f'{template["title"]}\n{template["opening"]}\n'
                f'记录对象 {site}-{serial:03d}：本次读数 {values}，'
                f'复核间隔 {rng.randint(3, 37)} 分钟，责任班组 {rng.randint(1, 8)}。\n'
                f'{template["zh"]}\n{template["en"]}\n')
        # Every section has independently generated structured content. No old
        # passage constants are imported, and a nonce is not the source of novelty.
        for row in range(32):
            numbers = [rng.randint(10, 990) / 10 for _ in template['fields']]
            record = dict(zip(template['fields'], numbers))
            text += (f'\nSection {row}: {site} observation {serial * 32 + row}\n'
                     f'{json.dumps(record, ensure_ascii=False)}\n'
                     f'复核意见：第 {row + 1} 项保留 {rng.randint(2, 7)} 份副本，'
                     f'在 {rng.randint(6, 23)} 时比较相邻记录；'
                     f'允许偏差 {rng.randint(1, 9)}，不要用缺失值替代实测值。\n'
                     f'{template["en"]}\n{template["code"]}'
                     + template['log'].format(site=site, row=row,
                                              value=rng.randint(120, 780), count=rng.randint(1, 9))
                     + '\n')
        text += '\n' + template['task']
        ids = tokenizer.encode(text, add_special_tokens=False)
        require(len(ids) >= length, 'authored structured document too short')
        key = f'{split}-document-{serial:03d}'
        documents[key] = dict(source_text=text, prompt_token_ids=ids[:length],
                              prompt_text=tokenizer.decode(ids[:length]),
                              template_sha256=digest(template))
        return key

    def tokens(key):
        return documents[key]['prompt_token_ids']

    def case(name, keys):
        return dict(id=name, dataset=name, scenario='synthetic_structured_causal_cache_trace',
                    concurrency=1, jobs=[dict(job=i, arrival_s=0., prompt_token_ids=tokens(key),
                                              max_tokens=MAX_TOKENS, document_id=key)
                                         for i, key in enumerate(keys)])

    old, new, cold = [doc() for _ in range(4)], [doc() for _ in range(4)], [doc() for _ in range(8)]
    phase = old * 2
    for i in range(16):
        phase.append(new[i % 4])
        if i % 2:
            phase.append(cold[i // 2])
    cyclic = [doc() for _ in range(4)]
    hot, cold = [doc() for _ in range(4)], [doc() for _ in range(8)]
    mixed = [key for i in range(8) for key in (hot[i % 4], cold[i], hot[(i + 1) % 4])]
    unique = [doc() for _ in range(12)]
    common = doc(512)
    branches = []
    for _ in range(4):
        suffix = doc(513)
        ids = tokens(common) + tokens(suffix)
        key = f'{split}-branch-{len(branches)}'
        documents[key] = dict(prompt_token_ids=ids, prompt_text=tokenizer.decode(ids),
                              source_parts=[common, suffix], template_sha256=digest(template))
        branches.append(key)
    cases = [case('phase_shift', phase), case('first_cpu_reuse', cyclic * 4),
             case('mixed_hot_cold', mixed), case('unique', unique),
             case('shared_branches', branches * 4)]
    validate_cases(cases)
    return cases, documents


def validate_cases(cases):
    require([c['id'] for c in cases] == list(CASE_JOBS), 'full fixed case order is required')
    for case in cases:
        require(case['concurrency'] == 1 and len(case['jobs']) == CASE_JOBS[case['id']],
                'case request count/concurrency changed')
        for i, job in enumerate(case['jobs']):
            require(job['job'] == i and job['arrival_s'] == 0 and job['max_tokens'] == MAX_TOKENS,
                    'job identity/arrival/output budget changed')
            ids = job['prompt_token_ids']
            require(len(ids) == 1025 and all(type(t) is int and t >= 0 for t in ids),
                    'all prompts must contain exactly 1025 real token IDs')
    by_name = {c['id']: c for c in cases}
    cyclic = by_name['first_cpu_reuse']
    distinct = {digest(j['prompt_token_ids']) for j in cyclic['jobs']}
    require(len(distinct) == 4 and len(prefix_hashes([cyclic])['prefix_chain']) == 16 > GPU_PAGES,
            'four cyclic 1025-token documents must exceed the 12 physical GPU pages')
    require(all(cyclic['jobs'][i]['prompt_token_ids'] == cyclic['jobs'][i % 4]['prompt_token_ids']
                for i in range(16)), 'first_cpu_reuse must be four full cycles')
    shared = by_name['shared_branches']['jobs']
    require(len({digest(j['prompt_token_ids'][:512]) for j in shared}) == 1
            and len({digest(j['prompt_token_ids']) for j in shared}) == 4,
            'shared branches require 512 common tokens and four distinct suffixes')
    require(len(prefix_hashes([by_name['unique']])['first256']) == 12,
            'unique case must have 12 distinct first pages')


def file_inventory(directory):
    require(directory.is_dir(), f'missing source/model directory: {directory}')
    paths = [p for p in sorted(directory.rglob('*')) if p.is_file()
             and not any(part in ('__pycache__', '.git', '.cache', '.locks')
                         for part in p.relative_to(directory).parts)
             and p.suffix not in ('.pyc', '.pyo')]
    require(paths, f'empty inventory: {directory}')
    return {p.relative_to(directory).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in paths}


def model_binding(data):
    model = absolute(data['model'])
    inventory = file_inventory(model)
    require('config.json' in inventory and data['model_files'], 'missing inherited model identity')
    for name, identity in data['model_files'].items():
        require(name in inventory and inventory[name]['sha256'] == identity['sha256']
                and inventory[name]['bytes'] == identity['bytes'], f'inherited model changed: {name}')
    return dict(path=str(model), files=inventory, inherited_identity=data['model_files'])


def protocol_document(root):
    return dict(schema='nano-joint-experiment-protocol-v1', root=str(root),
                preregistered_protocol=str(root / 'protocol.json'),
                preregistered_protocol_sha256=sha(root / 'protocol.json'),
                benchmark_sha256=sha(__file__), variants=list(VARIANTS),
                branches={v: branch_for(v) for v in VARIANTS}, seeds=SEEDS,
                cases=CASE_JOBS, recipe=RECIPE, rounds=dict(dev=1, final=ROUNDS), max_tokens=MAX_TOKENS,
                capacity=dict(gpu_pages=GPU_PAGES, cpu_pages=CPU_PAGES, page_bytes=PAGE_BYTES,
                              page_tokens=PAGE_TOKENS, active_requests=1, engine_max_num_seqs=4, max_model_len=2048,
                              prefill_chunk_size=128, normal_capacity='all cases, automatic GPU KV capacity'),
                legacy='original async legacy transfer, max_inflight=4, no cost policy',
                joint='same legacy configuration plus host_kv_joint_residency_path',
                graph='ordinary decode Graph on; preserve runtime capture defaults',
                timing='one warmup then reset and one measured trial per case; full host/GPU drain '
                       'included in seconds; no diagnostic profiler or step trace',
                guards='two stable existing-guard observations before each worker and each trial',
                gates=GATES, absolute_slo_separate=ABSOLUTE_SLO, numerics=NUMERICS_LABEL,
                dev='one fixed candidate; one paired round; failed screen permits diagnostic final only',
                final_used_for_tuning=False,
                note='Fresh synthetic structured Chinese/English/code/log documents. No production claim.')


def prepare(a):
    require(not a.output.exists() and not a.protocol.exists(), 'prepare outputs must both be new')
    priors = {path.parent.name: read(path) for path in old_paths(a.root)}
    inherited = next(iter(priors.values()))
    for prior in priors.values():
        require(absolute(prior['model']) == absolute(inherited['model'])
                and prior['model_files'] == inherited['model_files'], 'older model identities disagree')
    data = dict(schema='nano-joint-inputs-v1', model=str(absolute(inherited['model'])),
                model_files=copy.deepcopy(inherited['model_files']), seeds=SEEDS,
                older_inputs={str(path): sha(path) for path in old_paths(a.root)})
    data['model_binding_sha256'] = digest(model_binding(data))
    from transformers import AutoTokenizer
    _, performance, _ = helpers(a.root)
    tokenizer = AutoTokenizer.from_pretrained(data['model'], local_files_only=True)
    built = {split: build_split(tokenizer, split, seed) for split, seed in SEEDS.items()}
    data.update(splits={s: result[0] for s, result in built.items()},
                documents={s: result[1] for s, result in built.items()})
    data['overlap'] = audit_inputs(data['splits'], priors)
    data.update(split_sha256={s: digest(cases) for s, cases in data['splits'].items()},
                authored_templates=TEMPLATES, protocol=str(a.protocol),
                protocol_sha256=digest(protocol_document(a.root)), environment=performance.environment(),
                note='Independent synthetic data, no production traffic or quality claim. '
                     'Within-case repetitions are deliberate; no old passage pool is reused.')
    write(a.protocol, protocol_document(a.root))
    data['protocol_file_sha256'] = sha(a.protocol)
    write(a.output, data)
    print(json.dumps(dict(status='PREPARED_CPU_ONLY', inputs=str(a.output), protocol=str(a.protocol),
                          audit=data['overlap']), ensure_ascii=False))


def validate_cost(cost, data):
    require(cost.get('schema') == 'nano-cache-cost-v1'
            and cost.get('status') == 'PASS_CALIBRATION_ONLY', 'original measured cost calibration required')
    identity = cost['identity']
    expected = dict(model=data['model'], config_sha256=data['model_files']['config.json']['sha256'],
                    transport='legacy', chunk_size=128, block_size=256,
                    linear_backend='torch', numerical_backend='default', rmsnorm_backend='triton')
    require(all(identity.get(k) == v for k, v in expected.items()), 'legacy cost identity mismatch')
    require(str(identity.get('gpu_uuid', '')).startswith('GPU-'), 'cost model must identify physical GPU UUID')
    require(cost.get('query_ms') and all(cost.get('transfer_ms', {}).get(d) for d in ('h2d', 'd2h')),
            'missing measured legacy cost tables')


def fitted_document(data, cost, cost_sha):
    # Deliberately access only TRAIN. No DEV/final statistics or observations are
    # inputs to the sole fitted parameter. Hashing inputs.json is provenance only.
    train = data['splits']['train']
    validate_cases(train)
    pages = sorted((len(job['prompt_token_ids']) - 1) // PAGE_TOKENS
                   for case in train for job in case['jobs'])
    p90 = pages[math.ceil(.9 * len(pages)) - 1]
    return dict(schema='nano-joint-residency-v1', cost_model=copy.deepcopy(cost),
                victim_scan_limit=min(64, max(8, 4 * p90)), history_capacity=4096,
                training_provenance=dict(split='train', seed=SEEDS['train'],
                    train_sha256=digest(train), requests=len(pages), p90_complete_prefix_pages=p90,
                    page_tokens=PAGE_TOKENS, recipe=RECIPE, measured_cost_model_sha256=cost_sha,
                    dev_used=False, final_used=False))


def fit(a):
    require(a.output == a.root / 'calibration-joint.json', 'joint recipe has one authoritative output path')
    require(not a.output.exists(), 'calibration output already exists')
    data, cost = read(a.inputs), read(a.root / 'calibration-cost.json')
    validate_cost(cost, data)
    document = fitted_document(data, cost, sha(a.root / 'calibration-cost.json'))
    document['training_provenance']['inputs_sha256'] = sha(a.inputs)
    write(a.output, document)
    print(json.dumps(dict(status='FIT_TRAIN_ONLY_CPU_ONLY', output=str(a.output),
                          victim_scan_limit=document['victim_scan_limit'], recipe=RECIPE)))


def config_for(variant, data, root, normal=False):
    common, _, _ = helpers(root)
    config = common.base_config(data['model'], normal)
    config.update(kvcache_block_size=PAGE_TOKENS)
    require(config['max_num_seqs'] == 4, 'preserve the inherited four-slot resource budget; traces are C1')
    if variant == 'gpu_only':
        config['host_kv_cache_bytes'] = 0
    else:
        config.update(host_kv_cache_bytes=CPU_PAGES * PAGE_BYTES, host_kv_async_prefetch=True,
                      host_kv_prefetch_max_inflight_blocks=4)
    if variant == 'joint':
        config['host_kv_joint_residency_path'] = str(root / 'calibration-joint.json')
    return config


def collect_bindings(root, inputs, protocol):
    data = read(inputs)
    require(data.get('schema') == 'nano-joint-inputs-v1' and data['seeds'] == SEEDS,
            'wrong new input schema/seeds')
    require(data['protocol'] == str(protocol) and data['protocol_file_sha256'] == sha(protocol)
            and read(protocol) == protocol_document(root)
            and data['protocol_sha256'] == digest(read(protocol)), 'protocol/benchmark changed since prepare')
    require(data['older_inputs'] == {str(p): sha(p) for p in old_paths(root)}, 'older audit inputs changed')
    require(all(data['split_sha256'][s] == digest(cases) for s, cases in data['splits'].items()),
            'input split hashes changed')
    require(data['overlap']['status'] == 'PASS_ZERO_CROSS_SPLIT_AND_OLDER_OVERLAP'
            and not any(n for row in data['overlap']['comparisons'].values() for n in row.values()),
            'input separation audit failed')
    cost_path, joint_path = root / 'calibration-cost.json', root / 'calibration-joint.json'
    cost = read(cost_path)
    validate_cost(cost, data)
    expected = fitted_document(data, cost, sha(cost_path))
    expected['training_provenance']['inputs_sha256'] = sha(inputs)
    require(read(joint_path) == expected, 'calibration differs from the single TRAIN-only recipe')
    model = model_binding(data)
    require(digest(model) == data['model_binding_sha256'], 'model/tokenizer inventory changed')
    sources, benchmark, shared = {}, {}, {}
    for branch in BRANCHES:
        checkout = root / branch
        sources[branch] = file_inventory(checkout / 'nanovllm')
        require('config.py' in sources[branch] and 'engine/block_manager.py' in sources[branch],
                'incomplete runtime source inventory')
        benchmark[branch] = sha(checkout / 'benchmarks' / Path(__file__).name)
        shared[branch] = {name: sha(checkout / 'benchmarks' / name) for name in HELPERS}
    require(set(benchmark.values()) == {sha(__file__)}, 'deploy the identical benchmark into both branches')
    require(shared['baseline'] == shared['candidate'], 'shared timing/guard/config/reset helpers differ')
    configs = {mode: {v: config_for(v, data, root, mode == 'normal') for v in VARIANTS}
               for mode in ('bounded', 'normal')}
    return dict(schema='nano-joint-bindings-v1', root=str(root), sources=sources,
                benchmark_sha256=sha(__file__), branch_benchmark_sha256=benchmark, helpers=shared,
                inputs=str(inputs), inputs_sha256=sha(inputs), split_sha256=data['split_sha256'],
                protocol=str(protocol), protocol_sha256=sha(protocol), model=model,
                calibrations={str(p): sha(p) for p in (cost_path, joint_path)},
                gpu_uuid=cost['identity']['gpu_uuid'], configs=configs,
                variants=list(VARIANTS), rounds=dict(dev=1, final=ROUNDS), cases=CASE_JOBS,
                sampling=SAMPLING, gates=GATES, absolute_slo=ABSOLUTE_SLO, recipe=RECIPE)


def rounds_for(split):
    return 1 if split == 'dev' else ROUNDS


def check_frozen(document, bindings):
    require(document.get('schema') == 'nano-joint-freeze-v1'
            and document['bindings'] == bindings
            and document['bindings_sha256'] == digest(bindings), 'frozen source/model/protocol binding changed')
    screen = document['dev_screen']
    require(sha(screen['analysis']) == screen['analysis_sha256']
            and sha(screen['index']) == screen['index_sha256'], 'frozen DEV evidence changed')
    require(document['diagnostic_only'] == (not screen['passed'])
            and document['final_used_for_tuning'] is False, 'invalid frozen screening decision')


def stable_guard(common, path, uuid):
    """Reuse the existing ownership guard; never stop or signal foreign work."""
    observations = []
    for observation in range(2):
        snapshot = common.guard(path)
        require(snapshot['gpu']['uuid'] == uuid and snapshot['status'] == 'AVAILABLE',
                'physical GPU changed or ownership guard failed')
        observations.append(snapshot)
        if observation == 0:
            time.sleep(.25)
    require(observations[0]['gpu']['index'] == observations[1]['gpu']['index'],
            'GPU identity not stable at admission')
    return observations


def reset_engine(common, engine):
    common.reset(engine)
    manager = engine.scheduler.block_manager
    cache = manager.host_cache
    # The existing reset clears GPU indexes, CPU pages and legacy history. If a
    # runtime attaches the new policy separately, use its own cold-reset API;
    # do not invent parent/depth metadata or carry warmup request history forward.
    policies = {id(value): value for owner in (manager, cache) if owner is not None
                for value in vars(owner).values() if type(value).__name__ == 'JointResidencyPolicy'}
    for policy in policies.values():
        clear = getattr(policy, 'reset', None) or getattr(policy, 'clear', None)
        if callable(clear):
            clear()
        require(not getattr(policy, 'history', None) and not getattr(policy, 'restore_pins', None),
                'joint history/source pins survived the reused reset')


def numeric_delta(before, after):
    result = {}
    for key, right in after.items():
        left = before.get(key, {} if isinstance(right, dict) else 0)
        if isinstance(right, dict) and isinstance(left, dict):
            result[key] = numeric_delta(left, right)
        elif type(right) in (int, float) and type(left) in (int, float):
            result[key] = right - left
    return result


def measured_trial(engine, case, performance, slo, torch):
    """Keep old event/TTFT definitions; add the explicit complete transfer drain.

    Old run_trial synchronizes CUDA before stopping its timer but does not call
    flush_host_stores/reap_host_loads. Charge that remaining work to throughput;
    preserve the old duration and event times as raw evidence. Reporting/reset
    overhead outside the old timer is not relabelled as request processing time.
    """
    scheduler_before = copy.deepcopy(engine.scheduler.stats())
    trial = performance.run_trial(engine, case, SAMPLING, TRIAL_TIMEOUT, trace_steps=False)
    trial['request_loop_seconds'] = trial['seconds']
    trial['inputs'] = copy.deepcopy(case)
    trial['sampling'] = dict(SAMPLING)
    drain_started = time.perf_counter()
    try:
        manager = engine.scheduler.block_manager
        manager.flush_host_stores()
        manager.reap_host_loads(blocking=True)
        cache = manager.host_cache
        if cache is not None and callable(getattr(cache, 'reap', None)):
            cache.reap(blocking=True)
        torch.cuda.synchronize()
        require(not manager.used_block_ids and not manager._reservations
                and not manager._host_load_leases and not getattr(manager, '_pending_host_stores', ()),
                'state/transfer reservations survived final drain')
        if cache is not None:
            require(not getattr(cache, 'inflight', ()) and not getattr(cache, 'pending_stores', ()),
                    'host transfers survived drain')
            require(not getattr(getattr(cache, 'policy', None), 'restore_pins', ()),
                    'joint restore source pins survived drain')
    except BaseException:
        trial.update(status='FAIL', drain_error=traceback.format_exc())
    trial['drain_seconds'] = time.perf_counter() - drain_started
    trial['seconds'] = trial['request_loop_seconds'] + trial['drain_seconds']
    trial['tokens_per_second'] = trial['output_tokens'] / trial['seconds']
    trial['timing_scope'] = 'original request-loop seconds plus explicit complete host/GPU drain'
    trial['scheduler_before'] = scheduler_before
    trial['scheduler_after_drain'] = copy.deepcopy(engine.scheduler.stats())
    trial['scheduler_numeric_deltas'] = numeric_delta(scheduler_before, trial['scheduler_after_drain'])
    trial['runtime_after_drain'] = performance.snapshot(engine)
    trial['runtime_numeric_deltas_including_drain'] = numeric_delta(
        trial['runtime_before'], trial['runtime_after_drain'])
    trial['cache_after_drain'] = trial['runtime_after_drain'].get('host_cache', {})
    before_cache = trial['runtime_before'].get('host_cache', {})
    trial['cache_numeric_deltas'] = numeric_delta(before_cache, trial['cache_after_drain'])
    runtime = trial['runtime_numeric_deltas_including_drain'].get('runtime', {})
    trial['work_counters'] = dict(
        prefill_tokens=runtime.get('scheduled_prefill', trial['scheduler_numeric_deltas'].get('prefill_tokens')),
        forward_calls=runtime.get('forward_calls'),
        prefill_steps=trial['scheduler_numeric_deltas'].get('prefill_steps'),
        decode_steps=trial['scheduler_numeric_deltas'].get('decode_steps'))
    trial['cpu_resources_after_drain'] = performance.cpu_resources()
    trial['slo'] = slo.evaluate(trial, ABSOLUTE_SLO['ttft_ms'], ABSOLUTE_SLO['tpot_ms'])
    return trial


def effective_config(engine, requested, variant, normal):
    names = set(requested) | {'host_kv_cost_model_path', 'host_kv_joint_residency_path',
        'host_kv_transfer_backend', 'host_kv_transfer_idle_fastpath',
        'host_kv_prefetch_max_inflight_blocks', 'allow_runtime_graph_capture', 'model_dtype'}
    actual = {name: getattr(engine.config, name, None) for name in sorted(names)}
    for name, expected in requested.items():
        require(actual[name] == expected, f'engine ignored requested setting: {name}')
    require(actual['enforce_eager'] is False and actual['enable_mixed_graph'] is False
            and actual['max_num_seqs'] == 4 and actual['max_kvcache_blocks'] == (None if normal else 12),
            'Graph/resource budget changed')
    require(actual['host_kv_transfer_backend'] == 'legacy'
            and actual['host_kv_transfer_idle_fastpath'] is False
            and actual['host_kv_cost_model_path'] is None, 'legacy transport/no-cost baseline changed')
    if variant != 'joint':
        require(actual['host_kv_joint_residency_path'] is None, 'joint flag leaked into a baseline arm')
    return actual


def worker(a):
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='nano-joint-worker-v1', status='RUNNING', variant=a.variant,
                  branch=branch_for(a.variant), split=a.split, round=a.round,
                  normal_capacity=a.normal_capacity, trials=[], warmups=[], started=time.time(),
                  output=str(a.output), benchmark_sha256=sha(__file__))
    engine, error = None, None
    try:
        require(0 <= a.round < rounds_for(a.split), 'invalid fixed round number')
        require(absolute(sys.executable) == absolute(SERVER_PYTHON), 'use the authoritative server Python')
        branch = branch_for(a.variant)
        require(absolute(__file__) == a.root / branch / 'benchmarks' / Path(__file__).name,
                'worker must execute the benchmark from its own branch')
        # First imports in a fresh worker are always its active branch. config_for
        # uses this immutable helper instance; coordinator helper SHA equality was
        # checked before configs for multiple arms were constructed.
        common, performance, slo = helpers(a.root, branch)
        bindings = collect_bindings(a.root, a.inputs, a.protocol)
        if a.binding:
            bound = read(a.binding)
            require(bound['bindings'] == bindings and bound['bindings_sha256'] == digest(bindings),
                    'worker source/model/input binding mismatch')
        if a.freeze:
            check_frozen(read(a.freeze), bindings)
        require(a.split != 'final' or a.freeze is not None, 'final requires a pre-final freeze')
        data = read(a.inputs)
        cases = data['splits'][a.split]
        validate_cases(cases)
        config = bindings['configs']['normal' if a.normal_capacity else 'bounded'][a.variant]
        report.update(bindings_sha256=digest(bindings), inputs=str(a.inputs), inputs_sha256=sha(a.inputs),
                      protocol=str(a.protocol), protocol_sha256=sha(a.protocol), config=config,
                      binding=str(a.binding) if a.binding else None,
                      freeze=str(a.freeze) if a.freeze else None,
                      freeze_sha256=sha(a.freeze) if a.freeze else None,
                      sampling=SAMPLING, case_sha256={c['id']: digest(c) for c in cases},
                      environment=performance.environment(), host_load_before_startup=performance.cpu_resources())
        # Own the append-only guard file exclusively before handing it to guard().
        (a.output / 'guard.jsonl').open('x', encoding='utf-8').close()
        report['gpu_admission'] = stable_guard(common, a.output / 'guard.jsonl', bindings['gpu_uuid'])
        import nanovllm
        import torch
        require(absolute(nanovllm.__file__) == a.root / branch / 'nanovllm' / '__init__.py',
                'runtime imported from another branch')
        report['import_path'] = str(absolute(nanovllm.__file__))
        start = time.perf_counter()
        options = dict(config)
        model = options.pop('model')
        engine = nanovllm.LLM(model, **options)
        report.update(startup_seconds=time.perf_counter() - start,
                      startup_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      startup_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      effective_config=effective_config(engine, config, a.variant, a.normal_capacity),
                      effective_runtime=engine.config.runtime_summary(),
                      physical_kv_blocks=engine.config.num_kvcache_blocks,
                      physical_kv_bytes=engine.model_runner.kv_cache.numel()
                                        * engine.model_runner.kv_cache.element_size(),
                      host_load_after_startup=performance.cpu_resources())
        blocks = report['physical_kv_blocks']
        require(blocks > 0 and (a.normal_capacity or blocks == GPU_PAGES), 'physical GPU page cap not honored')
        require(report['physical_kv_bytes'] == blocks * PAGE_BYTES, 'model KV page is not 28 MiB')
        cache = engine.scheduler.block_manager.host_cache
        require((cache is None) if a.variant == 'gpu_only' else
                (cache is not None and cache.capacity == CPU_PAGES and cache.page_bytes == PAGE_BYTES),
                'physical CPU cache budget differs from eight pages')
        report['startup_runtime'] = performance.snapshot(engine)
        for case in cases:
            for phase in ('warmup', 'measured'):
                reset_engine(common, engine)
                stable_guard(common, a.output / 'guard.jsonl', bindings['gpu_uuid'])
                # Preserve ordinary Graph/capture defaults. There is deliberately
                # no set_graph_capture_enabled() call and no profiling context.
                trial = measured_trial(engine, case, performance, slo, torch)
                trial.update(phase=phase, variant=a.variant, split=a.split, round=a.round,
                             normal_capacity=a.normal_capacity, case_sha256=digest(case),
                             bindings_sha256=digest(bindings))
                filename = a.output / (case['id'] + ('-warmup.json' if phase == 'warmup' else '.json'))
                write(filename, trial)
                report['warmups' if phase == 'warmup' else 'trials'].append(
                    dict(path=str(filename), sha256=sha(filename), case=case['id']))
                require(trial['status'] == 'PASS' and trial['failures'] == 0
                        and trial['missing_requests'] == 0 and trial['max_active_observed'] == 1,
                        f'{phase} failed: {case["id"]}; preserve raw evidence')
        require(collect_bindings(a.root, a.inputs, a.protocol) == bindings,
                'source/model/input changed during worker')
        report.update(status='PASS_EXECUTION_ONLY', runtime=performance.snapshot(engine),
                      host_load_at_end=performance.cpu_resources())
    except BaseException as exception:
        error = exception
        report.update(status='FAIL', traceback=traceback.format_exc())
    finally:
        try:
            if engine is not None:
                engine.exit()
        except BaseException as exception:
            error = error or exception
            report.update(status='FAIL', exit_traceback=traceback.format_exc())
        report['finished'] = time.time()
        if (a.output / 'guard.jsonl').exists():
            report['guard'] = dict(path=str(a.output / 'guard.jsonl'), sha256=sha(a.output / 'guard.jsonl'))
        write(a.output / 'report.json', report)
    require(error is None, f'worker failed; see {a.output / "report.json"}')
    print(json.dumps(dict(status=report['status'], report=str(a.output / 'report.json'))), flush=True)


def suite(a):
    require(absolute(sys.executable) == absolute(SERVER_PYTHON), 'use the authoritative server Python')
    require(a.split != 'final' or a.freeze is not None, 'final requires a pre-final freeze')
    bindings = collect_bindings(a.root, a.inputs, a.protocol)
    frozen = read(a.freeze) if a.freeze else None
    if frozen:
        check_frozen(frozen, bindings)
    a.output.mkdir(parents=True, exist_ok=False)
    common, performance, _ = helpers(a.root)
    binding_file = a.output / 'binding.json'
    write(binding_file, dict(schema='nano-joint-suite-binding-v1', bindings=bindings,
                             bindings_sha256=digest(bindings)))
    index = dict(schema='nano-joint-index-v1', status='RUNNING', split=a.split,
                 normal_capacity=a.normal_capacity, rounds=rounds_for(a.split), variants=list(VARIANTS),
                 cases=list(CASE_JOBS), reports=[], schedule=[], failures=[],
                 inputs=str(a.inputs), inputs_sha256=sha(a.inputs),
                 binding=str(binding_file), binding_sha256=sha(binding_file),
                 bindings_sha256=digest(bindings),
                 freeze=str(a.freeze) if a.freeze else None,
                 freeze_sha256=sha(a.freeze) if a.freeze else None,
                 diagnostic_only=frozen['diagnostic_only'] if frozen else True,
                 dev_screen=frozen['dev_screen'] if frozen else None,
                 host_load_before=performance.cpu_resources(), started=time.time())
    (a.output / 'coordinator-guard.jsonl').open('x', encoding='utf-8').close()
    try:
        for round_ in range(rounds_for(a.split)):
            order = list(VARIANTS[round_ % len(VARIANTS):] + VARIANTS[:round_ % len(VARIANTS)])
            if round_ % 2:
                order.reverse()
            index['schedule'].append(dict(round=round_, order=order))
            for variant in order:
                stable_guard(common, a.output / 'coordinator-guard.jsonl', bindings['gpu_uuid'])
                branch = a.root / branch_for(variant)
                out = a.output / f'r{round_:02d}-{variant}'
                command = [str(SERVER_PYTHON), '-B', str(branch / 'benchmarks' / Path(__file__).name),
                           'worker', '--root', str(a.root), '--inputs', str(a.inputs),
                           '--protocol', str(a.protocol), '--variant', variant, '--split', a.split,
                           '--round', str(round_), '--binding', str(binding_file), '--output', str(out)]
                if a.freeze:
                    command += ['--freeze', str(a.freeze)]
                if a.normal_capacity:
                    command += ['--normal-capacity']
                write(a.output / f'r{round_:02d}-{variant}-launch.json',
                      dict(command=command, cwd=str(branch), host_load=performance.cpu_resources()))
                with (a.output / f'r{round_:02d}-{variant}.log').open('x', encoding='utf-8') as log:
                    result = subprocess.run(command, cwd=branch, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=WORKER_TIMEOUT, check=False)
                require(result.returncode == 0, f'worker failed ({result.returncode}); preserve {out}')
                report_path = out / 'report.json'
                index['reports'].append(dict(round=round_, variant=variant,
                                              path=str(report_path), sha256=sha(report_path)))
                write(a.output / f'progress-{len(index["reports"]):02d}.json',
                      dict(reports=index['reports'], schedule=index['schedule']))
                print(json.dumps(dict(event='WORKER_DONE', split=a.split, round=round_, variant=variant)), flush=True)
        index['status'] = 'COMPLETE_MEASUREMENTS_PENDING_ANALYSIS'
    except BaseException:
        index['status'] = 'FAIL'
        index['failures'].append(traceback.format_exc())
        raise
    finally:
        index.update(finished=time.time(), host_load_after=performance.cpu_resources())
        index['guard'] = dict(path=str(a.output / 'coordinator-guard.jsonl'),
                              sha256=sha(a.output / 'coordinator-guard.jsonl'))
        write(a.output / 'index.json', index)


def checked_file(row):
    path = Path(row['path'])
    require(path.is_absolute() and sha(path) == row['sha256'], f'nonabsolute/changed evidence file: {path}')
    return path


def check_guard_file(row, uuid, minimum):
    path = checked_file(row)
    observations = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    require(len(observations) >= minimum and all(
        o['status'] == 'AVAILABLE' and o['gpu']['uuid'] == uuid and not o['foreign_processes']
        for o in observations), f'incomplete/failed guard evidence: {path}')


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def p95(values):
    # Exactly the existing nanovllm.metrics.percentile(..., 95) index convention;
    # keep analysis CPU-only without importing the runtime package's __init__.
    require(values and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
            'missing or nonfinite event latency evidence')
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(.95 * (len(ordered) - 1))))]


def ratio(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    return 1. if numerator == 0 else None


def load_evidence(index_path):
    index = read(index_path)
    require(index.get('schema') == 'nano-joint-index-v1'
            and index['status'] == 'COMPLETE_MEASUREMENTS_PENDING_ANALYSIS' and not index['failures'],
            'incomplete suite cannot pass analysis')
    split, normal = index['split'], index['normal_capacity']
    require(split in ('dev', 'final') and index['rounds'] == rounds_for(split)
            and index['variants'] == list(VARIANTS) and index['cases'] == list(CASE_JOBS),
            'suite matrix is not the fixed complete matrix')
    require(Path(index['binding']).is_absolute() and sha(index['binding']) == index['binding_sha256'],
            'suite binding file changed')
    bound = read(index['binding'])
    bindings = bound['bindings']
    require(bound['bindings_sha256'] == index['bindings_sha256'] == digest(bindings)
            and bindings['benchmark_sha256'] == sha(__file__), 'analysis benchmark/binding mismatch')
    require(index['inputs'] == bindings['inputs'] and Path(index['inputs']).is_absolute()
            and sha(index['inputs']) == index['inputs_sha256'] == bindings['inputs_sha256'],
            'analysis input binding changed')
    data = read(index['inputs'])
    cases = data['splits'][split]
    validate_cases(cases)
    require(digest(cases) == bindings['split_sha256'][split], 'measured split changed')
    if split == 'final':
        require(index['freeze'] and Path(index['freeze']).is_absolute()
                and sha(index['freeze']) == index['freeze_sha256'], 'missing/changed pre-final freeze')
        frozen = read(index['freeze'])
        check_frozen(frozen, bindings)
        require(index['diagnostic_only'] == frozen['diagnostic_only']
                and index['dev_screen'] == frozen['dev_screen'], 'DEV failure label lost')
    expected_schedule = []
    for round_ in range(rounds_for(split)):
        order = list(VARIANTS[round_ % 3:] + VARIANTS[:round_ % 3])
        expected_schedule.append(dict(round=round_, order=list(reversed(order)) if round_ % 2 else order))
    require(index['schedule'] == expected_schedule, 'rotate/reverse schedule changed')
    expected_rows = [(s['round'], v) for s in expected_schedule for v in s['order']]
    require([(r['round'], r['variant']) for r in index['reports']] == expected_rows,
            'missing/duplicate/out-of-order fresh workers')
    check_guard_file(index['guard'], bindings['gpu_uuid'], 2 * len(expected_rows))
    reports, trials, evidence_hashes = {}, {}, {str(index_path): sha(index_path)}
    for row in index['reports']:
        report_path = checked_file(row)
        report = read(report_path)
        round_, variant = row['round'], row['variant']
        config = bindings['configs']['normal' if normal else 'bounded'][variant]
        require(report['status'] == 'PASS_EXECUTION_ONLY' and report['variant'] == variant
                and report['round'] == round_ and report['split'] == split
                and report['normal_capacity'] == normal and report['branch'] == branch_for(variant)
                and report['config'] == config and report['sampling'] == SAMPLING
                and report['bindings_sha256'] == digest(bindings)
                and report['benchmark_sha256'] == bindings['benchmark_sha256']
                and report['inputs_sha256'] == bindings['inputs_sha256']
                and report['freeze_sha256'] == index['freeze_sha256'], 'worker identity/config mismatch')
        require(positive(report['startup_seconds']) and report['physical_kv_blocks'] > 0
                and report['physical_kv_bytes'] == report['physical_kv_blocks'] * PAGE_BYTES
                and (normal or report['physical_kv_blocks'] == GPU_PAGES), 'invalid physical memory/startup evidence')
        check_guard_file(report['guard'], bindings['gpu_uuid'], 2 + 4 * len(cases))
        reports[round_, variant] = report
        evidence_hashes[str(report_path)] = row['sha256']
        for phase, field in (('warmup', 'warmups'), ('measured', 'trials')):
            require([t['case'] for t in report[field]] == list(CASE_JOBS), 'missing/duplicate warmup/measured case')
            for case, trial_row in zip(cases, report[field]):
                trial_path = checked_file(trial_row)
                trial = read(trial_path)
                require(trial['phase'] == phase and trial['variant'] == variant
                        and trial['round'] == round_ and trial['split'] == split
                        and trial['normal_capacity'] == normal and trial['case'] == case['id']
                        and trial['inputs'] == case and trial['case_sha256'] == digest(case)
                        and report['case_sha256'][case['id']] == digest(case)
                        and trial['bindings_sha256'] == digest(bindings) and trial['sampling'] == SAMPLING,
                        'raw case/input identity changed between arms')
                require(trial['status'] == 'PASS' and trial['failures'] == trial['missing_requests'] == 0
                        and trial['used_blocks_after'] == trial['kv_ref_total_after'] == 0
                        and trial['max_active_observed'] == 1 and trial['step_trace'] is None
                        and positive(trial['request_loop_seconds']) and positive(trial['seconds'])
                        and trial['drain_seconds'] >= 0
                        and math.isclose(trial['seconds'], trial['request_loop_seconds'] + trial['drain_seconds']),
                        'failed/undrained/profiled trial cannot count as timed evidence')
                requests = trial['requests']
                require([q['job'] for q in requests] == list(range(len(case['jobs'])))
                        and trial['output_tokens'] == len(requests) * MAX_TOKENS, 'missing request/token count')
                require(all(not q['failed'] and q['output_tokens'] == q['expected_tokens'] == MAX_TOKENS
                            and q['input_tokens'] == 1025 and q['finish_reason'] == 'length'
                            and len(q['output_token_ids']) == MAX_TOKENS
                            and all(type(token) is int and token >= 0 for token in q['output_token_ids'])
                            for q in requests), 'invalid raw exact-token evidence')
                evidence_hashes[str(trial_path)] = trial_row['sha256']
                if phase == 'measured':
                    trials[round_, variant, case['id']] = trial
    # An automatic-capacity comparison must retain equal actual GPU resources,
    # as well as equal configuration/CPU budgets. No subgroup is silently dropped.
    for round_ in range(rounds_for(split)):
        require(len({reports[round_, v]['physical_kv_blocks'] for v in VARIANTS}) == 1,
                'physical automatic KV capacities differ across paired arms')
    return index, bindings, reports, trials, evidence_hashes


def latency_values(rows, name):
    if name == 'ttft_ms':
        return [q[name] for trial in rows for q in trial['requests']]
    return [gap for trial in rows for q in trial['requests'] for gap in q['event_gap_ms']]


def comparisons_for(trials, split):
    rounds, comparisons = list(range(rounds_for(split))), []
    for reference in ('legacy_async', 'gpu_only'):
        for case in ('pooled', *CASE_JOBS):
            chosen = list(CASE_JOBS) if case == 'pooled' else [case]
            left = [trials[r, reference, c] for r in rounds for c in chosen]
            right = [trials[r, 'joint', c] for r in rounds for c in chosen]
            paired = [sum(trials[r, reference, c]['seconds'] for c in chosen)
                      / sum(trials[r, 'joint', c]['seconds'] for c in chosen) for r in rounds]
            exact = all({q['job']: q['output_token_ids'] for q in x['requests']}
                        == {q['job']: q['output_token_ids'] for q in y['requests']}
                        for x, y in zip(left, right))
            ttft_left, ttft_right = p95(latency_values(left, 'ttft_ms')), p95(latency_values(right, 'ttft_ms'))
            gap_left, gap_right = p95(latency_values(left, 'event_gap_ms')), p95(latency_values(right, 'event_gap_ms'))
            ttft_ratio, gap_ratio = ratio(ttft_right, ttft_left), ratio(gap_right, gap_left)
            criteria = dict(median_speed=statistics.median(paired) >= GATES['median_paired_speed_min'],
                            every_observed_round_faster=all(speed > 1 for speed in paired),
                            ttft=ttft_ratio is not None and ttft_ratio <= GATES['ttft_p95_ratio_max'],
                            event_gap=gap_ratio is not None and gap_ratio <= GATES['event_gap_p95_ratio_max'],
                            exact_tokens=exact)
            comparisons.append(dict(candidate='joint', reference=reference, case=case,
                rounds=rounds, paired_speed_ratios=paired, median_speed_ratio=statistics.median(paired),
                all_rounds_faster=criteria['every_observed_round_faster'],
                all_five_faster=len(rounds) == 5 and criteria['every_observed_round_faster'],
                ttft_p95_ratio=ttft_ratio, event_gap_p95_ratio=gap_ratio,
                candidate_ttft_p95_ms=ttft_right, reference_ttft_p95_ms=ttft_left,
                candidate_event_gap_p95_ms=gap_right, reference_event_gap_p95_ms=gap_left,
                greedy_exact=exact, gate_details=criteria,
                passes_relative_gates=len(rounds) == 5 and all(criteria.values()),
                passes_dev_screen=split == 'dev' and len(rounds) == 1 and all(criteria.values())))
    return comparisons


def optional_sum(values):
    values = list(values)
    return sum(values) if values and all(v is not None for v in values) else None


def optional_max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def summaries_for(reports, trials, split):
    summaries, rounds = [], list(range(rounds_for(split)))
    for variant in VARIANTS:
        for case in ('pooled', *CASE_JOBS):
            chosen = list(CASE_JOBS) if case == 'pooled' else [case]
            rows = [trials[r, variant, c] for r in rounds for c in chosen]
            worker_reports = [reports[r, variant] for r in rounds]
            round_rates = [sum(trials[r, variant, c]['output_tokens'] for c in chosen)
                           / sum(trials[r, variant, c]['seconds'] for c in chosen) for r in rounds]
            costs = {name: sum(t['cache_numeric_deltas'].get(name, 0) for t in rows)
                     for name in ('stores', 'loads', 'd2h_bytes', 'h2d_bytes')}
            work = {name: optional_sum(t['work_counters'].get(name) for t in rows)
                    for name in ('prefill_tokens', 'forward_calls', 'prefill_steps', 'decode_steps')}
            policy_keys = {name for t in rows for name, value in
                           t['cache_numeric_deltas'].get('policy', {}).items()
                           if type(value) in (int, float)}
            policy = {name: optional_sum(t['cache_numeric_deltas'].get('policy', {}).get(name, 0) for t in rows)
                      for name in sorted(policy_keys)}
            met, total = sum(t['slo']['slo_met_requests'] for t in rows), sum(len(t['requests']) for t in rows)
            summaries.append(dict(variant=variant, case=case, round_tokens_s=round_rates,
                median_round_tokens_s=statistics.median(round_rates),
                ttft_p95_ms=p95(latency_values(rows, 'ttft_ms')),
                event_gap_p95_ms=p95(latency_values(rows, 'event_gap_ms')),
                costs=costs, work_counters=work, policy_counters=policy,
                counter_note='Missing work counters are null, not fabricated zero; raw scheduler/runtime deltas retained.',
                absolute_slo=dict(slo_met_requests=met, total_requests=total, all_met=met == total,
                                  attainment=met / total, limits=ABSOLUTE_SLO,
                                  separate_from_relative_gates=True),
                drain_seconds=sum(t['drain_seconds'] for t in rows),
                startup_median_s=statistics.median(r['startup_seconds'] for r in worker_reports),
                startup_peak_allocated_bytes=optional_max(r['startup_peak_allocated_bytes'] for r in worker_reports),
                startup_peak_reserved_bytes=optional_max(r['startup_peak_reserved_bytes'] for r in worker_reports),
                peak_allocated_bytes=optional_max(t['memory'].get('max_memory_allocated') for t in rows),
                peak_reserved_bytes=optional_max(t['memory'].get('max_memory_reserved') for t in rows),
                physical_kv_blocks=[r['physical_kv_blocks'] for r in worker_reports],
                physical_kv_bytes=[r['physical_kv_bytes'] for r in worker_reports],
                startup_host_load=[r['host_load_before_startup'] for r in worker_reports]))
    return summaries


def analyze_index(index_path):
    index, bindings, reports, trials, evidence = load_evidence(index_path)
    split, normal = index['split'], index['normal_capacity']
    comparisons = comparisons_for(trials, split)
    summaries = summaries_for(reports, trials, split)
    legacy_control = next(s for s in summaries if s['variant'] == 'legacy_async' and s['case'] == 'first_cpu_reuse')
    cpu_reuse_exercised = normal or legacy_control['costs']['loads'] > 0
    exact = all(c['greedy_exact'] for c in comparisons)
    dev_passed = split == 'dev' and not normal and cpu_reuse_exercised and all(c['passes_dev_screen'] for c in comparisons)
    formal_passed = split == 'final' and cpu_reuse_exercised and all(c['passes_relative_gates'] for c in comparisons)
    eligible = formal_passed and not index['diagnostic_only']
    if not exact:
        status = 'REVIEW_REQUIRED_TOKEN_MISMATCH'
    elif split == 'dev':
        status = 'DEV_SCREEN_PASS' if dev_passed else 'DEV_SCREEN_FAIL_DIAGNOSTIC_ONLY'
    elif index['diagnostic_only']:
        status = 'FORMAL_DIAGNOSTIC_ONLY_DEV_FAILED'
    else:
        status = 'RELATIVE_GATES_PASS_REVIEW_REQUIRED' if formal_passed else 'RELATIVE_GATES_FAIL'
    return dict(schema='nano-joint-analysis-v1', status=status, split=split, normal_capacity=normal,
                rounds=list(range(rounds_for(split))), variants=list(VARIANTS), cases=list(CASE_JOBS),
                index=str(index_path), index_sha256=sha(index_path), evidence_sha256=evidence,
                bindings_sha256=digest(bindings), inputs_sha256=bindings['inputs_sha256'],
                benchmark_sha256=bindings['benchmark_sha256'], gates=GATES,
                comparisons=comparisons, summaries=summaries,
                first_cpu_reuse_control=dict(required=not normal, exercised=cpu_reuse_exercised,
                                             legacy_loads=legacy_control['costs']['loads'],
                                             distinct_complete_prefix_pages=16, physical_cap=GPU_PAGES),
                dev_screen_passed=dev_passed, formal_relative_gates_passed=formal_passed,
                diagnostic_only=index['diagnostic_only'], eligible_under_relative_performance_gates=eligible,
                recommended=False, numerical_label=NUMERICS_LABEL,
                decision_note='No automatic recommendation. Failed DEV is permanently diagnostic-only; '
                              'exact tokens do not certify logits/KV. Absolute SLO is reported separately. '
                              'Bounded and normal-capacity final reports must both be reviewed.',
                final_used_for_tuning=False,
                note='Synthetic Chinese/English/code/log traces only, no production claim. '
                     'Pooled ratios use sums of full drained seconds within each paired round, '
                     'then median of paired ratios; per-case comparisons use the same five gates.')


def analyze(a):
    require(not a.output.exists(), 'analysis output must be new')
    try:
        result = analyze_index(a.index)
    except BaseException:
        write(a.output, dict(schema='nano-joint-analysis-v1', status='INVALID_EVIDENCE',
                              recommended=False, index=str(a.index), traceback=traceback.format_exc()))
        raise
    write(a.output, result)
    print(json.dumps(dict(status=result['status'], output=str(a.output),
                          pooled=[c for c in result['comparisons'] if c['case'] == 'pooled']), indent=2))


def freeze(a):
    require(not a.output.exists(), 'freeze output must be new')
    analysis = read(a.dev_analysis)
    require(analysis.get('schema') == 'nano-joint-analysis-v1' and analysis['split'] == 'dev'
            and analysis['normal_capacity'] is False, 'freeze requires the single bounded DEV screen')
    # Recompute the screen from immutable raw evidence, not an editable PASS label.
    index_path = Path(analysis['index'])
    require(index_path.is_absolute() and analysis == analyze_index(index_path), 'DEV analysis/evidence changed')
    bindings = collect_bindings(a.root, a.inputs, a.protocol)
    require(analysis['bindings_sha256'] == digest(bindings), 'sources/model/recipe changed after DEV')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == bindings['gpu_uuid'],
            'freeze requires the measured physical GPU UUID in CUDA_VISIBLE_DEVICES')
    passed = analysis['dev_screen_passed']
    document = dict(schema='nano-joint-freeze-v1',
                    status='FROZEN_BEFORE_FINAL' if passed else 'FROZEN_DIAGNOSTIC_ONLY_DEV_FAILED',
                    bindings=bindings, bindings_sha256=digest(bindings),
                    dev_screen=dict(analysis=str(a.dev_analysis), analysis_sha256=sha(a.dev_analysis),
                                    index=str(index_path), index_sha256=sha(index_path), passed=passed),
                    diagnostic_only=not passed, recommended=False,
                    final_plan=dict(capacity_modes=['bounded', 'normal'], rounds_per_mode=5,
                                    workers_per_mode=15, total_workers=30, all_cases=True),
                    final_used_for_tuning=False, created=time.time())
    write(a.output, document)
    print(json.dumps(dict(status=document['status'], output=str(a.output),
                          diagnostic_only=document['diagnostic_only'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    for mode in ('prepare', 'fit', 'worker', 'suite', 'analyze', 'freeze'):
        command = sub.add_parser(mode)
        command.add_argument('--root', type=Path, default=SERVER_ROOT)
        command.add_argument('--output', type=Path, required=True)
        if mode != 'analyze':
            command.add_argument('--inputs', type=Path, help='default: R/inputs.json')
            command.add_argument('--protocol', type=Path, help='default: R/harness-protocol.json; R/protocol.json is read only')
        if mode in ('worker', 'suite'):
            command.add_argument('--split', choices=('dev', 'final'), default='dev')
            command.add_argument('--freeze', type=Path)
            command.add_argument('--normal-capacity', action='store_true',
                                 help='automatic KV capacity for ALL cases, equal four-slot base resource budget')
        if mode == 'worker':
            command.add_argument('--variant', choices=VARIANTS, required=True)
            command.add_argument('--round', type=int, default=0)
            command.add_argument('--binding', type=Path)
        elif mode == 'analyze':
            command.add_argument('--index', type=Path, required=True)
        elif mode == 'freeze':
            command.add_argument('--dev-analysis', type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, absolute(value))
    if args.action != 'analyze':
        args.inputs = args.inputs or args.root / 'inputs.json'
        args.protocol = args.protocol or args.root / 'harness-protocol.json'
        require(args.protocol != args.root / 'protocol.json', 'preregistered R/protocol.json must remain untouched')
    if args.action in ('worker', 'suite'):
        require(args.split != 'final' or args.freeze is not None, '--split final requires --freeze')
    globals()[args.action](args)


if __name__ == '__main__':
    main()
