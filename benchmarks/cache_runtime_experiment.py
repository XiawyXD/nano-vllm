"""Independent cache calibration, frozen workloads, interleaved five-round evidence.

Uses the existing guard, run_trial, SLO evaluator and raw event definitions.
Never changes correctness/speed gates or promotes a configuration automatically.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable
VARIANTS=('gpu_only','legacy_sync','legacy_async','cost_only','batched_only','combined')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for b in iter(lambda:file.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as file:json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)


def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'nanovllm').rglob('*.py'))}


def base_config(model,normal=False):
    return dict(model=model,tensor_parallel_size=1,max_model_len=2048,max_num_seqs=4,
        max_num_batched_tokens=256,prefill_chunk_size=128,max_prefill_steps=1,
        gpu_memory_utilization=.85,max_kvcache_blocks=None if normal else 12,
        rmsnorm_backend='triton',linear_backend='torch',numerical_backend='default',
        metadata_backend='reuse',enforce_eager=False,enable_prefix_caching=True,
        batching_mode='separate',speculative_policy='off',enable_mixed_graph=False)


def config_for(variant,model,calibration_dir,normal=False):
    c=base_config(model,normal)
    if variant!='gpu_only':c['host_kv_cache_bytes']=2**30
    if variant in ('legacy_async','batched_only','combined'):c['host_kv_async_prefetch']=True
    if variant in ('batched_only','combined'):
        c.update(host_kv_transfer_backend='batched',host_kv_transfer_batch_blocks=4,
                 host_kv_transfer_slots=2,host_kv_transfer_max_bytes=256*2**20)
    if variant in ('cost_only','combined'):
        c['host_kv_cost_model_path']=str(Path(calibration_dir)/('batched.json' if variant=='combined' else 'legacy.json'))
    return c


def build_cases(tokenizer,seed):
    rng=random.Random(seed)
    pool=tokenizer.encode('Memory cache requests prefix scheduling. 缓存、会话与资源回收。 def merge(items): return items\n',add_special_tokens=False)
    def document(n):return [rng.choice(pool) for _ in range(n)]
    docs=[document(1536) for _ in range(4)]
    def case(name,prompts,concurrency=1,arrivals=None):
        return dict(id=name,dataset=name,scenario='synthetic_capacity_trace',concurrency=concurrency,
            jobs=[dict(job=i,arrival_s=0. if arrivals is None else arrivals[i],
                       prompt_token_ids=p,max_tokens=16) for i,p in enumerate(prompts)])
    common=document(512)
    return [case('cyclic',[docs[i%4][:1025] for i in range(16)]),
            case('session',[docs[s][:513+turn*256] for turn in range(4) for s in range(4)]),
            case('branch',[common+docs[i%4][:513] for i in range(16)]),
            case('unique',[document(1025) for _ in range(16)]),
            case('burst',[docs[i%4][:1025] for i in range(12)],2,[(i//4)*.05 for i in range(12)])]


def prepare(a):
    from transformers import AutoTokenizer
    from benchmarks.runtime_v2_performance import environment
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    splits={name:build_cases(tokenizer,seed) for name,seed in (
        ('calibration',2026102801),('dev',2026102802),('formal',2026102803))}
    prompts={s:{digest(j['prompt_token_ids']) for c in cases for j in c['jobs']} for s,cases in splits.items()}
    prefixes={s:{digest(j['prompt_token_ids'][:256]) for c in cases for j in c['jobs']} for s,cases in splits.items()}
    overlap={f'{x}/{y}':dict(prompts=len(prompts[x]&prompts[y]),first_prefix_pages=len(prefixes[x]&prefixes[y]))
        for x,y in (('calibration','dev'),('calibration','formal'),('dev','formal'))}
    assert not any(v['prompts'] or v['first_prefix_pages'] for v in overlap.values())
    files={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p))
        for p in Path(a.model).iterdir() if p.is_file() and p.suffix in ('.json','.safetensors')}
    write(a.output,dict(schema='cache-runtime-inputs-v1',model=str(Path(a.model).resolve()),
        splits=splits,overlap=overlap,model_files=files,environment=environment(),
        note='New deterministic synthetic traces for cache cost/pressure, not natural-language quality or production sampling'))
    print(json.dumps(dict(status='PREPARED_CPU_ONLY',path=str(a.output),overlap=overlap)))


def guard(path):
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    return ensure_gpu1_available(evidence_path=path)


def calibrate(a):
    a.output.parent.mkdir(parents=True,exist_ok=True)
    admission=guard(a.output.with_suffix('.guard.jsonl'))
    from nanovllm import LLM,SamplingParams
    from nanovllm.engine.sequence import Sequence
    import torch
    c=base_config(a.model)
    c.update(host_kv_cache_bytes=2**30,host_kv_async_prefetch=a.transport=='batched')
    if a.transport=='batched':c['host_kv_transfer_backend']='batched'
    model=c.pop('model');engine=None
    report=dict(schema='nano-cache-cost-v1',status='RUNNING',query_ms=[],transfer_ms={},raw={},source_sha256=sources())
    # Train one global first-reuse prior on calibration traces only. Neither DEV
    # nor formal output is accessed, and this is not a model-quality objective.
    from collections import Counter
    counts=Counter()
    for case in read(a.inputs)['splits']['calibration']:
        for job in case['jobs']:
            ids=job['prompt_token_ids']
            for end in range(256,len(ids),256):counts[case['id'],digest(ids[:end])]+=1
    report['cold_reuse_probability']=sum(n>1 for n in counts.values())/len(counts)
    report['prior_provenance']=dict(inputs_sha256=sha(a.inputs),split='calibration',
        distinct_prefixes=len(counts),reused_prefixes=sum(n>1 for n in counts.values()),
        definition='fraction of distinct prefix chains observed in more than one request; one conservative future reuse')
    try:
        engine=LLM(model,**c);manager=engine.scheduler.block_manager;cache=manager.host_cache
        report['identity']=dict(model=str(Path(model).resolve()),config_sha256=sha(Path(model)/'config.json'),
            chunk_size=128,block_size=256,linear_backend='torch',numerical_backend='default',rmsnorm_backend='triton',
            transport=a.transport,gpu_uuid=admission['gpu']['uuid'],dtype=str(engine.config.hf_config.dtype),
            model_dtype=engine.config.model_dtype,batch_blocks=engine.config.host_kv_transfer_batch_blocks,
            slots=engine.config.host_kv_transfer_slots)
        rng=random.Random(2026102801)
        tokens=engine.tokenizer.encode('Calibration of memory scheduling and cache transfer. ',add_special_tokens=False)
        for past in (0,256,512,1024,1536):
            for q in (1,32,64,128,256):
                guard(a.output.with_suffix('.guard.jsonl'))
                ids=[rng.choice(tokens) for _ in range(past+q)]
                seq=Sequence(ids,SamplingParams(temperature=0,max_tokens=1));manager.allocate(seq,0)
                try:
                    seq.num_cached_tokens=0
                    while seq.num_cached_tokens<past:
                        seq.num_scheduled_tokens=min(128,past-seq.num_cached_tokens)
                        engine.model_runner.run([seq],True)
                        seq.num_cached_tokens+=seq.num_scheduled_tokens
                    seq.num_scheduled_tokens=q
                    engine.model_runner.run([seq],True)
                    values=[]
                    for _ in range(5):
                        start=time.perf_counter();engine.model_runner.run([seq],True)
                        values.append((time.perf_counter()-start)*1000)
                    report['query_ms'].append(dict(q=q,past=past,ms=statistics.median(values),samples=values))
                finally:manager.deallocate(seq)
        # Calibrate whole transfer critical-path cost, including pack/scatter.
        for direction in ('d2h','h2d'):
            rows=[]
            for n in (1,2,4):
                values=[]
                for repeat in range(6):
                    guard(a.output.with_suffix('.guard.jsonl'))
                    manager.reap_host_loads(blocking=True);cache.clear()
                    keys=[cache.key(None,[100+repeat,i]) for i in range(n)]
                    engine.model_runner.kv_cache[:,:,:n].fill_(repeat+.25)
                    torch.cuda.synchronize()
                    if direction=='h2d':
                        if getattr(cache,'batched',False):cache.put_many(list(zip(keys,range(n))));cache.reap(blocking=True)
                        else:
                            for key,bid in zip(keys,range(n)):cache.put(key,bid)
                    start=time.perf_counter()
                    if direction=='d2h':
                        if getattr(cache,'batched',False):cache.put_many(list(zip(keys,range(n))));cache.reap(blocking=True)
                        else:
                            for key,bid in zip(keys,range(n)):cache.put(key,bid)
                    else:
                        pairs=list(zip(keys,range(4,4+n)))
                        if getattr(cache,'batched',False):
                            cache.load_many(pairs);cache.wait_for_blocks(range(4,4+n));cache.reap(blocking=True)
                        else:
                            for key,bid in pairs:cache.load(key,bid)
                    torch.cuda.synchronize()
                    if repeat:values.append((time.perf_counter()-start)*1000)
                rows.append([n,statistics.median(values)])
                report['raw'][f'{direction}_{n}']=values
            report['transfer_ms'][direction]=rows
        report.update(status='PASS_CALIBRATION_ONLY',config=engine.config.runtime_summary())
        from benchmarks.runtime_v2_performance import cpu_resources
        report['host_load']=cpu_resources()
    except BaseException:
        report.update(status='FAIL',traceback=traceback.format_exc());raise
    finally:
        if engine is not None:engine.exit()
        write(a.output,report)
    print(json.dumps(dict(status=report['status'],transport=a.transport,transfer_ms=report['transfer_ms'])))


def reset(engine):
    m=engine.scheduler.block_manager;m.flush_host_stores();m.reap_host_loads(blocking=True)
    assert not m.used_block_ids and not m._reservations and not m._host_load_leases
    if m.host_cache is not None:
        m.host_cache.clear();m.host_cache.counters.clear()
        if getattr(m.host_cache,'policy',None) is not None:m.host_cache.policy.counters.clear()
    for bid in list(m.free_block_ids):m._remove_prefix_index(bid)


def worker(a):
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',variant=a.variant,split=a.split,round=a.round,trials=[],source_sha256=sources())
    engine=None
    try:
        report['gpu']=guard(a.output/'guard.jsonl')
        data=read(a.inputs)
        if a.freeze:
            frozen=read(a.freeze)
            assert frozen['source_sha256']==sources() and frozen['inputs_sha256']==sha(a.inputs)
            assert frozen['gpu_uuid']==report['gpu']['gpu']['uuid']
            assert frozen['benchmark_sha256']==sha(__file__)
            for name,expected in frozen['calibration_sha256'].items():assert sha(Path(a.calibration_dir)/name)==expected
        for name,identity in data['model_files'].items():
            p=Path(data['model'])/name
            assert p.stat().st_size==identity['bytes'] and p.stat().st_mtime_ns==identity['mtime_ns']
        c=config_for(a.variant,data['model'],a.calibration_dir,a.normal_capacity)
        report['config']=dict(c)
        from nanovllm import LLM
        from benchmarks.runtime_v2_performance import run_trial,environment
        from benchmarks.slo_goodput import evaluate
        import torch
        model=c.pop('model');start=time.perf_counter();engine=LLM(model,**c)
        report.update(startup_seconds=time.perf_counter()-start,environment=environment(),
            physical_kv_blocks=engine.config.num_kvcache_blocks,
            physical_kv_bytes=engine.model_runner.kv_cache.numel()*engine.model_runner.kv_cache.element_size(),
            startup_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            startup_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        cases=data['splits'][a.split]
        if a.normal_capacity:cases=[cases[0]]
        if a.cases:cases=[case for case in cases if case['id'] in a.cases]
        sampling=dict(temperature=0,ignore_eos=True,speculative_method='off')
        for case in cases:
            reset(engine)
            warm=run_trial(engine,case,sampling,180)
            write(a.output/(case['id']+'-warmup.json'),warm)
            if warm['status']!='PASS':raise RuntimeError(warm['error'] or 'warmup state leak')
            reset(engine);guard(a.output/'guard.jsonl')
            engine.set_graph_capture_enabled(False)
            trial=run_trial(engine,case,sampling,180)
            trial['slo']=evaluate(trial,1000,30)
            engine.scheduler.block_manager.reap_host_loads(blocking=True)
            trial['cache_after_drain']=engine.runtime_metrics()['host_cache']
            trial['scheduler']=engine.scheduler.stats()
            write(a.output/(case['id']+'.json'),trial)
            report['trials'].append(case['id']+'.json')
            if trial['status']!='PASS':raise RuntimeError(trial['error'] or 'trial state leak')
        report['status']='PASS_EXECUTION_ONLY'
        report['runtime']=engine.runtime_metrics()
        measured=[read(a.output/name) for name in report['trials']]
        report['peak_allocated_bytes']=max(t['memory']['max_memory_allocated'] for t in measured)
        report['peak_reserved_bytes']=max(t['memory']['max_memory_reserved'] for t in measured)
    except BaseException:
        report.update(status='FAIL',traceback=traceback.format_exc());raise
    finally:
        try:
            if engine is not None:engine.exit()
        finally:write(a.output/'report.json',report)
    print(json.dumps(dict(status=report['status'],variant=a.variant,output=str(a.output))))


def freeze(a):
    data=read(a.inputs)
    for name,identity in data['model_files'].items():assert sha(Path(data['model'])/name)==identity['sha256']
    write(a.output,dict(status='FROZEN_BEFORE_FORMAL',source_sha256=sources(),benchmark_sha256=sha(__file__),
        inputs_sha256=sha(a.inputs),calibration_sha256={n:sha(Path(a.calibration_dir)/n) for n in ('legacy.json','batched.json')},
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],variants=list(VARIANTS),rounds=5,
        gates=dict(speed=1.03,all_rounds_faster=True,ttft_p95_ratio=1.10,event_gap_p95_ratio=1.10,
                   logits_atol=.01,logits_rtol=.01,kv_bitwise=True),created=time.time()))


def suite(a):
    frozen=read(a.freeze);a.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for round_ in range(frozen['rounds']):
        order=list(frozen['variants'])
        order=order[round_%len(order):]+order[:round_%len(order)]
        if round_%2:order.reverse()
        if a.normal_capacity:order=['gpu_only','combined'] if round_%2==0 else ['combined','gpu_only']
        for variant in order:
            guard(a.output/'coordinator-guard.jsonl')
            out=a.output/f'r{round_:02d}-{variant}'
            cmd=[PY,'-B','-m','benchmarks.cache_runtime_experiment','worker','--variant',variant,
                 '--split','formal','--round',str(round_),'--inputs',str(a.inputs),
                 '--calibration-dir',str(a.calibration_dir),'--freeze',str(a.freeze),'--output',str(out)]
            if a.normal_capacity:cmd+=['--normal-capacity']
            with (a.output/f'r{round_:02d}-{variant}.log').open('x') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'worker failed; preserve partial round {out}')
            reports.append(dict(round=round_,variant=variant,path=str(out/'report.json')))
            print(json.dumps(dict(event='WORKER_DONE',round=round_,variant=variant)),flush=True)
            write(a.output/f'progress-{len(reports)}.json',dict(reports=reports))
    write(a.output/'index.json',dict(status='COMPLETE_MEASUREMENTS_PENDING_ANALYSIS',reports=reports,
        normal_capacity=a.normal_capacity,freeze=str(a.freeze),freeze_sha256=sha(a.freeze)))


def analyze(a):
    from nanovllm.metrics import percentile
    index=read(a.index);groups={};reports={}
    for row in index['reports']:
        report=read(row['path']);reports[row['round'],row['variant']]=report
        for filename in report['trials']:
            trial=read(Path(row['path']).parent/filename)
            groups[row['round'],row['variant'],trial['case']]=trial
    variants=sorted({v for _,v,_ in groups});rounds=sorted({r for r,_,_ in groups})
    cases=sorted({c for _,_,c in groups})
    numerics=[]
    for (r,v,c),trial in groups.items():
        for reference in ('gpu_only','legacy_sync'):
            base=groups.get((r,reference,c))
            if base is None:continue
            same=(len(base['requests'])==len(trial['requests']) and all(
                x['job']==y['job'] and x['output_token_ids']==y['output_token_ids']
                for x,y in zip(base['requests'],trial['requests'])))
            numerics.append(dict(round=r,variant=v,reference=reference,case=c,all_tokens_equal=same,
                matched_requests=sum(x['output_token_ids']==y['output_token_ids']
                    for x,y in zip(base['requests'],trial['requests'])),requests=len(trial['requests'])))
    summaries=[]
    for v in variants:
        rows=[t for (r,x,c),t in groups.items() if x==v]
        rr=[reports[r,v] for r in rounds if (r,v) in reports]
        summaries.append(dict(variant=v,
            median_round_tokens_s=statistics.median(sum(groups[r,v,c]['output_tokens'] for c in cases)
                /sum(groups[r,v,c]['seconds'] for c in cases) for r in rounds if (r,v) in reports),
            ttft_p95_ms=percentile([q['ttft_ms'] for t in rows for q in t['requests']],95),
            event_gap_p95_ms=percentile([g for t in rows for q in t['requests'] for g in q['event_gap_ms']],95),
            slo_requests=sum(t['slo']['slo_met_requests'] for t in rows),
            total_requests=sum(len(t['requests']) for t in rows),
            startup_median_s=statistics.median(x['startup_seconds'] for x in rr),
            peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in rr),
            peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in rr),
            physical_kv_blocks=[x['physical_kv_blocks'] for x in rr],
            raw_cache_metrics=[t['cache_after_drain'] for t in rows]))
    comparisons=[]
    pairs=[('combined','legacy_sync'),('batched_only','legacy_sync'),('batched_only','legacy_async'),
           ('cost_only','legacy_sync'),('combined','batched_only'),('combined','gpu_only')]
    for candidate,reference in pairs:
        if candidate not in variants or reference not in variants:continue
        for subgroup in [None,*cases]:
            selected=cases if subgroup is None else [subgroup]
            paired=[sum(groups[r,reference,c]['seconds'] for c in selected)
                    /sum(groups[r,candidate,c]['seconds'] for c in selected) for r in rounds]
            def values(v,kind):
                rows=[groups[r,v,c] for r in rounds for c in selected]
                if kind=='ttft':return [q['ttft_ms'] for t in rows for q in t['requests']]
                return [g for t in rows for q in t['requests'] for g in q['event_gap_ms']]
            ttft=percentile(values(candidate,'ttft'),95)/percentile(values(reference,'ttft'),95)
            gap=percentile(values(candidate,'gap'),95)/percentile(values(reference,'gap'),95)
            speed=statistics.median(paired)
            correct=all(len(groups[r,reference,c]['requests'])==len(groups[r,candidate,c]['requests']) and all(
                x['output_token_ids']==y['output_token_ids'] for x,y in zip(
                    groups[r,reference,c]['requests'],groups[r,candidate,c]['requests']))
                for r in rounds for c in selected)
            passed=(len(rounds)>=5 and speed>=1.03 and all(r>1 for r in paired)
                    and ttft<=1.10 and gap<=1.10 and correct)
            candidate_rows=[groups[r,candidate,c] for r in rounds for c in selected]
            absolute=all(t['slo']['slo_met_requests']==t['slo']['total_requests'] for t in candidate_rows)
            comparisons.append(dict(candidate=candidate,reference=reference,case=subgroup or 'pooled',
                paired_speed_ratios=paired,median_speed_ratio=speed,all_rounds_faster=all(r>1 for r in paired),
                ttft_p95_ratio=ttft,event_gap_p95_ratio=gap,greedy_exact=correct,
                passes_relative_gates=passed,absolute_slo_all_met=absolute,passes_original_gates=passed and absolute))
    old_exact=all(n['all_tokens_equal'] for n in numerics if n['reference']=='legacy_sync' and n['variant']!='gpu_only')
    gpu_exact=all(n['all_tokens_equal'] for n in numerics if n['reference']=='gpu_only')
    status='ANALYZED' if gpu_exact else 'ANALYZED_WITH_BASELINE_REVIEW' if old_exact else 'REVIEW_REQUIRED'
    write(a.output,dict(status=status,rounds=rounds,summaries=summaries,comparisons=comparisons,
        numerics=numerics,cached_reference_exact=old_exact,gpu_only_reference_exact=gpu_exact,
        index_sha256=sha(a.index),note='Each speed claim uses a paired median and its own output reference. GPU-only/cache scheduling differences remain REVIEW, not a relaxed numeric gate. Subgroup failures and absolute SLO retained.'))
    print(json.dumps(dict(status=status,comparisons=[r for r in comparisons if r['case']=='pooled']),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--model',required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('calibrate');q.add_argument('--model',required=True);q.add_argument('--inputs',type=Path,required=True);q.add_argument('--transport',choices=('legacy','batched'),required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('analyze');q.add_argument('--index',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    for mode in ('worker','freeze','suite'):
        q=sub.add_parser(mode);q.add_argument('--inputs',type=Path,required=True);q.add_argument('--calibration-dir',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        if mode!='freeze':q.add_argument('--freeze',type=Path)
        if mode=='worker':
            q.add_argument('--variant',choices=VARIANTS,required=True);q.add_argument('--split',choices=('dev','formal'),default='dev');q.add_argument('--round',type=int,default=-1);q.add_argument('--cases',nargs='+')
        if mode!='freeze':q.add_argument('--normal-capacity',action='store_true')
    a=p.parse_args();globals()[a.action](a)


if __name__=='__main__':main()
