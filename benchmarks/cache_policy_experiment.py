"""Bounded cache-policy/idle-path experiment; old gates and engine timing reused."""
import argparse
from collections import defaultdict
import copy
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

from benchmarks.cache_runtime_experiment import read,write,sha,digest,sources,guard,base_config,reset
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('legacy_async','v1_legacy','v2_legacy','v1_batch','v2_batch','idle_fastpath')
PAGE_BYTES=28*2**20

def branch_for(variant):
    return 'baseline' if variant in ('legacy_async','v1_legacy','v1_batch') else 'candidate-fastpath' if variant=='idle_fastpath' else 'candidate'


def prepare(a):
    from transformers import AutoTokenizer
    from benchmarks.runtime_v2_performance import CHINESE_TASKS,NATURAL_TASKS,CODE_TASKS,environment
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    passages=list(CHINESE_TASKS)+list(NATURAL_TASKS)+list(CODE_TASKS)
    # 原创材料各段只拼一次；split/nonce改变开头，背景材料共享须明确记录。
    def build(split,seed):
        rng=random.Random(seed);serial=0
        def doc(length):
            nonlocal serial
            serial+=1;start=rng.randrange(len(passages))
            content=(f'Document {split} {rng.getrandbits(96):024x} {serial}.\n'+
                     '\n\n'.join(passages[start:]+passages[:start]))
            tokens=tokenizer.encode(content,add_special_tokens=False)
            assert len(tokens)>length
            return tokens[:length]+tokenizer.encode('\n',add_special_tokens=False)[:1]
        def documents(n):return [doc((1024,512,256,1024)[i%4]) for i in range(n)]
        def case(name,prompts):
            return dict(id=name,dataset=name,scenario='causal_synthetic_cache_trace',concurrency=1,
                jobs=[dict(job=i,arrival_s=0.,prompt_token_ids=p,max_tokens=16) for i,p in enumerate(prompts)])
        cases=[]
        hot=documents(4);cold=documents(16)
        mixed=hot*2
        for i in range(8):mixed += [hot[i%4],cold[2*i],cold[2*i+1]]
        cases.append(case('mixed_hot_cold',mixed))
        old=documents(4);new=documents(4);cold=documents(8)
        phase=old*2
        for i in range(16):
            phase.append(new[i%4])
            if i%2==1:phase.append(cold[i//2])
        cases.append(case('phase_shift',phase))
        hot=documents(4);cases.append(case('first_reuse',hot*4))
        cases.append(case('unique',documents(16)))
        common=doc(512)[:-1];branches=documents(4)
        cases.append(case('shared_branches',[common+p[:512]+p[-1:] for p in branches]*4))
        return cases
    splits={s:build(s,seed) for s,seed in [('train',2026102804),('dev',2026102805),('final',2026102806)]}
    def hashes(cases,prefix=False):return {digest(j['prompt_token_ids'][:256] if prefix else j['prompt_token_ids']) for c in cases for j in c['jobs']}
    audit={}
    for s,t in [('train','dev'),('train','final'),('dev','final')]:
        audit[s+'/'+t]=dict(prompts=len(hashes(splits[s])&hashes(splits[t])),first_pages=len(hashes(splits[s],True)&hashes(splits[t],True)))
    old=read(a.root.parent/'cache-runtime-20261028-001/inputs.json')
    prior=[c for cases in old['splits'].values() for c in cases]
    for s in splits:audit[s+'/old']=dict(prompts=len(hashes(splits[s])&hashes(prior)),first_pages=len(hashes(splits[s],True)&hashes(prior,True)))
    assert not any(x['prompts'] or x['first_pages'] for x in audit.values())
    write(a.output,dict(schema='cache-policy-inputs-v2',model=a.model,splits=splits,overlap=audit,
        model_files=old['model_files'],environment=environment(),
        note='Synthetic causal hot/cold traces, not production traffic; disjoint prompt/first-page hashes but authored background passages shared. Prefix lengths balanced across hot/cold; future labels never enter runtime.'))
    print(json.dumps(dict(status='PREPARED_CPU_ONLY',overlap=audit)))


def fit(a):
    data=read(a.inputs);distances=[];chains=[]
    for case in data['splits']['train']:
        observations=defaultdict(list)
        for i,job in enumerate(case['jobs']):
            ids=job['prompt_token_ids']
            for end in range(256,len(ids),256):observations[digest(ids[:end])].append(i)
        chains.extend(observations.values())
        distances.extend(b-a for positions in observations.values() for a,b in zip(positions,positions[1:]))
    assert distances
    ordered=sorted(distances);p90=ordered[math.ceil(.9*len(ordered))-1]
    window=min(64,max(4,2*p90))
    reused=sum(any(b-a<=window for a,b in zip(p,p[1:])) for p in chains)
    prior=reused/len(chains)
    a.output.mkdir(exist_ok=False)
    for transport in ('legacy','batched'):
        source=a.root/'calibration-v1'/f'{transport}.json';doc=read(source)
        doc['cold_reuse_probability']=prior
        doc['replacement']=dict(kind='capacity_value',reuse_window_requests=window,victim_scan_limit=64)
        doc['policy_fit']=dict(inputs_sha256=sha(a.inputs),split='train',p90_request_distance=p90,
            window_rule='clip(2*TRAIN p90 finite request reuse distance,4,64)',
            distinct_prefixes=len(chains),reused_within_window=reused,
            measured_cost_table_reused_sha256=sha(source),recipe='one fixed capacity-value candidate, no grid')
        write(a.output/f'{transport}.json',doc)
    print(json.dumps(dict(status='FIT_TRAIN_ONLY',window=window,cold_prior=prior,prefixes=len(chains))))


def config(variant,data,root,normal=False):
    c=base_config(data['model'],normal)
    c['host_kv_cache_bytes']=8*PAGE_BYTES+64
    if variant=='legacy_async':c['host_kv_async_prefetch']=True
    if variant in ('v1_batch','v2_batch','idle_fastpath'):
        c.update(host_kv_async_prefetch=True,host_kv_transfer_backend='batched')
    if variant!='legacy_async':
        directory='calibration-v2' if variant.startswith('v2_') else 'calibration-v1'
        transport='batched' if variant in ('v1_batch','v2_batch','idle_fastpath') else 'legacy'
        c['host_kv_cost_model_path']=str(root/directory/(transport+'.json'))
    if variant=='idle_fastpath':c['host_kv_transfer_idle_fastpath']=True
    return c


def worker(a):
    a.output.mkdir(exist_ok=False,parents=True);engine=None
    report=dict(status='RUNNING',variant=a.variant,split=a.split,round=a.round,trials=[],source_sha256=sources(),benchmark_sha256=sha(__file__))
    try:
        report['gpu']=guard(a.output/'guard.jsonl')
        data=read(a.inputs);c=config(a.variant,data,a.root,a.normal_capacity);report['config']=dict(c)
        if a.freeze:
            frozen=read(a.freeze)
            assert frozen['inputs_sha256']==sha(a.inputs) and frozen['benchmark_sha256']==sha(__file__)
            branch=branch_for(a.variant)
            assert frozen['sources'][branch]==sources()
            for name,value in frozen['calibrations'].items():assert sha(a.root/name)==value
        for name,identity in data['model_files'].items():
            path=Path(data['model'])/name
            assert path.stat().st_size==identity['bytes'] and path.stat().st_mtime_ns==identity['mtime_ns']
        from nanovllm import LLM
        import nanovllm,torch
        from benchmarks.runtime_v2_performance import run_trial,environment
        from benchmarks.slo_goodput import evaluate
        model=c.pop('model');start=time.perf_counter();engine=LLM(model,**c)
        report.update(startup_seconds=time.perf_counter()-start,environment=environment(),
            import_path=nanovllm.__file__,physical_kv_blocks=engine.config.num_kvcache_blocks,
            physical_kv_bytes=engine.model_runner.kv_cache.numel()*engine.model_runner.kv_cache.element_size())
        cases=data['splits'][a.split]
        if a.cases:cases=[case for case in cases if case['id'] in a.cases]
        if a.normal_capacity:cases=[case for case in cases if case['id']=='first_reuse']
        assert cases
        sampling=dict(temperature=0,ignore_eos=True,speculative_method='off')
        for case in cases:
            reset(engine);warm=run_trial(engine,case,sampling,180)
            write(a.output/(case['id']+'-warmup.json'),warm);assert warm['status']=='PASS'
            reset(engine);guard(a.output/'guard.jsonl');engine.set_graph_capture_enabled(False)
            trial=run_trial(engine,case,sampling,180);trial['slo']=evaluate(trial,1000,30)
            engine.scheduler.block_manager.reap_host_loads(blocking=True)
            trial['cache_after_drain']=engine.runtime_metrics()['host_cache']
            write(a.output/(case['id']+'.json'),trial);report['trials'].append(case['id']+'.json')
            assert trial['status']=='PASS',trial.get('error')
        report['status']='PASS_EXECUTION_ONLY'
    except BaseException:
        report.update(status='FAIL',traceback=traceback.format_exc());raise
    finally:
        try:
            if engine is not None:engine.exit()
        finally:write(a.output/'report.json',report)
    print(json.dumps(dict(status=report['status'],variant=a.variant,output=str(a.output))))


def suite(a):
    a.output.mkdir(exist_ok=False,parents=True);reports=[]
    for round_ in range(a.rounds):
        order=a.variants[round_%len(a.variants):]+a.variants[:round_%len(a.variants)]
        if round_%2:order=list(reversed(order))
        for variant in order:
            guard(a.output/'coordinator-guard.jsonl')
            branch=branch_for(variant)
            directory=a.root/branch;out=a.output/f'r{round_:02d}-{variant}'
            cmd=[sys.executable,'-B','-m','benchmarks.cache_policy_experiment','worker','--root',str(a.root),
                '--inputs',str(a.inputs),'--variant',variant,'--split',a.split,'--round',str(round_),'--output',str(out)]
            if a.freeze:cmd+=['--freeze',str(a.freeze)]
            if a.normal_capacity:cmd+=['--normal-capacity']
            if a.cases:cmd+=['--cases',*a.cases]
            with (a.output/f'r{round_:02d}-{variant}.log').open('x') as log:
                result=subprocess.run(cmd,cwd=directory,stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode==0,f'worker failed; preserve {out}'
            reports.append(dict(round=round_,variant=variant,path=str(out/'report.json')))
            write(a.output/f'progress-{len(reports)}.json',dict(reports=reports))
            print(json.dumps(dict(event='WORKER_DONE',round=round_,variant=variant)),flush=True)
    write(a.output/'index.json',dict(status='COMPLETE_MEASUREMENTS_PENDING_ANALYSIS',reports=reports,
        split=a.split,normal_capacity=a.normal_capacity,freeze=str(a.freeze) if a.freeze else None))


def analyze(a):
    from nanovllm.metrics import percentile
    index=read(a.inputs);trials={};rpts={}
    for row in index['reports']:
        rpt=read(row['path']);rpts[row['round'],row['variant']]=rpt
        for filename in rpt['trials']:trials[row['round'],row['variant'],Path(filename).stem]=read(Path(row['path']).parent/filename)
    variants=sorted({v for _,v,_ in trials});rounds=sorted({r for r,_,_ in trials});cases=sorted({c for _,_,c in trials})
    pairs=[('v2_legacy','v1_legacy'),('v2_legacy','legacy_async'),('v2_batch','v1_batch'),('idle_fastpath','v1_batch'),('idle_fastpath','legacy_async')]
    comparisons=[]
    for candidate,reference in pairs:
        if candidate not in variants or reference not in variants:continue
        for case in ['pooled',*cases]:
            chosen=cases if case=='pooled' else [case]
            paired=[sum(trials[r,reference,c]['seconds'] for c in chosen)/sum(trials[r,candidate,c]['seconds'] for c in chosen) for r in rounds]
            left=[trials[r,reference,c] for r in rounds for c in chosen];right=[trials[r,candidate,c] for r in rounds for c in chosen]
            exact=all([q['output_token_ids'] for q in x['requests']]==[q['output_token_ids'] for q in y['requests']] for x,y in zip(left,right))
            def p95(rows,key):
                values=[q[key] for t in rows for q in t['requests']] if key=='ttft_ms' else [g for t in rows for q in t['requests'] for g in q['event_gap_ms']]
                return percentile(values,95)
            ttft=p95(right,'ttft_ms')/p95(left,'ttft_ms');gap=p95(right,'gap')/p95(left,'gap')
            passed=len(rounds)>=5 and statistics.median(paired)>=1.03 and all(x>1 for x in paired) and ttft<=1.10 and gap<=1.10 and exact
            comparisons.append(dict(candidate=candidate,reference=reference,case=case,
                paired_speed_ratios=paired,median_speed_ratio=statistics.median(paired),all_rounds_faster=all(x>1 for x in paired),
                ttft_p95_ratio=ttft,event_gap_p95_ratio=gap,greedy_exact=exact,passes_relative_gates=passed))
    summaries=[]
    for variant in variants:
        rows=[t for (r,v,c),t in trials.items() if v==variant]
        def total(name):return sum(t['cache_after_drain'].get(name,0) for t in rows)
        def policy(name):return sum(t['cache_after_drain'].get('policy',{}).get(name,0) for t in rows)
        summaries.append(dict(variant=variant,stores=total('stores'),loads=total('loads'),d2h_bytes=total('d2h_bytes'),h2d_bytes=total('h2d_bytes'),
            capacity_rejects=policy('capacity_rejects'),capacity_replacements=policy('capacity_replacements'),
            admission_checks=policy('admission_checks'),admission_accepts=policy('admission_accepts'),
            restore_checks=policy('restore_checks'),restore_accepts=policy('restore_accepts'),
            slo_requests=sum(t['slo']['slo_met_requests'] for t in rows),total_requests=sum(len(t['requests']) for t in rows),
            startup_median_s=statistics.median(rpts[r,variant]['startup_seconds'] for r in rounds),
            peak_allocated_bytes=max(t['memory']['max_memory_allocated'] for t in rows),
            peak_reserved_bytes=max(t['memory']['max_memory_reserved'] for t in rows),
            physical_kv_blocks=[rpts[r,variant]['physical_kv_blocks'] for r in rounds]))
    status='REVIEW_REQUIRED' if any(not c['greedy_exact'] for c in comparisons) else 'ANALYZED'
    write(a.output,dict(status=status,rounds=rounds,comparisons=comparisons,summaries=summaries,
        index_sha256=sha(a.inputs),note='Same-backend paired equality; existing unrelated shape counterexample remains REVIEW. No parameter tuning from final.'))
    print(json.dumps(dict(status=status,comparisons=[c for c in comparisons if c['case']=='pooled'],summaries=summaries),indent=2))


def freeze(a):
    source_maps={}
    for branch in ('baseline','candidate','candidate-fastpath'):
        source_maps[branch]={str(p.relative_to(a.root/branch)):sha(p) for p in sorted((a.root/branch/'nanovllm').rglob('*.py'))}
    data=read(a.inputs)
    for name,identity in data['model_files'].items():assert sha(Path(data['model'])/name)==identity['sha256']
    write(a.output,dict(status='FROZEN_BEFORE_FINAL',sources=source_maps,inputs_sha256=sha(a.inputs),
        benchmark_sha256=sha(__file__),calibrations={str(p.relative_to(a.root)):sha(p) for d in ('calibration-v1','calibration-v2') for p in sorted((a.root/d).glob('*.json'))},
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],protocol_sha256=sha(a.root/'protocol.json'),created=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    for mode in ('prepare','fit','worker','suite','analyze','freeze'):
        q=s.add_parser(mode);q.add_argument('--root',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        if mode=='prepare':q.add_argument('--model',required=True)
        else:q.add_argument('--inputs',type=Path,required=True)
        if mode in ('worker','suite'):
            q.add_argument('--freeze',type=Path);q.add_argument('--split',choices=('dev','final'),default='dev')
            q.add_argument('--cases',nargs='+');q.add_argument('--normal-capacity',action='store_true')
        if mode=='worker':q.add_argument('--variant',choices=VARIANTS,required=True);q.add_argument('--round',type=int,default=0)
        if mode=='suite':q.add_argument('--variants',choices=VARIANTS,nargs='+',required=True);q.add_argument('--rounds',type=int,default=1)
    a=p.parse_args();globals()[a.action](a)


if __name__=='__main__':main()
