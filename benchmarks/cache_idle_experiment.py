"""Same-source fastpath toggle plus original baselines; no change to policy freeze."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from benchmarks import cache_policy_experiment as base
from benchmarks.cache_runtime_experiment import read,write,sha,sources,guard

ARMS=('idle_control','idle_fastpath','legacy_async')

def worker(a):
    frozen=read(a.freeze)
    assert frozen['idle_benchmark_sha256']==sha(__file__)
    arm=a.arm
    a.variant='legacy_async' if arm=='legacy_async' else 'idle_fastpath'
    original=base.config
    def configure(*args,**kwargs):
        c=original(*args,**kwargs)
        if arm=='idle_control':c['host_kv_transfer_idle_fastpath']=False
        return c
    base.config=configure
    try:base.worker(a)
    finally:
        base.config=original
        if (a.output/'report.json').exists():
            write(a.output/'arm.json',dict(arm=arm,report_sha256=sha(a.output/'report.json'),
                idle_benchmark_sha256=sha(__file__),interpretation='Same frozen candidate-fastpath source for both toggle arms; original fastest legacy async remains independent.'))

def suite(a):
    a.output.mkdir(exist_ok=False,parents=True);reports=[]
    arms=['idle_control','idle_fastpath'] if a.normal_capacity else list(ARMS)
    for r in range(a.rounds):
        order=arms[r%len(arms):]+arms[:r%len(arms)]
        if r%2:order=list(reversed(order))
        for arm in order:
            guard(a.output/'guard.jsonl')
            branch='baseline' if arm=='legacy_async' else 'candidate-fastpath'
            out=a.output/f'r{r:02d}-{arm}'
            command=[sys.executable,'-B','-m','benchmarks.cache_idle_experiment','worker',
                '--root',str(a.root),'--inputs',str(a.inputs),'--freeze',str(a.freeze),
                '--split',a.split,'--round',str(r),'--arm',arm,'--output',str(out)]
            if a.normal_capacity:command+=['--normal-capacity']
            with (a.output/f'r{r:02d}-{arm}.log').open('x') as log:
                p=subprocess.run(command,cwd=a.root/branch,stdout=log,stderr=subprocess.STDOUT)
            assert p.returncode==0,f'preserve failed worker {out}'
            reports.append(dict(round=r,variant=arm,path=str(out/'report.json')))
            write(a.output/f'progress-{len(reports)}.json',dict(reports=reports))
            print(json.dumps(dict(event='WORKER_DONE',round=r,arm=arm)),flush=True)
    write(a.output/'index.json',dict(status='COMPLETE',reports=reports,split=a.split,normal_capacity=a.normal_capacity))

def analyze(a):
    from nanovllm.metrics import percentile
    index=read(a.inputs);t={};reports={}
    for row in index['reports']:
        r,v=row['round'],row['variant'];report=read(row['path']);reports[r,v]=report
        for name in report['trials']:t[r,v,Path(name).stem]=read(Path(row['path']).parent/name)
    rounds=sorted({r for r,v,c in t});cases=sorted({c for r,v,c in t});variants=sorted({v for r,v,c in t})
    comparisons=[]
    for ref in ('idle_control','legacy_async'):
        if ref not in variants:continue
        for case in ['pooled',*cases]:
            selected=cases if case=='pooled' else [case]
            ratio=[sum(t[r,ref,c]['seconds'] for c in selected)/sum(t[r,'idle_fastpath',c]['seconds'] for c in selected) for r in rounds]
            old=[t[r,ref,c] for r in rounds for c in selected];new=[t[r,'idle_fastpath',c] for r in rounds for c in selected]
            exact=all([q['output_token_ids'] for q in x['requests']]==[q['output_token_ids'] for q in y['requests']] for x,y in zip(old,new))
            def p95(rows,gaps):return percentile([g for x in rows for q in x['requests'] for g in q['event_gap_ms']] if gaps else [q['ttft_ms'] for x in rows for q in x['requests']],95)
            ttft=p95(new,False)/p95(old,False);gap=p95(new,True)/p95(old,True);speed=statistics.median(ratio)
            comparisons.append(dict(candidate='idle_fastpath',reference=ref,case=case,
                paired_speed_ratios=ratio,median_speed_ratio=speed,all_rounds_faster=all(x>1 for x in ratio),
                ttft_p95_ratio=ttft,event_gap_p95_ratio=gap,greedy_exact=exact,
                passes_relative_gates=len(rounds)>=5 and speed>=1.03 and all(x>1 for x in ratio) and ttft<=1.10 and gap<=1.10 and exact))
    summaries=[]
    for v in variants:
        rows=[x for (r,variant,c),x in t.items() if variant==v]
        summaries.append(dict(variant=v,stores=sum(x['cache_after_drain'].get('stores',0) for x in rows),
            loads=sum(x['cache_after_drain'].get('loads',0) for x in rows),
            peak_allocated_bytes=max(x['memory']['max_memory_allocated'] for x in rows),
            peak_reserved_bytes=max(x['memory']['max_memory_reserved'] for x in rows),
            startup_median_s=statistics.median(reports[r,v]['startup_seconds'] for r in rounds),
            slo_requests=sum(x['slo']['slo_met_requests'] for x in rows),total_requests=sum(len(x['requests']) for x in rows),
            physical_kv_blocks=[reports[r,v]['physical_kv_blocks'] for r in rounds]))
    result=dict(status='ANALYZED' if all(x['greedy_exact'] for x in comparisons) else 'REVIEW_REQUIRED',
        comparisons=comparisons,summaries=summaries,rounds=rounds,index_sha256=sha(a.inputs))
    write(a.output,result)
    print(json.dumps(dict(status=result['status'],comparisons=[c for c in comparisons if c['case']=='pooled'],summaries=summaries),indent=2))

def freeze(a):
    doc=read(a.inputs)
    assert sha(base.__file__)==doc['benchmark_sha256']
    for branch,mapping in doc['sources'].items():
        actual={str(p.relative_to(a.root/branch)):sha(p) for p in sorted((a.root/branch/'nanovllm').rglob('*.py'))}
        assert mapping==actual
    doc['idle_benchmark_sha256']=sha(__file__)
    doc['idle_control_note']='Adds same-source flag-off control; unchanged algorithm/data/numeric/performance gates. Frozen before idle DEV/final.'
    write(a.output,doc)

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    for mode in ('worker','suite','analyze','freeze'):
        q=s.add_parser(mode);q.add_argument('--root',type=Path,required=True);q.add_argument('--inputs',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        if mode in ('worker','suite'):
            q.add_argument('--freeze',type=Path,required=True);q.add_argument('--split',choices=('dev','final'),default='dev');q.add_argument('--normal-capacity',action='store_true');q.set_defaults(cases=None)
        if mode=='worker':q.add_argument('--arm',choices=ARMS,required=True);q.add_argument('--round',type=int,default=0)
        if mode=='suite':q.add_argument('--rounds',type=int,default=1)
    a=p.parse_args();globals()[a.action](a)

if __name__=='__main__':main()
