"""Guarded TP1 GPU bridge protocol: same-checkpoint numerical/lifecycle checks."""
import argparse,hashlib,json,pathlib,sys,traceback,gc
ROOT=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['legacy','device'],required=True);p.add_argument('--compare');p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--source',type=pathlib.Path,default=ROOT);p.add_argument('--model');p.add_argument('--draft');p.add_argument('--packed-linear',action='store_true');a=p.parse_args();C=a.source.resolve();sys.path.insert(0,str(C))
from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
report=dict(status='RUNNING',variant=a.variant,benchmark_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),source_sha256={str(f.relative_to(C)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (C/'nanovllm').rglob('*.py')},rounds=[],requests=[],numerics=[],faults=[])
assert not a.output.exists()
engine=None
def guard():return ensure_gpu1_available(evidence_path=a.output.with_suffix('.guard.jsonl'))
try:
    report['admission']=guard()
    import torch
    from nanovllm import LLM,SamplingParams
    from nanovllm.utils.context import get_context
    from tests.regression.check_draft_model import common_history_logits,compare_logits,same_path_kv_check
    common_history_logits=torch.inference_mode()(common_history_logits)
    same_path_kv_check=torch.inference_mode()(same_path_kv_check)
    config=json.loads((C/'configs/fp16_draft.json').read_text())['config'];model=config.pop('model')
    if a.model:model=a.model
    if a.draft:config['draft_model']=a.draft
    config.update(draft_prefill_graph=True,draft_prefill_tail_graph=True,draft_verify_metadata=True,
        draft_async_metadata=True,draft_device_verification=a.variant=='device')
    if a.packed_linear:config['stable_linear_packed']=True
    report['config']=config
    def empty(e):
        m=e.scheduler.block_manager;o=e.model_runner.draft_runner
        assert not e.scheduler._active_items and not e.scheduler._reservations and not m._reservations
        assert not m.used_block_ids and sum(b.ref_count for b in m.blocks)==0
        assert not o.proposer.cached_tokens and o._pending_device_proposal is None
        assert get_context().slot_mapping is None
    def make():
        guard();e=LLM(model,**config);return e
    def kv_hash(kv,slots):
        ids=torch.tensor(slots,device=kv.device,dtype=torch.int64)
        bits=kv.flatten(2,3).index_select(2,ids).contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(bits).hexdigest()
    engine=make();runner=engine.model_runner;owner=runner.draft_runner
    report['effective']=engine.config.runtime_summary()
    # Same fixed-history numeric thresholds; snapshots never enter timed tests.
    for prefix,q in [(1,2),(255,4),(256,4),(257,8),(507,4)]:
        guard();history=[7+(i*31)%1000 for i in range(prefix+8)]
        x=common_history_logits(engine,history,prefix,1);y=common_history_logits(engine,history,prefix,q)
        row=dict(prefix=prefix,Q=q,**compare_logits(x,y));assert row['strict_allclose'],row
        row['reference_sha256']=hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        row['query_sha256']=hashlib.sha256(y.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        report['numerics'].append(row)
    check,snapshots=same_path_kv_check(engine,[7+(i*31)%1000 for i in range(264)],256)
    assert check['kv_bitwise_equal'] and check['logits']['strict_allclose'];report['same_path_kv']=check
    del snapshots
    current={};captured=set();orig=runner.run_device_verification if a.variant=='device' else runner.run_plan
    def record(case,history,proposed,predictions,end,pages):
        if current.get('skip_record',False):return
        row=dict(case=case,history=list(history),draft=list(proposed),predictions=list(predictions),
            cached=list(owner.proposer.cached_tokens))
        if True:  # 每轮检查，包括拒绝/全接受后的有效前缀，不过滤慢样本。
            block=engine.config.kvcache_block_size;slots=[pages[i//block]*block+i%block for i in range(end)]
            row['target_valid_kv_sha256']=kv_hash(runner.kv_cache,slots)
            row['draft_valid_kv_sha256']=kv_hash(owner.kv,range(len(owner.proposer.cached_tokens)))
            captured.add(case)
            logits=runner.native_verification_graphs.entries[len(proposed)+1][-1]
            row['target_logits_sha256']=hashlib.sha256(logits.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        report['rounds'].append(row)
    if a.variant=='device':
        def wrapped(plan,seqs):
            result=orig(plan,seqs);item=plan.items[0]
            record(current['case'],seqs[0].token_ids,result.draft,result.predictions,item.end_pos,item.block_table)
            return result
        runner.run_device_verification=wrapped
    else:
        def wrapped(plan,seqs,*args):
            result=orig(plan,seqs,*args)
            if plan.has_verification:
                item=plan.items[0];record(current['case'],seqs[0].token_ids,item.draft,result[item.request_id],item.end_pos,item.block_table)
            return result
        runner.run_plan=wrapped
    for k in [1,2,4,8]:
        for n in [32,255,256,257,768]:
            guard();case=f'K{k}_N{n}';current.update(case=case,n=n,K=k)
            history=[2+i%97 for i in range(n)]
            sp=SamplingParams(temperature=0,ignore_eos=True,max_tokens=16,speculative_method='draft_model',speculative_max_draft_tokens=k)
            result=engine.generate([history],sp,use_tqdm=False)[0]['token_ids']
            assert len(result)==16;empty(engine)
            report['requests'].append(dict(case=case,input_ids=history,tokens=result))
    # Same input, early stop inside a multi-token verification commit.
    anchor=report['requests'][0];history=anchor['input_ids'];known=anchor['tokens']
    eos_index=next((i for i,t in enumerate(known) if i>0 and t not in known[:i]),None)
    if eos_index is not None:
        original_eos=engine.scheduler.eos;engine.scheduler.eos=known[eos_index]
        current.update(case='eos',n=32,K=4)
        result=engine.generate([history],SamplingParams(temperature=0,max_tokens=16,speculative_method='draft_model',speculative_max_draft_tokens=4),use_tqdm=False)[0]
        assert result['token_ids']==known[:eos_index+1]
        report['requests'].append(dict(case='eos',tokens=result['token_ids']));engine.scheduler.eos=original_eos;empty(engine)
    stop=engine.tokenizer.decode(known[:4]);assert stop
    stop_end=next(i for i in range(1,len(known)+1) if stop in engine.tokenizer.decode(known[:i]))
    current.update(case='stop',n=32,K=4)
    result=engine.generate([history],SamplingParams(temperature=0,ignore_eos=True,max_tokens=16,stop=stop,speculative_method='draft_model',speculative_max_draft_tokens=4),use_tqdm=False)[0]
    assert result['token_ids']==known[:stop_end];report['requests'].append(dict(case='stop',tokens=result['token_ids']));empty(engine)
    # Cancel only at an actual step boundary; next request must remain isolated.
    current.update(case='cancel',n=32,K=4)
    seq=engine.add_request(history,SamplingParams(temperature=0,ignore_eos=True,max_tokens=16,speculative_method='draft_model',speculative_max_draft_tokens=4))
    engine.step();engine.step();assert engine.cancel_request(seq.seq_id);empty(engine)
    current['skip_record']=True
    again=engine.generate([history],SamplingParams(temperature=0,ignore_eos=True,max_tokens=16,speculative_method='draft_model',speculative_max_draft_tokens=4),use_tqdm=False)[0]['token_ids']
    assert again==known;empty(engine);current['skip_record']=False
    report['after_cancel_request_isolation_exact']=True
    if a.variant=='device':
        current.update(case='planner_fault',n=32,K=4)
        seq=engine.add_request(history,SamplingParams(temperature=0,ignore_eos=True,max_tokens=16,speculative_method='draft_model',speculative_max_draft_tokens=4));engine.step();before=seq.token_ids.copy()
        upgrade=engine.scheduler.upgrade_device_verification
        def fail_upgrade(*args):upgrade(*args);raise RuntimeError('injected upgraded planner failure')
        engine.scheduler.upgrade_device_verification=fail_upgrade
        try:
            try:engine.step()
            except RuntimeError as exc:assert str(exc)=='injected upgraded planner failure'
            else:raise AssertionError('planner fault swallowed')
        finally:engine.scheduler.upgrade_device_verification=upgrade
        assert seq.token_ids==before and not runner._failed
        assert engine.cancel_request(seq.seq_id);empty(engine);report['faults'].append('planner_rollback_pass')
        current['skip_record']=True
        again=engine.generate([history],SamplingParams(temperature=0,ignore_eos=True,max_tokens=16,speculative_method='draft_model',speculative_max_draft_tokens=4),use_tqdm=False)[0]['token_ids']
        assert again==known;empty(engine);current['skip_record']=False
        report['after_planner_failure_isolation_exact']=True
    engine.exit();engine=None;runner=None;owner=None;orig=None;wrapped=None;obj=None;original=None;gc.collect();torch.cuda.empty_cache()
    if a.variant=='device':
        for stage in ['draft','target']:
            engine=make();runner=engine.model_runner;owner=runner.draft_runner
            seq=engine.add_request(history,SamplingParams(temperature=0,ignore_eos=True,max_tokens=16,speculative_method='draft_model',speculative_max_draft_tokens=4));engine.step();before=seq.token_ids.copy()
            obj=owner if stage=='draft' else runner.native_verification_graphs
            method='_rollout' if stage=='draft' else 'replay_prepared';original=getattr(obj,method)
            def fail(*args,**kwargs):original(*args,**kwargs);raise RuntimeError('injected '+stage+' after GPU enqueue')
            setattr(obj,method,fail)
            try:
                try:engine.step()
                except RuntimeError as exc:assert str(exc)=='injected '+stage+' after GPU enqueue',str(exc)
                else:raise AssertionError(stage+' fault swallowed')
            finally:setattr(obj,method,original)
            assert seq.token_ids==before and runner._failed
            assert not owner.proposer.cached_tokens and owner._pending_device_proposal is None
            assert engine.cancel_request(seq.seq_id);empty(engine)
            report['faults'].append(stage+'_fence_and_rollback_pass');engine.exit();engine=None;runner=None;owner=None;orig=None;wrapped=None;obj=None;original=None;gc.collect();torch.cuda.empty_cache()
    if a.compare:
        old=json.loads(pathlib.Path(a.compare).read_text());assert old['status']=='PASS_GPU_CONTRACTS' and old['variant']=='legacy'
        assert old['source_sha256']==report['source_sha256']
        assert old['benchmark_sha256']==report['benchmark_sha256']
        assert old['numerics']==report['numerics'];assert old['same_path_kv']==report['same_path_kv']
        assert old['requests']==report['requests'];assert old['rounds']==report['rounds']
        report['cross_process_tokens_logit_hashes_target_and_draft_kv_exact']=True
    report['status']='PASS_GPU_CONTRACTS'
except BaseException:report.update(status='FAIL',traceback=traceback.format_exc());raise
finally:
    if engine is not None:engine.exit()
    with a.output.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k in ['status','variant','faults','cross_process_tokens_logit_hashes_target_and_draft_kv_exact','traceback']},indent=2))
