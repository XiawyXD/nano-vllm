"""Independent integration regression; only this benchmark is copied into main.

CPU mode uses the existing real planner/fake physical-KV fixture. GPU mode must
have an explicit exclusive lease for the selected GPU, runs one engine/config per process, and
requires the existing guard before CUDA initialization. No performance claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import MethodType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def require(value,message):
    if not value:
        raise AssertionError(message)


def reject(fn):
    try:
        fn()
    except (ValueError,RuntimeError,TypeError):
        return
    raise AssertionError('invalid input was accepted')


def guarded_snapshot(report,stage):
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    if not os.environ.get('NANOVLLM_BENCH_GPU_INDEX'):
        raise ValueError('explicit approved physical GPU index required')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('explicit approved complete GPU UUID required')
    # The existing guard independently resolves physical index -> full UUID and
    # rejects every foreign process. No ordinal fallback or occupancy bypass.
    snapshot=ensure_gpu1_available()
    selected=dict(index=snapshot['gpu']['index'],uuid=snapshot['gpu']['uuid'])
    if report.get('selected_gpu',selected)!=selected:
        raise ValueError('selected GPU changed during test')
    report['selected_gpu']=selected
    report['guards'].append(dict(stage=stage,**snapshot))


def cpu_guard_contract():
    """Synthetic guard inputs only; never query or initialize an actual GPU."""
    from unittest.mock import patch
    from benchmarks import runtime_v2_gpu_guard as guard
    uuid='GPU-00000000-0000-0000-0000-000000000001'
    gpu=dict(index='1',uuid=uuid,**{'memory.used':'0','utilization.gpu':'0'})
    foreign=[dict(gpu_uuid=uuid,pid='999999999',used_memory='3872')]
    def query(fields,kind):
        return [gpu] if kind=='gpu' else []
    with patch.object(guard,'_query',side_effect=query), patch.dict(os.environ,
            {'NANOVLLM_BENCH_GPU_INDEX':'1','CUDA_VISIBLE_DEVICES':uuid}):
        evidence={'guards':[]}
        guarded_snapshot(evidence,'synthetic_valid')
        require(evidence['selected_gpu']==dict(index='1',uuid=uuid),'guard selection not recorded')
        for changes in ({'CUDA_VISIBLE_DEVICES':'1'}, {'CUDA_VISIBLE_DEVICES':'GPU-wrong'},
                        {'NANOVLLM_BENCH_GPU_INDEX':'2'}, {'NANOVLLM_BENCH_GPU_INDEX':''}):
            with patch.dict(os.environ,changes):
                reject(lambda: guarded_snapshot({'guards':[]},'synthetic_invalid'))
        with patch.object(guard,'_query',side_effect=lambda fields,kind: [gpu] if kind=='gpu' else foreign):
            reject(lambda: guarded_snapshot({'guards':[]},'synthetic_foreign'))
        reject(lambda: guarded_snapshot({'guards':[],'selected_gpu':dict(index='2',uuid='GPU-other')},'synthetic_changed'))
    return dict(valid_selected_pair=1,rejected_cases=6,scope='synthetic CPU mocks; not live GPU evidence')


def params(**kwargs):
    from nanovllm.sampling_params import SamplingParams
    defaults=dict(temperature=.8,max_tokens=24,ignore_eos=True,seed=123,
                  speculative_method='draft_model',speculative_sampling='rejection',
                  speculative_max_draft_tokens=4)
    defaults.update(kwargs)
    return SamplingParams(**defaults)


def config_kwargs(args):
    return dict(draft_model=args.draft,enable_stochastic_speculation=True,
        numerical_backend='consistent',batching_mode='mixed',speculative_policy='fixed',
        max_num_seqs=1,max_model_len=512,max_num_batched_tokens=256,prefill_chunk_size=256,
        kvcache_block_size=256,max_kvcache_blocks=4,gpu_memory_utilization=.7,
        enforce_eager=False,draft_execution=args.draft_execution,draft_verify_graph=True,
        draft_graph_max_tokens=8,metadata_backend='reuse',enable_mixed_graph=False)


def option_checks(args):
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.config import Config
    require(SamplingParams().speculative_sampling=='greedy','request default changed')
    for change in ({'temperature':0},{'temperature':1e-10},{'top_k':1},{'top_p':.9},
                   {'repetition_penalty':1.1},{'speculative_max_draft_tokens':9},
                   {'speculative_method':'off'},{'speculative_sampling':'unknown'}):
        reject(lambda: params(**change))
    config=Config(args.model,**config_kwargs(args))
    require(config.enable_stochastic_speculation and config.linear_backend=='triton_stable'
            and config.rmsnorm_backend=='triton','opt-in effective backend wrong')
    for change in ({'numerical_backend':'default'},{'draft_model':None},{'tensor_parallel_size':2},
                   {'max_num_seqs':2},{'kv_cache_dtype':'fp8'},{'enable_stochastic_speculation':1}):
        options=config_kwargs(args)|change
        reject(lambda: Config(args.model,**options))
    ordinary=Config(args.model,max_model_len=512,max_num_batched_tokens=256,max_num_seqs=1)
    require(not ordinary.enable_stochastic_speculation,'Config default changed')
    return dict(status='PASS',invalid_sampling_cases=8,invalid_config_cases=6,
                default_config_enabled=False,default_request_sampling='greedy')


def resource_state(engine):
    manager=engine.scheduler.block_manager
    owner=getattr(engine.model_runner,'draft_runner',None)
    wrapper=getattr(owner,'stochastic',None)
    return dict(used_blocks=len(manager.used_block_ids),refs=sum(b.ref_count for b in manager.blocks),
        manager_reservations=len(manager._reservations),planner_reservations=len(engine.scheduler._reservations),
        planner_active=len(engine.scheduler._active_items),
        draft_cached=0 if owner is None else len(owner.proposer.cached_tokens),
        rng_active=False if wrapper is None else wrapper.generators is not None)


def require_empty(engine):
    state=resource_state(engine)
    require(not any(state.values()),'resource leak: '+str(state))
    return state


def planner_failure_case(engine,owner,seq,*,after_reservation):
    """The next step must finish a real probability rollout, then fail planning.

    CPU uses the real StochasticDraft with a small fake logits model; GPU uses
    the loaded real model. The same cleanup assertions apply to both modes.
    """
    from nanovllm.utils.context import get_context
    wrapper=owner.stochastic
    schedule=engine.scheduler.schedule_plan
    tokens_before=list(seq.token_ids)
    table_before=list(seq.block_table)
    cached_before=seq.num_cached_tokens
    transfers_before=wrapper.counters['candidate_host_transfers']
    observed={}
    marker='injected planning failure after probability proposal'
    def fail_schedule(*args,**kwargs):
        observed['proposal_completed']=(wrapper.counters['candidate_host_transfers']==transfers_before+1)
        observed['rng_active']=wrapper.generators is not None
        observed['draft_cached']=len(owner.proposer.cached_tokens)
        require(observed['proposal_completed'] and observed['rng_active'] and observed['draft_cached']>0,
                'planning injection did not follow a complete probability proposal')
        if after_reservation:
            plan=schedule(*args,**kwargs)
            observed['verification_reserved']=plan.has_verification
            observed['reservations']=len(engine.scheduler._reservations)
            require(plan.has_verification and observed['reservations']>0,'planning fixture never reserved verification')
        raise RuntimeError(marker)
    engine.scheduler.schedule_plan=fail_schedule
    try:
        try:
            engine.step()
        except RuntimeError as error:
            require(str(error)==marker,'unexpected planning failure: '+str(error))
        else:
            raise AssertionError('injected planning failure swallowed')
    finally:
        engine.scheduler.schedule_plan=schedule
    require(seq.token_ids==tokens_before,'planning failure changed committed tokens')
    require(seq.block_table==table_before and seq.num_cached_tokens==cached_before,
            'planning failure did not restore target reservation state')
    require(not engine.scheduler._reservations and not engine.scheduler._active_items
            and not engine.scheduler.block_manager._reservations,'planning failure retained reservations')
    require(wrapper.generators is None and wrapper.request_id is None and owner.request_id is None
            and not owner.proposer.cached_tokens,'planning failure retained draft state/RNG')
    require(get_context().slot_mapping is None,'planning failure leaked Context')
    require(not engine.model_runner._failed,'CPU planning failure poisoned model runner')
    require(engine.cancel_request(seq.seq_id),'planning-failed request did not cancel')
    result=dict(stage='after_reservation' if after_reservation else 'before_reservation',
                observed=observed,resources=require_empty(engine))
    engine.forget_request(seq.seq_id)
    return result


def cpu_checks(args,report):
    # This fixture explicitly sets CUDA_VISIBLE_DEVICES='' on import. Never
    # import it in GPU mode, even after the guard.
    from tests.regression.check_cache_draft import draft_engine
    from nanovllm.engine.packed_runner import StochasticVerification
    from nanovllm.engine.execution_plan import ExecutionRole
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.stochastic_draft import StochasticDraft
    import torch
    require(not torch.cuda.is_initialized(),'CPU fixture initialized CUDA')
    report['cpu_guard_contract']=cpu_guard_contract()
    report['options']=option_checks(args)
    def fixture(accepted=0,bad=None):
        engine,draft=draft_engine()
        engine.config.enable_stochastic_speculation=True
        runner=engine.model_runner
        runner.draft_runner=draft
        original=runner.call
        runner.draft_propose_probabilities=lambda rid,tokens,limit,temp,seed: SimpleNamespace(
            token_ids=draft.propose(rid,tokens,limit))
        def call(method,plan,sequences,proposals=None):
            output=original(method,plan,sequences)
            for item in plan.items:
                if item.role is ExecutionRole.VERIFY:
                    count=min(accepted,len(item.draft))
                    value=StochasticVerification(item.draft[:count]+(9,),count)
                    if bad=='list': value=list(value.tokens)
                    if bad=='bool_count': value=StochasticVerification((9,),False)
                    if bad=='length': value=StochasticVerification((9,9),0)
                    if bad=='prefix': value=StochasticVerification((99,9),1)
                    if bad=='negative': value=StochasticVerification((-1,),0)
                    if bad=='bool_token': value=StochasticVerification((True,),0)
                    if bad=='float_token': value=StochasticVerification((1.5,),0)
                    if bad=='none_token': value=StochasticVerification((None,),0)
                    if bad=='out_of_vocab': value=StochasticVerification((engine.config.hf_config.vocab_size,),0)
                    output[item.request_id]=value
            return output
        runner.run_plan=lambda plan,sequences,proposals=None: call('run_plan',plan,sequences,proposals)
        runner._failed=False
        runner.world_size=1
        runner.rank=0
        # Exercise the actual fail-closed dispatch as well as real engine/planner.
        runner.call=MethodType(ModelRunner.call,runner)
        return engine,draft
    rows=report['cpu_cases']=[]
    report['planner_exception_cases']=[]
    for after_reservation in (False,True):
        engine,owner=fixture()
        owner.device=torch.device('cpu')
        owner.capacity=256
        owner.chunk_size=8
        owner.workspace_bytes=0
        owner.block_table=torch.zeros((1,128),dtype=torch.int32)
        vocab=engine.config.hf_config.vocab_size
        class CPUModel:
            def __init__(self):
                self.lm_head=SimpleNamespace(weight=torch.zeros((vocab,1)))
            def __call__(self,ids,positions):
                logits=torch.zeros((ids.numel(),vocab))
                logits.scatter_(1,((ids+1)%vocab).reshape(-1,1),2.)
                return logits
            def compute_logits(self,hidden):
                return hidden[-1:]
        owner.model=CPUModel()
        wrapper=owner.stochastic=StochasticDraft(owner,use_graph=False)
        engine.model_runner.draft_propose_probabilities=lambda rid,tokens,limit,temp,seed: wrapper.propose(tokens,limit,temp,seed,rid)
        engine.model_runner.draft_forget=lambda rid: wrapper.clear() if rid==owner.request_id else None
        try:
            seq=engine.add_request([1,2,3],params())
            engine.step()
            report['planner_exception_cases'].append(planner_failure_case(engine,owner,seq,after_reservation=after_reservation))
        finally:
            wrapper.close()
    for accepted in (0,1,2,4):
        for stopping in ('length','eos','stop'):
            engine,draft=fixture(accepted)
            if stopping=='eos': engine.scheduler.eos=9
            seq=engine.add_request([1,2,3],params(ignore_eos=stopping!='eos',stop='9' if stopping=='stop' else None))
            seen_verify=False
            for _ in range(100):
                if seq.is_finished: break
                engine.step()
                seen_verify |= engine.model_runner.plans[-1].has_verification
                if not seq.is_finished:
                    require(seq.num_cached_tokens==len(seq)-1,'target rollback cached length wrong')
                    require(draft.proposer.cached_tokens==seq.token_ids[:len(draft.proposer.cached_tokens)],
                            'draft rollback retained rejected token')
            require(seq.is_finished and seen_verify,'typed verification case did not finish')
            require(seq.finish_reason==('length' if stopping=='length' else 'stop'),'wrong finish reason')
            require(not draft.proposer.cached_tokens and draft.request_id is None,'fake owner not forgotten')
            if stopping=='stop': require('9' not in engine.completion_text(seq),'stop text leaked')
            if stopping=='length': require(seq.num_completion_tokens==24,'max_tokens not exact')
            rows.append(dict(accepted=accepted,stop=stopping,resources=require_empty(engine)))
    invalid_findings=report['invalid_token_findings']=[]
    for bad in ('list','bool_count','length','prefix','negative','bool_token','float_token','none_token','out_of_vocab','target_failure'):
        engine,draft=fixture(0,bad)
        seq=engine.add_request([1,2,3],params())
        engine.step()
        before=list(seq.token_ids)
        if bad=='target_failure': engine.model_runner.fail_after_write=True
        if bad=='out_of_vocab':
            try:
                engine.step()
            except (ValueError,RuntimeError,TypeError):
                pass
            else:
                invalid_findings.append(dict(case=bad,token=engine.config.hf_config.vocab_size,
                    committed=list(seq.token_ids[len(before):]),issue='typed correction accepts token >= vocabulary size'))
                engine.cancel_request(seq.seq_id)
                rows.append(dict(invalid=bad,rejected=False,resources=require_empty(engine)))
                continue
        else:
            reject(engine.step)
        require(seq.token_ids==before,'failed verification partially committed sequence')
        require(not engine.scheduler._reservations and not engine.scheduler.block_manager._reservations,
                'failed verification retained reservations')
        require(not draft.proposer.cached_tokens and draft.request_id is None,'failed verification retained draft')
        if bad=='target_failure':
            require(engine.model_runner._failed,'failed model runner was reusable')
            reject(lambda: engine.model_runner.call('run_plan',None,None))
        engine.cancel_request(seq.seq_id)
        rows.append(dict(invalid=bad,resources=require_empty(engine)))
    engine,draft=fixture(2)
    seq=engine.add_request([1,2,3],params())
    engine.step()
    engine.step()
    require(bool(draft.proposer.cached_tokens),'cancel fixture never activated draft')
    require(engine.cancel_request(seq.seq_id),'active cancellation failed')
    require(not draft.proposer.cached_tokens and draft.request_id is None,'cancellation retained draft owner')
    rows.append(dict(cancel='active_after_verify',resources=require_empty(engine)))
    engine,draft=fixture()
    engine.config.enable_stochastic_speculation=False
    reject(lambda: engine.add_request([1,2,3],params()))
    require(engine.scheduler.is_finished(),'disabled request entered scheduler')
    report['cpu_cases']=rows
    report['cpu_scope']='real planner/typed result/physical fake KV; no model or probability distribution claim'
    require(not torch.cuda.is_initialized(),'CPU tests initialized CUDA')
    require(not invalid_findings,'out-of-vocabulary typed correction was not rejected; see invalid_token_findings')


def fresh_q_check(torch,owner,history,proposal,temperature):
    """Cold first proposal, identical prefill partition; teacher-force actual IDs.

    Uses a fresh-zero physical cache in the same allocation, then restores every
    KV value. No sampling or wrapper logits helper is used in the reference.
    No word-vocabulary D2H; only error/equality scalars are reported.
    """
    from nanovllm.utils.context import set_context,reset_context
    from nanovllm.engine.speculative_sampling import temperature_probabilities
    saved=owner.kv.clone()
    rows=[]
    def forward(tokens,start):
        ids=torch.tensor(tokens,device=owner.device,dtype=torch.int64)
        positions=torch.arange(start,start+len(tokens),device=owner.device)
        q=torch.tensor([0,len(tokens)],device=owner.device,dtype=torch.int32)
        k=torch.tensor([0,start+len(tokens)],device=owner.device,dtype=torch.int32)
        try:
            set_context(True,q,k,len(tokens),start+len(tokens),positions.int(),block_tables=owner.block_table)
            return owner.model.compute_logits(owner.model(ids,positions))[-1]
        finally:
            reset_context()
    try:
        owner.kv.zero_()
        for offset in range(0,len(history),owner.chunk_size):
            logits=forward(history[offset:offset+owner.chunk_size],offset)
        for index,token in enumerate(proposal.token_ids):
            expected=temperature_probabilities(logits,temperature)
            actual=proposal.probabilities[index]
            row=dict(row=index,exact=bool(torch.equal(actual,expected)),
                     max_abs=float((actual-expected).abs().max()),
                     sampled_token_positive_mass=bool(actual[token]>0))
            rows.append(row)
            if index+1<len(proposal.token_ids):
                logits=forward([token],len(history)+index)
    finally:
        owner.kv.copy_(saved)
        reset_context()
    require(all(row['exact'] and row['sampled_token_positive_mass'] for row in rows),'actual q differs from teacher-history reference')
    return dict(history=len(history),K=len(rows),rows=rows)


def gpu_checks(args,report,guard):
    import torch
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.engine.packed_runner import StochasticVerification
    from nanovllm.utils.context import get_context
    report['options']=option_checks(args)
    guard('before_engine')
    engine=None
    try:
        engine=LLMEngine(args.model,**config_kwargs(args))
        runner=engine.model_runner
        owner=runner.draft_runner
        require(owner.stochastic is not None,'enabled wrapper missing')
        require(bool(owner.stochastic.graphs)==(args.draft_execution=='graph'),'wrong probability graph activation')
        require(len(engine.scheduler.block_manager.blocks)==4,'physical KV count not four')
        report['effective']=engine.config.runtime_summary()
        traces=report['verification_events']=[]
        qchecks=report['actual_q_checks']=[]
        run_plan=runner.run_plan
        probability=runner.draft_propose_probabilities
        inject=False
        def traced(plan,sequences,proposals=None):
            result=run_plan(plan,sequences,proposals)
            if plan.has_verification:
                traces.append(dict(path=runner.last_plan_stats.get('path'),
                    items=[dict(K=len(item.draft),start=item.start_pos,
                        accepted=result[item.request_id].accepted if isinstance(result[item.request_id],StochasticVerification) else None)
                        for item in plan.items]))
                if inject:
                    raise RuntimeError('injected after real target verification KV write')
            return result
        def checked_probability(request_id,history,limit,temperature,seed):
            cold=owner.request_id!=request_id
            result=probability(request_id,history,limit,temperature,seed)
            if cold and len(qchecks)<8:
                with torch.inference_mode():
                    qchecks.append(fresh_q_check(torch,owner,list(history),result,temperature))
            return result
        runner.run_plan=traced
        runner.draft_propose_probabilities=checked_probability
        rows=report['requests']=[]
        def generate(name,prompt,sampling):
            guard(name)
            seq=engine.add_request(prompt,sampling)
            start=len(traces)
            for _ in range(600):
                if seq.is_finished: break
                engine.step()
                require(not engine.scheduler._reservations,'step retained planner reservation')
                if not seq.is_finished:
                    require(seq.num_cached_tokens<=len(seq)-1,'target cached beyond committed history')
                    require(owner.proposer.cached_tokens==seq.token_ids[:len(owner.proposer.cached_tokens)],
                            'draft cache diverged from committed history')
            require(seq.is_finished,'bounded request did not finish')
            rows.append(dict(name=name,tokens=list(seq.completion_token_ids),reason=seq.finish_reason,
                text=engine.completion_text(seq),verification_rounds=len(traces)-start,resources=require_empty(engine)))
            engine.forget_request(seq.seq_id)
            return rows[-1]
        prompt=engine.tokenizer.encode('The capital of France is')
        baseline=generate('greedy_plain',prompt,params(temperature=0,speculative_method='off',speculative_sampling='greedy'))
        greedy=generate('greedy_draft',prompt,params(temperature=0,speculative_sampling='greedy'))
        require(baseline['tokens']==greedy['tokens'],'greedy output changed')
        first=generate('seed_a_first',prompt,params())
        generate('seed_b_interleaved',prompt,params(seed=991))
        second=generate('seed_a_repeated',prompt,params())
        require(first['tokens']==second['tokens'],'request seed state leaked across requests')
        require(first['verification_rounds']>0,'rejection request bypassed verification')
        for length in (255,256,507):
            result=generate(f'prefix{length}',[7+(i*31)%1000 for i in range(length)],params(max_tokens=min(12,512-length)))
            require(len(result['tokens'])==min(12,512-length),'length bound wrong')
        eos_before=engine.scheduler.eos
        try:
            engine.scheduler.eos=first['tokens'][0]
            result=generate('forced_eos',prompt,params(ignore_eos=False))
            require(result['tokens']==first['tokens'][:1] and result['reason']=='stop','EOS handling failed')
        finally:
            engine.scheduler.eos=eos_before
        stop=engine.tokenizer.decode(first['tokens'][:2],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        require(bool(stop),'stop fixture unexpectedly empty')
        result=generate('stop_string',prompt,params(stop=stop))
        require(result['reason']=='stop' and stop not in result['text'],'stop handling failed')
        report['planner_exception_cases']=[]
        for after_reservation in (False,True):
            guard('planner_exception_'+str(after_reservation))
            seq=engine.add_request(prompt,params())
            engine.step()
            report['planner_exception_cases'].append(planner_failure_case(engine,owner,seq,after_reservation=after_reservation))
        recovered=generate('after_planner_exception',prompt,params())
        require(recovered['tokens']==first['tokens'],'planning exception leaked into next request')
        for name,failure in (('cancel',False),('target_exception',True)):
            guard(name)
            seq=engine.add_request(prompt,params())
            engine.step()
            if failure:
                before=list(seq.token_ids)
                inject=True
                try:
                    try: engine.step()
                    except RuntimeError as error:
                        require(str(error)=='injected after real target verification KV write','unexpected error')
                    else: raise AssertionError('target failure not raised')
                finally: inject=False
                require(seq.token_ids==before,'failed target committed tokens')
                require(owner.stochastic.generators is None and not owner.proposer.cached_tokens,'error retained request RNG/draft')
            else:
                engine.step()
            require(engine.cancel_request(seq.seq_id),'active request did not cancel')
            require(get_context().slot_mapping is None,'cancel/error context leaked')
            rows.append(dict(name=name,resources=require_empty(engine)))
            engine.forget_request(seq.seq_id)
            if not failure:
                recovered=generate('after_cancel',prompt,params())
                require(recovered['tokens']==first['tokens'],'cancellation leaked into next request')
            else:
                # Production ModelRunner.call deliberately poisons the engine
                # after any model failure. Cleanup is required; reuse is not.
                require(runner._failed,'model exception did not fail closed')
                reject(lambda: runner.call('run_plan',None,None))
                report['exception_policy']='resources released; failed runner rejects reuse; recreate engine'
        require(all(event['path']=='native_verification_graph' for event in traces),'requested target verification graph bypassed')
        report['draft_stats']=owner.stats()
        report['runtime_counts']=dict(engine._runtime_counts)
        report['scope']='bounded integration, seed checks for isolation only, not distribution validation or throughput'
    finally:
        if engine is not None:
            engine.exit()
        report['distributed_cleared']=not torch.distributed.is_initialized()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--cpu',action='store_true')
    modes.add_argument('--gpu',action='store_true')
    parser.add_argument('--model',default='/home/hrxu/huggingface/Qwen3-1.7B')
    parser.add_argument('--draft',default='/home/hrxu/huggingface/Qwen3-0.6B')
    parser.add_argument('--draft-execution',choices=('eager','graph'),default='eager')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=dict(status='RUNNING',mode='gpu' if args.gpu else 'cpu',guards=[])
    with Path(args.output).open('x',encoding='utf-8') as output:
        def guard(stage):
            guarded_snapshot(report,stage)
        try:
            if args.gpu:
                guard('before_cuda_initialization')
            else:
                require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU mode must explicitly hide all GPUs')
            report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((ROOT/'nanovllm').rglob('*.py'))}
            report['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            if args.gpu: gpu_checks(args,report,guard)
            else: cpu_checks(args,report)
            report['status']='PASS'
        except BaseException:
            report['status']='FAIL'
            report['traceback']=traceback.format_exc()
        finally:
            json.dump(report,output,indent=2)
            output.write('\n')
        print(json.dumps({k:report[k] for k in ('status','mode','options','traceback') if k in report},indent=2))
        return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
