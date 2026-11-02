"""Guarded TP1 greedy FP16/Marlin draft lifecycle; no quality/perf claim.

Run one fresh process per eager/graph mode. Graph mode additionally compares
identical-history proposal tokens and valid KV against eager using the same
loaded model, without creating any graph after target KV budgeting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tests.regression.check_stochastic_engine import guarded_snapshot, require, require_empty


def bitwise_equal(left,right):
    """Raw bits, including signed zero; numerical equality alone is insufficient."""
    import torch
    return (left.shape==right.shape and left.dtype==right.dtype
            and bool(torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))))


def run(args,report,guard):
    import torch
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.context import get_context
    guard('before_engine')
    engine=None
    try:
        quantized=args.draft_quantization=='awq'
        engine=LLMEngine(args.model,draft_model=args.draft,draft_quantization='awq' if quantized else None,
            draft_dtype='float16',draft_quant_backend='marlin' if quantized else 'triton',marlin_library=args.marlin_library,
            numerical_backend='consistent',batching_mode='mixed',speculative_policy='fixed',
            max_num_seqs=1,max_model_len=512,max_num_batched_tokens=256,prefill_chunk_size=256,
            kvcache_block_size=256,max_kvcache_blocks=4,gpu_memory_utilization=args.gpu_memory_utilization,
            draft_execution=args.execution,draft_verify_graph=True,draft_graph_max_tokens=8,
            draft_async_metadata=args.async_metadata, draft_prefill_graph=args.prefill_graph, draft_prefill_tail_graph=args.tail_graph,
            metadata_backend='reuse',enable_mixed_graph=False,enforce_eager=False,
            enable_stochastic_speculation=False)
        runner=engine.model_runner
        owner=runner.draft_runner
        report['effective']=engine.config.runtime_summary()
        require(report['effective']['draft_async_metadata'] is args.async_metadata,
                'Config did not receive explicit draft_async_metadata')
        require(owner.async_metadata is args.async_metadata,'DraftRunner async flag differs from Config')
        require(owner.prefill_graph_enabled is args.prefill_graph,'DraftRunner prefill graph flag differs from Config')
        require(owner.prefill_tail_graph_enabled is args.tail_graph,'tail graph flag differs')
        if args.prefill_graph:
            require(bool(owner.prefill_graphs),'prefill capture entries missing')
        if args.tail_graph:
            require((1,257,True) in owner.prefill_graphs,'Q257 tail graph missing')
        require(owner.stochastic is None,'greedy test unexpectedly enabled stochastic wrapper')
        layers=[m for m in owner.model.modules() if hasattr(m,'qweight')]
        if quantized:
            require(layers and all(m.backend=='awq_marlin' for m in layers),'quant layers silently fell back')
        else:
            require(not layers,'FP16 lifecycle unexpectedly loaded packed quantized projections')
        report['marlin_layers']=len(layers)
        require(owner.kv.dtype==torch.float16,'draft KV did not use explicit FP16 dtype')
        workspace_counts=[]
        for layer in layers:
            states=[layer._marlin_state,*layer._marlin_graph_states]
            pointers=[state.workspace.data_ptr() for state in states]
            require(len(set(pointers))==len(pointers),'Marlin graph workspaces alias')
            workspace_counts.append(len(states)-1)
        report['graph_workspaces_per_layer']=workspace_counts
        if args.execution=='graph' and quantized:
            require(all(count>=2 for count in workspace_counts),'Q1/Q2 dedicated graph workspaces missing')
        if args.execution=='graph':
            graphs=[owner.graph,*owner.suffix_graphs.values()]
            report['graph_async_metadata']=[dict(enabled=g.async_metadata,pinned=g.host_state.is_pinned(),
                has_copy_event=g.metadata_copy_event is not None) for g in graphs]
            require(all(g.async_metadata is args.async_metadata for g in graphs),'token/suffix async mode differs')
            if args.async_metadata:
                require(all(g.host_state.is_pinned() and g.metadata_copy_event is not None for g in graphs),
                        'async graphs lack pinned storage or event')
        report['prefill_graph_keys']=sorted(owner.prefill_graphs)
        probes=report['proposal_probes']=[]
        @torch.inference_mode()
        def proposal_pair(history,k,name):
            before_kv=owner.kv.clone()
            before_cache=list(owner.proposer.cached_tokens)
            before_request=owner.request_id
            request=before_request if before_request is not None else -1000
            original_rollout=owner.proposer.rollout
            actual=owner.propose(request,history,k)
            actual_cache=list(owner.proposer.cached_tokens)
            length=len(actual_cache)
            actual_kv=owner.kv.flatten(2,3)[:,:,:length].clone()
            row=dict(name=name,history=len(history),K=k,tokens=actual,cached=length)
            if args.execution=='graph':
                owner.kv.copy_(before_kv)
                owner.proposer.cached_tokens[:]=before_cache
                owner.request_id=before_request
                owner.proposer.rollout=None
                try:
                    reference=owner.propose(request,history,k)
                finally:
                    owner.proposer.rollout=original_rollout
                compared=owner.kv.flatten(2,3)[:,:,:length]
                row.update(tokens_equal=actual==reference,kv_exact=bitwise_equal(actual_kv,compared),
                    kv_max_abs=float((actual_kv.float()-compared.float()).abs().max()),
                    cache_equal=owner.proposer.cached_tokens==actual_cache)
                probes.append(row)
                require(row['tokens_equal'] and row['kv_exact'] and row['cache_equal'],
                        'Marlin same-history eager/graph mismatch: '+name)
            else:
                probes.append(row)
            require(get_context().slot_mapping is None,'proposal leaked Context')
            return actual
        with torch.inference_mode():
            for length in ((1,32,33,255,256,257,507) if (args.prefill_graph or args.extended_numerics) else (1,255,256,507)):
                guard('proposal_prefix_'+str(length))
                owner.forget(owner.request_id)
                owner.kv.zero_()
                history=[7+(i*31)%1000 for i in range(length)]
                proposal_pair(history,4,f'prefix{length}')
            for accepted in (0,2,4):
                owner.forget(owner.request_id)
                owner.kv.zero_()
                history=[7+(i*31)%1000 for i in range(255)]
                draft=owner.propose(-2000,history,4)
                correction=(draft[accepted]+1)%owner.model.lm_head.weight.shape[0] if accepted<4 else 17
                committed=history+draft[:accepted]+[correction]
                owner.synchronize(-2000,committed)
                require(owner.proposer.cached_tokens==committed[:len(owner.proposer.cached_tokens)],
                        'rollback retained rejected candidate')
                require(len(committed)-len(owner.proposer.cached_tokens)==(2 if accepted==4 else 1),
                        'wrong committed suffix after forced rollback/bonus fixture')
                proposal_pair(committed,4,f'controlled_accepted{accepted}')
            owner.forget(owner.request_id)
        require_empty(engine)
        events=report['verification_events']=[]
        original_run=runner.run_plan
        fail_target=False
        def trace(plan,sequences,proposals=None):
            result=original_run(plan,sequences,proposals)
            if plan.has_verification:
                for item in plan.items:
                    if not item.draft: continue
                    prediction=result[item.request_id]
                    accepted=0
                    while accepted<len(item.draft) and prediction[accepted]==item.draft[accepted]:
                        accepted+=1
                    events.append(dict(K=len(item.draft),accepted=accepted,path=runner.last_plan_stats['path']))
                if fail_target:
                    raise RuntimeError('injected Marlin target verification failure after KV write')
            return result
        runner.run_plan=trace
        rows=report['requests']=[]
        report['request_tokens']={}
        def sampling(**options):
            fields=dict(temperature=0,max_tokens=24,ignore_eos=True,speculative_method='draft_model',
                        speculative_sampling='greedy',speculative_max_draft_tokens=4)
            fields.update(options)
            return SamplingParams(**fields)
        def generate(name,prompt,**options):
            guard(name)
            seq=engine.add_request(prompt,sampling(**options))
            initial=len(events)
            for _ in range(600):
                if seq.is_finished: break
                engine.step()
                require(not engine.scheduler._reservations,'step retained reservations')
                if not seq.is_finished:
                    require(owner.proposer.cached_tokens==seq.token_ids[:len(owner.proposer.cached_tokens)],
                            'greedy verification retained rejected draft KV')
            require(seq.is_finished,'bounded request did not finish')
            row=dict(name=name,tokens=list(seq.completion_token_ids),text=engine.completion_text(seq),
                reason=seq.finish_reason,verification_rounds=len(events)-initial,resources=require_empty(engine))
            rows.append(row)
            report['request_tokens'][name]=row['tokens']
            engine.forget_request(seq.seq_id)
            return row
        prompt=engine.tokenizer.encode('The capital of France is')
        baseline=generate('plain_reference',prompt,speculative_method='off')
        first=generate('quant_draft',prompt)
        require(first['tokens']==baseline['tokens'],'quant draft changed target greedy tokens')
        require(first['verification_rounds']>0,'quant draft verification bypassed')
        for length in ((255,256,257,507) if args.extended_numerics else (255,256,507)):
            history=[7+(i*31)%1000 for i in range(length)]
            maximum=min(12,512-length)
            reference=generate(f'plain_prefix{length}',history,max_tokens=maximum,speculative_method='off')
            candidate=generate(f'draft_prefix{length}',history,max_tokens=maximum)
            require(candidate['tokens']==reference['tokens'] and len(candidate['tokens'])==maximum,
                    'cross-page/length target output mismatch')
        repeated=generate('request_isolation',prompt)
        require(repeated['tokens']==first['tokens'],'new request sees previous draft KV')
        old_eos=engine.scheduler.eos
        try:
            engine.scheduler.eos=first['tokens'][0]
            stopped=generate('eos',prompt,ignore_eos=False)
            require(stopped['tokens']==first['tokens'][:1] and stopped['reason']=='stop','EOS handling failed')
        finally:
            engine.scheduler.eos=old_eos
        stop=engine.tokenizer.decode(first['tokens'][:2],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        require(bool(stop),'empty stop fixture')
        stopped=generate('stop',prompt,stop=stop)
        require(stopped['reason']=='stop' and stop not in stopped['text'],'stop handling failed')
        if args.extended_numerics:
            # 保留既有 .01/.01 logits 阈值及 KV/token exact 要求；不替换原失败用例。
            from tests.regression.check_draft_model import common_history_logits, compare_logits, same_path_kv_check
            numeric=report['fixed_history']=[]
            tensors={}
            with torch.inference_mode():
                for prefix,chunk in ((1,2),(255,4),(256,4),(257,8),(507,4)):
                    guard('fixed_history_'+str(prefix))
                    history=[7+(i*31)%1000 for i in range(min(prefix+8,512))]
                    reference=common_history_logits(engine,history,prefix,1)
                    candidate=common_history_logits(engine,history,prefix,chunk)
                    row=dict(prefix=prefix,chunk=chunk,**compare_logits(reference,candidate))
                    numeric.append(row)
                    tensors[str(prefix)]=dict(history=history,reference=reference,candidate=candidate)
                    # 即使下一条断言失败也保留原始 logits，不能只留下汇总误差。
                    raw_path=Path(args.output).with_name(Path(args.output).stem+f'_prefix{prefix}.pt')
                    with raw_path.open('xb') as raw:
                        torch.save(tensors[str(prefix)],raw)
                    require(row['strict_allclose'],'fixed-history logits exceed existing .01/.01 threshold')
                history=[7+(i*31)%1000 for i in range(264)]
                report['target_same_path_kv'],snapshots=same_path_kv_check(engine,history,256)
                require(report['target_same_path_kv']['kv_bitwise_equal'],'same-path target KV differs')
                require(len(snapshots)==2 and bitwise_equal(*snapshots),'same-path target KV raw bits differ')
                report['target_same_path_kv']['raw_bits_equal']=True
                tensors['same_path_kv']=snapshots
            tensor_path=Path(args.output).with_suffix('.pt')
            require(not tensor_path.exists(),'refusing to overwrite fixed-history evidence')
            torch.save(tensors,tensor_path)
            report['fixed_history_tensors']=str(tensor_path)
            report['fixed_history_sha256']=hashlib.sha256(tensor_path.read_bytes()).hexdigest()
            if args.compare_report:
                old_report=json.loads(Path(args.compare_report).read_text())
                require(hashlib.sha256(Path(old_report['fixed_history_tensors']).read_bytes()).hexdigest()
                        ==old_report['fixed_history_sha256'],'comparison tensor file fingerprint differs')
                old=torch.load(old_report['fixed_history_tensors'],map_location='cpu',weights_only=True)
                require(old.keys()==tensors.keys(),'comparison prefix set differs')
                require(len(old['same_path_kv'])==len(snapshots)==2,'comparison KV snapshots incomplete')
                comparisons=[]
                report['cross_process_fixed_history']=comparisons
                for prefix in ('1','255','256','257','507'):
                    require(old[prefix]['history']==tensors[prefix]['history'],'comparison history differs')
                    for path in ('reference','candidate'):
                        reference=old[prefix][path];candidate=tensors[prefix][path]
                        row=dict(prefix=prefix,path=path,exact=bitwise_equal(reference,candidate),
                                 **compare_logits(reference,candidate))
                        comparisons.append(row)
                        require(row['exact'],'cross-process fixed-history logits not bitwise identical')
                report['cross_process_fixed_history']=comparisons
                report['cross_process_target_kv_exact']=all(bitwise_equal(x,y) for x,y in zip(old['same_path_kv'],snapshots))
                require(report['cross_process_target_kv_exact'],'cross-process target KV differs')
            require_empty(engine)
        lifecycle=report['lifecycle']=[]
        for stage in ('cancel','planner','draft','target'):
            guard('exception_'+stage)
            seq=engine.add_request(prompt,sampling())
            engine.step()
            before=list(seq.token_ids)
            original_schedule=engine.scheduler.schedule_plan
            original_rollout=owner.proposer.rollout
            original_forward=owner.proposer.forward
            marker='injected Marlin '+stage+' failure'
            observed={}
            def bad_schedule(*a,**kw):
                observed['draft_published']=len(owner.proposer.cached_tokens)
                plan=original_schedule(*a,**kw)
                require(plan.has_verification,'planner injection lacked verification')
                raise RuntimeError(marker)
            def bad_draft(*a,**kw):
                callback=original_rollout if original_rollout is not None else original_forward
                callback(*a,**kw)
                observed['real_draft_forward_completed']=True
                raise RuntimeError(marker)
            if stage=='planner': engine.scheduler.schedule_plan=bad_schedule
            if stage=='draft':
                if original_rollout is None: owner.proposer.forward=bad_draft
                else: owner.proposer.rollout=bad_draft
            fail_target=stage=='target'
            try:
                if stage=='cancel':
                    engine.step()
                else:
                    try: engine.step()
                    except RuntimeError as error:
                        expected='injected Marlin target verification failure after KV write' if stage=='target' else marker
                        require(str(error)==expected,'unexpected lifecycle error: '+str(error))
                    else: raise AssertionError('injected lifecycle exception swallowed')
                    require(seq.token_ids==before,'exception partially committed tokens')
                    require(not owner.proposer.cached_tokens and owner.request_id is None,'exception retained draft owner')
                    require(not engine.scheduler._reservations and not engine.scheduler.block_manager._reservations,
                            'exception retained reservations')
            finally:
                engine.scheduler.schedule_plan=original_schedule
                owner.proposer.rollout=original_rollout
                owner.proposer.forward=original_forward
                fail_target=False
            require(engine.cancel_request(seq.seq_id),'cancel failed')
            require(get_context().slot_mapping is None,'exception/cancel leaked Context')
            lifecycle.append(dict(stage=stage,observed=observed,resources=require_empty(engine)))
            engine.forget_request(seq.seq_id)
            if stage!='target':
                recovered=generate('after_'+stage,prompt)
                require(recovered['tokens']==baseline['tokens'],'lifecycle failure contaminated next request')
            else:
                require(runner._failed,'model failure did not fail closed')
        require(any(event['accepted']<event['K'] for event in events),'no actual target rejection observed')
        report['target_rejection_rounds']=sum(event['accepted']<event['K'] for event in events)
        report['target_full_accept_rounds']=sum(event['accepted']==event['K'] for event in events)
        report['draft_stats']=owner.stats()
        if args.compare_report:
            reference=json.loads(Path(args.compare_report).read_text())
            require(reference['status']=='PASS' and reference['execution']=='eager','comparison report is not successful eager')
            require(reference['source_sha256']==report['source_sha256'],'eager/graph core source differs')
            require(reference['request_tokens']==report['request_tokens'],'eager/graph end-to-end output mismatch')
            require([p['tokens'] for p in reference['proposal_probes']]==[p['tokens'] for p in probes],
                    'eager/graph probe tokens differ across fresh processes')
            report['cross_execution_exact']=True
    finally:
        if engine is not None: engine.exit()
        report['distributed_cleared']=not torch.distributed.is_initialized()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='/home/hrxu/huggingface/Qwen3-1.7B')
    parser.add_argument('--draft',default='/home/hrxu/huggingface/Qwen3-0.6B-AWQ-Orion-zhen-tokenizer-overlay')
    parser.add_argument('--draft-quantization',choices=('awq','none'),default='awq',
                        help='Default preserves original Marlin test; none validates actual FP16 draft with all lifecycle/numeric checks')
    parser.add_argument('--marlin-library')
    parser.add_argument('--gpu-memory-utilization',type=float,default=.7,
                        help='Admission budget only: 8B+draft uses .90; numerical checks are unchanged')
    parser.add_argument('--execution',choices=('eager','graph'),required=True)
    parser.add_argument('--prefill-graph',action='store_true',help='Opt-in exact-shape startup draft-prefill graphs')
    parser.add_argument('--tail-graph',action='store_true',help='Opt-in true Q1 tail of chunked cold prefill')
    parser.add_argument('--async-metadata',action='store_true',
                        help='Explicit experimental pinned/event H2D; graph only, default OFF')
    parser.add_argument('--compare-report')
    parser.add_argument('--extended-numerics',action='store_true',help='Add 257 boundary, fixed-history Q=1/2/4/8 and target KV checks without weakening original assertions')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.tail_graph and not args.prefill_graph:
        parser.error('--tail-graph requires --prefill-graph')
    if args.prefill_graph and args.execution!='graph':
        parser.error('--prefill-graph requires --execution graph')
    if args.async_metadata and args.execution!='graph':
        parser.error('--async-metadata requires --execution graph; eager reference remains OFF')
    if args.draft_quantization=='awq' and not args.marlin_library:
        parser.error('AWQ lifecycle requires --marlin-library')
    report=dict(status='RUNNING',execution=args.execution,async_metadata=args.async_metadata,
        draft_quantization=args.draft_quantization,guards=[],scope='greedy lifecycle, not model quality or performance')
    with Path(args.output).open('x',encoding='utf-8') as output:
        def guard(stage): guarded_snapshot(report,stage)
        try:
            guard('before_cuda_initialization')
            report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((ROOT/'nanovllm').rglob('*.py'))}
            report['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            report['helper_sha256']={name:hashlib.sha256((ROOT/('tests/regression' if name.startswith('check_') else 'benchmarks')/name).read_bytes()).hexdigest()
                for name in ('check_draft_model.py','check_stochastic_engine.py','runtime_v2_gpu_guard.py')}
            report['extended_numerics']=args.extended_numerics
            report['marlin_library']=str(Path(args.marlin_library).resolve()) if args.marlin_library else None
            from benchmarks.cache_draft_experiment import file_record
            report['checkpoint_identity']={name:dict(path=str(Path(path).resolve()),
                files={p.name:file_record(p) for p in sorted(Path(path).iterdir())
                       if p.is_file() and p.suffix in ('.safetensors','.json')})
                for name,path in (('target',args.model),('draft',args.draft))}
            report['gpu_memory_utilization']=args.gpu_memory_utilization
            if args.compare_report:
                reference=json.loads(Path(args.compare_report).read_text())
                require(reference['status']=='PASS' and reference['execution']=='eager','comparison must be a PASS eager report')
                require(reference.get('async_metadata',False) is False,'eager reference must retain async OFF')
                for field in ('source_sha256','benchmark_sha256','helper_sha256','extended_numerics','marlin_library',
                              'checkpoint_identity','gpu_memory_utilization','draft_quantization'):
                    require(reference[field]==report[field],'comparison evidence identity differs: '+field)
            run(args,report,guard)
            report['status']='PASS'
        except BaseException:
            report['status']='FAIL'
            report['traceback']=traceback.format_exc()
        finally:
            json.dump(report,output,indent=2)
            output.write('\n')
        print(json.dumps({k:report[k] for k in ('status','execution','traceback') if k in report},indent=2))
        return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
