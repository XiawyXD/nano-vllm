"""Same-history model/KV/cache boundary regression and cross-process comparison."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('legacy','batched'),required=True)
    p.add_argument('--batching-mode',choices=('separate','mixed'),default='separate')
    p.add_argument('--policy',type=Path)
    p.add_argument('--joint-policy',type=Path)
    p.add_argument('--idle-fastpath',action='store_true')
    p.add_argument('--model',default='/home/hrxu/huggingface/Qwen3-1.7B')
    p.add_argument('--compare',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    tensor_path=a.output.with_suffix('.pt')
    if tensor_path.exists():raise FileExistsError(tensor_path)
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    report=dict(status='RUNNING',backend=a.backend,guards=[],probes=[])
    report['guards'].append(ensure_gpu1_available())
    import torch
    from nanovllm import LLM,SamplingParams
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import reset_context,get_context
    from tests.regression.check_draft_model import compare_logits
    from benchmarks.cache_runtime_experiment import reset,sources,sha
    engine=None;logit_records={}
    try:
        config=dict(tensor_parallel_size=1,max_model_len=2048,max_num_seqs=1,
            max_num_batched_tokens=256,prefill_chunk_size=128,enforce_eager=False,
            rmsnorm_backend='triton',linear_backend='torch',numerical_backend='default',
            metadata_backend='reuse',max_kvcache_blocks=12,gpu_memory_utilization=.85,
            batching_mode=a.batching_mode,speculative_policy='off',host_kv_cache_bytes=2**30,
            host_kv_async_prefetch=a.backend=='batched',host_kv_transfer_backend=a.backend)
        if a.policy:config['host_kv_cost_model_path']=str(a.policy)
        if a.joint_policy:
            config['host_kv_joint_residency_path']=str(a.joint_policy)
            config['host_kv_async_prefetch']=True
        if a.idle_fastpath:config['host_kv_transfer_idle_fastpath']=True
        engine=LLM(a.model,**config);m=engine.scheduler.block_manager;c=m.host_cache;r=engine.model_runner
        report.update(model=a.model,config=config,source_sha256=sources(),model_config_sha256=sha(Path(a.model)/'config.json'))

        @torch.no_grad()
        def run(tokens,restore):
            seq=Sequence(tokens,SamplingParams(temperature=0,max_tokens=1))
            loaded=m.restore_host_prefix(seq) if restore else 0
            m.allocate(seq,m.can_allocate(seq) if restore else 0)
            last=None
            try:
                while seq.num_cached_tokens<len(tokens):
                    m.flush_host_stores()
                    if getattr(c,'asynchronous',False):c.wait_for_blocks(seq.block_table)
                    seq.num_scheduled_tokens=min(128,len(tokens)-seq.num_cached_tokens)
                    try:
                        ids,pos=r.prepare_prefill([seq]);last=r.run_model(ids,pos,True).detach().cpu()
                    finally:reset_context()
                    m.observe_cache_request(seq)
                    m.hash_blocks(seq)
                    seq.num_cached_tokens+=seq.num_scheduled_tokens;seq.num_scheduled_tokens=0
                slots=torch.tensor([seq.block_table[i//256]*256+i%256 for i in range(len(tokens))],device='cuda')
                valid=r.kv_cache.flatten(2,3).index_select(2,slots).cpu().contiguous()
                return last,valid,loaded
            finally:
                m.deallocate(seq);m.reap_host_loads(blocking=True)

        @torch.no_grad()
        def pressure():
            ids=[m._allocate_block() for _ in range(len(m.free_block_ids))]
            m.flush_host_stores()
            r.kv_cache.zero_()  # only this engine's own KV; proves CPU data is used
            for bid in ids:
                m.blocks[bid].ref_count=0;m._deallocate_block(bid)
            m.reap_host_loads(blocking=True)

        for length in (255,256,257,513,1025):
            report['guards'].append(ensure_gpu1_available())
            reset(engine)
            tokens=[100+(i*31)%1000 for i in range(length)]
            first,kv1,_=run(tokens,False)
            pressure()
            second,kv2,loads=run(tokens,True)
            numeric=compare_logits(first,second)
            assert numeric['strict_allclose'] and torch.equal(kv1,kv2),(length,numeric)
            if length>256:assert loads>0,'no real CPU prefix restored'
            key=str(length)
            logit_records[key]=second
            report['probes'].append(dict(length=length,restored_blocks=loads,logits=numeric,
                valid_kv_sha256=hashlib.sha256(kv2.view(torch.uint8).numpy().tobytes()).hexdigest(),
                valid_kv_bitwise_equal=True))
            del kv1,kv2
        reset(engine)
        params=SamplingParams(temperature=0,max_tokens=16,ignore_eos=True)
        generator=engine.generate_stream('请解释缓存的用途。',params)
        next(generator);generator.close()
        assert not engine._requests and not m.used_block_ids
        report['cancel_after_event']='PASS'
        ordinary_call=r.call
        def force_eos(method,*args):
            result=ordinary_call(method,*args)
            if method=='run_plan':return {rid:[engine.config.eos]*len(ids) for rid,ids in result.items()}
            return [engine.config.eos]*len(result) if method=='run' else result
        with patch.object(r,'call',side_effect=force_eos):
            events=list(engine.generate_stream('EOS lifecycle test.',SamplingParams(temperature=0,max_tokens=8)))
        assert len(events)==1 and events[-1].finished and events[-1].finish_reason=='stop'
        report['real_forward_injected_eos']='PASS'
        with patch.object(r,'call',side_effect=RuntimeError('injected pre-forward failure')):
            try:list(engine.generate_stream('Fault cleanup test.',params))
            except RuntimeError as error:assert 'injected' in str(error)
            else:raise AssertionError('failure was not propagated')
        m.reap_host_loads(blocking=True)
        assert not engine._requests and not m.used_block_ids and not m._host_load_leases and not m._reservations
        assert not getattr(engine.scheduler,'_reservations',{}) and not getattr(engine.scheduler,'_active_items',{})
        assert get_context().slot_mapping is None
        report['fault_cleanup']='PASS'
        if a.compare:
            previous=json.loads(a.compare.read_text())
            assert previous['status']=='PASS' and previous['model_config_sha256']==report['model_config_sha256']
            assert previous['source_sha256']==report['source_sha256']
            reference_tensors=Path(previous['tensor_path'])
            if not reference_tensors.is_absolute():
                # 每个报告的 .pt 与 JSON 同目录；不能依赖生成报告时的 cwd。
                reference_tensors=a.compare.resolve().parent/reference_tensors.name
            tensors=torch.load(reference_tensors,weights_only=True,map_location='cpu')
            for old,new in zip(previous['probes'],report['probes']):
                assert old['length']==new['length'] and old['valid_kv_sha256']==new['valid_kv_sha256']
                check=compare_logits(tensors[str(new['length'])],logit_records[str(new['length'])])
                assert check['strict_allclose'],check
            report['cross_process']='PASS'
        with tensor_path.open('xb') as handle:torch.save(logit_records,handle)
        report.update(status='PASS',tensor_path=str(tensor_path.resolve()),cache=c.stats())
    except BaseException:
        import traceback
        report.update(status='FAIL',traceback=traceback.format_exc());raise
    finally:
        try:
            if engine is not None:engine.exit()
        finally:
            with a.output.open('x') as handle:json.dump(report,handle,indent=2)
            print(json.dumps({k:report[k] for k in ('status','backend','cross_process','traceback') if k in report},indent=2))


if __name__=='__main__':main()
