"""Guarded bitwise KV transfer/lease/fault tests, not an end-to-end speed claim."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    report=dict(status='RUNNING',checks=[])
    report['admission']=ensure_gpu1_available()
    import torch
    from nanovllm.engine.cache_transfer import CacheTransferWorkspace,BatchedHostPrefixCache
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.sequence import Sequence

    def make(shape,dtype=torch.float16,host_pages=8):
        kv=torch.randn(shape,device='cuda',dtype=dtype)
        page_shape=tuple(shape[:2])+tuple(shape[3:])
        w=CacheTransferWorkspace(page_shape,dtype,batch_blocks=4,slots=2,max_bytes=256*2**20,device=kv.device)
        w.warmup(kv)
        c=BatchedHostPrefixCache(kv,host_pages*w.page_bytes+w.host_index_bytes,'gpu-regression',
            workspace=w,min_restore_tokens=shape[3],max_restore_blocks=8,max_inflight_blocks=4)
        return kv,w,c

    def equal(x,y):
        assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))

    try:
        for dtype,shape in [(torch.float16,(2,3,12,3,1,7)),
                            (torch.float16,(2,3,12,256,2,32)),
                            (torch.bfloat16,(2,28,12,256,8,128))]:
            ensure_gpu1_available()
            kv,w,c=make(shape,dtype)
            expected=kv[:,:,:4].clone()
            with torch.cuda.stream(w.write_stream):torch.cuda._sleep(30_000_000)
            c.put_many([(bytes([i]),i) for i in range(8)])
            for slot in w.slots:
                if slot.job and not slot.done.query():
                    assert all(key not in c.pages for key in slot.job['keys'])
            assert c.counters['bounded_store_skips']==4
            # Immediate same-stream source overwrite must not corrupt backup.
            kv[:,:,:4].fill_(-19)
            c.reap(blocking=True)
            assert len(c.pages)==4
            c.load_many([(bytes([i]),i+4) for i in range(4)])
            c.wait_for_blocks(range(4,8))
            equal(kv[:,:,4:8],expected)
            assert set(c.reap(blocking=True))==set(range(4,8))
            assert not c.source_pins
            c.mark_consumed(range(4,8))
            assert c.counters['consumed_prefetch_bytes']==4*w.page_bytes
            assert c.stats()['pinned_pool_bytes']<=c.max_bytes
            assert c.stats()['staging_bytes']<=256*2**20
            for pairs in ([(b'x',99)],[(b'x',0),(b'y',0)]):
                try:c.put_many(pairs)
                except ValueError:pass
                else:raise AssertionError('invalid page batch accepted')
            side=torch.cuda.Stream()
            with torch.cuda.stream(side):
                try:c.put(b'cross-stream',0)
                except RuntimeError:pass
                else:raise AssertionError('cross-stream cache call accepted')
            c.clear()
            report['checks'].append(dict(case='bitwise_roundtrip',dtype=str(dtype),shape=shape,status='PASS'))
            del c,w,kv,expected
            torch.cuda.empty_cache()

        # Read sources must remain pinned even if all host entries are LRU victims.
        kv,w,c=make((2,2,12,256,1,16),host_pages=4)
        expected=kv[:,:,:4].clone()
        c.put_many([(bytes([i]),i) for i in range(4)]);c.reap(blocking=True)
        with torch.cuda.stream(w.read_stream):torch.cuda._sleep(30_000_000)
        c.load_many([(bytes([i]),i+4) for i in range(4)])
        c.put_many([(bytes([i+10]),i) for i in range(4)])
        assert len(c.source_pins)==4 and c.counters['bounded_store_skips']>=4
        c.wait_for_blocks(range(4,8));equal(kv[:,:,4:8],expected);c.reap(blocking=True)
        # Failure after DMA enqueue must fence before dropping source ownership.
        with patch('nanovllm.layers.kv_transfer.move_pages',side_effect=RuntimeError('injected scatter failure')):
            try:c.load_many([(b'\x00',8)])
            except RuntimeError as error:assert 'injected' in str(error)
            else:raise AssertionError('injected failure was not raised')
        assert c.failed and not c.inflight and not c.source_pins
        c.clear()
        report['checks'].append(dict(case='source_pin_and_exception_fence',status='PASS'))
        del kv,w,c,expected
        torch.cuda.empty_cache()

        # 新容量策略不能淘汰 H2D 尚在读取的 CPU 源，即使新页价值更高。
        from nanovllm.engine.cache_cost import CacheCostModel,CacheCostPolicy
        from tests.regression.check_cache_cost import fixture
        doc=fixture();doc.update(cold_reuse_probability=.25,
            replacement=dict(kind='capacity_value',reuse_window_requests=12,victim_scan_limit=64))
        policy=CacheCostPolicy(CacheCostModel(doc))
        kv,w,c=make((2,2,12,256,1,16),host_pages=4);c.policy=policy
        old_keys=[f'old{i}'.encode() for i in range(4)]
        new_keys=[f'new{i}'.encode() for i in range(4)]
        policy.observe(1,old_keys,1024);policy.observe(2,old_keys,1024)
        c.put_many(list(zip(old_keys,range(4))));c.reap(blocking=True)
        for request in range(3,9):policy.observe(request,new_keys,1024)
        expected=kv[:,:,:4].clone()
        with torch.cuda.stream(w.read_stream):torch.cuda._sleep(100_000_000)
        c.load_many(list(zip(old_keys,range(4,8))))
        c.put_many(list(zip(new_keys,range(4))))
        assert len(c.source_pins)==4 and set(c.pages)==set(old_keys)
        assert c.counters['capacity_store_skips']==4
        c.wait_for_blocks(range(4,8));equal(kv[:,:,4:8],expected);c.reap(blocking=True)
        assert not c.source_pins
        # 复制完成后，同一批新页可以替换低价值旧页；验证真正搬回的字节。
        c.put_many(list(zip(new_keys,range(4))));c.reap(blocking=True)
        assert set(c.pages)==set(new_keys)
        assert policy.counters['capacity_replacements']==4
        kv[:,:,:4].fill_(-17)
        c.load_many(list(zip(new_keys,range(8,12))))
        c.wait_for_blocks(range(8,12));equal(kv[:,:,8:12],expected);c.reap(blocking=True)
        c.clear()
        report['checks'].append(dict(case='capacity_policy_source_pins_and_real_replacement',status='PASS'))
        del kv,w,c,expected
        torch.cuda.empty_cache()

        kv,w,c=make((2,2,8,256,1,16))
        old=Sequence.block_size;Sequence.block_size=256
        try:
            manager=BlockManager(8,256);manager.host_cache=c;manager.defer_host_stores=True
            tokens=[100+i%17 for i in range(1025)]
            seq=Sequence(tokens);manager.allocate(seq,0)
            for i,bid in enumerate(seq.block_table):kv[:,:,bid].fill_(i+5)
            expected=[kv[:,:,bid].clone() for bid in seq.block_table[:4]]
            seq.num_scheduled_tokens=len(tokens);manager.hash_blocks(seq);manager.deallocate(seq)
            pressure=[manager._allocate_block() for _ in range(8)]
            manager.flush_host_stores();kv.fill_(-3)
            for bid in pressure:
                manager.blocks[bid].ref_count=0;manager._deallocate_block(bid)
            manager.reap_host_loads(blocking=True)
            with torch.cuda.stream(w.read_stream):torch.cuda._sleep(30_000_000)
            again=Sequence(tokens)
            assert manager.restore_host_prefix(again)==4
            protected=set(manager._host_load_leases)
            assert len(protected)==4
            manager.allocate(again,manager.can_allocate(again))
            restored=list(again.block_table[:4])
            # Cancellation removes request references; unfinished copies retain leases.
            manager.deallocate(again)
            other=[manager._allocate_block() for _ in range(len(manager.free_block_ids))]
            assert not protected.intersection(other)
            for bid in other:
                manager.blocks[bid].ref_count=0;manager._deallocate_block(bid)
            manager.reap_host_loads(blocking=True)
            for bid,value in zip(restored,expected):equal(kv[:,:,bid],value)
            assert not manager.used_block_ids and not manager._host_load_leases
            assert not manager._pending_host_stores and all(b.ref_count==0 for b in manager.blocks)
            c.clear()
        finally:Sequence.block_size=old
        report['checks'].append(dict(case='cancel_copy_page_reuse',status='PASS'))
        report['status']='PASS'
    except BaseException:
        import traceback
        report.update(status='FAIL',traceback=traceback.format_exc())
        raise
    finally:
        a.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
