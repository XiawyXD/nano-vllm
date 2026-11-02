"""空传输快路径：真实调度器、逻辑 KV 和事件/lease 状态的 CPU 回归。"""
from collections import deque
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
from unittest.mock import Mock,patch
import torch
from nanovllm.engine.cache_transfer import BatchedHostPrefixCache
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from tests.regression.check_cache_draft import DelayedHostCache,PhysicalPlanRunner,seed_and_evict,assert_drained
from tests.regression.check_planned_engine import make_engine,params


class IdleCache(DelayedHostCache):
    def __init__(self,*args,fast=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.idle_fastpath=fast
        self.pending_stores={};self.completed_loads=deque();self.unconsumed={}
        self._check=Mock()

    @property
    def needs_reap(self):return bool(self.inflight or self.pending_stores or self.completed_loads)

    def mark_consumed(self,blocks):
        for bid in blocks:self.unconsumed.pop(bid,None)


class LegacyRunner(PhysicalPlanRunner):
    def call(self,method,seqs,is_prefill):
        assert method=='run'
        result=[];self.calls+=1
        for seq in seqs:
            assert set(seq.block_table)&self.host.inflight.keys() <= self.host.waited
            start=seq.num_cached_tokens if is_prefill else len(seq)-1
            end=start+seq.num_scheduled_tokens if is_prefill else len(seq)
            for pos in range(start):
                bid,off=seq.block_table[pos//2],pos%2
                assert int(self.kv[0,0,bid,off,0,0])==seq[pos]
            for pos in range(start,end):self.kv[:,:,seq.block_table[pos//2],pos%2].fill_(seq[pos])
            result.append(seq[end-1]%3+1)
        return result


def engine(mode,fast=True):
    e=make_engine(blocks=6,block_size=2,budget=2,chunk=2,policy='off')
    if mode=='separate':
        e._packed_enabled=False;e.config.batching_mode='separate';e.scheduler=Scheduler(e.config)
    kv=torch.zeros(2,2,6,2,1,2)
    cache=IdleCache(kv,4096,'idle-test',fast=fast,min_restore_tokens=2)
    e.scheduler.block_manager.host_cache=cache
    e.model_runner=(LegacyRunner if mode=='separate' else PhysicalPlanRunner)(kv,cache)
    return e,kv,cache


class IdleTests(unittest.TestCase):
    def setUp(self):self.old_block=Sequence.block_size
    def tearDown(self):Sequence.block_size=self.old_block

    def test_completed_queue_is_not_idle(self):
        c=BatchedHostPrefixCache.__new__(BatchedHostPrefixCache)
        c.pending_stores={};c.inflight={};c.completed_loads=deque()
        self.assertFalse(c.needs_reap)
        c.completed_loads.append(7);self.assertTrue(c.needs_reap)
        c.completed_loads.clear();c.pending_stores[b'x']=1;self.assertTrue(c.needs_reap)
        c.pending_stores.clear();c.inflight[3]=object();self.assertTrue(c.needs_reap)

    def test_idle_steps_skip_unused_helpers_and_preserve_tokens(self):
        for mode in ('separate','mixed'):
            outputs=[]
            for fast in (False,True):
                e,kv,c=engine(mode,fast);m=e.scheduler.block_manager
                seq=e.add_request([1,2,3],params(maximum=3))
                with patch.object(m,'flush_host_stores',wraps=m.flush_host_stores) as flush, \
                     patch.object(c,'wait_for_blocks',wraps=c.wait_for_blocks) as wait, \
                     patch.object(c,'mark_consumed',wraps=c.mark_consumed) as consume, \
                     patch.object(m,'reap_host_loads',wraps=m.reap_host_loads) as reap:
                    for _ in range(20):
                        if e.is_finished():break
                        e.step()
                    self.assertTrue(e.is_finished());assert_drained(self,m)
                    if fast:
                        self.assertEqual((flush.call_count,wait.call_count,consume.call_count,reap.call_count),(0,0,0,0))
                        self.assertGreater(c._check.call_count,0)
                    else:self.assertGreater(wait.call_count,0)
                outputs.append(seq.completion_token_ids)
            self.assertEqual(*outputs)

    def test_inflight_copy_is_waited_and_lease_reaped(self):
        for mode in ('separate','mixed'):
            e,kv,c=engine(mode);m=e.scheduler.block_manager
            seed_and_evict(m,kv)
            seq=e.add_request([1,2,3,4,9],params(maximum=3))
            e.step()
            self.assertTrue(c.waited,'real consumed copy must keep its event dependency')
            c.ready.update(c.inflight)
            for _ in range(20):
                if e.is_finished():break
                e.step()
            self.assertTrue(seq.is_finished);assert_drained(self,m)

    def test_failed_cache_is_checked_even_when_empty(self):
        for mode in ('separate','mixed'):
            e,kv,c=engine(mode);seq=e.add_request([1],params(maximum=2))
            c._check.side_effect=RuntimeError('injected failed cache')
            with self.assertRaisesRegex(RuntimeError,'injected failed cache'):e.step()
            self.assertEqual(e.model_runner.calls,0)
            e.cancel_request(seq.seq_id);e.forget_request(seq.seq_id)
            assert_drained(self,e.scheduler.block_manager)


if __name__=='__main__':unittest.main()
