"""CPU-only cost decisions, bounded history, validation and baseline KV fallback."""
import copy
import unittest
from nanovllm.engine.cache_cost import CacheCostModel, CacheCostPolicy


def fixture():
    return dict(schema='nano-cache-cost-v1',identity=dict(chunk_size=128,block_size=256),
        query_ms=[dict(q=q,past=p,ms=0.5+q*.02+q*p*.000001)
                  for q in (1,32,128,256) for p in (0,256,512,1024,2048)],
        transfer_ms=dict(h2d=[[1,.2],[2,.35],[4,.6],[8,1.1]],
                         d2h=[[1,.2],[2,.35],[4,.6],[8,1.1]]))


class CostTests(unittest.TestCase):
    def capacity_policy(self,window=8,scan=64):
        doc=fixture();doc['cold_reuse_probability']=.25
        doc['replacement']=dict(kind='capacity_value',reuse_window_requests=window,victim_scan_limit=scan)
        return CacheCostPolicy(CacheCostModel(doc))

    def test_capacity_keeps_first_reuse_when_space_is_free(self):
        policy=self.capacity_policy();policy.observe(1,[b'cold'],1024)
        self.assertTrue(policy.admit(b'cold'))
        self.assertEqual(policy.plan_admission([b'cold'],[b'hot'],1),([b'cold'],[]))
        self.assertEqual(policy.counters['victim_keys_scanned'],0)

    def test_capacity_rejects_cold_pollution_and_replaces_cold_with_hot(self):
        policy=self.capacity_policy()
        policy.observe(1,[b'hot'],1024);policy.observe(2,[b'hot'],1024)
        policy.observe(3,[b'cold'],1024)
        self.assertEqual(policy.plan_admission([b'cold'],[b'hot'],0),([],[]))
        self.assertEqual(policy.plan_admission([b'hot'],[b'cold'],0),([b'hot'],[b'cold']))

    def test_capacity_uses_cheapest_eligible_victim_and_scan_bound(self):
        policy=self.capacity_policy(scan=2)
        policy.observe(1,[b'hot'],1024);policy.observe(2,[b'hot'],1024)
        policy.observe(3,[b'a',b'b'],1024)
        self.assertEqual(policy.plan_admission([b'hot'],[b'a',b'b',b'unseen'],0),([b'hot'],[b'a']))
        self.assertEqual(policy.counters['victim_keys_scanned'],2)
        self.assertEqual(policy.plan_admission([b'hot'],[],0),([],[]))

    def test_capacity_recent_window_ages_out_former_hotset(self):
        policy=self.capacity_policy(window=4)
        policy.observe(1,[b'old'],1024);policy.observe(2,[b'old'],1024)
        old=policy.retention_value(b'old');self.assertGreater(old,0)
        for i in range(3,7):policy.observe(i,[],0)
        self.assertEqual(policy.retention_value(b'old'),0)
        policy.observe(7,[b'new'],1024)
        self.assertEqual(policy.plan_admission([b'new'],[b'old'],0),([b'new'],[b'old']))

    def test_capacity_expensive_write_still_rejected_with_free_space(self):
        doc=fixture();doc.update(cold_reuse_probability=.25,
            replacement=dict(kind='capacity_value',reuse_window_requests=8,victim_scan_limit=64))
        doc['transfer_ms']['d2h']=[[1,1000],[4,4000]]
        policy=CacheCostPolicy(CacheCostModel(doc));policy.observe(1,[b'new'],1024)
        self.assertFalse(policy.admit(b'new'))

    def test_capacity_host_cache_rejection_does_not_copy_or_evict(self):
        from unittest.mock import patch
        import torch
        from nanovllm.engine.tiered_cache import HostPrefixCache
        policy=self.capacity_policy()
        policy.observe(1,[b'hot'],1024);policy.observe(2,[b'hot'],1024)
        cache=HostPrefixCache(torch.zeros(2,2,4,2,1,2),64,'test',policy=policy)
        self.assertEqual(cache.capacity,1)
        cache.put(b'hot',0)
        policy.observe(3,[b'cold'],1024)
        with patch.object(cache,'_copy') as copy:
            cache.put(b'cold',1);copy.assert_not_called()
        self.assertEqual(list(cache.pages),[b'hot'])
        self.assertEqual(cache.counters['capacity_store_skips'],1)

    def test_capacity_value_memo_is_bounded_and_cold_metadata_validated(self):
        policy=self.capacity_policy()
        for length in range(1,140):policy._page_saving(length)
        self.assertEqual(len(policy.value_cache),128)
        for options in (dict(kind='unknown'),dict(kind='capacity_value',reuse_window_requests=0),
                        dict(kind='capacity_value',victim_scan_limit=65)):
            doc=fixture();doc['replacement']=options
            with self.assertRaises(ValueError):CacheCostModel(doc)

    def test_prefill_counts_suffix_and_past_k(self):
        model=CacheCostModel(fixture())
        self.assertAlmostEqual(model.prefill_ms(257,256),model.query_ms(1,256))
        self.assertGreater(model.prefill_ms(1025),model.prefill_ms(1025,1024))
        self.assertGreater(model.query_ms(128,1024),model.query_ms(128,0))

    def test_history_counts_distinct_requests_only(self):
        policy=CacheCostPolicy(CacheCostModel(fixture()))
        policy.observe(1,[b'a',b'b'],1024)
        self.assertFalse(policy.admit(b'a'))
        policy.observe(1,[b'a'],1024)
        self.assertEqual(policy.history[b'a']['count'],1)
        policy.observe(2,[b'a'],1024)
        self.assertTrue(policy.admit(b'a'))

    def test_history_is_bounded(self):
        policy=CacheCostPolicy(CacheCostModel(fixture()),history_capacity=2)
        for i in range(10):
            policy.observe(i,[str(i).encode()],1024)
        self.assertEqual(len(policy.history),2)
        self.assertEqual(policy.counters['history_evictions'],8)

    def test_recompute_can_win_and_backlog_can_reverse_decision(self):
        policy=CacheCostPolicy(CacheCostModel(fixture()))
        self.assertTrue(policy.restore(1025,1024,0,4))
        self.assertFalse(policy.restore(1025,1024,0,4,queued_blocks=1000))
        self.assertFalse(policy.restore(1025,1024,1024,1))
        slow=fixture();slow['transfer_ms']['h2d']=[[1,100],[4,400]]
        self.assertFalse(CacheCostPolicy(CacheCostModel(slow)).restore(1025,1024,0,4))

    def test_calibrated_prior_can_admit_expensive_cold_prefix(self):
        doc=fixture();doc['cold_reuse_probability']=.25
        policy=CacheCostPolicy(CacheCostModel(doc))
        policy.observe(1,[b'long'],1024)
        self.assertTrue(policy.admit(b'long'))
        self.assertEqual(policy.counters['admission_prior_accepts'],1)
        doc['transfer_ms']['d2h']=[[1,1000],[4,4000]]
        policy=CacheCostPolicy(CacheCostModel(doc));policy.observe(1,[b'long'],1024)
        self.assertFalse(policy.admit(b'long'))

    def test_invalid_numbers_and_incomplete_grid_rejected(self):
        for mutate in (lambda d:d['query_ms'][0].update(ms=float('nan')),
                       lambda d:d['query_ms'].pop(),
                       lambda d:d['transfer_ms'].update(h2d=[[1,-1]]),
                       lambda d:d.update(schema='unknown')):
            doc=copy.deepcopy(fixture());mutate(doc)
            with self.assertRaises(ValueError):CacheCostModel(doc)

    def test_fragmented_host_slots_keep_lane_mapping(self):
        from nanovllm.engine.cache_transfer import contiguous_segments
        self.assertEqual(contiguous_segments([3,4,7,1,2]),[(0,2,3,5),(2,3,7,8),(3,5,1,3)])
        self.assertEqual(contiguous_segments([]),[])

    def test_restore_bypass_does_not_allocate_or_publish(self):
        from tests.regression.check_tiered_cache import TieredTests
        state=TieredTests();state.setUp()
        try:
            state.commit([1,2,3,4,5],20);state.evict()
            slow=fixture();slow['identity']['block_size']=2
            slow['transfer_ms']['h2d']=[[1,1000],[4,4000]]
            state.cache.policy=CacheCostPolicy(CacheCostModel(slow))
            before=list(state.manager.free_block_ids)
            self.assertEqual(state.manager.restore_host_prefix(state.seq([1,2,3,4,9])),0)
            self.assertEqual(before,list(state.manager.free_block_ids))
            self.assertFalse(state.manager.hash_to_block_id)
            state.invariant()
            state.cache.policy=None
            self.assertEqual(state.manager.restore_host_prefix(state.seq([1,2,3,4,9])),2)
            state.invariant()
        finally:state.tearDown()

    def test_cold_admission_avoids_copy(self):
        from unittest.mock import patch
        import torch
        from nanovllm.engine.tiered_cache import HostPrefixCache
        policy=CacheCostPolicy(CacheCostModel(fixture()))
        cache=HostPrefixCache(torch.zeros(2,2,4,2,1,2),1024,'test',policy=policy)
        policy.observe(1,[b'a'],1024)
        with patch.object(cache,'_copy') as copy:
            cache.put(b'a',0)
            copy.assert_not_called()
        self.assertFalse(cache.pages)

    def test_identity_rejects_dtype_gpu_and_batch_changes(self):
        import hashlib,os,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'config.json').write_text('{}')
            cfg=SimpleNamespace(model=folder,prefill_chunk_size=128,max_num_batched_tokens=256,
                kvcache_block_size=256,linear_backend='torch',numerical_backend='default',
                rmsnorm_backend='triton',model_dtype='checkpoint',host_kv_transfer_backend='batched',
                host_kv_transfer_batch_blocks=4,host_kv_transfer_slots=2)
            doc=fixture();doc['identity'].update(model=str(path.resolve()),
                config_sha256=hashlib.sha256(b'{}').hexdigest(),linear_backend='torch',
                numerical_backend='default',rmsnorm_backend='triton',model_dtype='checkpoint',
                transport='batched',batch_blocks=4,slots=2,gpu_uuid='GPU-test')
            model=CacheCostModel(doc)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-test'):
                model.check_config(cfg)
                for field,value in [('model_dtype','float16'),('host_kv_transfer_batch_blocks',2)]:
                    old=getattr(cfg,field);setattr(cfg,field,value)
                    with self.assertRaises(ValueError):model.check_config(cfg)
                    setattr(cfg,field,old)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-other'):
                with self.assertRaises(ValueError):model.check_config(cfg)

    def test_staging_budget_rejected_before_cuda_allocation(self):
        import torch
        from nanovllm.engine.cache_transfer import CacheTransferWorkspace
        with self.assertRaises(ValueError):
            CacheTransferWorkspace((2,28,256,8,128),torch.bfloat16,batch_blocks=4,slots=2,
                                   max_bytes=1,device='cuda')


if __name__=='__main__':
    unittest.main()
