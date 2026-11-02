"""CPU protocol regressions, real planner/reservations with fake GPU execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from tests.regression.check_cache_draft import draft_engine,assert_drained
from tests.regression.check_draft_graph import GraphTests
from nanovllm.config import Config
from nanovllm.engine.device_draft import DeviceVerificationItem,DeviceVerificationResult
from nanovllm.engine.execution_plan import ExecutionPlan,ExecutionItem,ExecutionRole
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

def fake_device_engine(rule=lambda t:t%3+1):
    e,d=draft_engine(rule);e.config.draft_device_verification=True
    old=e.model_runner.call
    def call(method,plan,seqs):
        if method!='run_device_verification':return old(method,plan,seqs)
        ticket=plan.items[0]
        assert isinstance(ticket,DeviceVerificationItem)
        proposed=tuple(d.propose(ticket.request_id,seqs[0].token_ids,ticket.count))
        item=ExecutionItem(ticket.request_id,ExecutionRole.VERIFY,(ticket.last_token,)+proposed,
            ticket.start_pos,ticket.block_table,proposed,ticket.logit_positions)
        pred=old('run_plan',ExecutionPlan((item,),plan.step_id),seqs)
        return DeviceVerificationResult(ticket.request_id,proposed,tuple(pred[ticket.request_id]))
    e.model_runner.call=call
    return e,d

class DeviceDraftTests(unittest.TestCase):
    def setUp(self):self.block=Sequence.block_size
    def tearDown(self):Sequence.block_size=self.block
    def params(self,**kw):return SamplingParams(temperature=0,speculative_method='draft_model',speculative_max_draft_tokens=4,**kw)
    def test_default_off_and_no_fake_cpu_ids(self):
        self.assertIs(Config.__dataclass_fields__['draft_device_verification'].default,False)
        item=DeviceVerificationItem(1,7,4,255,(1,2,3))
        self.assertEqual(item.query_len,5);self.assertEqual(item.end_pos,260)
        self.assertEqual(item.logit_positions,(0,1,2,3,4))
        for attr in ['input_token_ids','draft']:
            with self.assertRaisesRegex(RuntimeError,'not CPU materialized'):getattr(item,attr)
        for k in [0,-1,True,1.5]:
            with self.assertRaises(ValueError):DeviceVerificationItem(1,7,k,0,(1,))
    def test_end_to_end_all_partial_rejection(self):
        for rule in [lambda t:t%3+1,lambda t:3,lambda t:99]:
            e,d=fake_device_engine(rule);seq=e.add_request([1,2,3],self.params(max_tokens=19))
            while not e.is_finished():
                e.step();self.assertEqual(d.proposer.cached_tokens,seq.token_ids[:len(d.proposer.cached_tokens)])
            self.assertEqual(seq.completion_token_ids,[1,2,3]*6+[1])
            self.assertTrue(any(p.has_verification for p in e.model_runner.plans))
            self.assertFalse(d.proposer.cached_tokens);assert_drained(self,e.scheduler.block_manager)
    def test_eos_stop_and_length_one(self):
        for eos,kw,expected in [(-1,dict(max_tokens=1),[1]),(2,dict(max_tokens=9),[1,2]),(-1,dict(max_tokens=9,stop='23'),[1,2,3])]:
            e,d=fake_device_engine();e.config.eos=e.scheduler.eos=eos
            result=e.generate([[1,2,3]],self.params(**kw),use_tqdm=False)
            self.assertEqual(result[0]['token_ids'],expected);self.assertFalse(d.proposer.cached_tokens)
            assert_drained(self,e.scheduler.block_manager)
    def test_cancel_and_repeated_request_isolation(self):
        e,d=fake_device_engine();seq=e.add_request([1,2,3],self.params(max_tokens=9));e.step();e.step()
        self.assertTrue(e.cancel_request(seq.seq_id));self.assertFalse(d.proposer.cached_tokens)
        assert_drained(self,e.scheduler.block_manager)
        for prompt in [[2],[3],[1,2,3]]:
            result=e.generate([prompt],self.params(max_tokens=5),use_tqdm=False)
            t=prompt[-1];expected=[]
            for _ in range(5):t=t%3+1;expected.append(t)
            self.assertEqual(result[0]['token_ids'],expected);assert_drained(self,e.scheduler.block_manager)
    def test_target_failure_rolls_back_real_pages(self):
        e,d=fake_device_engine();seq=e.add_request([1,2,3],self.params(max_tokens=9));e.step();old=seq.token_ids.copy()
        e.model_runner.fail_after_write=True
        with self.assertRaisesRegex(RuntimeError,'temporary KV writes'):e.step()
        self.assertEqual(seq.token_ids,old);self.assertFalse(e.scheduler._reservations)
        self.assertFalse(e.scheduler.block_manager._reservations);self.assertFalse(d.proposer.cached_tokens)
        e.cancel_request(seq.seq_id);assert_drained(self,e.scheduler.block_manager)
    def test_planner_upgrade_failure_before_gpu(self):
        e,d=fake_device_engine();seq=e.add_request([1,2,3],self.params(max_tokens=9));e.step();old=seq.token_ids.copy()
        def fail(*a,**kw):raise MemoryError('upgrade failed')
        e.scheduler.block_manager.extend_reservation=fail
        with self.assertRaisesRegex(MemoryError,'upgrade failed'):e.step()
        self.assertEqual(seq.token_ids,old);self.assertFalse(e.scheduler._active_items)
        self.assertFalse(e.scheduler._reservations);self.assertFalse(d.proposer.cached_tokens)
        e.cancel_request(seq.seq_id);assert_drained(self,e.scheduler.block_manager)
    def test_materialization_provenance_and_double_use(self):
        e,d=fake_device_engine();seq=e.add_request([1,2,3],self.params(max_tokens=9));e.step()
        plan=e.scheduler.schedule_plan(allow_verification=False)
        ticket=e.scheduler.upgrade_device_verification(plan,4)
        good=DeviceVerificationResult(seq.seq_id,(2,3,1,2),(2,3,1,2,3))
        for bad in [replace(good,request_id=seq.seq_id+1),DeviceVerificationResult(seq.seq_id,(2,),(2,3))]:
            with self.assertRaises(ValueError):e.scheduler.materialize_device_verification(ticket,bad)
        actual=e.scheduler.materialize_device_verification(ticket,good)
        with self.assertRaises(ValueError):e.scheduler.materialize_device_verification(ticket,good)
        self.assertEqual(actual.items[0].draft,good.draft)
        e.scheduler.rollback_plan();e.cancel_request(seq.seq_id);assert_drained(self,e.scheduler.block_manager)
    def test_device_owner_publication_abort_and_reuse(self):
        r=GraphTests().make_runner();r._pending_device_proposal=None;r._device_work_stream=None
        h=SimpleNamespace(request_id=7,history=(2,5,7),prefix=0,count=4)
        r.request_id=7;r._pending_device_proposal=h;r._device_work_stream=Mock()
        with self.assertRaisesRegex(RuntimeError,'must finish'):r.propose(8,[2],1)
        with self.assertRaisesRegex(RuntimeError,'provenance'):r.finish_device_proposal(object(),[1,2,3,4])
        r.finish_device_proposal(h,[1,2,3,4]);self.assertEqual(r.proposer.cached_tokens,[2,5,7,1,2,3])
        self.assertIsNone(r._pending_device_proposal)
        with self.assertRaisesRegex(RuntimeError,'provenance'):r.finish_device_proposal(h,[1,2,3,4])
        r._pending_device_proposal=h;stream=r._device_work_stream=Mock()
        r.forget(7);stream.synchronize.assert_called_once_with();self.assertFalse(r.proposer.cached_tokens)
        self.assertIsNone(r._pending_device_proposal)

if __name__=='__main__':unittest.main()
