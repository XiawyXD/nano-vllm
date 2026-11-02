"""CPU utility checks only; cannot replace check_runtime_v2_engine on a real GPU."""
from __future__ import annotations
import json
from types import SimpleNamespace as NS
import unittest

from tests.regression.check_runtime_v2_engine import (FixedDraft,accepted_predictions,physical_slots,
                                               prediction_history,index_logit_histories,state_invariants,
                                               format_device_scope)
from tests.regression.check_packed_runtime import make_specs


class Checks(unittest.TestCase):
    def test_greedy_acceptance_first_middle_all(self):
        self.assertEqual(accepted_predictions([1,2],[9,2,3]),([9],0))
        self.assertEqual(accepted_predictions([1,2],[1,9,3]),([1,9],1))
        self.assertEqual(accepted_predictions([1,2],[1,2,3]),([1,2,3],2))

    def test_proposer_respects_limit(self):
        draft=FixedDraft([1,2,3,4])
        self.assertEqual(draft.propose([9],2),[1,2])
        self.assertEqual(draft.calls,1)

    def test_cross_page_slots_are_not_contiguous_physically(self):
        self.assertEqual(physical_slots([7,2],2,6,4),[30,31,8,9])

    def test_device_scope_comes_from_guarded_physical_gpu(self):
        scope=format_device_scope(
            {"index":"2","uuid":"GPU-fae44001-5277-3d60-bd3a-0e634ed0e6a8"},
            {"tensor_parallel_size":1,"kv_cache_dtype":"model"},
            "torch.bfloat16",
        )
        self.assertIn("GPU2",scope)
        self.assertIn("GPU-fae44001-5277-3d60-bd3a-0e634ed0e6a8",scope)
        self.assertIn("TP1",scope)
        self.assertNotIn("GPU1",scope)

    def test_device_scope_rejects_unverified_gpu(self):
        with self.assertRaises(ValueError):
            format_device_scope({}, {"tensor_parallel_size":1,"kv_cache_dtype":"model"},
                                "torch.bfloat16")

    def test_uniform_varlen_option_preserves_mixed_case_roles(self):
        specs=make_specs(False,"varlen")
        backends={spec["name"]:spec["backend"] for spec in specs}
        self.assertEqual(backends,{"pd":"varlen","dv":"varlen",
                                   "pure_d_varlen":"varlen","pd_partial":"varlen"})
        with self.assertRaises(ValueError):make_specs(False,"decode_only")

    def test_prediction_history_uses_only_query_prefix(self):
        self.assertEqual(prediction_history([10,11,12],2,[12,20,21],0),(10,11,12))
        self.assertEqual(prediction_history([10,11,12],2,[12,20,21],1),(10,11,12,20))
        self.assertEqual(prediction_history([1,2,3,4],2,[3,4],1),(1,2,3,4))
        with self.assertRaises(ValueError):prediction_history([1],2,[3],0)

    def test_compact_logit_rows_ignore_partial_prefill(self):
        step=dict(logits_tensor_key="test",before={11:dict(tokens=[1,2,3]),12:dict(tokens=[8,9])},
            plan=dict(items=[dict(request_id=11,start_pos=0,input_token_ids=[1,2],logit_positions=[]),
                             dict(request_id=12,start_pos=1,input_token_ids=[9],logit_positions=[0])]))
        result=index_logit_histories([step],{11:0,12:1})
        self.assertEqual(result,{(1,(8,9)):("test",0)})
        with self.assertRaises(AssertionError):index_logit_histories([step,step],{11:0,12:1})

    def make_engine(self):
        blocks=[NS(block_id=0,ref_count=2),NS(block_id=1,ref_count=0)]
        manager=NS(blocks=blocks,used_block_ids={0},free_block_ids=[1],block_size=4,_reservations={})
        sequences={key:NS(block_table=[0],is_finished=False,num_cached_tokens=4) for key in (11,12)}
        class Sequence:
            def __init__(self,obj): self.__dict__.update(obj.__dict__)
            def __len__(self):return 5
        sequences={key:Sequence(seq) for key,seq in sequences.items()}
        return NS(scheduler=NS(block_manager=manager,_reservations={},_active_items={}),_requests=sequences)

    def test_shared_prefix_references_match_live_tables(self):
        engine=self.make_engine()
        self.assertEqual(state_invariants(engine)["kv_ref_total"],2)
        engine.scheduler.block_manager.blocks[0].ref_count=1
        with self.assertRaises(AssertionError):state_invariants(engine)

    def test_reservation_and_free_partition_fail_closed(self):
        engine=self.make_engine()
        engine.scheduler._reservations[1]=object()
        with self.assertRaises(AssertionError):state_invariants(engine)
        engine=self.make_engine()
        engine.scheduler.block_manager.free_block_ids=[0,1]
        with self.assertRaises(AssertionError):state_invariants(engine)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    print(json.dumps(dict(status="PASS" if result.wasSuccessful() else "FAIL",tests=result.testsRun,
        scope="CPU utility checks only; no real LLM/PlannedScheduler GPU acceptance")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
