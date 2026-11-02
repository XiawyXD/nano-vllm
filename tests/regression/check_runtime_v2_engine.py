"""Real LLMEngine + PlannedScheduler GPU correctness, not a performance benchmark.

Two tracks stay separate: untouched greedy trajectories, and explicitly injected
sampling/proposals AFTER actual target forward/sampling for protocol edge cases.
Run only through the configured physical-GPU guard (GPU1 by default), with a new
output JSON. No GPU is used by --help.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from tests.regression.check_packed_runtime import (digest, write_new, tensor_digest, logical_kv,
    metadata_snapshot, legacy_decode_snapshot, assert_metadata, compare, provenance)


class FixedDraft:
    """Explicit protocol stimulus, never presented as genuine n-gram quality."""
    def __init__(self, tokens):
        self.tokens, self.calls = tuple(tokens), 0

    def propose(self, history, max_tokens):
        self.calls += 1
        return list(self.tokens[:max_tokens])


def accepted_predictions(draft, predictions):
    """Independent oracle for contiguous greedy acceptance (not a production patch)."""
    count = 0
    while count < len(draft) and draft[count] == predictions[count]:
        count += 1
    return list(draft[:count]) + [predictions[count]], count


def physical_slots(table, start, end, block_size):
    return [table[position//block_size]*block_size + position%block_size for position in range(start,end)]


def format_device_scope(gpu, config, model_dtype):
    """Describe the physical device proven by the guard and the active model/KV modes."""
    physical_index = gpu.get("index")
    uuid = gpu.get("uuid")
    if not physical_index or not uuid:
        raise ValueError("device scope requires the guarded physical GPU index and UUID")
    return (f"GPU{physical_index} uuid={uuid} TP{config['tensor_parallel_size']} "
            f"model_dtype={model_dtype} kv_cache_dtype={config['kv_cache_dtype']}")


def prediction_history(before_tokens, start_pos, query, local_row):
    """History that conditions one logits row; rejected draft histories stay distinct."""
    if not 0<=local_row<len(query) or not 0<=start_pos<=len(before_tokens):
        raise ValueError("invalid query/history coordinate")
    return tuple(before_tokens[:start_pos])+tuple(query[:local_row+1])


def index_logit_histories(steps, request_indices):
    rows={}
    for step in steps:
        key=step.get("logits_tensor_key")
        if key is None:continue
        compact=0
        for item in step["plan"]["items"]:
            request_id=item["request_id"]
            before=step["before"][request_id]["tokens"]
            for local in item["logit_positions"]:
                history=prediction_history(before,item["start_pos"],item["input_token_ids"],local)
                identity=(request_indices[request_id],history)
                if identity in rows:raise AssertionError("same request/history produced duplicate observed logits")
                rows[identity]=(key,compact)
                compact+=1
    return rows


def state_invariants(engine):
    """Compare references to the actual live page tables, including shared prefixes."""
    manager, scheduler = engine.scheduler.block_manager, engine.scheduler
    references = Counter(page for seq in engine._requests.values() for page in seq.block_table)
    for block in manager.blocks:
        if block.ref_count != references[block.block_id]:
            raise AssertionError(f"physical page {block.block_id} reference count disagrees with live requests")
    used = {number for number,count in references.items() if count}
    if used != manager.used_block_ids:
        raise AssertionError("used page set differs from live references")
    free = list(manager.free_block_ids)
    if len(set(free)) != len(free) or used & set(free) or used | set(free) != set(range(len(manager.blocks))):
        raise AssertionError("free/used physical page partition is invalid")
    if scheduler._reservations or scheduler._active_items or manager._reservations:
        raise AssertionError("a completed engine.step left an active reservation")
    for seq in engine._requests.values():
        if len(set(seq.block_table)) != len(seq.block_table):
            raise AssertionError("one request aliases two logical pages to the same physical page")
        if seq.is_finished:
            if seq.block_table:
                raise AssertionError("finished request still holds physical KV pages")
        else:
            if not 0 <= seq.num_cached_tokens <= len(seq):
                raise AssertionError("cached frontier outside committed logical sequence")
            needed = (seq.num_cached_tokens+manager.block_size-1)//manager.block_size
            if len(seq.block_table) != needed:
                raise AssertionError("page table was not trimmed to the committed KV frontier")
    return dict(kv_ref_total=sum(block.ref_count for block in manager.blocks),
                used_blocks=len(used), engine_records=len(engine._requests), reservations=0)


class EngineTrace:
    """Observe REAL run_plan and original sampler, then optionally override named rows.

    The model hook counts eager correctness calls only. CPU copies synchronize and
    must not be used to report throughput or controller timing benefits.
    """
    def __init__(self, engine, raw, tensors, guard_path=None):
        import nanovllm.engine.packed_runner as packed
        self.engine, self.runner, self.packed = engine, engine.model_runner, packed
        self.raw, self.tensors = raw, tensors
        self.guard_path = guard_path
        self.original_run, self.original_sampler = self.runner.run_plan, self.runner.sampler
        self.original_prepare = packed.prepare_plan
        self.original_decode = self.runner.prepare_decode
        self.active, self.policy, self.case, self.index = False, None, "", 0
        self.case_steps, self.case_events, self.last = [], [], None
        def prepare(*args, **kwargs):
            result = self.original_prepare(*args, **kwargs)
            if self.active:
                self.current["metadata"] = metadata_snapshot(result)
            return result
        def prepare_decode(seqs):
            result = self.original_decode(seqs)
            if self.active:
                self.current["metadata"] = legacy_decode_snapshot(*result, seqs)
            return result
        def sample(logits, owners):
            original = self.original_sampler(logits, owners)
            if not self.active:
                return original
            import torch
            if not bool(torch.isfinite(logits).all()):
                raise AssertionError("real model produced non-finite logits")
            original_ids = original.detach().cpu().tolist()
            effective = list(original_ids)
            offsets = Counter()
            for row, seq in enumerate(owners):
                local = offsets[seq.seq_id]
                offsets[seq.seq_id] += 1
                if self.policy is not None:
                    value = self.policy(self.plan, seq, local, original_ids[row])
                    if value is not None:
                        if not 0 <= value < self.engine.config.hf_config.vocab_size:
                            raise AssertionError("injected token is outside real vocabulary")
                        effective[row] = value
            key = f"{self.case}/step{self.index}/logits"
            if self.policy is None:
                self.tensors[key] = logits.detach().cpu().clone()
            self.current.update(original_sampled_ids=original_ids, effective_sampled_ids=effective,
                sampler_owner_ids=[seq.seq_id for seq in owners], injected=self.policy is not None,
                logits_sha256=tensor_digest(logits.detach().cpu()),
                logits_tensor_key=key if self.policy is None else None)
            return original if self.policy is None else torch.tensor(effective,dtype=original.dtype,device=original.device)
        def run(plan, seqs):
            import torch
            self.plan, self.calls = plan, 0
            self.current = dict(plan=asdict(plan), before={seq.seq_id:dict(tokens=list(seq.token_ids),
                completion_tokens=seq.num_completion_tokens,cached=seq.num_cached_tokens) for seq in seqs})
            prefixes = {item.request_id:logical_kv(self.runner,seq,item.start_pos)
                        for item,seq in zip(plan.items,seqs)}
            self.active = True
            try:
                result = self.original_run(plan,seqs)
            finally:
                self.active = False
            assert_metadata(self.current["metadata"], {"plan":plan}, self.runner.block_size)
            if self.calls != 1:
                raise AssertionError(f"real eager packed plan executed {self.calls} transformer calls, expected1")
            if self.current.get("sampler_owner_ids",[]) != self.current["metadata"]["owners"]:
                raise AssertionError("sampler did not receive the plan's compact row owners")
            if set(result) != set(plan.request_ids):
                raise AssertionError("run_plan returned unexpected request ids")
            kv = {}
            for item,seq in zip(plan.items,seqs):
                after = logical_kv(self.runner,seq,item.end_pos)
                if not torch.equal(after[:,:,:item.start_pos],prefixes[item.request_id]):
                    raise AssertionError("actual GPU forward overwrote the committed logical KV prefix")
                kv[item.request_id] = dict(prefix_length=item.start_pos,prefix_exact=True,
                    prefix_sha256=tensor_digest(prefixes[item.request_id]),
                    query_kv_sha256=tensor_digest(after[:,:,item.start_pos:]),
                    reserved_pages=list(item.block_table))
            self.current.update(predictions=result, kv=kv, actual_transformer_calls=self.calls,
                                runner_stats=dict(self.runner.last_plan_stats))
            return result
        def count(module,args):
            if self.active:
                self.calls += 1
        packed.prepare_plan, self.runner.prepare_decode = prepare, prepare_decode
        self.runner.sampler, self.runner.run_plan = sample, run
        self.handle = self.runner.model.register_forward_pre_hook(count)

    def begin(self,name,policy=None):
        self.case,self.policy,self.case_steps,self.case_events = name,policy,[],[]

    def step(self):
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        from nanovllm.engine.execution_plan import ExecutionRole
        ensure_gpu1_available(evidence_path=self.guard_path)
        self.current = None
        outputs,stats = self.engine.step()
        if self.current is None:
            raise AssertionError("LLMEngine.step did not call real ModelRunner.run_plan")
        if stats.forward_calls != 1:
            raise AssertionError("StepStats disagrees with observed one forward")
        post = {}
        for item in self.plan.items:
            seq = self.engine.get_request(item.request_id)
            old = self.current["before"][item.request_id]
            generated = seq.completion_token_ids[old["completion_tokens"]:]
            rows = self.current["predictions"][item.request_id]
            expected,accepted = accepted_predictions(item.draft,rows) if item.role is ExecutionRole.VERIFY else (rows,0)
            if generated != expected[:len(generated)]:
                raise AssertionError("engine did not commit the accepted prefix/correction in order")
            if (rows and not generated) or (not seq.is_finished and generated != expected):
                raise AssertionError("engine silently dropped predictions without a stop condition")
            if seq.token_ids[:len(old["tokens"])] != old["tokens"]:
                raise AssertionError("engine mutated committed token history")
            frontier = item.end_pos if item.role is ExecutionRole.PREFILL else item.start_pos+len(generated)
            if not seq.is_finished and seq.num_cached_tokens != frontier:
                raise AssertionError("actual commit frontier does not match executed/accepted tokens")
            post[item.request_id] = dict(generated=generated,tokens=list(seq.token_ids),
                cached=seq.num_cached_tokens,expected_live_frontier=frontier,
                pages=list(seq.block_table),finished=seq.is_finished,finish_reason=seq.finish_reason,
                accepted_before_stop=accepted)
        self.last = dict(self.current,case=self.case,step=self.index,stats=asdict(stats),after=post,
                         invariants=state_invariants(self.engine),events=[asdict(event) for event in self.engine.step_events()])
        self.case_steps.append(self.last)
        self.case_events.extend(self.last["events"])
        self.raw.write(json.dumps(self.last,ensure_ascii=False)+"\n"); self.raw.flush()
        self.index += 1
        return self.last

    def close(self):
        self.runner.run_plan, self.runner.sampler = self.original_run, self.original_sampler
        self.runner.prepare_decode, self.packed.prepare_plan = self.original_decode, self.original_prepare
        self.handle.remove()


def cleanup(engine):
    for seq in list(engine._requests.values()):
        if not seq.is_finished:
            engine.cancel_request(seq.seq_id)
        engine.forget_request(seq.seq_id)
    state = state_invariants(engine)
    if any(state[key] for key in ("kv_ref_total","used_blocks","engine_records","reservations")):
        raise AssertionError("case cleanup left KV references/engine records")
    if engine._stream_texts or engine._event_times or engine.adaptive_controller._requests:
        raise AssertionError("case cleanup left streaming/controller per-request records")
    return dict(status="PASS",**state)


def parameters(*,spec=False,maximum=24,eos=False,stop=()):
    from nanovllm import SamplingParams
    return SamplingParams(temperature=0,ignore_eos=not eos,max_tokens=maximum,stop=list(stop),
        speculative_ngram_size=3 if spec else 0,speculative_max_draft_tokens=4)


def run_until(trace,predicate,maximum=80):
    for _ in range(maximum):
        if predicate(): return
        trace.step()
    raise AssertionError("bounded engine test did not reach expected state")


def alphabet(tokenizer):
    result = {}
    for text in ("A","B","C","D","X","Y","Z"):
        ids = tokenizer.encode(text,add_special_tokens=False)
        if len(ids)!=1 or tokenizer.decode(ids,clean_up_tokenization_spaces=False)!=text:
            raise RuntimeError(f"protocol fixture requires reversible single token {text}")
        result[text] = ids[0]
    if tokenizer.decode([result["A"],result["B"]],clean_up_tokenization_spaces=False)!="AB":
        raise RuntimeError("tokenizer is not suitable for exact cross-token stop fixture")
    return result


def control_mixed(trace,letters):
    """Real GPU KV + forced responses prove state transitions, never logits accuracy."""
    engine,block = trace.engine,trace.engine.config.kvcache_block_size
    a,b,c,d,x,y,z = (letters[key] for key in "ABCDXYZ")
    spec = engine.add_request(([a,b,c,d]*block)[:block-1],parameters(spec=True,maximum=24))
    ordinary = engine.add_request([y]*31,parameters(maximum=48))
    spec._ngram_proposer = FixedDraft([a,b,c,d])
    verify_rounds = 0
    def policy(plan,seq,row,original):
        if seq.seq_id != spec.seq_id: return y
        item = next(item for item in plan.items if item.request_id==seq.seq_id)
        if not item.draft: return x
        expected = ([z]*item.query_len if verify_rounds==0 else
                    ([a,b,z]+[z]*(item.query_len-3)) if verify_rounds==1 else list(item.draft)+[x])
        return expected[row]
    trace.begin("controlled_mixed_reject_cross_page",policy)
    saw_pd,saw_dv,trimmed,continue_position = False,False,False,None
    for _ in range(40):
        row = trace.step()
        stats=row["stats"]
        saw_pd |= stats["scheduled_prefill"]>0 and stats["query_decode"]>0
        saw_dv |= stats["query_verify"]>0 and stats["query_decode"]>0
        item = next((item for item in trace.plan.items if item.request_id==spec.seq_id),None)
        if item and item.draft:
            post=row["after"][spec.seq_id]
            if verify_rounds==0:
                if item.start_pos!=block-1 or post["generated"]!=[z] or post["cached"]!=block:
                    raise AssertionError("first rejection did not commit only the boundary base query")
                trimmed=len(item.block_table)==2 and len(post["pages"])==1
                if not trimmed: raise AssertionError("rejected cross-page draft allocation was not trimmed")
                continue_position=post["cached"]
            elif verify_rounds==1:
                if item.start_pos!=continue_position or post["generated"]!=[a,b,z]:
                    raise AssertionError("reject continuation used the wrong position/accepted prefix")
            elif verify_rounds==2:
                if post["generated"]!=list(item.draft)+[x]:
                    raise AssertionError("fully accepted draft did not include target bonus")
            verify_rounds += 1
            if verify_rounds==3:
                spec.speculative_ngram_size=0
                break
    if not saw_pd or not saw_dv or verify_rounds!=3:
        raise AssertionError("actual P+D/D+V/three acceptance cases were not all exercised")
    previous=spec.num_completion_tokens
    trace.step();trace.step()
    if spec.num_completion_tokens!=previous+2:
        raise AssertionError("ordinary continuation after rejection did not progress")
    return dict(status="PASS",scope="injected protocol after actual target/sampler",saw_pd=saw_pd,saw_dv=saw_dv,
                reject_trimmed_temporary_page=trimmed,verify_rounds=verify_rounds,cleanup=cleanup(engine))


def control_stops(trace,letters):
    engine=trace.engine
    a,b,c,d,x,y,z=(letters[key] for key in "ABCDXYZ")
    results={}
    for kind in ("cross_token_stop","eos_inside_accepted_draft"):
        seq=engine.add_request([y]*29,parameters(spec=kind.startswith("eos"),maximum=16,
            eos=kind.startswith("eos"),stop=("AB",) if kind.startswith("cross") else ()))
        other=engine.add_request([x]*17,parameters(maximum=24))
        seq._ngram_proposer=FixedDraft([a,engine.scheduler.eos,b,c]) if kind.startswith("eos") else None
        def policy(plan,owner,row,original):
            if owner.seq_id!=seq.seq_id:return y
            item=next(item for item in plan.items if item.request_id==owner.seq_id)
            if item.draft:return (list(item.draft)+[x])[row]
            return x if owner.num_completion_tokens==0 else a
        trace.begin(f"controlled_{kind}",policy)
        run_until(trace,lambda:seq.num_completion_tokens>=1)
        if kind.startswith("cross"):
            trace.step()  # actual A output must remain buffered as a possible stop prefix
            if seq.completion_token_ids!=[x,a]:raise AssertionError("stop fixture failed to emit its first two tokens")
            shown="".join(event["text"] for event in trace.case_events if event["request_id"]==seq.seq_id)
            if shown!="X":raise AssertionError("cross-token stop prefix leaked to stream")
            seq.speculative_ngram_size=3
            seq._ngram_proposer=FixedDraft([b,c,d])
        run_until(trace,lambda:seq.is_finished)
        expected=[x,a,b] if kind.startswith("cross") else [x,a,engine.scheduler.eos]
        if seq.completion_token_ids!=expected or seq.finish_reason!="stop":
            raise AssertionError("EOS/stop did not truncate speculative commit at exact token")
        if kind.startswith("cross"):
            text="".join(event["text"] for event in trace.case_events if event["request_id"]==seq.seq_id)
            if text!="X" or engine.completion_text(seq)!="X":raise AssertionError("stop text leaked/duplicated")
        before=other.num_completion_tokens
        trace.step()
        if other.num_completion_tokens!=before+1:raise AssertionError("one request stopping blocked its batch peer")
        results[kind]=dict(status="PASS",injected=True,generated=expected,cleanup=cleanup(engine))
    return results


def control_cancel_prefix(trace,letters):
    engine,block=trace.engine,trace.engine.config.kvcache_block_size
    x,y=letters["X"],letters["Y"]
    policy=lambda plan,seq,row,original:x
    results={}
    for kind,length in (("waiting",17),("partial_prefill",block*2+3),("running",19)):
        trace.begin(f"cancel_{kind}",policy)
        seq=engine.add_request([y]*length,parameters(maximum=12))
        if kind!="waiting":trace.step()
        if kind=="partial_prefill" and not 0<seq.num_cached_tokens<len(seq):raise AssertionError("partial-prefill cancellation not exercised")
        if kind=="running":trace.step()
        if not engine.cancel_request(seq.seq_id):raise AssertionError("live request cancellation failed")
        results[kind]=dict(status="PASS",cleanup=cleanup(engine))
    prompt=([letters["A"],letters["B"],x,y]*(block+7))[:block+7]
    trace.begin("prefix_first",policy)
    first=engine.add_request(prompt,parameters(maximum=2))
    run_until(trace,lambda:first.is_finished)
    first_pages=list(trace.plan.items[0].block_table)
    cleanup(engine)
    trace.begin("prefix_reuse_shared",policy)
    second=engine.add_request(prompt,parameters(maximum=12))
    trace.step()
    first_item=next(item for item in trace.plan.items if item.request_id==second.seq_id)
    if first_item.start_pos<block or first_item.block_table[0]!=first_pages[0]:
        raise AssertionError("cached full prefix page was not reused by the actual planner")
    third=engine.add_request(prompt,parameters(maximum=12))
    trace.step()
    shared=second.block_table[0]
    if third.block_table[0]!=shared or engine.scheduler.block_manager.blocks[shared].ref_count!=2:
        raise AssertionError("concurrent prefix reuse did not hold two references to one physical page")
    state_invariants(engine)
    results["prefix_reuse"]=dict(status="PASS",reused_page=shared,shared_ref_count=2,
        cached_start=first_item.start_pos,cleanup=cleanup(engine))
    return results


def natural_track(trace,quick):
    engine=trace.engine
    tokenizer=engine.tokenizer
    prompts=["Continue this pattern using the same words: red blue green red blue green red blue green",
        "Write a short Python function that sums the even integers in a list. Explain one edge case."]
    encoded=[tokenizer.encode(text,add_special_tokens=False) for text in prompts]
    result={}
    history_maps={}
    engine.scheduler.block_manager.enable_prefix_caching=False
    for policy in ("off","fixed"):
        engine.config.speculative_policy=policy
        trace.begin(f"actual_greedy_{policy}",None)
        seqs=[engine.add_request(ids,parameters(spec=True,maximum=8 if quick else 24)) for ids in encoded]
        run_until(trace,engine.is_finished,maximum=128)
        history_maps[policy]=index_logit_histories(trace.case_steps,{seq.seq_id:i for i,seq in enumerate(seqs)})
        result[policy]=dict(outputs=[list(seq.completion_token_ids) for seq in seqs],
            steps=[dict(step=row["step"],plan_kind=row["metadata"]["batch_kind"],
                        stats=row["stats"],logits_tensor_key=row.get("logits_tensor_key"),
                        before_histories=row["before"],logits_layout=row["metadata"]["owners"])
                   for row in trace.case_steps],runtime=engine.runtime_metrics(),cleanup=cleanup(engine))
    engine.config.speculative_policy="fixed"
    engine.scheduler.block_manager.enable_prefix_caching=True
    mismatches=[]
    for index,(baseline,candidate) in enumerate(zip(result["off"]["outputs"],result["fixed"]["outputs"])):
        if baseline!=candidate:
            first=next((i for i,(a,b) in enumerate(zip(baseline,candidate)) if a!=b),min(len(baseline),len(candidate)))
            mismatches.append(dict(request=index,first_different_output=first,baseline=baseline,candidate=candidate))
    proposed=sum(row["stats"]["query_verify"] for row in result["fixed"]["steps"])
    numerical=[]
    for identity,(reference_key,reference_row) in history_maps["off"].items():
        candidate=history_maps["fixed"].get(identity)
        if candidate is None:continue  # Never compare logits after histories have diverged.
        key,row=candidate
        difference=compare(trace.tensors[key][row:row+1],trace.tensors[reference_key][reference_row:reference_row+1],logits=True)
        numerical.append(dict(request=identity[0],history=list(identity[1]),comparison=difference,
                              baseline_tensor=reference_key,baseline_row=reference_row,candidate_tensor=key,candidate_row=row))
    numerical_ok=bool(numerical) and all(row["comparison"].get("allclose") and row["comparison"].get("argmax_equal") for row in numerical)
    result.update(status="REVIEW_REQUIRED" if mismatches or not proposed or not numerical_ok else "PASS_OBSERVED_TRAJECTORIES",
        mismatch=mismatches,actual_verification_query_tokens=proposed,
        common_history_logits=numerical,strict_common_history_status="PASS" if numerical_ok else "REVIEW_REQUIRED",
        skipped_nonmatching_histories=len(history_maps["off"])-len(numerical),
        scope="Original sampler/proposer. Strict unchanged thresholds apply only to exact common histories. Other histories are never compared as if equal; no distribution-preservation proof.")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--gpu-memory-utilization",type=float,default=.3)
    parser.add_argument("--linear-backend", choices=("torch", "triton_stable"), default="torch")
    parser.add_argument("--rmsnorm-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--mixed-attention-backend", choices=("split", "varlen"), default="split")
    args=parser.parse_args()
    if args.output.suffix.lower()!=".json" or not 0<args.gpu_memory_utilization<1:
        parser.error("new .json output and 0<gpu-memory-utilization<1 required")
    out=args.output.resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    for path in (out,out.with_suffix(".pt"),out.with_suffix(".events.jsonl"),out.with_suffix(".started.json"),out.with_suffix(".gpu.jsonl")):
        if path.exists():raise FileExistsError(path)
    write_new(out.with_suffix(".started.json"),dict(at=time.time(),pid=os.getpid(),args={**vars(args),"output":str(out)}))
    report=dict(status="RUNNING",device_scope="pending GPU guard",protocol={},real_greedy={},
        numerical_contract="Protocol injection never counts as numerical accuracy; existing strict gates unchanged",
        timing_note="Correctness hooks synchronize/copy and are not performance timing",gpu_executed=False)
    engine=trace=None
    tensors={}
    with out.with_suffix(".events.jsonl").open("x",encoding="utf-8") as raw:
        try:
            from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
            from benchmarks.runtime_v2_soak import source_hashes
            report["gpu_guard"]=ensure_gpu1_available(evidence_path=out.with_suffix(".gpu.jsonl"))
            report["source_sha256"]=source_hashes()
            report["model_config_sha256"]=digest(Path(args.model)/"config.json")
            import torch
            from nanovllm import LLM
            from nanovllm.engine.planned_scheduler import PlannedScheduler
            config=dict(runtime_preset="baseline",batching_mode="mixed",speculative_policy="fixed",metadata_backend="reuse",
                enable_mixed_graph=False,enforce_eager=True,
                mixed_attention_backend=args.mixed_attention_backend,sampler_backend="torch",
                mlp_backend="torch",rmsnorm_backend=args.rmsnorm_backend,linear_backend=args.linear_backend,
                kv_cache_dtype="model",tensor_parallel_size=1,
                max_model_len=2048,max_num_seqs=4,max_num_batched_tokens=128,prefill_chunk_size=64,
                gpu_memory_utilization=args.gpu_memory_utilization,enable_prefix_caching=True)
            report["config"]=config
            engine=LLM(args.model,**config)
            report["gpu_executed"]=True
            if not isinstance(engine.scheduler,PlannedScheduler):raise AssertionError("real PlannedScheduler not active")
            report["device"]=dict(name=torch.cuda.get_device_name(),dtype=str(next(engine.model_runner.model.parameters()).dtype))
            if next(engine.model_runner.model.parameters()).dtype!=torch.bfloat16:
                raise AssertionError("this numeric acceptance scope requires the model BF16 path")
            report["device_scope"] = format_device_scope(
                report["gpu_guard"]["gpu"], config, report["device"]["dtype"])
            report["provenance"]=provenance(args)
            trace=EngineTrace(engine,raw,tensors,out.with_suffix(".gpu.jsonl"))
            letters=alphabet(engine.tokenizer)
            report["real_greedy"]=natural_track(trace,args.quick)
            report["protocol"]["mixed_rejection"]=control_mixed(trace,letters)
            report["protocol"]["stop_eos"]=control_stops(trace,letters)
            report["protocol"]["cancel_prefix"]=control_cancel_prefix(trace,letters)
            report["final_cleanup"]=cleanup(engine)
            report["status"]="REVIEW_REQUIRED" if report["real_greedy"]["status"]=="REVIEW_REQUIRED" else "PASS"
        except BaseException as error:
            report.update(status="FAIL",error=f"{type(error).__name__}: {error}")
        finally:
            if trace:
                try:
                    trace.close()
                except BaseException as error:
                    report.update(status="FAIL",trace_restore_error=f"{type(error).__name__}: {error}")
            if engine:
                try:
                    cleanup(engine)
                except BaseException as error:
                    report.update(status="FAIL",cleanup_error=f"{type(error).__name__}: {error}")
                finally:
                    try:
                        engine.exit()
                    except BaseException as error:
                        report.update(status="FAIL",engine_exit_error=f"{type(error).__name__}: {error}")
            if tensors:
                import torch
                torch.save(tensors,out.with_suffix(".pt"))
                report["tensor_sha256"]=digest(out.with_suffix(".pt"))
            raw.flush()
            report["events_sha256"]=digest(out.with_suffix(".events.jsonl"))
            if out.with_suffix(".gpu.jsonl").exists():
                report["gpu_observations_sha256"]=digest(out.with_suffix(".gpu.jsonl"))
            report["finished_at"]=time.time()
            write_new(out,report)
    print(json.dumps(dict(status=report["status"],output=str(out),gpu_executed=report["gpu_executed"])))
    return 0 if report["status"]=="PASS" else 2 if report["status"]=="REVIEW_REQUIRED" else 1


if __name__=="__main__":
    raise SystemExit(main())
