"""阶段5：CPU热路径诊断与完全无debug hooks的端到端消融，二者进程隔离。

python -m benchmarks.runtime_v2_profile run --model MODEL --output NEW_DIR --quick
python -m benchmarks.runtime_v2_profile self-test

正式长测由runtime_v2_suite启动，负责完整Linux进程组回收。timing worker不安装
Python monkeypatch/forward hook，不开启cProfile/torch.profiler，也不逐step插入同步。
diagnostic worker复用profile_runtime.py的作用域包裹方法：只有诊断才能安装标签。
嵌套host wall包含等待/子调用，不能相加或改名GPU时间；设备耗时只读可选CUDA trace。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import cProfile
import functools
import io
import json
import math
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import time

from benchmarks.runtime_v2_performance import (
    ROOT, build_workloads, cpu_resources, environment, load, metric_deltas,
    object_sha, sha, snapshot, terminate_child, write_new,
)
from benchmarks.release_matrix import provenance

VARIANTS = ("dynamic_torch", "reuse_torch", "dynamic_auto", "reuse_auto")


def configurations(args, cases) -> dict:
    """2×2消融；除metadata/MLP两个开关外，输入、调度、Graph和采样完全相同。"""
    base = dict(runtime_preset="baseline", batching_mode="mixed", speculative_policy="off",
        enable_mixed_graph=False, enforce_eager=False, tensor_parallel_size=1,
        sampler_backend="torch", rmsnorm_backend="torch", decode_attention_backend="flash",
        mixed_attention_backend="split", kv_cache_dtype="model", enable_prefix_caching=False,
        max_model_len=max(len(job["prompt_token_ids"]) + case["output_tokens"] for case in cases for job in case["jobs"]),
        max_num_batched_tokens=args.token_budget, prefill_chunk_size=args.chunk_size,
        max_num_seqs=max(case["batch"] for case in cases), max_prefill_steps=1,
        gpu_memory_utilization=args.gpu_memory_utilization)
    return {name: dict(base, metadata_backend=name.split("_")[0], mlp_backend=name.split("_")[1]) for name in VARIANTS}


def make_cases(tokenizer, args) -> list[dict]:
    """D-heavy保持固定batch；P-heavy输出1token，仍是完整首token流水线，不伪称GEMM时间。"""
    from types import SimpleNamespace
    specifications = [(f"d{batch}", "decode_heavy", batch, args.decode_prompt_tokens, args.decode_output_tokens,
                       "code" if batch in (8,32) else "english") for batch in args.decode_batches]
    prefill = [(1,128)] if args.quick else [(1,128),(1,512),(8,256)]
    specifications += [(f"p{batch}_{length}","prefill_heavy",batch,length,1,"chinese" if batch == 1 else "code")
                       for batch,length in prefill]
    cases=[]
    for name,kind,batch,length,output_tokens,dataset in specifications:
        helper=SimpleNamespace(datasets=[dataset],scenarios=["short"],concurrencies=[batch],requests_per_case=batch,
            short_tokens=length,long_tokens=length,output_tokens=output_tokens,arrival_interval=0.0)
        case=build_workloads(tokenizer,helper)[0]
        case.update(id=name,kind=kind,batch=batch,output_tokens=output_tokens)
        cases.append(case)
    return cases


def validate_fixed_decode_shapes(args, cases):
    """先让同批全部短prompt一起完成P，再进入固定B的D；否则D32可能实际只跑D8。"""
    for case in cases:
        if case["kind"] != "decode_heavy":
            continue
        lengths=[len(row["prompt_token_ids"]) for row in case["jobs"]]
        if max(lengths)>args.chunk_size or sum(lengths)>args.token_budget:
            raise ValueError(f"{case['id']}: fixed decode batch requires chunk-size >= {max(lengths)} "
                             f"and token-budget >= {sum(lengths)} for one complete prefill round")


def check_decode_schedule(case, delta):
    """由生产StepStats聚合核实形状；只检查整轮计数，不向正式路径加逐step hook。"""
    if case["kind"] != "decode_heavy":
        return None
    expected={"steps":case["output_tokens"],"batch_P_PREFILL":1,
        "batch_D_NORMAL":case["output_tokens"]-1,"batch_PD_PACKED":0,
        "query_decode":case["batch"]*(case["output_tokens"]-1),"query_verify":0,
        "scheduled_prefill":sum(len(row["prompt_token_ids"]) for row in case["jobs"])}
    mismatch={key:dict(expected=value,actual=delta.get(key,0)) for key,value in expected.items()
              if delta.get(key,0)!=value}
    return dict(status="FAIL" if mismatch else "PASS",expected=expected,mismatch=mismatch)


@contextmanager
def scoped_labels(targets, record_function=None):
    """诊断专用函数包装；无逐调用CUDA同步，finally精确恢复实例/模块原属性布局。

    record_function由torch.profiler提供；cProfile-only时不导入/开启设备Profiler。
    包裹scheduler和其内部page操作会形成嵌套，host_ms是inclusive而非可相加的分解。
    """
    totals=defaultdict(lambda:dict(calls=0,host_wall_ms=0.0))
    originals=[]
    try:
        for obj,name,label in targets:
            old=getattr(obj,name)
            had_local=name in getattr(obj,"__dict__",{})
            def factory(original,tag):
                @functools.wraps(original)
                def measured(*args,**kwargs):
                    started=time.perf_counter()
                    context=record_function("runtime_v2::"+tag) if record_function else nullcontext()
                    try:
                        with context:
                            return original(*args,**kwargs)
                    finally:
                        totals[tag]["calls"]+=1
                        totals[tag]["host_wall_ms"]+=(time.perf_counter()-started)*1000
                return measured
            setattr(obj,name,factory(old,label))
            originals.append((obj,name,old,had_local))
        yield totals
    finally:
        for obj,name,old,had_local in reversed(originals):
            if had_local:
                setattr(obj,name,old)
            else:
                delattr(obj,name)


def diagnostic_targets(llm):
    """按当前真实调用路径标注packing/page/H2D/sampler/text，关键接口缺失不静默忽略。"""
    from nanovllm.engine import packed_runner
    runner=llm.model_runner
    scheduler=llm.scheduler
    manager=scheduler.block_manager
    pool=runner.metadata_buffers
    targets=[(scheduler,"schedule_plan","scheduler_and_page_reservation"),
        (packed_runner,"prepare_plan","input_packing_page_tables_and_h2d_enqueue"),
        (pool,"begin","metadata_frame_begin_possible_host_wait"),
        (pool,"tensor","metadata_fill_and_h2d_enqueue"),(pool,"end","metadata_event_record"),
        (runner.model,"forward","transformer_host_dispatch"),(runner.model,"compute_logits","lm_head_host_dispatch"),
        (runner.sampler,"forward","sampler_host_dispatch_without_outer_tolist"),
        (scheduler,"commit_item","commit_tokens"),(llm,"_decode_completion","tokenizer_decode"),
        (llm,"completion_text","completion_text"),(llm,"_stream_delta","stream_delta"),
        (llm,"_make_step_events","event_construction")]
    for name in ("probe_prefix","required_new_blocks","reserve_to","extend_reservation","commit","rollback","deallocate"):
        targets.append((manager,name,"page_"+name))
    missing=[label for obj,name,label in targets if not callable(getattr(obj,name,None))]
    if missing:
        raise RuntimeError(f"diagnostic integration changed: missing {missing}")
    return targets


def assert_no_debug_hooks(llm) -> dict:
    """正式计时进程拒绝外部Python trace/profile及注册forward hooks；不自行安装观测hook。"""
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("timing worker is running under Python trace/profile")
    count=0
    for module in llm.model_runner.model.modules():
        count+=len(getattr(module,"_forward_hooks",{}))+len(getattr(module,"_forward_pre_hooks",{}))
    if count:
        raise RuntimeError("registered debug forward hooks would contaminate end-to-end timing")
    return dict(python_profiler=False,python_trace=False,registered_forward_hooks=0,
                scoped_labels_installed=False,torch_profiler_enabled=False)


def generate_once(llm,case,*,diagnostic=False,params_factory=None,device=None,clock=time.perf_counter) -> dict:
    """generate完整调用只在边界同步；内部tolist本身会等待，不能另逐轮插入synchronize。"""
    if params_factory is None:
        from nanovllm import SamplingParams
        params_factory=SamplingParams
    if device is None:
        import torch
        device=torch.cuda
    params=params_factory(temperature=0,ignore_eos=True,max_tokens=case["output_tokens"],speculative_ngram_size=0)
    before=snapshot(llm)
    cpu_before=cpu_resources()
    outputs,error=[],None
    device.synchronize()
    device.reset_peak_memory_stats()
    began=clock()
    try:
        outputs=llm.generate([row["prompt_token_ids"] for row in case["jobs"]],params,use_tqdm=False)
        device.synchronize()
        if len(outputs)!=case["batch"] or any(len(row["token_ids"])!=case["output_tokens"] for row in outputs):
            raise AssertionError("fixed-length greedy generation returned wrong batch/token count")
    except BaseException as exception:
        error=f"{type(exception).__name__}: {exception}"
    seconds=max(clock()-began,1e-12)
    after=snapshot(llm)
    manager=llm.scheduler.block_manager
    refs=sum(block.ref_count for block in manager.blocks)
    if refs or manager.used_block_ids or getattr(llm,"_requests",{}):
        error=error or "generate did not release request records/KV references"
    count=sum(len(row["token_ids"]) for row in outputs)
    runtime_left,runtime_right=before.get("runtime",{}),after.get("runtime",{})
    runtime_delta={key:runtime_right.get(key,0)-runtime_left.get(key,0) for key in runtime_right}
    shape_check=check_decode_schedule(case,runtime_delta)
    if shape_check and shape_check["status"]!="PASS":
        error=error or "observed scheduling did not preserve the requested fixed decode batch"
    memory={}
    for key in ("memory_allocated","memory_reserved","max_memory_allocated","max_memory_reserved"):
        try: memory[key]=getattr(device,key)()
        except Exception: memory[key]=None
    return dict(status="FAIL" if error else "PASS",error=error,case=case["id"],kind=case["kind"],batch=case["batch"],
        expected_generated_tokens=case["batch"]*case["output_tokens"],generated_tokens=count,
        seconds=seconds,tokens_per_second=count/seconds,outputs=outputs,diagnostic=diagnostic,
        timing_interpretation="Instrumented diagnostic; NOT a performance result" if diagnostic else "Uninstrumented end-to-end generate including initial prefill, sampling waits, event/text decoding and result assembly",
        runtime_before=before,runtime_after=after,runtime_delta=runtime_delta,
        fixed_decode_schedule_check=shape_check,
        graph_and_acceptance_delta=metric_deltas(before,after),
        cpu_resources_before=cpu_before,cpu_resources_after=cpu_resources(),gpu_memory=memory,
        used_blocks_after=len(manager.used_block_ids),kv_ref_total_after=refs,
        timing_note="host_prepare_ms/model_enqueue_ms/sample_and_wait_ms are CPU wall scopes. sample_and_wait includes waiting for earlier GPU work; none is standalone GPU time.")


def save_cprofile(profile,path):
    """保存原始.prof、完整函数表和按累计耗时排序文本，便于定位Python热点而非只截前几项。"""
    binary=path.with_suffix(".prof")
    if binary.exists(): raise FileExistsError(binary)
    profile.dump_stats(str(binary))
    stream=io.StringIO()
    stats=pstats.Stats(profile,stream=stream).sort_stats("cumulative")
    stats.print_stats()
    text_path=path.with_suffix(".txt")
    with text_path.open("x",encoding="utf-8") as file: file.write(stream.getvalue())
    rows=[]
    for (filename,line,function),(primitive,total,self_seconds,cumulative_seconds,callers) in stats.stats.items():
        rows.append(dict(file=filename,line=line,function=function,primitive_calls=primitive,total_calls=total,
            self_host_seconds=self_seconds,cumulative_host_seconds=cumulative_seconds,
            callers={repr(key):value for key,value in callers.items()}))
    write_new(path.with_suffix(".json"),dict(scope="cProfile CPU call timings, including blocking native calls; not GPU kernel time",
        functions=sorted(rows,key=lambda row:row["cumulative_host_seconds"],reverse=True)))
    return dict(raw=binary.name,raw_sha256=sha(binary),table=path.with_suffix(".json").name,text=text_path.name)


def torch_diagnostic(llm,case,work,mode):
    """独立于cProfile的torch.profiler诊断；只有CUDA活动事件才能用于设备时间归因。"""
    import torch
    activities=[torch.profiler.ProfilerActivity.CPU]
    if mode=="cuda": activities.append(torch.profiler.ProfilerActivity.CUDA)
    with scoped_labels(diagnostic_targets(llm),torch.profiler.record_function) as stages:
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=True) as profile:
            result=generate_once(llm,case,diagnostic=True)
    trace=work / f"{case['id']}.torch_{mode}.trace.json"
    if trace.exists(): raise FileExistsError(trace)
    profile.export_chrome_trace(str(trace))
    operations=[]
    for op in profile.key_averages():
        operations.append(dict(name=op.key,calls=op.count,self_cpu_us=op.self_cpu_time_total,
            cpu_total_us=op.cpu_time_total,self_device_us=getattr(op,"self_device_time_total",None),
            device_total_us=getattr(op,"device_time_total",None)))
    device_events=sum("cuda" in str(event.device_type).lower() for event in profile.events())
    if mode=="cuda" and device_events==0:
        result.update(status="FAIL",error="CUDA profiler requested but no CUDA device events were captured")
    write_new(work / f"{case['id']}.torch_{mode}.json",dict(result=result,host_scopes=dict(stages),operations=operations,
        trace_file=trace.name,trace_sha256=sha(trace),cuda_device_event_count=device_events,
        interpretation="Nested host scopes overlap. Memcpy device duration and kernel execution come from CUDA trace only; CPU-only mode cannot infer them. Graph replay bypasses Python layer calls."))
    return result


def prepare(args):
    """冻结性能语料、四组合配置和环境；不借用质量holdout，不改自动MLP白名单。"""
    from transformers import AutoTokenizer
    output=args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    cases=make_cases(tokenizer,args)
    validate_fixed_decode_shapes(args,cases)
    configs=configurations(args,cases)
    selected=args.profile_cases or (["d8","p1_128"] if args.quick else ["d8","p8_256"])
    if not set(selected)<={case["id"] for case in cases}: raise ValueError("profile-cases must exist in frozen matrix")
    env=environment()
    model=Path(args.model).resolve()
    manifest=dict(schema=1,model=str(model),cases_sha256=object_sha(cases),configs=configs,config_sha256=object_sha(configs),
        provenance=provenance(),environment=env,environment_sha256=object_sha(env),profile_cases=selected,
        model_sha256={path.name:sha(path) for path in sorted(model.iterdir()) if path.is_file() and path.suffix in {".json",".safetensors",".model",".txt",".tiktoken"}},
        args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        scope="Metadata dynamic/reuse x MLP Torch/auto end-to-end ablation. Auto remains the current offline whitelist; this script does not tune or extend it.",
        numerical_scope="No fixed-history strict logits check in this script. Use check_tensorcore_model and runtime_v2 numerical/quality tools. Generated-token divergence stays visible, not treated as proof of a bug or ignored.",
        diagnostics_scope="Separate worker processes from formal timing; cProfile scopes are nested host wall, torch.profiler CUDA events are device evidence. Model initialization/JIT/ordinary decode Graph capture is reported separately.")
    output.mkdir(parents=True)
    write_new(output/"cases.json",cases)
    write_new(output/"profile_manifest.json",manifest)
    return output


def load_prepared(directory):
    manifest=load(directory/"profile_manifest.json")
    cases=load(directory/"cases.json")
    if object_sha(cases)!=manifest["cases_sha256"] or object_sha(manifest["configs"])!=manifest["config_sha256"]:
        raise RuntimeError("frozen workload/config changed")
    if provenance()["source_sha256"]!=manifest["provenance"]["source_sha256"] or object_sha(environment())!=manifest["environment_sha256"]:
        raise RuntimeError("source/environment changed")
    for name,expected in manifest["model_sha256"].items():
        if sha(Path(manifest["model"])/name)!=expected: raise RuntimeError(f"model file changed: {name}")
    return manifest,cases


def worker(directory,variant,phase,round_id):
    """timing与diagnostic用不同进程：诊断包装不可能污染正式五轮计时模型。"""
    manifest,cases=load_prepared(directory)
    work=directory/phase/f"round_{round_id:02d}"/variant
    work.mkdir(parents=True,exist_ok=False)
    write_new(work/"started.json",dict(event="WORKER_STARTED",variant=variant,phase=phase,round=round_id))
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    ensure_gpu1_available(evidence_path=work/"gpu_observations.jsonl")
    import torch
    from nanovllm import LLM
    report=dict(status="RUNNING",variant=variant,phase=phase,round=round_id,files=[],
        manifest_sha256=sha(directory/"profile_manifest.json"),device=dict(name=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),cuda=torch.version.cuda))
    llm=None
    def run_guarded(case,kind,diagnostic=False):
        ensure_gpu1_available(evidence_path=work/"gpu_observations.jsonl")
        result=generate_once(llm,case,diagnostic=diagnostic)
        result.update(variant=variant,round=round_id,phase=kind)
        path=work/f"{case['id']}.{kind}.json"
        write_new(path,result)
        report["files"].append(dict(path=path.relative_to(directory).as_posix(),sha256=sha(path),kind=kind))
        if result["status"]!="PASS": raise RuntimeError(result["error"])
        return result
    try:
        with torch.inference_mode():
            began=time.perf_counter()
            llm=LLM(manifest["model"],**manifest["configs"][variant])
            report["initialization_seconds_including_graph_capture_and_warmup"]=time.perf_counter()-began
            report["effective_runtime"]=llm.config.runtime_summary()
            mlps=[layer.mlp for layer in llm.model_runner.model.model.layers]
            if not all(mlp.mlp_backend==manifest["configs"][variant]["mlp_backend"] for mlp in mlps):
                raise AssertionError("MLP backend did not reach model layers")
            first=mlps[0]
            report["model_mlp"]=dict(dtype=str(first.gate_up_proj.weight.dtype),gate_up_shape=list(first.gate_up_proj.weight.shape),
                configured_backend=first.mlp_backend,device_capability=first.mlp_device_capability,
                routing_note="Static config/shape evidence only, not counted Python forwards. Timing has no debug hook; CUDA diagnostic trace may reveal fused kernels. Current auto whitelist fallback is retained for all other shapes.")
            if phase=="timing":
                report["no_debug_instrumentation"]=assert_no_debug_hooks(llm)
                for case in cases:
                    run_guarded(case,"first_use")
                    for number in range(manifest["args"]["warmups"]): run_guarded(case,f"warmup_{number}")
                    assert_no_debug_hooks(llm)
                    run_guarded(case,"steady")
            else:
                for original in cases:
                    if original["id"] not in manifest["profile_cases"]: continue
                    case=dict(original,output_tokens=min(original["output_tokens"],manifest["args"]["profile_output_tokens"]))
                    run_guarded(case,"diagnostic_warmup",diagnostic=True)
                    ensure_gpu1_available(evidence_path=work/"gpu_observations.jsonl")
                    profile=cProfile.Profile()
                    with scoped_labels(diagnostic_targets(llm)) as stages:
                        profile.enable()
                        try: result=generate_once(llm,case,diagnostic=True)
                        finally: profile.disable()
                    paths=save_cprofile(profile,work/f"{case['id']}.cprofile")
                    write_new(work/f"{case['id']}.cprofile_result.json",dict(result=result,host_scopes=dict(stages),files=paths,
                        note="host_wall_ms includes children and GPU wait; do not sum nested stages or label them device time. Sampler.forward excludes outer .tolist; runtime_delta.sample_and_wait_ms includes it."))
                    if result["status"]!="PASS": raise RuntimeError(result["error"])
                    if manifest["args"]["torch_profiler"]!="none":
                        ensure_gpu1_available(evidence_path=work/"gpu_observations.jsonl")
                        traced=torch_diagnostic(llm,case,work,manifest["args"]["torch_profiler"])
                        if traced["status"]!="PASS": raise RuntimeError(traced["error"])
            report["status"]="PASS_EXECUTION_ONLY"
    except BaseException as error:
        report.update(status="FAIL",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if llm is not None: llm.exit()
        except Exception as error:
            report.update(status="FAIL",cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["artifact_sha256"]={path.relative_to(work).as_posix():sha(path) for path in sorted(work.iterdir()) if path.is_file()}
            write_new(work/"report.json",report)


def aggregate(directory,failures):
    """只用无hooks的steady算倍率；诊断耗时、微kernel速度与端到端速度保持三者独立。"""
    manifest=load(directory/"profile_manifest.json")
    cases=load(directory/"cases.json")
    trials,workers,diagnostics=[],[],[]
    for path in sorted(directory.glob("*/round_*/*/report.json")):
        report=load(path)
        if report["manifest_sha256"]!=sha(directory/"profile_manifest.json"): raise RuntimeError("worker manifest changed")
        for relative,digest in report["artifact_sha256"].items():
            if sha(path.parent/relative)!=digest: raise RuntimeError("profile artifact changed")
        workers.append(dict(path=path.relative_to(directory).as_posix(),status=report["status"],phase=report["phase"]))
        for row in report["files"]:
            if sha(directory/row["path"])!=row["sha256"]: raise RuntimeError("trial changed")
            trial=load(directory/row["path"])
            if row["kind"]=="steady":
                if trial["diagnostic"]: raise RuntimeError("diagnostic result cannot enter timing aggregate")
                trials.append(trial)
        if report["phase"]=="diagnostic": diagnostics.append(dict(variant=report["variant"],directory=path.parent.relative_to(directory).as_posix()))
    groups={}
    for trial in trials: groups.setdefault((trial["case"],trial["variant"]),[]).append(trial)
    rows={}
    for (case,variant),items in groups.items():
        times=[row["seconds"] for row in items]
        rows[f"{case}/{variant}"]=dict(case=case,variant=variant,rounds=len(items),seconds_by_round=times,
            median_seconds=statistics.median(times),minimum_seconds=min(times),maximum_seconds=max(times),
            generated_tokens_per_trial=items[0]["generated_tokens"],
            median_tokens_per_second=statistics.median(row["tokens_per_second"] for row in items))
    comparisons={}
    pairs=(("metadata_torch","dynamic_torch","reuse_torch"),("metadata_auto","dynamic_auto","reuse_auto"),
           ("mlp_dynamic","dynamic_torch","dynamic_auto"),("mlp_reuse","reuse_torch","reuse_auto"),
           ("combined","dynamic_torch","reuse_auto"))
    for case in cases:
        for label,baseline,candidate in pairs:
            left,right=rows.get(f"{case['id']}/{baseline}"),rows.get(f"{case['id']}/{candidate}")
            if left and right:
                factor=left["median_seconds"]/right["median_seconds"]
                comparisons[f"{case['id']}/{label}"]=dict(reference=baseline,candidate=candidate,
                    end_to_end_speedup=factor,regression=factor<1,rounds=min(left["rounds"],right["rounds"]),
                    meaning="Entire generate wall ratio, not isolated MLP/GEMM kernel speedup")
    baseline={(row["case"],row["round"]):row for row in trials if row["variant"]=="dynamic_torch"}
    differences=[]
    for trial in trials:
        reference=baseline.get((trial["case"],trial["round"]))
        if reference and [r["token_ids"] for r in reference["outputs"]]!=[r["token_ids"] for r in trial["outputs"]]:
            differences.append(dict(case=trial["case"],round=trial["round"],variant=trial["variant"]))
    args=manifest["args"]
    timing_expected=args["rounds"]*len(VARIANTS) if args["mode"] in ("timing","both") else 0
    diagnostics_expected=len(VARIANTS) if args["mode"] in ("diagnostic","both") else 0
    execution_ok=(not failures and len(workers)==timing_expected+diagnostics_expected
        and len(trials)==timing_expected*len(cases) and all(row["status"]=="PASS_EXECUTION_ONLY" for row in workers))
    full=(not args["quick"] and args["rounds"]>=5 and set(args["decode_batches"])=={1,8,16,32}
          and args["mode"] in ("timing","both") and args["warmups"]>=1
          and args["decode_output_tokens"]>=64 and args["decode_prompt_tokens"]>=64)
    status="FAIL" if not execution_ok else "REVIEW_REQUIRED" if differences else "PASS_MEASUREMENT_ONLY" if full else "NOT_RUN"
    return dict(status=status,execution_status="PASS" if execution_ok else "FAIL",measurement_scope="FULL" if full else "NOT_EVALUATED",
        uninstrumented_aggregate=rows,comparisons=comparisons,diagnostic_directories=diagnostics,workers=workers,failures=failures,
        observed_generation_differences=differences,
        numerical_note="Strict fixed-history logits remain a separate required gate; matching generated sequences alone is not proof of numerical equivalence.",
        interpretation="Negative speedups/regressions remain in tables. PASS_MEASUREMENT_ONLY means the requested measurement completed, not that auto/reuse won or that release acceptance passed.")


def run(args):
    directory=prepare(args)
    failures,process=[],None
    commands=[]
    if args.mode in ("timing","both"):
        for round_id in range(args.rounds):
            order=VARIANTS[round_id%4:]+VARIANTS[:round_id%4]
            commands.extend(("timing",round_id,variant) for variant in order)
    if args.mode in ("diagnostic","both"):
        commands.extend(("diagnostic",0,variant) for variant in VARIANTS)
    try:
        for phase,round_id,variant in commands:
            from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
            ensure_gpu1_available(evidence_path=directory/"gpu_observations.jsonl")
            command=[sys.executable,"-m","benchmarks.runtime_v2_profile","worker","--prepared",str(directory),
                     "--variant",variant,"--phase",phase,"--round",str(round_id)]
            with (directory/f"{phase}_{round_id:02d}_{variant}.log").open("x",encoding="utf-8") as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write_new(directory/f"{phase}_{round_id:02d}_{variant}_process.json",dict(pid=process.pid,command=command,event="STARTED"))
                try: code=process.wait(timeout=args.worker_timeout)
                except subprocess.TimeoutExpired:
                    terminate_child(process)
                    code=124
                if code:
                    failures.append(dict(phase=phase,round=round_id,variant=variant,exit_code=code))
                    raise RuntimeError("worker failed; stopping new experiments")
            process=None
    except BaseException as error:
        failures.append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        if process is not None: terminate_child(process)
        try: report=aggregate(directory,failures)
        except Exception as error:
            report=dict(status="FAIL",execution_status="FAIL",measurement_scope="NOT_EVALUATED",failures=failures,
                        aggregation_error=f"{type(error).__name__}: {error}",raw_evidence="Preserved unchanged in output directory")
        write_new(directory/"summary.json",report)
        print(json.dumps(dict(status=report["status"],output=str(directory),scope=report["measurement_scope"])))
    return 0 if report["status"]=="PASS_MEASUREMENT_ONLY" else 1 if report["status"]=="FAIL" else 2


def self_test():
    """仅CPU fake：证明wrapper恢复、无hooks门槛、同步次数与错误记录；不代表GPU计时通过。"""
    from types import SimpleNamespace
    from unittest.mock import patch
    import unittest
    class Tests(unittest.TestCase):
        def test_scoped_restore_after_exception(self):
            class Subject:
                def call(self): raise ValueError("expected")
            subject=Subject()
            original=subject.call.__func__
            with scoped_labels([(subject,"call","target")]) as counters:
                with self.assertRaises(ValueError): subject.call()
            self.assertEqual(subject.call.__func__,original)
            self.assertNotIn("call",subject.__dict__)
            self.assertEqual(counters["target"]["calls"],1)
        def test_nested_counters_do_not_drop_calls(self):
            class Subject:
                def outer(self): return self.inner()+1
                def inner(self): return 3
            subject=Subject()
            with scoped_labels([(subject,"outer","outer"),(subject,"inner","inner")]) as totals:
                self.assertEqual(subject.outer(),4)
            self.assertGreaterEqual(totals["outer"]["host_wall_ms"],totals["inner"]["host_wall_ms"])
        def test_hook_gate(self):
            module=SimpleNamespace(_forward_hooks={},_forward_pre_hooks={})
            llm=SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(modules=lambda:[module])))
            assert_no_debug_hooks(llm)
            module._forward_hooks[1]=object()
            with self.assertRaises(RuntimeError): assert_no_debug_hooks(llm)
        def test_generation_sync_only_boundaries_and_failure(self):
            for fail in (False,True):
                sync=[]
                device=SimpleNamespace(synchronize=lambda:sync.append(1),reset_peak_memory_stats=lambda:None)
                class Engine:
                    _requests={}
                    scheduler=SimpleNamespace(block_manager=SimpleNamespace(blocks=[],used_block_ids=set()))
                    counts={}
                    def runtime_metrics(self): return dict(runtime=dict(self.counts))
                    def generate(self,prompts,params,use_tqdm=False):
                        # 模拟多轮生成；helper不能按每一轮插同步，只能包围整个generate。
                        for _ in range(8): pass
                        self.counts=dict(steps=4,batch_P_PREFILL=1,batch_D_NORMAL=3,query_decode=3,scheduled_prefill=2)
                        return [dict(text="ok",token_ids=[1]*(1 if fail else 4)) for prompt in prompts]
                case=dict(id="d1",kind="decode_heavy",batch=1,output_tokens=4,jobs=[dict(prompt_token_ids=[2,3])])
                row=generate_once(Engine(),case,params_factory=lambda **kw:kw,device=device)
                self.assertEqual(len(sync),2)
                self.assertEqual(row["status"],"FAIL" if fail else "PASS")
                self.assertFalse(row["diagnostic"])
        def test_fixed_decode_shape_admission_and_observed_gate(self):
            case=dict(id="d32",kind="decode_heavy",batch=32,output_tokens=64,
                      jobs=[dict(prompt_token_ids=[1]*96) for _ in range(32)])
            args=SimpleNamespace(token_budget=1024,chunk_size=256)
            with self.assertRaises(ValueError): validate_fixed_decode_shapes(args,[case])
            args.token_budget=4096
            validate_fixed_decode_shapes(args,[case])
            self.assertEqual(check_decode_schedule(case,{})["status"],"FAIL")
            expected=check_decode_schedule(case,{})["expected"]
            self.assertEqual(check_decode_schedule(case,expected)["status"],"PASS")
            expected["batch_PD_PACKED"]=1
            self.assertEqual(check_decode_schedule(case,expected)["status"],"FAIL")
        def test_cprofile_raw_evidence_and_no_overwrite(self):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                profile=cProfile.Profile()
                profile.runcall(sum,[1,2,3])
                path=Path(directory)/"example.cprofile"
                files=save_cprofile(profile,path)
                self.assertEqual(sha(Path(directory)/files["raw"]),files["raw_sha256"])
                self.assertTrue(load(path.with_suffix(".json"))["functions"])
                with self.assertRaises(FileExistsError): save_cprofile(profile,path)
        def test_factorial_only_changes_intended_fields(self):
            args=SimpleNamespace(token_budget=1024,chunk_size=128,gpu_memory_utilization=.4)
            cases=[dict(batch=8,output_tokens=64,jobs=[dict(prompt_token_ids=[1]*96)])]
            configs=configurations(args,cases)
            reference=configs["dynamic_torch"]
            for config in configs.values():
                self.assertEqual({k:v for k,v in config.items() if k not in ("mlp_backend","metadata_backend")},
                                 {k:v for k,v in reference.items() if k not in ("mlp_backend","metadata_backend")})
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status="PASS" if result.wasSuccessful() else "FAIL",tests=result.testsRun,scope="CPU fake only, no CUDA experiment")))
    if not result.wasSuccessful(): raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    command=sub.add_parser("run")
    command.add_argument("--model",required=True)
    command.add_argument("--output",required=True,type=Path)
    command.add_argument("--quick",action="store_true")
    command.add_argument("--mode",choices=("timing","diagnostic","both"),default="both")
    command.add_argument("--torch-profiler",choices=("none","cpu","cuda"),default="none")
    command.add_argument("--profile-cases",default=None)
    command.add_argument("--profile-output-tokens",type=int,default=16)
    command.add_argument("--decode-batches",default=None)
    command.add_argument("--decode-prompt-tokens",type=int,default=96)
    command.add_argument("--decode-output-tokens",type=int,default=None)
    command.add_argument("--rounds",type=int,default=None)
    command.add_argument("--warmups",type=int,default=1)
    command.add_argument("--token-budget",type=int,default=4096)
    command.add_argument("--chunk-size",type=int,default=256)
    command.add_argument("--gpu-memory-utilization",type=float,default=.4)
    command.add_argument("--worker-timeout",type=float,default=1800)
    command=sub.add_parser("worker")
    command.add_argument("--prepared",required=True,type=Path)
    command.add_argument("--variant",required=True,choices=VARIANTS)
    command.add_argument("--phase",required=True,choices=("timing","diagnostic"))
    command.add_argument("--round",required=True,type=int)
    sub.add_parser("self-test")
    args=parser.parse_args()
    if args.command=="self-test": self_test(); return
    if args.command=="worker": worker(args.prepared.resolve(),args.variant,args.phase,args.round); return
    args.decode_batches=[int(value) for value in (args.decode_batches or ("1,8" if args.quick else "1,8,16,32")).split(",")]
    args.profile_cases=args.profile_cases.split(",") if args.profile_cases else None
    args.rounds=args.rounds if args.rounds is not None else (1 if args.quick else 5)
    args.decode_output_tokens=args.decode_output_tokens if args.decode_output_tokens is not None else (8 if args.quick else 64)
    if (len(set(args.decode_batches))!=len(args.decode_batches) or not set(args.decode_batches)<={1,8,16,32}
        or min([args.rounds,args.decode_prompt_tokens,args.decode_output_tokens,args.profile_output_tokens])<1
        or args.warmups<0 or not 0<args.chunk_size<=args.token_budget
        or not 0<args.gpu_memory_utilization<=1 or not math.isfinite(args.worker_timeout) or args.worker_timeout<=0
        or (args.profile_cases and len(set(args.profile_cases))!=len(args.profile_cases))):
        parser.error("invalid shape, repeats, profile selection or timeout")
    raise SystemExit(run(args))


if __name__=="__main__":
    main()
