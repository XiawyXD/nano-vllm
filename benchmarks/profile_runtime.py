"""定位 speculative 开销：区分无 profiler 耗时与有 profiler 的归因轨迹。

每个 stage 的 host_ms 包含等待和子调用，不能直接相加当作 GPU 时间。
GPU kernel 时间取 torch.profiler，Chrome trace 保存异步执行顺序。
"""
import argparse
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.incremental_ngram import IncrementalNGramProposer


@contextmanager
def instrumentation(llm):
    totals = defaultdict(lambda: {"calls":0,"host_ms":0.0})
    originals = []
    def wrap(obj, name, label):
        old = getattr(obj,name)
        def measured(*args,**kwargs):
            start=time.perf_counter()
            with torch.profiler.record_function("nano::"+label):
                try:
                    return old(*args,**kwargs)
                finally:
                    totals[label]["calls"]+=1
                    totals[label]["host_ms"]+=(time.perf_counter()-start)*1000
        setattr(obj,name,measured)
        originals.append((obj,name,old))
    runner=llm.model_runner
    for obj,name,label in [
        (llm.scheduler,"schedule","schedule"),
        (IncrementalNGramProposer,"propose","propose"),
        (runner,"prepare_prefill","prepare_prefill"),
        (runner,"prepare_decode","prepare_decode"),
        (runner,"run_model","model_forward"),
        (runner.model,"compute_logits","lm_head"),
        (runner.sampler,"forward","sampler"),
        (llm,"_restore_drafts","rollback"),
        (llm.scheduler,"postprocess","commit"),
        (llm,"_make_step_events","events"),
        (llm,"_stream_delta","text_decode"),
    ]:
        wrap(obj,name,label)
    try:
        yield totals
    finally:
        for obj,name,old in reversed(originals):
            setattr(obj,name,old)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="/home/hrxu/huggingface/Qwen3-1.7B")
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--output",required=True)
    ap.add_argument("--no-trace",action="store_true")
    args=ap.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    llm=LLM(args.model,max_model_len=512,max_num_batched_tokens=512,
            max_num_seqs=16,gpu_memory_utilization=.55,enable_prefix_caching=False)
    prompts=[[1,2,3]*64 for _ in range(args.batch)]
    params={"ordinary":SamplingParams(temperature=0,max_tokens=64,ignore_eos=True),
            "speculative":SamplingParams(temperature=0,max_tokens=64,ignore_eos=True,speculative_ngram_size=3)}
    result={"model":args.model,"batch":args.batch,"runs":[],"profiles":{}}
    def run(name):
        torch.cuda.synchronize()
        begin=time.perf_counter()
        outputs=llm.generate(prompts,params[name],use_tqdm=False)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        # 完成请求的元数据由消费者释放；避免基准重复运行混入无关内存增长。
        for rid in list(llm._requests):
            llm.forget_request(rid)
        return elapsed,outputs
    try:
        for name in params:
            for _ in range(2): run(name)
        for repeat in range(5):
            names=list(params) if repeat%2==0 else list(reversed(params))
            for name in names:
                elapsed,outputs=run(name)
                result["runs"].append(dict(mode=name,repeat=repeat,seconds=elapsed,
                    tokens_s=args.batch*64/elapsed,tokens=outputs[0]["token_ids"],stats=llm.speculative_stats()))
        if not args.no_trace:
            for name in params:
                with instrumentation(llm) as totals:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                        run(name)
                prof.export_chrome_trace(str(root/(name+".trace.json")))
                ops=sorted(prof.key_averages(),key=lambda x:x.self_device_time_total,reverse=True)
                result["profiles"][name]={"stages":dict(totals),"top_ops":[
                    dict(name=o.key,calls=o.count,self_cpu_us=o.self_cpu_time_total,
                         self_device_us=o.self_device_time_total) for o in ops[:25]]}
        (root/"summary.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result["profiles"],indent=2))
        for name in params:
            xs=[r["seconds"] for r in result["runs"] if r["mode"]==name]
            print(name,"mean_seconds",sum(xs)/len(xs))
    finally:
        llm.exit()


if __name__=="__main__":
    main()
