"""引擎级重复基准：固定到达表、预热、交错策略顺序，保存原始请求时间。

TTFT 从计划到达算起，包含等待；TPOT=(末 token 时间-首 token 时间)/(N-1)。
speculative 一次事件可含多个 token，记录真实事件间隔，不伪造逐 token 时间。
此脚本不包含网络延迟，不可把结果表述为 HTTP 端到端性能。
"""
import argparse
import json
import time
from pathlib import Path
import statistics
import torch
from nanovllm import LLM, SamplingParams


def summary(xs):
    ys = sorted(xs)
    def p(q):
        import math
        return ys[max(0, math.ceil(len(ys)*q)-1)] if ys else None
    return dict(n=len(xs), mean=statistics.mean(xs) if xs else None,
                p50=p(.5), p95=p(.95), p99=p(.99))


def workload(tokenizer, name):
    text = "Explain how a language model uses attention and caches previous keys and values. "
    base = tokenizer.encode(text * 200)
    # 明确保存到达表；不同策略使用相同的 prompt 和输出 token 数。
    lengths = [64]*4 + [1024] + [64]*7
    times = [0]*4 + [.06] + [.08 + .025*i for i in range(7)]
    return [(t, ([1,2,3]*400)[:n] if name == "repeat" else base[i:i+n], 32)
            for i,(t,n) in enumerate(zip(times,lengths))]


def run(llm, jobs, policy):
    settings = {"baseline":(None,0,0), "chunk128":(128,1,0),
                "chunk256":(256,1,0), "spec2":(256,1,2), "spec4":(256,1,4),
                "gated4":(256,1,4)}
    chunk, quota, draft = settings[policy]
    llm.scheduler.prefill_chunk_size = chunk
    llm.scheduler.max_prefill_steps = quota
    llm.scheduler.prefill_steps_since_decode = 0
    llm.config.speculative_max_batch_size = 1 if policy == "gated4" else 0
    llm._last_speculative_stats = {}
    records, seqs = {}, {}
    next_job = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    while next_job < len(jobs) or not llm.is_finished():
        now = time.perf_counter() - start
        while next_job < len(jobs) and jobs[next_job][0] <= now:
            arrival, prompt, count = jobs[next_job]
            seq = llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=count,
                ignore_eos=True, speculative_ngram_size=3 if draft else 0,
                speculative_max_draft_tokens=draft or 4))
            seqs[seq.seq_id] = seq
            records[seq.seq_id] = dict(arrival=arrival, submit=time.perf_counter()-start,
                input_tokens=len(prompt), events=[])
            next_job += 1
        if llm.is_finished():
            time.sleep(.001)
            continue
        llm.step()
        now = time.perf_counter() - start
        for event in llm.step_events():
            records[event.request_id]["events"].append([now, event.token_index+1])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    ttft, tpot, itl, completion = [], [], [], []
    for rid, item in records.items():
        events = item["events"]
        first, last = events[0][0], events[-1][0]
        count = seqs[rid].num_completion_tokens
        ttft.append((first-item["arrival"])*1000)
        tpot.append((last-first)/max(1,count-1)*1000)
        completion.append((last-item["arrival"])*1000)
        itl.extend((b[0]-a[0])*1000 for a,b in zip(events,events[1:]))
        item["output_tokens"] = count
        llm.forget_request(rid)
    assert not llm.scheduler.block_manager.used_block_ids
    return dict(policy=policy, seconds=seconds, tokens_per_second=sum(x["output_tokens"] for x in records.values())/seconds,
        ttft_ms=summary(ttft), tpot_ms=summary(tpot), event_gap_ms=summary(itl), latency_ms=summary(completion),
        requests=list(records.values()), spec=llm.speculative_stats(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), scheduler=llm.scheduler.stats())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-0.6B")
    ap.add_argument("--policies", default="baseline")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    llm = LLM(args.model, max_model_len=1536, max_num_batched_tokens=512,
              max_num_seqs=16, gpu_memory_utilization=.55, enable_prefix_caching=False)
    policies = args.policies.split(",")
    report = dict(model=args.model, repeats=args.repeats, cuda_graph=True, prefix_cache=False,
        device=torch.cuda.get_device_name(), torch=torch.__version__, trials=[])
    try:
        for name in ("mixed", "repeat"):
            jobs = workload(llm.tokenizer,name)
            for policy in policies:
                run(llm,jobs,policy)  # 每种形状与策略预热一次，不计入结果。
            for repeat in range(args.repeats):
                # 轮转顺序，降低时钟/温度随时间变化对某一策略的偏置。
                for policy in policies[repeat%len(policies):]+policies[:repeat%len(policies)]:
                    trial = run(llm,jobs,policy)
                    trial.update(workload=name, repeat=repeat)
                    report["trials"].append(trial)
                    print(json.dumps({k:trial[k] for k in ("workload","policy","repeat","tokens_per_second","ttft_ms","tpot_ms")}),flush=True)
                    Path(args.output).write_text(json.dumps(report,indent=2))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
