"""Measure TTFT, TPOT, tail latency and scheduler behavior.

The benchmark uses the public LLMEngine step loop instead of `generate()` so it
can timestamp every generated token while keeping the model implementation
unchanged. It is intentionally a baseline tool, not another scheduler.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.metrics import cuda_memory_snapshot, numeric_delta, summarize


@dataclass
class RequestTrace:
    """Host-side timestamps for one logical request."""

    seq_id: int
    submitted_at: float
    token_times: list[float] = field(default_factory=list)
    observed_completion_tokens: int = 0

    @property
    def ttft_ms(self) -> float | None:
        if not self.token_times:
            return None
        return (self.token_times[0] - self.submitted_at) * 1000.0

    @property
    def tpot_ms(self) -> float | None:
        # TPOT excludes the first token and measures intervals between later
        # tokens. A one-token completion has no TPOT sample.
        if len(self.token_times) < 2:
            return None
        intervals = [
            (right - left) * 1000.0
            for left, right in zip(self.token_times, self.token_times[1:])
        ]
        return mean(intervals)


def percentile(values: list[float], p: float) -> float | None:
    """Return a nearest-rank percentile without adding a dependency."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--num-initial", type=int, default=4)
    parser.add_argument("--short-prompt-tokens", type=int, default=64)
    parser.add_argument("--long-prompt-tokens", type=int, default=1024)
    parser.add_argument("--inject-after", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=None)
    parser.add_argument("--max-prefill-steps", type=int, default=0)
    parser.add_argument("--rmsnorm-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--decode-attention-backend", choices=("flash", "triton"), default="flash")
    parser.add_argument("--decode-attention-block-n", type=int, default=64)
    parser.add_argument("--decode-attention-num-warps", type=int, default=None)
    parser.add_argument("--kv-cache-dtype", choices=("model", "fp8"), default="model")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def add_tracked_request(llm, prompt, sampling_params, traces, sequence_refs):
    """Add one request and retain its Sequence object for timestamp sampling."""
    submitted_at = perf_counter()
    llm.add_request(prompt, sampling_params)
    # add_request appends to waiting; retaining this object is safe even after
    # Scheduler moves it to running or removes it as FINISHED.
    seq = llm.scheduler.waiting[-1]
    traces[seq.seq_id] = RequestTrace(seq.seq_id, submitted_at)
    sequence_refs.append(seq)


def observe_tokens(sequence_refs, traces, now):
    """Record one host timestamp for every newly visible completion token."""
    for seq in sequence_refs:
        trace = traces[seq.seq_id]
        completion_tokens = seq.num_completion_tokens
        while trace.observed_completion_tokens < completion_tokens:
            trace.token_times.append(now)
            trace.observed_completion_tokens += 1


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        max_prefill_steps=args.max_prefill_steps,
        rmsnorm_backend=args.rmsnorm_backend,
        decode_attention_backend=args.decode_attention_backend,
        decode_attention_block_n=args.decode_attention_block_n,
        decode_attention_num_warps=args.decode_attention_num_warps,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_prefix_caching=not args.disable_prefix_caching,
        enforce_eager=args.enforce_eager,
    )

    # Warm up model compilation and CUDA allocation. Warmup is excluded from
    # request latency and scheduler statistics reported below.
    warmup_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2)
    llm.generate([[1] * min(32, args.max_model_len)], warmup_params, use_tqdm=False)
    stats_before = llm.scheduler_stats()
    memory_before = cuda_memory_snapshot()

    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.max_tokens)
    short_prompt = [1] * args.short_prompt_tokens
    long_prompt = [1] * args.long_prompt_tokens
    traces = {}
    sequence_refs = []
    for _ in range(args.num_initial):
        add_tracked_request(llm, short_prompt, sampling_params, traces, sequence_refs)

    step_rows = []
    workload_start = perf_counter()
    step_id = 0
    while not llm.is_finished():
        if step_id == args.inject_after:
            add_tracked_request(llm, long_prompt, sampling_params, traces, sequence_refs)

        start = perf_counter()
        _, token_count = llm.step()
        end = perf_counter()
        observe_tokens(sequence_refs, traces, end)
        step_rows.append({
            "step": step_id,
            "kind": "prefill" if token_count > 0 else "decode",
            "scheduled_tokens": abs(token_count),
            "elapsed_ms": (end - start) * 1000.0,
        })
        step_id += 1
    workload_end = perf_counter()

    ttft = [trace.ttft_ms for trace in traces.values() if trace.ttft_ms is not None]
    tpot = [trace.tpot_ms for trace in traces.values() if trace.tpot_ms is not None]
    prefill_steps = [row["elapsed_ms"] for row in step_rows if row["kind"] == "prefill"]
    decode_steps = [row["elapsed_ms"] for row in step_rows if row["kind"] == "decode"]
    stats_after = llm.scheduler_stats()
    stats_delta = numeric_delta(stats_after, stats_before)
    memory_after = cuda_memory_snapshot()
    total_completion_tokens = sum(len(trace.token_times) for trace in traces.values())
    elapsed_s = workload_end - workload_start
    result = {
        "config": {
            "model": args.model,
            "num_initial": args.num_initial,
            "short_prompt_tokens": args.short_prompt_tokens,
            "long_prompt_tokens": args.long_prompt_tokens,
            "max_tokens": args.max_tokens,
            "prefill_chunk_size": args.prefill_chunk_size,
            "max_prefill_steps": args.max_prefill_steps,
            "rmsnorm_backend": args.rmsnorm_backend,
            "decode_attention_backend": args.decode_attention_backend,
            "decode_attention_block_n": args.decode_attention_block_n,
            "decode_attention_num_warps": args.decode_attention_num_warps,
            "kv_cache_dtype": args.kv_cache_dtype,
            "enable_prefix_caching": not args.disable_prefix_caching,
            "enforce_eager": args.enforce_eager,
        },
        "requests": len(traces),
        "completion_tokens": total_completion_tokens,
        "workload_seconds": elapsed_s,
        "tokens_per_second": total_completion_tokens / elapsed_s,
        "seed": args.seed,
        "ttft_ms": summarize(ttft),
        "tpot_ms": summarize(tpot),
        "step_ms": {
            "prefill": summarize(prefill_steps),
            "decode": summarize(decode_steps),
        },
        "scheduler_stats": stats_delta,
        "runtime_stats_after": stats_after,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "step_rows": step_rows,
    }
    serialized = json.dumps(result, ensure_ascii=False)
    print(serialized)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
