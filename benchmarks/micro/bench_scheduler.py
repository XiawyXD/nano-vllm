"""回放混合请求并记录 Prefill/Decode 调度行为。

示例：
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=. \
python benchmarks/micro/bench_scheduler.py --prefill-chunk-size 128 --max-prefill-steps 1
"""

import argparse
import json
import os
from time import perf_counter

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--num-initial", type=int, default=4)
    parser.add_argument("--short-prompt-tokens", type=int, default=64)
    parser.add_argument("--long-prompt-tokens", type=int, default=1024)
    parser.add_argument("--inject-after", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=None)
    parser.add_argument("--max-prefill-steps", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        max_prefill_steps=args.max_prefill_steps,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    short_prompt = [1] * args.short_prompt_tokens
    long_prompt = [1] * args.long_prompt_tokens

    for _ in range(args.num_initial):
        llm.add_request(short_prompt, sampling_params)

    rows = []
    step_id = 0
    while not llm.is_finished():
        if step_id == args.inject_after:
            llm.add_request(long_prompt, sampling_params)
            print(json.dumps({"event": "inject_long_request", "step": step_id}))

        start = perf_counter()
        _, token_count = llm.step()
        elapsed_ms = (perf_counter() - start) * 1000.0
        kind = "prefill" if token_count > 0 else "decode"
        row = {
            "step": step_id,
            "kind": kind,
            "scheduled_tokens": abs(token_count),
            "elapsed_ms": round(elapsed_ms, 3),
            "prefill_steps": llm.scheduler_stats()["prefill_steps"],
            "decode_steps": llm.scheduler_stats()["decode_steps"],
        }
        rows.append(row)
        print(json.dumps(row))
        step_id += 1

    print(json.dumps({
        "summary": llm.scheduler_stats(),
        "steps": len(rows),
    }))


if __name__ == "__main__":
    main()
