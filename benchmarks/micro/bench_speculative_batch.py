"""多请求 speculative batch 与普通 greedy 的 token-level 对照。"""

import argparse
import json
import os
from time import perf_counter

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    args = parser.parse_args()

    llm = LLM(args.model, max_model_len=256, max_num_batched_tokens=256, enforce_eager=True)
    prompts = [[1, 2, 3] * 32, [4, 5, 6] * 32]
    greedy_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    speculative_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        speculative_ngram_size=args.ngram_size,
        speculative_max_draft_tokens=args.max_draft_tokens,
    )
    start = perf_counter()
    greedy = llm.generate(prompts, greedy_params, use_tqdm=False)
    greedy_seconds = perf_counter() - start
    start = perf_counter()
    speculative = llm.generate(prompts, speculative_params, use_tqdm=False)
    speculative_seconds = perf_counter() - start
    assert [item["token_ids"] for item in speculative] == [item["token_ids"] for item in greedy]
    stats = llm.speculative_stats()
    result = {
        "status": "PASS",
        "request_count": len(prompts),
        "same_tokens_as_greedy": True,
        "greedy_seconds": greedy_seconds,
        "speculative_seconds": speculative_seconds,
        "speculative_speedup": greedy_seconds / speculative_seconds,
        "stats": stats,
        "completion_tokens": [len(item["token_ids"]) for item in speculative],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    llm.exit()


if __name__ == "__main__":
    main()
