"""对照 greedy 与 n-gram speculative 的输出和接受率。"""

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
    prompt = [1, 2, 3] * 32
    greedy_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    speculative_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        speculative_ngram_size=args.ngram_size,
        speculative_max_draft_tokens=args.max_draft_tokens,
    )

    start = perf_counter()
    greedy = llm.generate([prompt], greedy_params, use_tqdm=False)[0]
    greedy_seconds = perf_counter() - start
    start = perf_counter()
    speculative = llm.generate_speculative(prompt, speculative_params)[0]
    speculative_seconds = perf_counter() - start
    stats = llm.speculative_stats()
    assert speculative["token_ids"] == greedy["token_ids"], {
        "greedy": greedy["token_ids"],
        "speculative": speculative["token_ids"],
    }
    result = {
        "status": "PASS",
        "same_tokens_as_greedy": True,
        "greedy_seconds": greedy_seconds,
        "speculative_seconds": speculative_seconds,
        "speculative_speedup": greedy_seconds / speculative_seconds,
        "stats": stats,
        "completion_tokens": len(speculative["token_ids"]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    llm.exit()


if __name__ == "__main__":
    main()
