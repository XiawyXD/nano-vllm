"""Compare Torch and Triton RMSNorm backends end to end."""

import argparse
import os
import time

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--backend", choices=("torch", "triton"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    llm = LLM(
        args.model,
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=256,
        prefill_chunk_size=128,
        max_prefill_steps=1,
        rmsnorm_backend=args.backend,
    )
    prompts = ["Explain CUDA memory hierarchy in one sentence."] * 8
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16)

    # 先让模型和后端完成一次编译/缓存预热，再统计稳态运行时间。
    llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed.append(time.perf_counter() - start)

    print("backend:", args.backend)
    print("seconds:", [round(value, 4) for value in elapsed])
    print("mean_seconds:", round(sum(elapsed) / len(elapsed), 4))
    print("completion_tokens_per_request:", len(outputs[0]["token_ids"]))
    print("scheduler_stats:", llm.scheduler_stats())


if __name__ == "__main__":
    main()
