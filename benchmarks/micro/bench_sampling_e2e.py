"""在真实 Qwen3 continuous batch 中混合不同采样参数。"""

import json
import os

from nanovllm import LLM, SamplingParams


def main():
    model = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    llm = LLM(model, max_model_len=128, max_num_batched_tokens=128, enforce_eager=True)
    prompts = [[1, 2, 3, 4, 5]] * 3
    params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
        SamplingParams(temperature=0.8, top_k=1, ignore_eos=True, max_tokens=4, seed=7),
        SamplingParams(temperature=0.8, top_p=0.9, repetition_penalty=1.1, ignore_eos=True, max_tokens=4, seed=42),
    ]
    outputs = llm.generate(prompts, params, use_tqdm=False)
    assert len(outputs) == 3
    assert outputs[0]["token_ids"] == outputs[1]["token_ids"]
    assert all(len(output["token_ids"]) == 4 for output in outputs)
    print(json.dumps({
        "status": "PASS",
        "request_count": len(outputs),
        "sampling_modes": ["greedy", "top_k=1", "top_p=0.9+repetition_penalty"],
        "token_ids": [output["token_ids"] for output in outputs],
        "scheduler_stats": llm.scheduler_stats(),
    }, ensure_ascii=False))
    llm.exit()


if __name__ == "__main__":
    main()
