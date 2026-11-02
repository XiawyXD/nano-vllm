"""nano-vLLM 的简单吞吐率 benchmark。"""

import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    # 固定随机种子，保证每次生成的随机 prompt 长度一致。
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # enforce_eager=False 允许 decode 使用 CUDA Graph。
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 构造不同长度的 token-id prompt，模拟一批请求。
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # 每个请求可以拥有独立的采样参数和最大生成长度。
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 先进行一次小请求，预热 torch.compile/CUDA Graph 等运行路径。
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    # 正式测量时关闭 tqdm，避免终端刷新干扰计时。
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    # 这里统计的是计划生成的 completion token 总数。
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
