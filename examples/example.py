"""最小文本生成示例。"""

import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 使用本地下载的 Qwen3-0.6B checkpoint；expanduser 支持 `~` 路径。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 这里只用 tokenizer 的 chat template 构造输入，不参与模型推理。
    tokenizer = AutoTokenizer.from_pretrained(path)
    # enforce_eager=True 关闭 CUDA Graph，便于第一次运行调试。
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # temperature 控制随机性，max_tokens 限制每条 completion 的长度。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "which song of frank ocean is the most popular?",
    ]
    # Qwen3 是 chat 模型，需要把普通问题包装为 user 消息并添加生成提示。
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # nano-vLLM 会把多条 prompt 放入 scheduler，执行 continuous batching。
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        # output 同时包含可读文本和底层 token_ids，方便检查生成过程。
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    # 直接运行 `python examples/example.py` 时进入示例入口。
    main()
