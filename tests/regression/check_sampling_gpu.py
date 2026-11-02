"""在 GPU 上验证请求级完整采样参数。"""

from types import SimpleNamespace

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


def make_seq(**kwargs):
    values = dict(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=None,
        token_ids=[1, 2],
        num_completion_tokens=0,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def main():
    assert torch.cuda.is_available()
    sampler = Sampler().cuda()
    logits = torch.tensor([[0.0, 1.0, 3.0, 2.0]], device="cuda")
    greedy = sampler(logits, [make_seq(temperature=0.0)]).item()
    assert greedy == 2

    top_k = sampler(logits, [make_seq(temperature=1.0, top_k=1, seed=7)]).item()
    assert top_k == 2

    top_p = [
        sampler(logits, [make_seq(temperature=1.0, top_p=0.5, seed=i)]).item()
        for i in range(8)
    ]
    assert set(top_p) == {2}, top_p

    params = SamplingParams(
        temperature=0.7,
        top_k=8,
        top_p=0.9,
        repetition_penalty=1.1,
        seed=123,
        max_tokens=4,
    )
    assert params.top_k == 8 and params.top_p == 0.9
    print({
        "status": "PASS",
        "greedy_token": greedy,
        "top_k_token": top_k,
        "top_p_tokens": top_p,
        "validated_params": {
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "repetition_penalty": params.repetition_penalty,
            "seed": params.seed,
        },
    })


if __name__ == "__main__":
    main()
