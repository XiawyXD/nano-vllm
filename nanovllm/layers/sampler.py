"""从语言模型 logits 中采样下一个 token。"""

import torch
from torch import nn

from nanovllm.layers.fused_sampler import fused_greedy_argmax


class Sampler(nn.Module):
    """批量执行 temperature sampling。

    输入 logits 的形状通常是 [batch_size, vocab_size]，返回形状是
    [batch_size] 的 token id；每个请求本轮只采样一个下一个 token。
    """

    def __init__(self, backend: str = "torch", flashinfer_min_batch_size: int = 32):
        super().__init__()
        if backend not in {"torch", "triton", "flashinfer"}:
            raise ValueError("sampler backend must be 'torch', 'triton' or 'flashinfer'")
        # Torch保持默认；FlashInfer只加速经兼容性检查的随机行，其他行走原参考实现。
        self.backend = backend
        self.flashinfer = None
        if backend == "flashinfer":
            from nanovllm.layers.flashinfer_sampler import FlashInferSampler
            self.flashinfer = FlashInferSampler(min_batch_size=flashinfer_min_batch_size)

    def forward(self, logits: torch.Tensor, seqs):
        if logits.ndim != 2 or logits.size(0) != len(seqs):
            raise ValueError(f"logits must be [batch, vocab], got {logits.shape}")
        if not seqs:
            return torch.empty(0, dtype=torch.long, device=logits.device)
        # 纯 greedy 不需要 softmax 或逐请求 Python 循环。
        # Torch 是默认参考后端，自写 Triton 必须显式选择。
        if all(
            seq.temperature <= 1e-10
            and seq.top_k == 0
            and seq.top_p >= 1.0
            and seq.repetition_penalty == 1.0
            for seq in seqs
        ):
            if self.backend == "triton":
                return fused_greedy_argmax(logits)
            return logits.argmax(dim=-1)
        if self.flashinfer is not None and logits.is_cuda:
            return self.flashinfer(logits, seqs, self._reference)
        return self._reference(logits, seqs)

    @staticmethod
    def _reference(logits, seqs):
        """原Torch语义保留为单一参考；seed/penalty/边界回退都调用这里。"""
        tokens = []
        for row, seq in zip(logits, seqs):
            # float32 输入调用 float() 仍会共享存储，必须 clone 才能安全地
            # 原地施加 penalty/temperature/filter，避免污染调用方的 logits。
            row = row.float().clone()
            if seq.repetition_penalty != 1.0 and seq.token_ids:
                ids = torch.tensor(sorted(set(seq.token_ids)), device=row.device, dtype=torch.long)
                ids = ids[(ids >= 0) & (ids < row.numel())]
                if ids.numel():
                    selected = row.index_select(0, ids)
                    selected = torch.where(selected < 0, selected * seq.repetition_penalty, selected / seq.repetition_penalty)
                    row.scatter_(0, ids, selected)
            if seq.temperature <= 1e-10:
                tokens.append(row.argmax(dim=-1))
                continue
            row.div_(seq.temperature)
            if seq.top_k and seq.top_k < row.numel():
                kth = torch.topk(row, seq.top_k).values[-1]
                row.masked_fill_(row < kth, -float("inf"))
            if seq.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(row, descending=True)
                cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > seq.top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                sorted_logits.masked_fill_(remove, -float("inf"))
                row.fill_(-float("inf"))
                row.scatter_(0, sorted_indices, sorted_logits)
            probs = torch.softmax(row, dim=-1)
            if seq.seed is None:
                token = torch.multinomial(probs, 1)[0]
            else:
                generator = torch.Generator(device=row.device)
                generator.manual_seed(int(seq.seed) + seq.num_completion_tokens)
                token = torch.multinomial(probs, 1, generator=generator)[0]
            tokens.append(token)
        return torch.stack(tokens)
