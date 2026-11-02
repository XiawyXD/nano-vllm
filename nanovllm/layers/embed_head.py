"""词表并行的 embedding 和语言模型输出头。"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context
from nanovllm.layers.tensorcore_mlp import stable_bf16_linear


class VocabParallelEmbedding(nn.Module):
    """把词表沿 vocab 维切分到多个 Tensor Parallel rank。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 每个 rank 保存 vocab 的一个连续区间。
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # weight 形状是 [本 rank 的词表大小, hidden_size]。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整 embedding 权重中拷贝当前 rank 的 vocab 分片。"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 当前 rank 只对属于自己词表区间的 id 做有效 embedding。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 越界 id 临时平移为 0，避免 embedding 索引越界；之后会用 mask 清零。
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 非本 rank 的结果置零，再在所有 rank 上求和恢复完整 embedding。
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """词表并行的 LM head，输出 logits 时再跨 rank 拼回完整词表。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self.backend = "torch"

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.logits_indices is not None:
            # 显式行映射支持 P/D、D/V 混批；未完成的 prompt chunk 没有输出行。
            x = x.index_select(0, context.logits_indices)
        elif context.is_prefill and not context.return_all_logits:
            # prefill 输入可能含有所有 prompt token，但只需要每条序列
            # 最后一个位置的 hidden state 来预测下一个 token。
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # 当前 rank 计算本地 vocab 分片的 logits：[B, vocab_partition]。
        logits = (stable_bf16_linear(x, self.weight)
                  if self.backend == "triton_stable" else F.linear(x, self.weight))
        if self.tp_size > 1:
            # rank 0 收集所有 rank 的分片，沿词表维拼接成完整 logits。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
