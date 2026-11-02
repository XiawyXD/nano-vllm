"""Tensor Parallel 线性层。

线性层权重通常是 [out_features, in_features]。Column Parallel 把输出维
切开，Row Parallel 把输入维切开，并在必要处用 all_reduce 合并结果。
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.tensorcore_mlp import stable_bf16_linear


def divide(numerator, denominator):
    """只允许整除，保证每个 rank 拿到同样大小的分片。"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """所有并行线性层共享的参数和权重加载逻辑。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        backend: str = "torch",
    ):
        super().__init__()
        # tp_dim=0 表示沿 weight 的输出维切分；tp_dim=1 表示沿输入维切分。
        self.tp_dim = tp_dim
        if backend not in {"torch", "triton_stable"}:
            raise ValueError(f"unsupported linear backend: {backend}")
        if backend == "triton_stable" and bias:
            raise ValueError("triton_stable linear currently requires bias=False")
        self.backend = backend
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 把方法挂到 Parameter 上，loader 可以按模块类型自动调用。
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基类只定义接口，具体并行方式由子类实现。
        raise NotImplementedError

    def project(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """统一线性投影入口；实验后端不改变权重加载与 TP 分片布局。"""
        if self.backend == "triton_stable":
            if bias is not None:
                raise ValueError("triton_stable linear does not support bias")
            if getattr(self, "stable_packed_layout", False):
                from nanovllm.layers.stable_layout import packed_stable_bf16_linear
                return packed_stable_bf16_linear(x, self.weight)
            return stable_bf16_linear(x, self.weight)
        return F.linear(x, self.weight, bias)


class ReplicatedLinear(LinearBase):
    """每个 rank 都保存完整权重、独立计算的普通 Linear。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        backend: str = "torch",
    ):
        super().__init__(input_size, output_size, bias, backend=backend)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # replicated 参数不切片，直接复制完整权重。
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x, self.bias)


class ColumnParallelLinear(LinearBase):
    """沿输出维切分的 Linear。

    每个 rank 输出一段 channel，后续如果需要完整输出，通常由下一个
    RowParallelLinear 的 all_reduce 合并。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        backend: str = "torch",
    ):
        tp_size = dist.get_world_size()
        # weight 的第 0 维是输出维，所以只保存 output_size/tp_size。
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, backend)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 从完整权重的输出维中取当前 rank 的连续切片。
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """把多个逻辑 Column Linear（例如 gate/up）拼成一次矩阵乘。"""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        backend: str = "torch",
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, backend)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # loaded_shard_id 指示这是 output_sizes 中哪一个逻辑分支。
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 原始权重也按 TP 维切分，再拷贝到合并参数的对应区间。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """把 Q、K、V 的投影权重拼在一起并沿输出维并行。"""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        backend: str = "torch",
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 每个 rank 实际负责的 query/KV head 数。
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 输出布局为 [Q 分片, K 分片, V 分片]。
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, backend)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # 根据 q/k/v 选择合并权重中的目标区间。
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 再从原始权重中选出当前 rank 的 shard。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """沿输入维切分的 Linear。

    每个 rank 计算部分输入贡献，最后用 all_reduce 将各 rank 的部分和相加。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        backend: str = "torch",
    ):
        tp_size = dist.get_world_size()
        # weight 的第 1 维是输入维，所以只保存 input_size/tp_size。
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, backend)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # bias 是一维完整参数；矩阵权重则沿输入维取当前 rank 的 shard。
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只有 rank 0 加 bias，避免多 rank 重复相加。
        y = self.project(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # 把各 rank 对输出的部分贡献求和。
            dist.all_reduce(y)
        return y
