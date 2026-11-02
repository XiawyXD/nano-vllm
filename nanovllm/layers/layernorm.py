"""RMSNorm 层以及带 residual 融合的实现。"""

import torch
from torch import nn

from nanovllm.layers.fused_rmsnorm import fused_add_rmsnorm, fused_rmsnorm


class RMSNorm(nn.Module):
    """不减均值的 RMS normalization。

    相比 LayerNorm，RMSNorm 只根据均方根缩放输入，不额外计算均值和方差，
    更适合在 Transformer 推理中与 residual 相邻融合。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        if backend not in {"torch", "triton"}:
            raise ValueError(f"unknown RMSNorm backend: {backend}")
        self.eps = eps
        self.backend = backend
        # 每个 hidden channel 一个可学习缩放系数。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        # 用 float32 计算平方和，降低 fp16/bf16 下的数值误差。
        x = x.float()
        # var 这里不是中心化方差，而是最后一维的均方值 E[x^2]。
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt(var + eps) 等于 1 / sqrt(var + eps)，原地缩放输入。
        x.mul_(torch.rsqrt(var + self.eps))
        # 恢复原 dtype 后施加可学习的 RMSNorm weight。
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 先把主分支和 residual 相加；这是 Transformer 常见的 pre-norm 融合。
        x = x.float().add_(residual.float())
        # 保存未归一化的 residual，供下一子层继续使用。
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 没有 residual 时返回单个归一化结果；有 residual 时返回二元组。
        if self.backend == "triton":
            if residual is None:
                return fused_rmsnorm(x, self.weight, self.eps)
            return fused_add_rmsnorm(x, residual, self.weight, self.eps)
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
