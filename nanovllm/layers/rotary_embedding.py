"""Rotary Position Embedding（RoPE）。"""

from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """把最后一维拆成两半，并按位置角度执行二维旋转。

    对每一对通道 (x1, x2) 使用旋转矩阵：
    [x1*cos - x2*sin, x2*cos + x1*sin]。
    """
    # 用 float32 做三角变换和乘法，最后再转换回输入 dtype。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """预先计算并缓存所有位置的 cos/sin，forward 时只做索引。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # inv_freq 是不同二维通道对对应的角频率。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # t 表示位置 0, 1, ..., max_position_embeddings-1。
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积得到 [max_position_embeddings, rotary_dim/2] 的旋转角度。
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 拼成 [max_position_embeddings, 1, rotary_dim]，便于和 [N,H,D] 广播。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 不再让 FP16 draft 和 BF16 target 共用 Dynamo 的重编译预算。
        # 原先 @torch.compile 达到上限后回退 eager：compiled 的 FP32 FMA
        # 与 eager 的分离乘加舍入不同，甚至会改变固定历史下的 target token。
        # CUDA 使用一个融合 kernel，但明确禁止 FMA 收缩；运算顺序固定为
        # FP32 multiply -> FP32 add/sub -> cast，与下方 CPU/eager 参考一致。
        # 这不是关闭 CUDA Graph，也不修改全局 compiler limit 或验收阈值。
        if query.is_cuda:
            from nanovllm.layers.fused_rotary import fused_rotary
            return fused_rotary(positions, query, key, self.cos_sin_cache)
        # positions 选择本批次每个 token 对应的 cos/sin。
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # Q 和 K 使用相同位置旋转，从而把相对位置信息编码进点积。
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    # 相同超参数只创建一份 RoPE，避免重复构造大尺寸 cos/sin cache。
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
