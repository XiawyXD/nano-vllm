"""前馈网络使用的激活函数。"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """实现 SwiGLU 中的 `SiLU(gate) * up`。

    输入的最后一维包含两个等大的部分：前半部分经过 SiLU，后半部分
    作为 value/up 分支，逐元素相乘后把维度恢复到 intermediate_size。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 固定 eager 的算术语义：SiLU 先舍入到输入 dtype，再与 up 相乘。
        # 不共用 Dynamo 编译缓存：FP16 draft 的额外形状曾使 BF16 target
        # 发生编译回退，导致 Q=1 的已捕获路径与 Q>1 的 eager 路径数值不一致。
        # 这里保留两个可靠的 PyTorch GPU 原语，而非新引入近似 exp 实现。
        # 外层 CUDA Graph 仍会捕获/回放它们；没有关闭 target 或 draft Graph。
        # 将 [*, 2 * intermediate_size] 拆成两个 [*, intermediate_size]。
        x, y = x.chunk(2, -1)
        # SiLU(gate) 提供门控，再与另一支逐元素相乘。
        return F.silu(x) * y
