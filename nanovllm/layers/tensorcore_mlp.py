"""教学用 BF16 Tensor Core GEMM 与 gate/up + SwiGLU 融合算子。

这是显式选择的推理实验后端，不替换默认 Torch 路径，也不提供 backward。
输入 x[M,K]、权重 w[N,K] 都按 PyTorch Linear 的行主序保存。
`tl.dot` 的 BF16 输入、FP32 累加可在 Ada 上生成 Tensor Core MMA 指令。
是否更快必须看形状和实测，尤其不能把大 M 的收益外推到 M=1 的 decode。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_linear_kernel(
    X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """一个 program 计算 Y 的 BM×BN 小块，沿 K 分块累加。"""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + inner
        # X 小块为 [BM,BK]。W 存为 [N,K]，按转置视图读成 [BK,BN]。
        a = tl.load(X + rows[:, None] * K + ks[None, :],
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        b = tl.load(W + cols[None, :] * K + ks[:, None],
                    mask=(cols[None, :] < N) & (ks[:, None] < K), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _fused_swiglu_kernel(
    X, W, Y, M: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """同一块 X 同时乘 gate/up 两块权重，然后直接输出 SwiGLU。

W 前 I 行是 gate，后 I 行是 up；不会分配完整的 [M,2I] 中间 tensor。
两个 FP32 accumulator 会增加寄存器压力，这是融合必须实测的代价。
"""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    gate_acc = tl.zeros((BM, BN), dtype=tl.float32)
    up_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + inner
        a = tl.load(X + rows[:, None] * K + ks[None, :],
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        valid_w = (cols[None, :] < I) & (ks[:, None] < K)
        gate_w = tl.load(W + cols[None, :] * K + ks[:, None],
                         mask=valid_w, other=0.0)
        up_w = tl.load(W + (cols[None, :] + I) * K + ks[:, None],
                       mask=valid_w, other=0.0)
        gate_acc = tl.dot(a, gate_w, gate_acc)
        up_acc = tl.dot(a, up_w, up_acc)

    # 保留原 F.linear BF16 输出的中间舍入，不偷偷变成全 FP32 激活基线。
    # 即便这样，GEMM 归约顺序和 exp 实现仍可能与库实现有细小差异。
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    result = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(Y + rows[:, None] * I + cols[None, :], result.to(tl.bfloat16),
             mask=(rows[:, None] < M) & (cols[None, :] < I))


def _validate_inputs(x: torch.Tensor, weight: torch.Tensor) -> tuple[int, int, int]:
    """只检查 CPU 可见的元数据，不执行 .item()/GPU 同步或逐元素扫描。"""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2D tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("tensorcore experiment requires BF16 x and weight")
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        raise ValueError("x and weight must be on the same CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    m, k = x.shape
    n, weight_k = weight.shape
    if k != weight_k or k <= 0 or n <= 0:
        raise ValueError("positive matching K and positive output width are required")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise ValueError("inference-only kernel: use inference_mode/no_grad; no backward is implemented")
    return m, n, k


def _launch_config(m: int) -> tuple[int, int, int]:
    """固定的可解释起点，不在运行或 Graph capture 内偷偷做 autotune。

M 很小时 BM=16 减少 padding；较大 M 用 BM=32 复用权重。
这些不是宣称全局最优的参数。后续只能依据离线数据增加形状专用配置。
"""
    return (16 if m < 32 else 32), 64, 32


def bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """计算 x[M,K] @ weight[N,K].T；BF16 输出、FP32 累加、无 bias。

M=0 返回空 tensor；N/K=0、非连续和非 BF16 输入明确报错。
首次调用可能 JIT 编译；测量前预热，capture 前编译对应形状。
"""
    m, n, k = _validate_inputs(x, weight)
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m:
        bm, bn, bk = _launch_config(m)
        with torch.cuda.device(x.device):
            _bf16_linear_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                x, weight, output, m, n, k, bm, bn, bk, num_warps=4, num_stages=3,
            )
    return output


def stable_bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """推理实验后端：固定 GEMM tile，使同一行不依赖 batch 的 M。

    BM/BN/BK 固定为 16/128/64；不能按 M 在 decode/prefill 间切换
    kernel 配置，否则又会引入形状相关的 BF16 舍入。它只接受 BF16、
    CUDA、二维输入，不支持 autograd。词表 LM Head 继续使用 Torch。
    这保证的是本 kernel 的行级计算路径，不保证 Attention 或整个模型
    在不同调度历史下严格相等；端到端验证必须单独运行。
    """
    x = x.contiguous()
    weight = weight.contiguous()
    m, n, k = _validate_inputs(x, weight)
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m:
        bm, bn, bk = 16, 128, 64
        with torch.cuda.device(x.device):
            _bf16_linear_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                x, weight, output, m, n, k, bm, bn, bk,
                num_warps=4, num_stages=3,
            )
    return output


def fused_swiglu(x: torch.Tensor, gate_up_weight: torch.Tensor) -> torch.Tensor:
    """计算 SiLU(x @ W_gate.T) * (x @ W_up.T)，输出 [M,I]。

gate_up_weight 为 [2I,K]，前半 gate、后半 up；与当前 TP shard 的布局一致。
仅融合 MLP 前半部分，down projection/AllReduce 仍由原模型完成。
"""
    m, n, k = _validate_inputs(x, gate_up_weight)
    if n % 2:
        raise ValueError("gate_up_weight output width must be even (gate then up)")
    intermediate = n // 2
    output = torch.empty((m, intermediate), dtype=x.dtype, device=x.device)
    if m:
        bm, bn, bk = _launch_config(m)
        with torch.cuda.device(x.device):
            _fused_swiglu_kernel[(triton.cdiv(m, bm), triton.cdiv(intermediate, bn))](
                x, gate_up_weight, output, m, intermediate, k, bm, bn, bk,
                num_warps=4, num_stages=3,
            )
    return output


def can_use_fused_swiglu(x: torch.Tensor, weight: torch.Tensor, min_tokens: int) -> bool:
    """模型层的轻量回退判断；门槛可配置，但不表示此形状已经证明更快。

不适用的 dtype/layout/小 M 继续原 Torch 路径，避免实验开关破坏其他模型。
本函数只有元数据分支，可在普通 eager 与固定形状 CUDA Graph 中使用。
"""
    return (
        x.ndim == 2 and weight.ndim == 2 and x.is_cuda and weight.is_cuda
        and x.device == weight.device
        and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
        and x.is_contiguous() and weight.is_contiguous()
        and x.shape[0] >= min_tokens and x.shape[1] > 0
        and x.shape[1] == weight.shape[1]
        and weight.shape[0] > 0 and weight.shape[0] % 2 == 0
    )


def should_use_fused_swiglu(x, weight, backend, min_tokens, capability=None) -> bool:
    """auto只允许已通过端到端验收的形状；当前白名单为空。

    2026-09-21五轮无hook端到端消融否定了旧微基准的M=1/8/16/32收益：
    对应generate速度比为0.84/0.87/0.88/0.90，因此auto全部回退Torch。
    原始证据见RUNTIME_V2_GPU_ACCEPTANCE_PROGRESS.md，不能拿微基准倍率代替服务倍率。
    triton 是人工实验模式，使用min_tokens门槛；auto不用这个人工门槛。
    capability 在模型构造时读取一次，热路径不查询设备、不做autotune。
    """
    if backend == "torch":
        return False
    if backend == "triton":
        return can_use_fused_swiglu(x, weight, min_tokens)
    if backend == "auto":
        # 不删除实验算子：显式triton + min_tokens仍可复现旧形状。
        return False
    raise ValueError(f"unsupported MLP backend: {backend}")
