"""Tensor Core 实验共用的输入、独立参考、误差和版本记录。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import torch
import torch.nn.functional as F


def shapes(quick: bool):
    """返回 (案例名,M,K,I)；前者覆盖真实 Qwen3-1.7B，末尾故意不能整除tile。"""
    ms = (1, 16, 128) if quick else (1, 8, 16, 32, 128, 256, 512)
    result = [(f'qwen_m{m}', m, 2048, 6144) for m in ms]
    result.append(('odd_small', 3, 129, 97))
    if not quick:
        result.append(('odd_medium', 17, 255, 131))
    return result


def make_inputs(m, k, n, seed):
    """固定随机种子；输入方差1，权重方差1/K，防止尺寸变大只放大数值误差。"""
    generator = torch.Generator(device='cuda').manual_seed(seed)
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16, generator=generator)
    weight = torch.randn((n, k), device='cuda', dtype=torch.bfloat16, generator=generator)
    weight.mul_(1.0 / math.sqrt(k))
    return x, weight


def swiglu_activation(projected):
    """输入BF16 [M,2I]，前半gate后半up；供torch.compile生成基线融合激活。"""
    gate, up = projected.chunk(2, dim=-1)
    return F.silu(gate) * up


def compiled_activation():
    """直接使用项目真实SiluAndMul；保持其@torch.compile默认参数以公平对照。"""
    from nanovllm.layers.activation import SiluAndMul
    return SiluAndMul()


@contextmanager
def fp32_reference_mode():
    """关闭TF32参考近似，退出时恢复进程之前设置；不改候选BF16算子的实现。"""
    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def reference(kind, x, weight):
    """从确切BF16输入独立计算FP32参考；SwiGLU参考不插入BF16中间舍入。"""
    with fp32_reference_mode():
        projected = F.linear(x.float(), weight.float())
        if kind == 'fused_swiglu':
            gate, up = projected.chunk(2, dim=-1)
            return F.silu(gate) * up
        return projected


def compare(actual, expected, atol, rtol):
    """误差指标和pass判断分别保存；不因观测到误差就自动放宽预设公差。"""
    if actual.shape != expected.shape:
        return dict(shape_equal=False, actual_shape=list(actual.shape),
                    expected_shape=list(expected.shape), allclose=False)
    a, b = actual.float(), expected.float()
    difference = a-b
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if a.numel() == 0:
        return dict(shape_equal=True, finite=True, elements=0, max_absolute_error=0.0,
                    rmse=0.0, cosine=1.0, allclose=True, atol=atol, rtol=rtol)
    # 非有限输出应报告失败，不能把NaN指标写成仿佛可比较的JSON数字。
    if not finite:
        return dict(shape_equal=True, finite=False, elements=a.numel(),
                    max_absolute_error=None, rmse=None, cosine=None, allclose=False,
                    atol=atol, rtol=rtol)
    allowed = atol + rtol*b.abs()
    flat_a, flat_b = a.flatten().double(), b.flatten().double()
    return dict(shape_equal=True, finite=True, elements=a.numel(),
                max_absolute_error=difference.abs().max().item(),
                rmse=difference.square().mean().sqrt().item(),
                cosine=F.cosine_similarity(flat_a, flat_b, dim=0).item(),
                elements_exceeding_tolerance=int((difference.abs() > allowed).sum()),
                allclose=bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
                atol=atol, rtol=rtol)


def provenance():
    """记录当前代码哈希、设备、软件版本与基线算法，便于复现而非只认脚本名。"""
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    source_paths = [root/'nanovllm/layers/tensorcore_mlp.py'] + [
        root/name for name in ('benchmarks/tensorcore_utils.py', 'tests/regression/check_tensorcore.py', 'benchmarks/bench_tensorcore.py')]
    return dict(python=platform.python_version(), torch=torch.__version__,
                triton=importlib.metadata.version('triton'), cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source_paths if p.exists()},
                baseline_linear='torch.nn.functional.linear BF16 (PyTorch/CUDA backend)',
                baseline_swiglu='BF16 F.linear followed by project nanovllm.layers.activation.SiluAndMul(); original @torch.compile defaults, no dynamic/fullgraph override',
                fp32_reference='FP32 F.linear + FP32 activation, TF32 disabled; same exact BF16 input values')


def output_path(path):
    """创建父目录但拒绝覆盖旧证据；返回Path供脚本在finally保存完整或失败报告。"""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f'choose a new output file: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_report(target, report):
    """输出标准JSON；NaN必须显式转成失败/空指标，不能静默生成非标准JSON。"""
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def validate_cli(args):
    """验证数值检查配置；测试前固定阈值，整个测试过程不会自动修改它。"""
    for name in ('atol', 'rtol'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')
    if not torch.cuda.is_available():
        raise RuntimeError('this test requires one CUDA GPU')
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('BF16 Tensor Core experiment requires Ampere or newer')
