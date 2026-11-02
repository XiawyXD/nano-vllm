"""Triton fused RMSNorm kernels used by the optional runtime backend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    stride_x,
    stride_y,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)
    y = x * inv_rms * weight

    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    residual_out_ptr,
    stride_x,
    stride_residual,
    stride_y,
    stride_residual_out,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * stride_residual + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    summed = x + residual
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(
        residual_out_ptr + row * stride_residual_out + cols,
        summed,
        mask=mask,
    )

    mean_square = tl.sum(summed * summed, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)
    y = summed * inv_rms * weight
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


def _block_size(n_cols: int) -> int:
    if n_cols <= 0:
        raise ValueError("hidden size must be positive")
    return triton.next_power_of_2(n_cols)


def _num_warps(block_size: int) -> int:
    if block_size >= 1024:
        return 8
    if block_size >= 256:
        return 4
    return 2


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute RMSNorm for any tensor whose last dimension is hidden size."""
    if not x.is_cuda:
        raise ValueError("fused_rmsnorm requires a CUDA tensor")
    if not x.is_contiguous():
        x = x.contiguous()
    x_2d = x.reshape(-1, x.shape[-1])
    y_2d = torch.empty_like(x_2d)
    block_size = _block_size(x_2d.shape[1])
    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        weight,
        y_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        x_2d.shape[1],
        eps,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return y_2d.reshape_as(x)


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residual addition and RMSNorm in one Triton kernel."""
    if not x.is_cuda or not residual.is_cuda:
        raise ValueError("fused_add_rmsnorm requires CUDA tensors")
    if x.shape != residual.shape:
        raise ValueError("x and residual must have the same shape")
    if not x.is_contiguous():
        x = x.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()

    x_2d = x.reshape(-1, x.shape[-1])
    residual_2d = residual.reshape(-1, residual.shape[-1])
    y_2d = torch.empty_like(x_2d)
    residual_out_2d = torch.empty_like(residual_2d)
    block_size = _block_size(x_2d.shape[1])

    _fused_add_rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_2d,
        weight,
        y_2d,
        residual_out_2d,
        x_2d.stride(0),
        residual_2d.stride(0),
        y_2d.stride(0),
        residual_out_2d.stride(0),
        x_2d.shape[1],
        eps,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size),
    )
    return y_2d.reshape_as(x), residual_out_2d.reshape_as(residual)
