"""轻量级运行时 metrics 工具。

核心引擎不依赖第三方监控系统；benchmark 可以复用这里的百分位数、
延迟摘要和 CUDA memory snapshot，保证不同实验口径一致。
"""

from __future__ import annotations

from statistics import mean, median

import torch


def percentile(values: list[float], p: float) -> float | None:
    """返回 nearest-rank 风格的百分位数；空输入返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    """统一输出 count/mean/median/p95/p99，便于写入 JSON。"""
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def cuda_memory_snapshot() -> dict[str, int | None]:
    """读取当前 CUDA allocator 和设备剩余显存，单位均为 bytes。"""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
        "device_free": free,
        "device_total": total,
    }


def numeric_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    """对两个同构的计数器做 after-before，缺失字段按 0 处理。"""
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}
