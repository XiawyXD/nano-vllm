"""有界元数据搬运：动态参考路径与两个 pinned 暂存帧。

这里仅保存 token ID、位置、页表等小整数数组，不保存模型激活或 KV。
CPU 写 pinned memory 之前等待上一轮 H2D 完成；设备写入和模型消费始终使用
同一 CUDA stream，因此设备端不会在前一轮模型读完之前被下一轮覆盖。
"""
from __future__ import annotations

import math
from collections import Counter

import torch


class MetadataBuffers:
    """每个执行器独占一个有界池；不能跨线程或跨 CUDA stream 并发调用。"""

    def __init__(self, backend="dynamic", max_bytes=64 * 2**20, frames=2):
        if backend not in {"dynamic", "reuse"}:
            raise ValueError("metadata backend must be dynamic or reuse")
        if max_bytes <= 0 or frames != 2:
            raise ValueError("metadata pool requires a positive budget and two frames")
        self.backend = backend
        self.max_bytes = max_bytes
        self.frames = [dict() for _ in range(frames)]
        self.events = [None] * frames
        self.frame_id = -1
        self.stream_id = None
        self.bytes = 0
        self.counters = Counter()
        self._active = False
        self.grew = False

    def begin(self):
        """开始一个打包事务；只等待将要复用帧的 H2D，不同步整个设备。"""
        if self._active:
            raise RuntimeError("metadata begin called before previous end")
        stream = torch.cuda.current_stream()
        if self.stream_id is None:
            self.stream_id = stream.cuda_stream
        if self.stream_id != stream.cuda_stream:
            raise RuntimeError("metadata pool cannot be shared across CUDA streams")
        self.frame_id = (self.frame_id + 1) % len(self.frames)
        event = self.events[self.frame_id]
        if event is not None and not event.query():
            self.counters["host_waits"] += 1
            event.synchronize()
        self.grew = False
        self._active = True
        self.counters["rounds"] += 1

    def tensor(self, name, values, dtype, *, shape=None, max_elements):
        """把 Python 值填入命名缓冲；返回仅覆盖有效元素的设备 tensor view。

        max_elements 来自配置容量，不从用户输入无限增长；name 是固定字段名。
        在容量内，reuse 模式不会创建新的 pinned/device tensor 存储。
        """
        if not self._active:
            raise RuntimeError("metadata tensor requires begin/end")
        shape = tuple(shape) if shape is not None else (len(values),)
        count = math.prod(shape)
        if count > max_elements or count < 0:
            raise ValueError(f"metadata {name} exceeds configured capacity")
        itemsize = {torch.int32: 4, torch.int64: 8, torch.float32: 4, torch.bool: 1}.get(dtype)
        if itemsize is None:
            raise ValueError(f"unsupported metadata dtype: {dtype}")
        self.counters["h2d_copies"] += int(count > 0)
        self.counters["h2d_bytes"] += count * itemsize
        if self.backend == "dynamic":
            self.counters["allocations"] += 2
            return torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True).reshape(shape).cuda(non_blocking=True)
        frame = self.frames[self.frame_id]
        entry = frame.get(name)
        if entry is not None and entry[0].dtype != dtype:
            raise ValueError(f"metadata field {name} changed dtype")
        if entry is None or entry[0].numel() < count:
            if entry is None and len(frame) >= 32:
                raise ValueError("too many metadata buffer names")
            capacity = min(max_elements, max(1, 1 << (max(1, count) - 1).bit_length()))
            old_bytes = 0 if entry is None else 2 * entry[0].numel() * itemsize
            new_bytes = 2 * capacity * itemsize
            if self.bytes - old_bytes + new_bytes > self.max_bytes:
                raise MemoryError("metadata buffer budget exceeded")
            cpu = torch.empty(capacity, dtype=dtype, device="cpu", pin_memory=True)
            device = torch.empty(capacity, dtype=dtype, device="cuda")
            entry = (cpu, device)
            frame[name] = entry
            self.bytes += new_bytes - old_bytes
            self.counters["allocations"] += 2
            self.counters["growths"] += 1
            self.grew = True
        cpu_view = entry[0][:count].view(shape)
        device_view = entry[1][:count].view(shape)
        if count:
            # numpy view 与 pinned tensor 共享存储；避免每个字段额外构造 CPU tensor。
            cpu_view.numpy()[...] = values
            device_view.copy_(cpu_view, non_blocking=True)
        self.counters["reuse_calls"] += 1
        return device_view

    def end(self):
        """记录所有 H2D 的完成点；异常路径也必须调用，防止过早复用 CPU 存储。"""
        if not self._active:
            return
        if self.backend == "reuse":
            if self.events[self.frame_id] is None:
                self.events[self.frame_id] = torch.cuda.Event()
            self.events[self.frame_id].record(torch.cuda.current_stream())
        self._active = False

    def stats(self):
        """返回有界聚合，不泄漏输入 token 内容。"""
        return dict(self.counters, allocated_bytes=self.bytes,
                    max_bytes=self.max_bytes, frames=len(self.frames), backend=self.backend)
