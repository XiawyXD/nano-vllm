"""元数据双缓冲回归；默认纯CPU假torch/异步H2D，--gpu才运行真实设备。

CPU模式模拟尚未完成的复制，并在过早改写pinned源时主动报错。它能验证
生命周期/容量/字段更新，却不代替实际CUDA驱动、吞吐或Graph地址兼容验证。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class FakeEvent:
    """事件记录队列完成点，query不自动推进；synchronize才执行到该点。"""

    def __init__(self, cuda):
        self.cuda = cuda
        self.target = None
        self.waits = 0

    def record(self, stream):
        assert stream.cuda_stream == self.cuda.stream.cuda_stream
        self.target = len(self.cuda.copies)
        self.cuda.trace.append(("record", id(self), self.target))

    def query(self):
        return self.target is not None and self.cuda.completed >= self.target

    def synchronize(self):
        assert self.target is not None
        self.waits += 1
        self.cuda.trace.append(("wait", id(self), self.target))
        self.cuda.sync_to(self.target)


class FakeCuda:
    """有序流中只保存异步copy引用，不提前复制，暴露pinned源被覆盖的风险。"""

    def __init__(self):
        self.stream = SimpleNamespace(cuda_stream=7)
        self.copies = []
        self.completed = 0
        self.trace = []

    def current_stream(self):
        return self.stream

    def Event(self):
        return FakeEvent(self)

    def sync_to(self, target):
        for source, destination in self.copies[self.completed:target]:
            np.copyto(destination.array, source.array)
        self.completed = max(self.completed, target)

    def synchronize(self):
        self.sync_to(len(self.copies))

    def pending_source(self, root):
        return any(source.root is root for source, _ in self.copies[self.completed:])


class FakeTensor:
    """只实现MetadataBuffers使用的tensor接口，所有storage均为CPU numpy。"""

    def __init__(self, api, array, dtype, device="cpu", root=None):
        self.api, self.array, self.dtype, self.device = api, array, dtype, device
        self.root = self if root is None else root

    def numel(self):
        return self.array.size

    def __getitem__(self, selection):
        return FakeTensor(self.api, self.array[selection], self.dtype, self.device, self.root)

    def view(self, shape):
        return FakeTensor(self.api, self.array.reshape(shape), self.dtype, self.device, self.root)

    reshape = view

    def numpy(self):
        assert self.device == "cpu"
        if self.api.cuda.pending_source(self.root):
            raise AssertionError("host staging source overwritten before H2D event completion")
        self.api.cuda.trace.append(("host_write_view", id(self.root)))
        return self.array

    def copy_(self, source, non_blocking=False):
        assert self.device == "cuda" and source.device == "cpu" and non_blocking
        self.api.cuda.copies.append((source, self))
        self.api.cuda.trace.append(("copy", id(source.root)))
        return self

    def cuda(self, non_blocking=False):
        result = FakeTensor(self.api, np.empty_like(self.array), self.dtype, "cuda")
        return result.copy_(self, non_blocking=non_blocking)

    def data_ptr(self):
        return self.array.ctypes.data

    def tolist(self):
        self.api.cuda.synchronize()
        return self.array.tolist()


class FakeTorch:
    """dtype使用numpy认可的字符串；不导入或初始化真实torch/CUDA。"""

    int32, int64, float32, bool = "int32", "int64", "float32", "bool"

    def __init__(self):
        self.cuda = FakeCuda()

    def empty(self, count, dtype, device="cpu", pin_memory=False):
        assert device == "cuda" or pin_memory
        return FakeTensor(self, np.empty(count, dtype=dtype), dtype, device)

    def tensor(self, values, dtype, device="cpu", pin_memory=False):
        assert pin_memory and device == "cpu"
        return FakeTensor(self, np.asarray(values, dtype=dtype), dtype, device)


def load_module(torch_api):
    """按文件载入待测实现，绕过nanovllm顶层的模型依赖与GPU模块导入。"""
    path = Path(__file__).resolve().parents[2] / "nanovllm/engine/metadata_buffers.py"
    spec = importlib.util.spec_from_file_location("_metadata_buffers_regression", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch_api}):
        spec.loader.exec_module(module)
    return module


def expect(error, action):
    try:
        action()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def values_of(tensor):
    return tensor.tolist() if isinstance(tensor, FakeTensor) else tensor.cpu().tolist()


def check_updates_and_two_frame_addresses(module, api):
    """每轮改所有输入字段；每帧容量足够时地址必须保持不变，增长后仍有界。"""
    pool = module.MetadataBuffers("reuse", max_bytes=16384)
    pointers = {}
    total_elements = 0
    for turn in range(8):
        count = 3 if turn < 4 else 7 if turn < 6 else 6
        fields = {
            "input_ids": ([turn * 100 + index for index in range(count)], api.int64, None, 16),
            "positions": ([turn + index for index in range(count)], api.int64, None, 16),
            "slot_mapping": ([turn * 20 + index for index in range(count)], api.int32, None, 16),
            "context_lens": ([turn + 4, turn + 7], api.int32, None, 8),
            "pages_0": ([[turn + 1, turn + 2, -1], [turn + 3, -1, -1]], api.int32, (2, 3), 24),
            "cu_q_0": ([0, 1, count], api.int32, None, 9),
            "cu_k_0": ([0, turn + 4, 2 * turn + 11], api.int32, None, 9),
            "logits_indices": ([0, count - 1], api.int64, None, 16),
        }
        pool.begin()
        outputs = {}
        try:
            for name, (values, dtype, shape, maximum) in fields.items():
                outputs[name] = pool.tensor(name, values, dtype, shape=shape, max_elements=maximum)
                total_elements += np.asarray(values).size
                entry = pool.frames[pool.frame_id][name]
                key = pool.frame_id, name
                current = (entry[0].data_ptr(), entry[1].data_ptr(), entry[0].numel())
                if key in pointers and pointers[key][2] >= np.asarray(values).size:
                    assert current == pointers[key], ("storage changed within capacity", key)
                pointers[key] = current
        finally:
            pool.end()
        for name, output in outputs.items():
            assert values_of(output) == fields[name][0], ("stale metadata", turn, name)
        assert pool.bytes <= pool.max_bytes
    assert len(pool.frames) == 2
    assert pool.stats()["rounds"] == 8 and pool.stats()["h2d_copies"] == 64
    assert pool.stats()["reuse_calls"] == 64 and total_elements > 0


def check_cpu_event_lifecycle(module, api):
    """第三轮重用第一帧时，先等待其未完成事件，第二帧不得被误当作可改写。"""
    pool = module.MetadataBuffers("reuse", 1024)
    pool.begin()
    first = pool.tensor("ids", [1, 2], api.int64, max_elements=8)
    pool.end()
    first_event = pool.events[0]
    assert not first_event.query()
    pool.begin()
    second = pool.tensor("ids", [3, 4], api.int64, max_elements=8)
    pool.end()
    second_event = pool.events[1]
    assert not first_event.query() and not second_event.query()
    assert api.cuda.completed == 0
    pool.begin()
    assert pool.frame_id == 0 and first_event.waits == 1
    assert api.cuda.completed == 1 and not second_event.query()
    # 第一份结果已安全复制到GPU模拟存储，随后才允许覆盖同一帧CPU源。
    assert first.array.tolist() == [1, 2]
    third = pool.tensor("ids", [5, 6], api.int64, max_elements=8)
    pool.end()
    assert first.data_ptr() == third.data_ptr()
    assert second.data_ptr() != third.data_ptr()
    assert pool.stats()["host_waits"] == 1
    api.cuda.synchronize()
    assert third.array.tolist() == [5, 6] and second.array.tolist() == [3, 4]
    # 已完成的事件不能制造多余host wait。
    pool.begin()
    pool.tensor("ids", [7, 8], api.int64, max_elements=8)
    pool.end()
    assert pool.stats()["host_waits"] == 1


def check_capacity_and_state_guards(module, api):
    for kwargs in (dict(backend="unknown"), dict(max_bytes=0), dict(frames=1)):
        expect(ValueError, lambda kwargs=kwargs: module.MetadataBuffers(**kwargs))
    pool = module.MetadataBuffers("reuse", max_bytes=64)
    expect(RuntimeError, lambda: pool.tensor("x", [1], api.int64, max_elements=8))
    for _ in range(2):
        pool.begin()
        expect(RuntimeError, pool.begin)
        pool.tensor("x", [1, 2], api.int64, max_elements=8)
        pool.end()
    assert pool.bytes == 64
    before = [(entry[0].data_ptr(), entry[1].data_ptr()) for frame in pool.frames for entry in frame.values()]
    pool.begin()
    try:
        expect(MemoryError, lambda: pool.tensor("x", [1, 2, 3, 4], api.int64, max_elements=8))
        expect(ValueError, lambda: pool.tensor("x", [1, 2], api.float32, max_elements=8))
        expect(ValueError, lambda: pool.tensor("x", [1, 2, 3], api.int64, max_elements=2))
        assert pool.bytes == 64
    finally:
        pool.end()
    after = [(entry[0].data_ptr(), entry[1].data_ptr()) for frame in pool.frames for entry in frame.values()]
    assert before == after
    pool = module.MetadataBuffers("reuse", max_bytes=4096)
    pool.begin()
    try:
        for index in range(32):
            pool.tensor(f"field_{index}", [], api.int32, max_elements=8)
        expect(ValueError, lambda: pool.tensor("field_32", [], api.int32, max_elements=8))
    finally:
        pool.end()
    assert pool.stats()["h2d_copies"] == 0


def check_cpu_stream_guard(module, api):
    pool = module.MetadataBuffers("reuse", 1024)
    pool.begin()
    pool.tensor("x", [1], api.int64, max_elements=2)
    pool.end()
    api.cuda.stream = SimpleNamespace(cuda_stream=8)
    expect(RuntimeError, pool.begin)
    assert not pool._active


def check_dynamic_equivalence(module, api):
    pool = module.MetadataBuffers("dynamic", 1024)
    for values in ([1, 2, 3], [], [8, 9]):
        pool.begin()
        try:
            result = pool.tensor("x", values, api.int64, max_elements=8)
        finally:
            pool.end()
        assert values_of(result) == values
    assert pool.bytes == 0 and all(not frame for frame in pool.frames)
    assert pool.stats()["allocations"] == 6


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gpu", action="store_true", help="explicitly run real CUDA storage/value checks")
    parser.add_argument("--output", type=Path, help="write a new JSON result; existing files are never overwritten")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new evidence path")
    checks = [check_updates_and_two_frame_addresses, check_capacity_and_state_guards, check_dynamic_equivalence]
    if args.gpu:
        # 在import torch/任何CUDA操作之前验证设备UUID与当前进程树所有权。
        guard_path = (Path(__file__).resolve().parents[2] / "benchmarks/runtime_v2_gpu_guard.py")
        guard_spec = importlib.util.spec_from_file_location("_metadata_gpu_guard", guard_path)
        guard = importlib.util.module_from_spec(guard_spec)
        guard_spec.loader.exec_module(guard)
        guard.ensure_gpu1_available()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu requires an explicitly allocated CUDA device")
        api = torch
        module = load_module(api)
        for check in checks:
            check(module, api)
        torch.cuda.synchronize()
        device = torch.cuda.get_device_name()
        scope = "real CUDA allocation/address/value checks; not a throughput benchmark"
    else:
        checks += [check_cpu_event_lifecycle, check_cpu_stream_guard]
        for check in checks:
            api = FakeTorch()
            check(load_module(api), api)
        device = "CPU fake asynchronous H2D"
        scope = "simulated event lifecycle; no CUDA initialization or GPU performance claim"
    report = dict(status="PASS", device=device, checks=[check.__name__ for check in checks], scope=scope)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
