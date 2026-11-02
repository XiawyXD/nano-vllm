"""TP CPU 协议回归；不加载模型、不创建 CUDA tensor、不启动 NCCL GPU 作业。

运行 python -m tests.regression.check_tp_runtime。PASS 只证明配置/IPC/回收协议，
不替代 TP2 logits、CUDA Graph、NCCL 带宽或真实显存分配验收。
"""
import json
import os
import multiprocessing as mp
import tempfile
import time
import threading
from datetime import timedelta
from multiprocessing.connection import wait as connection_wait
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.tp_runtime import (
    common_kv_capacity, encode_call, decode_call, wait_worker_status, stop_processes,
)


def expect(error_type, action):
    """断言一个输入明确失败，避免测试静默接受非法参数。"""
    try:
        action()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def check_config():
    """用假的 Hugging Face 结构配置验证 TP 分片整除及端口/时限。"""
    hf = dict(num_attention_heads=4, num_key_value_heads=2, hidden_size=16,
              intermediate_size=32, vocab_size=64, max_position_embeddings=128)
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
        with patch("nanovllm.config.AutoConfig.from_pretrained", return_value=SimpleNamespace(**hf)):
            valid = Config(directory, tensor_parallel_size=2)
            assert valid.tp_master_port is None
            for key, value in (("tp_master_port", 0), ("tp_master_port", 65536),
                               ("tp_master_port", True), ("tp_timeout_seconds", float("nan")),
                               ("tp_startup_timeout_seconds", 0), ("tp_ipc_bytes", 4)):
                expect(ValueError, lambda k=key, v=value: Config(directory, **{k: v}))
            with patch.dict(os.environ, {"NANOVLLM_MASTER_PORT": "34567"}):
                assert Config(directory).tp_master_port == 34567
                assert Config(directory, tp_master_port=34568).tp_master_port == 34568
        for field in ("num_attention_heads", "num_key_value_heads", "hidden_size", "intermediate_size", "vocab_size"):
            bad = dict(hf, **{field: 3})
            with patch("nanovllm.config.AutoConfig.from_pretrained", return_value=SimpleNamespace(**bad)):
                expect(ValueError, lambda: Config(directory, tensor_parallel_size=2))


def check_capacity_min():
    """把临时 tensor 放 CPU，模拟 MIN 结果；零容量也必须先参加 collective。"""
    tensor = torch.tensor
    calls = []

    def cpu_tensor(*args, **kwargs):
        kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    def reduce_to(value):
        def reduction(capacity, op):
            calls.append(int(capacity.item()))
            assert op == dist.ReduceOp.MIN
            capacity.fill_(min(int(capacity.item()), value))
        return reduction

    with patch("torch.tensor", side_effect=cpu_tensor):
        with patch("torch.distributed.all_reduce", side_effect=reduce_to(7)):
            assert common_kv_capacity(12, 2) == 7
            assert common_kv_capacity(7, 2) == 7
        with patch("torch.distributed.all_reduce", side_effect=reduce_to(0)):
            expect(RuntimeError, lambda: common_kv_capacity(12, 2))
            expect(RuntimeError, lambda: common_kv_capacity(-3, 2))
        with patch("torch.distributed.all_reduce") as reduce:
            assert common_kv_capacity(9, 1) == 9
            reduce.assert_not_called()
    assert calls == [12, 7, 12, 0]


def cpu_worker(name, event, sender):
    """真实子进程只读取 CPU 共享内存；模拟 done/error，不执行模型。"""
    shm = SharedMemory(name=name)
    try:
        sender.send(("ready", None))
        for _ in range(2):
            if not event.wait(5):
                raise TimeoutError("test dispatch missing")
            method, args = decode_call(shm.buf)
            event.clear()
            assert method == "run" and args[0] in (1, 2)
            sender.send(("done", None))
    finally:
        shm.close()
        sender.close()


def check_protocol():
    """检查共享内存隔离、消息上限、两轮真实跨进程 ack，以及错误/EOF/超时。"""
    capacity = 4096
    message = encode_call("run", ([1, 2], True), capacity)
    buffer = bytearray(capacity)
    buffer[:len(message)] = message
    assert decode_call(buffer) == ("run", ([1, 2], True))
    expect(ValueError, lambda: encode_call("run", ("x" * capacity,), capacity))
    expect(ValueError, lambda: decode_call(bytearray(capacity)))
    buffer[:4] = capacity.to_bytes(4, "little")
    expect(ValueError, lambda: decode_call(buffer))
    expect(ValueError, lambda: encode_call("__dict__", (), capacity))

    ctx = mp.get_context("spawn")
    shm = SharedMemory(create=True, size=capacity)
    another = SharedMemory(create=True, size=capacity)
    process = None
    receiver, sender = ctx.Pipe(duplex=False)
    try:
        assert shm.name != another.name
        event = ctx.Event()
        process = ctx.Process(target=cpu_worker, args=(shm.name, event, sender))
        process.start()
        sender.close()
        wait_worker_status([(1, receiver)], "ready", 20)
        runner = ModelRunner.__new__(ModelRunner)
        runner.world_size, runner.rank = 2, 0
        runner.shm, runner.event = shm, [event]
        runner.worker_connections = [(1, receiver)]
        runner.config = SimpleNamespace(tp_timeout_seconds=5)
        runner._failed = False
        runner.run = lambda value: value * 10
        assert runner.call("run", 1) == 10
        assert runner.call("run", 2) == 20
        process.join(5)
        assert process.exitcode == 0
        assert bytes(another.buf) == bytes(capacity)
        # 序列化溢出发生在 dispatch 前；event 和运行时失败标记都不应改变。
        expect(ValueError, lambda: runner.call("run", "x" * capacity))
        assert not event.is_set() and not runner._failed
    finally:
        if process is not None:
            stop_processes([process], 1)
        receiver.close()
        sender.close()
        for handle in (shm, another):
            handle.close()
            handle.unlink()

    for case in ("error", "eof", "timeout"):
        receiver, sender = ctx.Pipe(duplex=False)
        try:
            if case == "error":
                sender.send(("error", "injected worker failure"))
            if case == "eof":
                sender.close()
            expected = TimeoutError if case == "timeout" else RuntimeError
            expect(expected, lambda: wait_worker_status([(1, receiver)], "done", 0.03))
        finally:
            receiver.close()
            sender.close()

    # DONE 尚未到达时确实进入阻塞 wait；发送后按就绪唤醒，不设脆弱微秒阈值。
    receiver, sender = ctx.Pipe(duplex=False)

    def delayed_done():
        time.sleep(0.05)
        sender.send(("done", None))

    thread = threading.Thread(target=delayed_done)
    try:
        thread.start()
        with patch("nanovllm.engine.tp_runtime.wait", wraps=connection_wait) as waited:
            wait_worker_status([(1, receiver)], "done", 2.0)
            assert waited.called
    finally:
        thread.join(2)
        receiver.close()
        sender.close()

    # exit ACK 使用独立 shutdown 时限，而非普通推理的 120 秒时限。
    runner = ModelRunner.__new__(ModelRunner)
    runner.world_size, runner.rank, runner._failed = 2, 0, False
    runner.shm = SimpleNamespace(buf=bytearray(capacity))
    runner.event = [ctx.Event()]
    runner.worker_connections = []
    runner.config = SimpleNamespace(tp_timeout_seconds=120.0, tp_shutdown_timeout_seconds=0.05)
    runner.exit = lambda: None
    with patch("nanovllm.engine.model_runner.wait_worker_status") as waited:
        runner.call("exit")
        waited.assert_called_once_with([], "done", 0.05)

    # rank0 的 forward 已返回、但 worker 回报错误时，也必须触发引擎清理。
    receiver, sender = ctx.Pipe(duplex=False)
    try:
        sender.send(("error", "injected post-forward worker failure"))
        runner = ModelRunner.__new__(ModelRunner)
        runner.world_size, runner.rank, runner._failed = 2, 0, False
        runner.shm = SimpleNamespace(buf=bytearray(capacity))
        runner.event = [ctx.Event()]
        runner.worker_connections = [(1, receiver)]
        runner.config = SimpleNamespace(tp_timeout_seconds=0.1)
        runner.run = lambda: None
        cleaned = []
        runner._failure_callback = lambda: cleaned.append(True)
        expect(RuntimeError, lambda: runner.call("run"))
        assert runner._failed and cleaned == [True]
        expect(RuntimeError, lambda: runner.call("run"))
        assert cleaned == [True]
    finally:
        receiver.close()
        sender.close()


def check_ports():
    """两个 CPU TCPStore 同时存活时，系统分配不同端口，无查找/关闭竞态。"""
    stores = [dist.TCPStore("127.0.0.1", 0, 1, is_master=True,
                           timeout=timedelta(seconds=3), wait_for_workers=False)
              for _ in range(2)]
    assert stores[0].port != stores[1].port
    for index, store in enumerate(stores):
        store.set("identity", str(index))
        assert store.get("identity") == str(index).encode()


def check_cleanup():
    """注入启动失败与运行失败，验证本引擎 SHM 回收、子进程有限终止、幂等退出。"""
    hf = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, hidden_size=16,
                         intermediate_size=32, vocab_size=64, max_position_embeddings=128)
    context = mp.get_context("spawn")

    class FailedProcess:
        def start(self):
            raise RuntimeError("injected spawn failure")

    fake_context = SimpleNamespace(Event=context.Event, Pipe=context.Pipe,
                                   Process=lambda **kwargs: FailedProcess())
    created = []

    def tracked_shm(**kwargs):
        handle = SharedMemory(**kwargs)
        created.append(handle.name)
        return handle

    with tempfile.TemporaryDirectory() as directory:
        with patch("nanovllm.config.AutoConfig.from_pretrained", return_value=hf), \
                patch("nanovllm.engine.llm_engine.AutoTokenizer.from_pretrained", return_value=object()), \
                patch("nanovllm.engine.llm_engine.torch.cuda.device_count", return_value=2), \
                patch("nanovllm.engine.llm_engine.mp.get_context", return_value=fake_context), \
                patch("nanovllm.engine.llm_engine.SharedMemory", side_effect=tracked_shm):
            engine = LLMEngine.__new__(LLMEngine)
            expect(RuntimeError, lambda: engine.__init__(directory, tensor_parallel_size=2))
            assert engine._tp_shm is None and engine._tp_store is None and engine.ps == []
    assert len(created) == 1
    expect(FileNotFoundError, lambda: SharedMemory(name=created[0]))

    # 只启动一个睡眠 CPU 子进程，用于检验失败路径不会永久 join。
    process = context.Process(target=time.sleep, args=(20,))
    process.start()
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(tp_shutdown_timeout_seconds=0.05)
    engine.ps, engine.events = [process], []
    engine._worker_connections, engine._tp_store = [], None
    engine._tp_shm = SharedMemory(create=True, size=4096)
    name = engine._tp_shm.name
    exits = []
    engine.model_runner = SimpleNamespace(_failed=True, exit=lambda: exits.append(True))
    start = time.monotonic()
    engine.exit()
    engine.exit()
    assert not process.is_alive() and time.monotonic() - start < 6
    assert exits == [True] and engine.ps == []
    expect(FileNotFoundError, lambda: SharedMemory(name=name))


def main():
    checks = [check_config, check_capacity_min, check_protocol, check_ports, check_cleanup]
    for check in checks:
        check()
    print(json.dumps({"status": "PASS", "device": "CPU", "checks": [fn.__name__ for fn in checks],
                      "tp2_gpu_validated": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
