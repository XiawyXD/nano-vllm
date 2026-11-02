"""TP 控制协议；导入模块本身不加载模型或初始化 CUDA。

只有显式调用 common_kv_capacity 时才创建用于 NCCL MIN 的小 CUDA tensor。
"""

import os
import pickle
import time
import traceback
from multiprocessing.connection import wait


HEADER_BYTES = 4


def common_kv_capacity(local_blocks: int, world_size: int) -> int:
    """先对全组执行 MIN，再共同拒绝零容量，确保 block id 在各卡都合法。"""
    import torch
    import torch.distributed as dist
    capacity = torch.tensor([max(0, local_blocks)], dtype=torch.int64, device="cuda")
    if world_size > 1:
        dist.all_reduce(capacity, op=dist.ReduceOp.MIN)
    blocks = int(capacity.item())
    if blocks <= 0:
        raise RuntimeError("no KV cache capacity on at least one TP rank; reduce memory use or increase budget")
    return blocks


def encode_call(method_name: str, args: tuple, capacity: int) -> bytes:
    """先完整编码并检查长度，超限时不改写共享内存，也不唤醒 worker。"""
    if method_name not in {"run", "run_speculative", "run_plan", "exit"}:
        raise ValueError(f"unsupported TP method: {method_name}")
    payload = pickle.dumps((method_name, args), protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > capacity - HEADER_BYTES:
        raise ValueError(f"TP IPC payload {len(payload)} exceeds capacity {capacity - HEADER_BYTES}")
    return len(payload).to_bytes(HEADER_BYTES, "little") + payload


def decode_call(buffer) -> tuple[str, tuple]:
    """校验长度后读取一次消息；pickle 仅供本机可信父子进程使用。"""
    length = int.from_bytes(buffer[:HEADER_BYTES], "little")
    if length <= 0 or length > len(buffer) - HEADER_BYTES:
        raise ValueError(f"invalid TP IPC length: {length}")
    method_name, args = pickle.loads(bytes(buffer[HEADER_BYTES:HEADER_BYTES + length]))
    if method_name not in {"run", "run_speculative", "run_plan", "exit"} or not isinstance(args, tuple):
        raise ValueError("invalid TP IPC call")
    return method_name, args


def wait_worker_status(connections, expected: str, timeout: float) -> None:
    """在一个总时限内等待所有 rank；错误和 EOF 立即转换为可读异常。"""
    deadline = time.monotonic() + timeout
    pending = dict(connections)
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"TP workers {sorted(pending)} timed out waiting for {expected}")
        # select/poll 阻塞到任一 Pipe 就绪，不能固定 sleep 给每个 decode 加 10ms。
        ready = wait(list(pending.values()), timeout=remaining)
        if not ready:
            raise TimeoutError(f"TP workers {sorted(pending)} timed out waiting for {expected}")
        ranks = {connection: rank for rank, connection in pending.items()}
        for connection in ready:
            rank = ranks[connection]
            try:
                message = connection.recv()
            except EOFError as error:
                raise RuntimeError(f"TP rank {rank} exited before {expected}") from error
            status, detail = message
            if status != expected:
                raise RuntimeError(f"TP rank {rank}: expected {expected}, received {status}: {detail}")
            del pending[rank]


def stop_processes(processes, timeout: float) -> None:
    """有限等待后仅终止本引擎创建的子进程；不触碰其他用户 GPU 进程。"""
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
    for process in alive:
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)


def tp_worker_entry(config, rank, event, status_connection, parent_pid) -> None:
    """worker 启动/运行异常回传父进程；致命 CUDA 错误要求重建引擎。"""
    runner = None
    try:
        from nanovllm.engine.model_runner import ModelRunner
        runner = ModelRunner(config, rank, event)
        status_connection.send(("ready", None))
        while True:
            # 服务可以无限空闲，但不能在父进程消失后永久等待 Event。
            if not event.wait(1.0):
                if os.getppid() != parent_pid:
                    break
                continue
            method_name, args = runner.read_shm()
            result = getattr(runner, method_name)(*args)
            del result
            status_connection.send(("done", None))
            if method_name == "exit":
                break
    except BaseException:
        try:
            status_connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        try:
            if runner is not None:
                runner.exit()
        finally:
            status_connection.close()
