"""默认物理GPU1；显式 NANOVLLM_BENCH_GPU_INDEX 可选择经用户批准的实验卡。

只读nvidia-smi和/proc，不向任何进程发信号。不凭用户名判定进程属于自己；
允许的PID必须是当前验收主进程或其仍可核验的后代。先检查再启动仍存在竞态，
所以长矩阵应在每个trial/worker边界再次检查，而非只在脚本开头检查一次。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


class GPUUnavailable(RuntimeError):
    """GPU被占用、环境指向错误或无法验证占用；调用方应保留证据并停止新增工作。"""


def _query(fields, kind):
    """解析固定nvidia-smi CSV；命令失败必须停止，不能把未知占用当空闲。"""
    try:
        result = subprocess.run(["nvidia-smi", f"--query-{kind}={','.join(fields)}",
                                 "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise GPUUnavailable(f"GPU occupancy query failed: {type(error).__name__}") from error
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != len(fields):
                raise GPUUnavailable("cannot parse GPU occupancy response")
            rows.append(dict(zip(fields, parts)))
    return rows


def _parent(pid):
    """读取Linux进程父PID；死亡、权限不足等情况不猜测所有权。"""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def is_descendant(pid, owner_pid):
    """沿当前OS父子链核实所有权；最多256层，循环或缺失均视为不属于自己。"""
    visited = set()
    for _ in range(256):
        if pid == owner_pid:
            return True
        if pid is None or pid <= 1 or pid in visited:
            return False
        visited.add(pid)
        pid = _parent(pid)
    return False


def ensure_gpu1_available(*, owner_pid=None, evidence_path=None):
    """返回只读快照；仅当前任务拥有的GPU进程可继续存在，其余一律拒绝。

    owner_pid默认当前进程。上层多进程验收可传自己的父验收PID，但不能传任意
    别人的PID来绕过检查。环境必须只可见选定物理卡（默认GPU1）的完整UUID；数字ordinal
    与nvidia-smi物理index的顺序未必一致，不能接受数字'1'。不自动改写环境。
    """
    owner_pid = os.getpid() if owner_pid is None else owner_pid
    if type(owner_pid) is not int or owner_pid <= 1:
        raise GPUUnavailable("GPU guard owner must be a concrete task PID greater than 1")
    if not is_descendant(os.getpid(), owner_pid):
        raise GPUUnavailable("GPU guard owner must be this process or its current ancestor")
    # 保留旧入口名兼容所有 harness；没有显式设置时仍严格限定 GPU1。
    selected = os.environ.get("NANOVLLM_BENCH_GPU_INDEX", "1")
    if not selected.isascii() or not selected.isdecimal():
        raise GPUUnavailable("NANOVLLM_BENCH_GPU_INDEX must be a nonnegative physical index")
    devices = _query(("index", "uuid", "memory.used", "utilization.gpu"), "gpu")
    gpu = next((row for row in devices if row["index"] == selected), None)
    if gpu is None:
        raise GPUUnavailable(f"physical GPU{selected} was not found")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu["uuid"]:
        raise GPUUnavailable(
            f"CUDA_VISIBLE_DEVICES must be the complete physical GPU{selected} UUID: {gpu['uuid']}; "
            "numeric ordinals (including '1') are not accepted")
    applications = _query(("gpu_uuid", "pid", "used_memory"), "compute-apps")
    applications = [row for row in applications if row["gpu_uuid"] == gpu["uuid"]]
    foreign = [row for row in applications if not is_descendant(int(row["pid"]), owner_pid)]
    snapshot = dict(timestamp=time.time(), owner_pid=owner_pid, gpu=gpu,
                    compute_processes=applications, foreign_processes=foreign,
                    status="AVAILABLE" if not foreign else "OCCUPIED")
    if evidence_path is not None:
        # 仅追加观察，不覆盖原始记录；该文件属于当前唯一验收进程。
        with Path(evidence_path).open("a", encoding="utf-8") as file:
            file.write(json.dumps(snapshot) + "\n")
    if foreign:
        raise GPUUnavailable(f"physical GPU{selected} is occupied by other computation; no new experiment started")
    return snapshot


if __name__ == "__main__":
    print(json.dumps(ensure_gpu1_available(), indent=2))
