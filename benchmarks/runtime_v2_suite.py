"""按显式计划运行验收阶段，支持严格resume及终态归档；默认不授权GPU。

计划JSON格式：{"fixtures":["benchmarks/workloads/runtime_v2_holdout.jsonl"],
 "stages":[{"name":"adaptive","entry":"adaptive","args":[],"timeout_seconds":120},
 {"name":"quality","entry":"quality","args":["run","--model","...",
 "--candidate-config","...","--output","{stage_output}"],"timeout_seconds":4000,
 "result_json":"{stage_output}/comparison.json"}]}

entry只允许已知benchmarks入口，绝不执行任意shell文本。每阶段一个独立进程，日志和
输出按attempt编号保存；FAIL重试用新路径。GPU阶段须明确--allow-gpu，否则NOT_RUN。
resume同时核对源码/计划/fixture哈希和真实PID出生标记；lock文件不是进程存活证据。
--stages只选择本次执行的阶段名，不裁剪冻结计划；selected_status与全计划status分开。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def entries() -> dict:
    """从现有统一入口复用白名单，再加入本轮纯CPU和质量入口。"""
    path = ROOT / "benchmarks/__main__.py"
    spec = importlib.util.spec_from_file_location("_runtime_suite_entries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapping = {name: target for name, (target, _) in module.ENTRIES.items()}
    mapping.update(adaptive="module:tests.regression.check_adaptive_speculation",
                   quality="module:benchmarks.runtime_v2_quality")
    return mapping


CPU_ENTRIES = {"adaptive", "unit", "api", "engine-hardening", "runtime-hardening", "service-hardening",
               "mlp-dispatch", "tp-runtime", "tensorcore-compare", "ngram", "execution-plan",
               "planned-engine", "packed-metadata-cpu", "metadata-cpu", "runtime-v2-harness", "gpu-guard", "evaluation-harness"}


def process_identity(pid: int) -> dict | None:
    """返回OS进程出生标记；PID复用不应把无关进程误认为旧验收进程。只读，不发信号。"""
    if sys.platform.startswith("linux"):
        try:
            # comm字段可以包含空格/括号，所以从最后一个右括号之后取stat字段。
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            return dict(pid=pid, birth=fields[19])
        except FileNotFoundError:
            return None
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return None
            raise RuntimeError(f"cannot safely inspect PID {pid}")
        try:
            created, exited, kernel_time, user_time = (ctypes.c_ulonglong() for _ in range(4))
            code = wintypes.DWORD()
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise RuntimeError("GetProcessTimes failed")
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError("GetExitCodeProcess failed")
            return dict(pid=pid, birth=str(created.value)) if code.value == 259 else None
        finally:
            kernel.CloseHandle(handle)
    raise RuntimeError("safe resume requires Linux /proc or Windows process creation time")


def ensure_terminated(identity: dict | None) -> None:
    """只有旧PID已消失/已终止/被复用时才允许继续；权限不足则拒绝resume。"""
    if identity and process_identity(identity["pid"]) == identity:
        raise RuntimeError(f"prior process is still alive: PID {identity['pid']}; do not resume concurrently")


def ensure_group_empty(group_id: int | None) -> None:
    """Linux独立会话的leader退出后仍可能有worker；只读检查整个旧进程组。"""
    if group_id is None or not sys.platform.startswith("linux"):
        return
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if fields[0] != "Z" and int(fields[2]) == group_id:
            raise RuntimeError(f"prior process group {group_id} still contains PID {path.name}")


@contextmanager
def exclusive_guard(directory: Path):
    """OS文件锁只防并发写，不能证明旧PID已结束；run/resume/archive还需独立查进程。"""
    with (directory / ".suite.guard").open("a+b") as file:
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another suite invocation holds the OS output lock") from error
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def save_state(path: Path, state: dict) -> None:
    """只替换本会话自己的进度文件；原始阶段日志/结果仍为只写一次。"""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def validate_successful_evidence(state: dict) -> None:
    """resume与归档都复核验收当时的证据SHA；不能只给被篡改后的文件重新算散列。"""
    for stage in state["stages"]:
        for attempt in stage["attempts"]:
            if attempt.get("status") != "PASS":
                continue
            if "log" not in attempt:
                raise RuntimeError("successful stage is missing its original log identity")
            for path_key, hash_key in (("log", "log_sha256"), ("result_json", "result_sha256")):
                if path_key in attempt:
                    path = Path(attempt[path_key])
                    if hash_key not in attempt or not path.is_file() or digest(path) != attempt[hash_key]:
                        raise RuntimeError("successful stage evidence changed or disappeared")


def fingerprint(plan_path: Path, plan: dict) -> dict:
    """源码、计划配置和语料必须共同匹配；运行时的日志文件不进入输入指纹。"""
    files = set((ROOT / "nanovllm").rglob("*.py")) | set((ROOT / "benchmarks").rglob("*.py")) | set(ROOT.glob("*.py"))
    fixtures = {}
    for value in plan.get("fixtures", ["benchmarks/workloads/runtime_v2_holdout.jsonl"]):
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
        fixtures[str(path.resolve())] = digest(path)
    configs = {}
    config_paths = list(plan.get("configs", []))
    for stage in plan["stages"]:
        arguments = stage.get("args", [])
        for value, parameter in option_pairs(arguments):
            if value in {"--candidate-config", "--config", "--corpus"}:
                config_paths.append(parameter)
            elif value == "--model":
                model = Path(parameter)
                model = model if model.is_absolute() else ROOT / model
                for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                    if (model / filename).is_file():
                        config_paths.append(str(model / filename))
    for value in config_paths:
        if "{" in value:
            raise ValueError("input config/fixture cannot depend on generated stage output")
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
        configs[str(path.resolve())] = digest(path)
    return dict(plan_sha256=digest(plan_path), fixtures_sha256=fixtures, configs_sha256=configs,
                source_sha256={path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(files)})


def option_pairs(arguments: list[str]):
    """同时识别argparse允许的--flag value和--flag=value，避免绕过输入/输出检查。"""
    for index, value in enumerate(arguments):
        if not value.startswith("--"):
            continue
        if "=" in value:
            yield tuple(value.split("=", 1))
        elif index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            yield value, arguments[index + 1]


def validate_plan(plan: dict, mapping: dict) -> None:
    """计划不能静默拼错入口、重复阶段名或取消超时上限。"""
    if not isinstance(plan, dict) or not isinstance(plan.get("stages"), list) or not plan["stages"]:
        raise ValueError("plan requires a nonempty stages list")
    names = set()
    for stage in plan["stages"]:
        name = stage.get("name", "")
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name) or name in names:
            raise ValueError("stage names must be unique lowercase alphanumeric/hyphen/underscore")
        names.add(name)
        if stage.get("entry") not in mapping or stage.get("entry") == "archive":
            raise ValueError("unknown/unsupported entry; archive is a separate terminal command")
        if not isinstance(stage.get("args", []), list) or not all(isinstance(arg, str) for arg in stage.get("args", [])):
            raise ValueError("args must be a list of strings, not shell text")
        if stage["entry"] == "metadata-cpu" and stage.get("args", []) not in ([], ["--help"]):
            raise ValueError("metadata-cpu only accepts no args or --help; use metadata-gpu for GPU checks")
        if stage["entry"] == "metadata-gpu" and "--gpu" not in stage.get("args", []):
            raise ValueError("metadata-gpu requires the explicit --gpu flag")
        if stage["entry"] not in CPU_ENTRIES and not isinstance(stage.get("result_json"), str):
            raise ValueError("non-CPU stages require result_json; exit code alone is not numerical/quality acceptance")
        timeout = stage.get("timeout_seconds", 600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("stage timeout must be finite and positive")


def select_stages(plan: dict, requested: str | None) -> list[str]:
    """只选择执行范围；名字须非空/不重复/已存在，并始终沿用冻结计划顺序。"""
    names = [stage["name"] for stage in plan["stages"]]
    if requested is None:
        return names
    if not isinstance(requested, str):
        raise ValueError("--stages must be a comma-separated list of stage names")
    selected = [name.strip() for name in requested.split(",")]
    if any(not name for name in selected):
        raise ValueError("--stages cannot be empty or contain empty names")
    if len(selected) != len(set(selected)):
        raise ValueError("--stages contains duplicate stage names")
    unknown = set(selected) - set(names)
    if unknown:
        raise ValueError(f"unknown --stages names: {', '.join(sorted(unknown))}")
    chosen = set(selected)
    return [name for name in names if name in chosen]


def aggregate_status(stages: list[dict], *, interrupted: bool = False) -> str:
    """只有范围内全部PASS才算PASS；部分未执行不能冒充整套验收通过。"""
    statuses = [stage["status"] for stage in stages]
    if interrupted or "FAIL" in statuses:
        return "FAIL"
    return "PASS" if statuses and all(status == "PASS" for status in statuses) else "NOT_RUN"


def stop_owned(process: subprocess.Popen) -> None:
    """只回收当前对象启动且尚未退出的子进程；不按端口/名称杀别人的服务。"""
    if os.name == "nt":
        if process.poll() is not None:
            return
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        return
    # Linux阶段拥有独立session，即使leader已退出，也继续回收仍占GPU的自有worker。
    def alive():
        process.poll()  # 顺便reap leader，避免zombie。
        expected = getattr(process, "_suite_identity", None)
        current = process_identity(process.pid)
        if expected and current is not None and expected != current:
            raise RuntimeError("refusing to signal a reused leader PID")
        try:
            ensure_group_empty(process.pid)
            return False
        except RuntimeError:
            return True
    for which, seconds in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (getattr(signal, "SIGKILL", 9), 10)):
        if not alive():
            return
        try:
            os.killpg(process.pid, which)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not alive():
                return
            time.sleep(0.05)
    if alive():
        raise RuntimeError("owned process group did not terminate after SIGKILL")


def result_gate(path: Path) -> tuple[str, list[str]]:
    """显式报告必须终态且无FAIL/REVIEW/NOT_RUN；不能把进程返回0硬编码为全部通过。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    statuses = []
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                # git_status is provenance text, not an acceptance verdict.
                # Keep all actual status fields strict, including nested reviews.
                if key == "git_status":
                    continue
                if (key == "status" or key.endswith("_status")) and isinstance(child, str):
                    statuses.append(child)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    if not statuses:
        return "FAIL", ["missing structured status"]
    if any(any(marker in status for marker in ("FAIL", "ERROR", "REVIEW", "RUNNING", "PENDING")) for status in statuses):
        return "FAIL", statuses
    if any("NOT_RUN" in status or "SKIP" in status for status in statuses):
        return "NOT_RUN", statuses
    if not all(status.startswith("PASS") for status in statuses):
        return "FAIL", statuses
    return "PASS", statuses


def render_argument(value: str, replacements: dict[str, str]) -> str:
    """仅替换约定占位符，不把prompt/JSON参数中的普通花括号当format表达式。"""
    for name, replacement in replacements.items():
        value = value.replace("{" + name + "}", replacement)
    return value


def execute_gated(gate: Path, command: list[str]) -> int:
    """先等父进程持久化PID，再执行模型命令；崩溃窗口中的无记录wrapper不会用GPU。"""
    deadline = time.monotonic() + 30
    while not gate.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("parent never granted persisted-PID startup permission")
        time.sleep(0.02)
    permission = json.loads(gate.read_text(encoding="utf-8"))
    if permission.get("child_pid") != os.getpid():
        raise RuntimeError("startup gate does not belong to this process")
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("empty gated command")
    if os.name != "nt":
        os.execv(command[0], command)  # 保持PID/进程组身份；resume仍检查相同的进程。
        raise AssertionError("execv returned")
    return subprocess.call(command)


def _run_locked(args) -> int:
    """串行运行选定阶段；失败后保留未运行阶段，恢复时只重试失败/未运行项。"""
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mapping = entries()
    validate_plan(plan, mapping)
    selected_names = select_stages(plan, getattr(args, "stages", None))
    selected = set(selected_names)
    if args.allow_gpu and os.name == "nt":
        raise RuntimeError("GPU suite execution requires Linux process-group cleanup; Windows supports CPU checks only")
    frozen = fingerprint(plan_path, plan)
    directory = args.output.resolve()
    state_path = directory / "suite.json"
    if args.resume:
        if (directory / "raw-data.tar.gz").exists() or (directory / "manifest.json").exists():
            raise RuntimeError("archived evidence is immutable; start a new suite directory")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != frozen:
            raise RuntimeError("resume rejected: source/config/fixture SHA256 differs")
        validate_successful_evidence(state)
        ensure_terminated(state.get("runner_identity"))
        for stage in state["stages"]:
            for attempt in stage["attempts"]:
                ensure_terminated(attempt.get("process_identity"))
                ensure_group_empty(attempt.get("process_group_id"))
                if attempt.get("status") == "RUNNING":
                    attempt["status"] = "FAIL"
                    attempt["error"] = "prior process terminated without recording a result"
            if stage["status"] == "RUNNING":
                stage["status"] = "FAIL"
        if "interruption" in state:
            state.setdefault("previous_interruptions", []).append(state.pop("interruption"))
    else:
        with (directory / "plan.json").open("x", encoding="utf-8") as file:
            json.dump(plan, file, ensure_ascii=False, indent=2)
        state = dict(schema=1, fingerprint=frozen, stages=[dict(name=row["name"], entry=row["entry"],
            status="NOT_RUN", attempts=[]) for row in plan["stages"]])
    state.update(status="RUNNING", selected_status="RUNNING", selected_stages=selected_names,
                 selection_explicit=getattr(args, "stages", None) is not None,
                 runner_identity=process_identity(os.getpid()), started_at=time.time())
    save_state(state_path, state)
    process = None
    try:
        for definition, stage in zip(plan["stages"], state["stages"]):
            if stage["name"] not in selected:
                # 首次未选项保持NOT_RUN；resume不能抹掉旧PASS/FAIL或历史attempt。
                continue
            if stage["status"] == "PASS":
                continue
            if not definition.get("enabled", True):
                stage["status"], stage["reason"] = "NOT_RUN", "disabled in frozen plan"
                continue
            needs_gpu = stage["entry"] not in CPU_ENTRIES
            if needs_gpu and not args.allow_gpu:
                stage["status"], stage["reason"] = "NOT_RUN", "GPU execution not authorized by --allow-gpu"
                continue
            if needs_gpu:
                from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
                ensure_gpu1_available(evidence_path=directory / "gpu_observations.jsonl")
            attempt_dir = directory / stage["name"] / f"attempt_{len(stage['attempts']) + 1:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            replacements = dict(stage_dir=str(attempt_dir), stage_output=str(attempt_dir / "output"), root=str(ROOT))
            tail = [render_argument(arg, replacements) for arg in definition.get("args", [])]
            for argument, parameter in option_pairs(tail):
                if argument in {"--output", "--output-dir"}:
                    destination = Path(parameter).resolve()
                    if not destination.is_relative_to(attempt_dir.resolve()) or destination.exists():
                        raise ValueError("stage output must be a new path inside its attempt directory")
            target = mapping[stage["entry"]]
            command = [sys.executable, "-m", target.removeprefix("module:")] if target.startswith("module:") else [sys.executable, str(ROOT / target)]
            command += tail
            attempt = dict(status="RUNNING", command=command, log=str(attempt_dir / "process.log"), started_at=time.time())
            stage["attempts"].append(attempt)
            stage["status"] = "RUNNING"
            save_state(state_path, state)
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            gate = attempt_dir / "startup_gate.json"
            launcher = [sys.executable, "-B", "-m", "benchmarks.runtime_v2_suite", "_exec_stage", "--gate", str(gate), "--", *command]
            attempt["launcher_command"] = launcher
            with Path(attempt["log"]).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(launcher, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=flags, start_new_session=os.name != "nt")
                attempt["process_identity"] = process_identity(process.pid)
                process._suite_identity = attempt["process_identity"]
                attempt["process_group_id"] = process.pid if os.name != "nt" else None
                save_state(state_path, state)
                if gate.exists():
                    raise FileExistsError(gate)
                save_state(gate, dict(child_pid=process.pid))
                try:
                    code = process.wait(timeout=definition.get("timeout_seconds", 600))
                except subprocess.TimeoutExpired:
                    attempt["error"] = "stage timeout"
                    stop_owned(process)
                    code = process.returncode if process.returncode else 124
            try:
                ensure_group_empty(attempt.get("process_group_id"))
            except RuntimeError as error:
                # 已知是本次Popen拥有的独立session，立即清理；resume只读检查不杀进程。
                attempt["error"] = str(error)
                stop_owned(process)
                code = code or 125
            attempt.update(exit_code=code, finished_at=time.time(), status="PASS" if code == 0 else "FAIL")
            attempt["log_sha256"] = digest(Path(attempt["log"]))
            if code == 0 and definition.get("result_json"):
                report_path = Path(render_argument(definition["result_json"], replacements)).resolve()
                if not report_path.is_relative_to(attempt_dir.resolve()):
                    raise ValueError("result_json must belong to this attempt directory")
                attempt["result_json"] = str(report_path)
                if not report_path.is_file():
                    attempt["status"], attempt["error"] = "FAIL", "expected structured result was not written"
                else:
                    attempt["status"], attempt["report_statuses"] = result_gate(report_path)
                    attempt["result_sha256"] = digest(report_path)
            elif code == 0:
                attempt["gate_scope"] = "CPU script assertions and exit code only"
            stage["status"] = attempt["status"]
            process = None
            save_state(state_path, state)
            if stage["status"] != "PASS" and not args.keep_going:
                break
    except BaseException as error:
        state["interruption"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if process is not None:
            stop_owned(process)
        for stage in state["stages"]:
            if stage["status"] == "RUNNING":
                stage["status"] = "FAIL"
                stage["attempts"][-1].update(status="FAIL", exit_code=process.returncode if process is not None else None,
                                              finished_at=time.time(), error="suite interrupted or child startup failed")
        interrupted = "interruption" in state
        state["status"] = aggregate_status(state["stages"], interrupted=interrupted)
        state["selected_status"] = aggregate_status(
            [row for row in state["stages"] if row["name"] in selected], interrupted=interrupted)
        state["finished_at"] = time.time()
        save_state(state_path, state)
    print(json.dumps(dict(status=state["status"], selected_status=state["selected_status"],
        selected_stages=selected_names, output=str(directory),
        stages=[(row["name"], row["status"]) for row in state["stages"]])))
    # 无--stages时选择全计划，沿用旧退出码；显式选择只据本次范围返回退出码。
    return 0 if state["selected_status"] == "PASS" else 2 if state["selected_status"] == "NOT_RUN" else 1


def run(args) -> int:
    """在整个执行周期持有OS锁，两个并发resume不会覆盖彼此的PID/进度记录。"""
    directory = args.output.resolve()
    if args.resume:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with exclusive_guard(directory):
        return _run_locked(args)


def _archive_locked(directory: Path) -> None:
    """确认所有自有进程结束后归档；历史RUNNING报告原样保留为中断证据，不伪造终态。"""
    directory = directory.resolve()
    state = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
    if state["status"] == "RUNNING":
        raise RuntimeError("suite is not terminal")
    validate_successful_evidence(state)
    ensure_terminated(state.get("runner_identity"))
    for stage in state["stages"]:
        for attempt in stage["attempts"]:
            ensure_terminated(attempt.get("process_identity"))
            ensure_group_empty(attempt.get("process_group_id"))
    if (directory / "raw-data.tar.gz").exists() or (directory / "manifest.json").exists():
        raise FileExistsError("archive already exists; do not rewrite experimental evidence")
    # 旧archive_release的standalone模式拒绝任意旧RUNNING JSON，无法保存SIGKILL现场。
    # 这里沿用tar+逐文件SHA校验，但“是否仍运行”以已检查的suite/PID/group为准。
    files = sorted(path for path in directory.rglob("*") if path.is_file() and not path.is_symlink()
                   and path.name != ".suite.guard")
    rows = [dict(path=path.relative_to(directory).as_posix(), bytes=path.stat().st_size,
                 sha256=digest(path)) for path in files]
    unfinished = []
    owners = [(Path(attempt["log"]).parent.resolve(), stage["name"], attempt["status"])
              for stage in state["stages"] for attempt in stage["attempts"] if "log" in attempt]
    def unfinished_owner(path):
        matches = [owner for owner in owners if path.resolve().is_relative_to(owner[0])]
        if len(matches) != 1 or matches[0][2] not in {"FAIL", "NOT_RUN"}:
            raise RuntimeError(f"unfinished evidence has no terminal failed attempt owner: {path}")
        return dict(stage=matches[0][1], attempt_status=matches[0][2])
    for path in files:
        if path.suffix == ".json":
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError):
                # 部分写入的文件也是原始失败证据，照常归档，不修饰内容。
                unfinished.append(dict(path=path.relative_to(directory).as_posix(), reason="partial_or_invalid_json",
                                       owner=unfinished_owner(path)))
            else:
                if isinstance(report, dict) and report.get("status") == "RUNNING":
                    unfinished.append(dict(path=path.relative_to(directory).as_posix(), reason="original_running_snapshot_processes_now_terminal",
                                           owner=unfinished_owner(path)))
    archive_path = directory / "raw-data.tar.gz"
    with tarfile.open(archive_path, "x:gz", compresslevel=5) as packed:
        for path in files:
            packed.add(path, arcname=path.relative_to(directory).as_posix(), recursive=False)
    with tarfile.open(archive_path, "r:gz") as packed:
        for row in rows:
            value = hashlib.sha256()
            with packed.extractfile(row["path"]) as stream:
                for block in iter(lambda: stream.read(8 * 2**20), b""):
                    value.update(block)
            if value.hexdigest() != row["sha256"]:
                raise RuntimeError(f"archive checksum mismatch: {row['path']}")
    manifest = dict(schema=1, archive=archive_path.name, archive_sha256=digest(archive_path), files=rows,
        suite_status=state["status"], execution_fingerprint=state["fingerprint"], unfinished_reports=unfinished,
        verification="Every member re-read and SHA256 verified; suite terminal and all recorded PID identities/groups checked empty.",
        interpretation="Archived unfinished snapshots are failure/interruption evidence, never reclassified as PASS.")
    with (directory / "manifest.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status="ARCHIVED", suite_status=state["status"], files=len(rows), unfinished_reports=len(unfinished))))


def archive(directory: Path) -> None:
    """归档与resume互斥；归档完成后run --resume会拒绝修改这份证据。"""
    directory = directory.resolve()
    with exclusive_guard(directory):
        _archive_locked(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--allow-gpu", action="store_true")
    command.add_argument("--keep-going", action="store_true")
    command.add_argument("--stages", help="comma-separated stage names; selection does not change the frozen plan")
    command = sub.add_parser("archive")
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("_exec_stage", help=argparse.SUPPRESS)
    command.add_argument("--gate", type=Path, required=True)
    command.add_argument("child_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "run":
        raise SystemExit(run(args))
    if args.command == "_exec_stage":
        raise SystemExit(execute_gated(args.gate, args.child_command))
    archive(args.output)


if __name__ == "__main__":
    main()
