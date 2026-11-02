"""Real HTTP runtime soak with owned-server initialization and cleanup evidence.

Run under runtime_v2_suite for an outer process-group watchdog. This module never
detaches children, never kills by port/name, and never claims recovery from a GPU
hang. Edge cases use real engines and HTTP, with explicitly different limits.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from benchmarks.release_soak import request_json, inspect_metrics, is_idle

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_new(path, value):
    """Raw snapshots are write-once; failure evidence is never replaced by a rerun."""
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


class JsonLines:
    """One session owns this append-only file; concurrent client writes use a lock."""
    def __init__(self, path):
        self.file = Path(path).open("x", encoding="utf-8")
        self.lock = threading.Lock()

    def put(self, value):
        with self.lock:
            self.file.write(json.dumps(value, ensure_ascii=False) + "\n")
            self.file.flush()

    def close(self):
        self.file.close()


def source_hashes():
    files = set((ROOT / "nanovllm").rglob("*.py")) | set((ROOT / "benchmarks").rglob("*.py")) | set(ROOT.glob("*.py"))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(files)}


def validate_manifest(manifest, *, verify_model=True):
    if source_hashes() != manifest["source_sha256"]:
        raise RuntimeError("source changed after session freeze")
    if object_sha(dict(engine=manifest["engine_config"], phases=manifest["phases"])) != manifest["config_sha256"]:
        raise RuntimeError("frozen engine/phase configuration changed")
    if object_sha(manifest["prompts"]) != manifest["prompts_sha256"]:
        raise RuntimeError("frozen HTTP workload changed")
    if {key:os.environ.get(key) for key in manifest["environment"]} != manifest["environment"]:
        raise RuntimeError("frozen device environment changed")
    if verify_model:
        for name, expected in manifest["model_sha256"].items():
            if sha(Path(manifest["model"])/name) != expected:
                raise RuntimeError(f"model/tokenizer file changed: {name}")


def process_info(pid, *, resources=True):
    """Linux birth time protects against PID reuse; missing PID is not guessed alive."""
    try:
        folder = Path(f"/proc/{pid}")
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        identity = dict(pid=int(pid), ppid=int(fields[1]), birth=fields[19], state=fields[0])
        if not resources or fields[0] == "Z":
            return dict(identity, rss_kib=0, threads=0, fd_count=0)
        status = {}
        for line in (folder / "status").read_text().splitlines():
            key, _, value = line.partition(":")
            status[key] = value.strip()
        try:
            fd_count, fd_error = len(list((folder / "fd").iterdir())), None
        except PermissionError:
            # 退出中的进程可能仍有可验证stat身份，但fd目录已无法读取。
            # 不能让可选遥测阻断仅针对已验证自有PID的清理，也不能把未知FD数假报为0。
            fd_count, fd_error = None, "PermissionError"
        return dict(identity, rss_kib=int(status.get("VmRSS", "0 kB").split()[0]),
                    threads=int(status.get("Threads", 0)), fd_count=fd_count, fd_error=fd_error)
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(pid, *, resources=True):
    """Read-only current ancestry; previously observed children retain birth identity."""
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                row = process_info(int(path.name), resources=False)
            except PermissionError:
                continue
            if row:
                rows[row["pid"]] = row
    owned = {pid}
    while True:
        found = {number for number, row in rows.items() if row["ppid"] in owned}
        if found <= owned:
            break
        owned |= found
    selected = [rows[number] for number in sorted(owned) if number in rows]
    if not resources:
        return selected
    return [row for item in selected if (row := process_info(item["pid"])) is not None
            and row["birth"] == item["birth"]]


def port_free(port):
    try:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


class Resources:
    """Sample from immediately after Popen, including imports/model initialization.

    Sampling timestamps/gaps are saved rather than claiming real-time precision.
    A guard failure stops new work. No signal is sent to external GPU processes.
    """
    def __init__(self, directory, abort, owner_pid):
        self.log = JsonLines(directory / "resources.jsonl")
        self.guard_log = directory / "gpu_guard.jsonl"
        self.abort, self.owner_pid = abort, owner_pid
        self.done = threading.Event()
        self.process = None
        self.leader_birth = None
        self.known = {}
        self.lock = threading.Lock()
        self.errors = []
        self.started = time.monotonic()
        self.samples, self.last_at, self.max_gap = 0, None, 0.0
        self.thread = threading.Thread(target=self.loop, name="soak-resources", daemon=True)

    def owned_snapshot(self, *, resources=True):
        """Discover descendants only while our original leader birth still matches."""
        discovered = []
        if self.process is not None and self.leader_birth is not None:
            rows = descendants(self.process.pid, resources=resources)
            leader = next((row for row in rows if row["pid"] == self.process.pid), None)
            if leader and leader["birth"] == self.leader_birth:
                discovered = rows
        with self.lock:
            for row in discovered:
                # Never replace a frozen identity if a child PID has been reused.
                self.known.setdefault(row["pid"], row["birth"])
            known = dict(self.known)
        result = []
        for pid, birth in known.items():
            row = process_info(pid, resources=resources)
            if row and row["birth"] == birth:
                result.append(row)
        return result

    def sample(self):
        from benchmarks.runtime_v2_gpu_guard import _query
        now = time.monotonic()
        rows = self.owned_snapshot()
        gpus = _query(("index", "uuid", "memory.used", "utilization.gpu"), "gpu")
        selected = os.environ.get("NANOVLLM_BENCH_GPU_INDEX", "1")
        gpu = next((row for row in gpus if row["index"] == selected), None)
        if gpu is None:
            raise RuntimeError(f"physical GPU{selected} missing during resource sample")
        apps = [row for row in _query(("gpu_uuid", "pid", "used_memory"), "compute-apps") if row["gpu_uuid"] == gpu["uuid"]]
        gap = None if self.last_at is None else now - self.last_at
        if gap is not None:
            self.max_gap = max(self.max_gap, gap)
        self.last_at, self.samples = now, self.samples + 1
        self.log.put(dict(at=time.time(), elapsed_s=now-self.started, interval_s=gap,
            server_pid=self.process.pid if self.process else None, processes=rows,
            aggregate=dict(rss_kib=sum(row["rss_kib"] for row in rows),
                           fd_count=sum(row["fd_count"] for row in rows if row["fd_count"] is not None),
                           fd_unavailable_processes=sum(row["fd_count"] is None for row in rows),
                           threads=sum(row["threads"] for row in rows)),
            gpu=gpu, gpu_compute_processes=apps))

    def check_guard(self):
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        return ensure_gpu1_available(owner_pid=self.owner_pid, evidence_path=self.guard_log)

    def loop(self):
        next_sample, next_guard = time.monotonic(), time.monotonic()
        while not self.done.is_set():
            now = time.monotonic()
            try:
                if now >= next_guard:
                    self.check_guard()
                    next_guard = time.monotonic() + 5.0
                if self.done.is_set():
                    break
                if now >= next_sample:
                    self.sample()
                    next_sample += 1.0
                    if next_sample < time.monotonic():
                        next_sample = time.monotonic() + 1.0
            except BaseException as error:
                item = dict(at=time.time(), error=f"{type(error).__name__}: {error}")
                self.errors.append(item)
                self.log.put(item)
                self.abort.set()
                # Keep measuring cleanup, but never clear abort after an occupancy loss.
                next_guard, next_sample = time.monotonic()+5, time.monotonic()+1
            self.done.wait(.05)

    def finish(self):
        self.done.set()
        self.thread.join(timeout=25)
        if self.thread.is_alive():
            raise RuntimeError("resource monitor failed to stop")
        self.log.close()


def stop_owned(process, monitor):
    """Bounded cleanup of this Popen and observed descendants; no kill-by-port/name.

    Children inherit the suite group. The outer suite remains the final watchdog
    if the orchestrator is killed or an unobserved grandchild escapes its ancestry.
    """
    report = dict(signals=[], survivors=[])
    alive, known = [], {}
    for which, timeout in ((signal.SIGINT, 20), (signal.SIGTERM, 8), (signal.SIGKILL, 8)):
        monitor.owned_snapshot(resources=False)
        with monitor.lock:
            known = dict(monitor.known)
        # The server gets a graceful exit first. Later stages also reap its children.
        pids = [process.pid] if which == signal.SIGINT and process.poll() is None else list(known)
        for pid in pids:
            row = process_info(pid, resources=False)
            if row and row["birth"] == known.get(pid) and row["state"] != "Z":
                try:
                    os.kill(pid, which)
                    report["signals"].append(dict(pid=pid, birth=row["birth"], signal=int(which)))
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            alive = []
            for pid, birth in known.items():
                row = process_info(pid, resources=False)
                if row and row["birth"] == birth and row["state"] != "Z":
                    alive.append(row)
            if not alive:
                break
            time.sleep(.05)
        if not alive:
            break
    process.poll()
    report.update(returncode=process.returncode, leader_reaped=process.returncode is not None,
                  survivors=alive, known_pid_births=known)
    return report


def server(args):
    """Benchmark-only host; unchanged production handlers except explicit slow test."""
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    phase = manifest["phases"][args.phase]
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    ensure_gpu1_available(owner_pid=args.owner_pid)
    from nanovllm import LLM
    from nanovllm.server import ServiceLimits, create_http_server
    engine = httpd = manager = None
    directory = args.manifest.parent / args.phase
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"owned server received signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        engine = LLM(manifest["model"], **manifest["engine_config"])
        httpd, manager = create_http_server(engine, "127.0.0.1", manifest["port"],
                                            limits=ServiceLimits(**phase["limits"]))
        if args.phase == "edge_slow":
            original = httpd.RequestHandlerClass
            delay = phase["handler_pause_seconds"]
            class SlowReader(original):
                def _stream_raw(self, record):
                    if self.headers.get("X-Benchmark-Slow-Consumer") == "1":
                        # Request already entered the REAL engine; only consumption pauses.
                        time.sleep(delay)
                    return super()._stream_raw(record)
            httpd.RequestHandlerClass = SlowReader
        write_new(directory / "ready.json", dict(pid=os.getpid(), birth=process_info(os.getpid())["birth"],
            at=time.time(), url=f'http://127.0.0.1:{manifest["port"]}', limits=asdict(manager.limits),
            engine_config=manifest["engine_config"], runtime=engine.config.runtime_summary(),
            owned_shm_names=[engine.config.tp_shm_name] if engine.config.tp_shm_name else [],
            handler_injection=phase.get("handler_pause_seconds", 0)))
        httpd.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup = dict(at=time.time(), errors=[])
        if manager is not None:
            try:
                manager.close()
                cleanup["manager_worker_alive"] = manager.worker.is_alive()
            except BaseException as error:
                cleanup["errors"].append(f"manager: {type(error).__name__}: {error}")
        if httpd is not None:
            httpd.server_close()
        if engine is not None:
            try:
                engine.exit()
            except BaseException as error:
                cleanup["errors"].append(f"engine: {type(error).__name__}: {error}")
        write_new(directory / "server_cleanup.json", cleanup)


def http_request(base, job, abort, raw, barrier=None):
    """Read actual SSE and actual request IDs; expected errors stay separate from failure."""
    result = dict(number=job["number"], kind=job["kind"], prompt_tokens=len(job["prompt"]),
                  expected_tokens=job["max_tokens"], events=[], request_id=None, error=None)
    if barrier:
        barrier.wait(timeout=5)
    if abort.is_set():
        return dict(result, outcome="NOT_SUBMITTED", seconds=0)
    started = time.monotonic()
    payload = dict(prompt=job["prompt"], sampling_params=dict(temperature=0, ignore_eos=True,
        max_tokens=job["max_tokens"], speculative_ngram_size=job.get("ngram", 0),
        speculative_max_draft_tokens=8), stream=True)
    request = urllib.request.Request(base + "/v1/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json", "X-Benchmark-Slow-Consumer":"1" if job["kind"] == "slow" else "0"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result["http_status"] = response.status
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                event = json.loads(data)
                event = dict(time_s=time.monotonic()-started, **event)
                result["events"].append(event)
                result["request_id"] = event.get("request_id", result["request_id"])
                raw.put(dict(record="event", number=job["number"], kind=job["kind"], **event))
                if "error" in event:
                    result["service_error"] = event["error"]
                    break
                if event.get("finished"):
                    result["finish_reason"] = event.get("finish_reason")
                    break
                if (job["kind"] == "cancel" or abort.is_set()) and "cancel_status" not in result and result["request_id"] is not None:
                    code, reply = request_json(base, f'/v1/requests/{result["request_id"]}/cancel', {}, timeout=4)
                    result.update(cancel_status=code, cancel_reply=reply)
    except urllib.error.HTTPError as error:
        result["http_status"] = error.code
        try:
            result["service_error"] = json.loads(error.read()).get("error", {})
        except (ValueError, AttributeError):
            result["error"] = "invalid HTTP error body"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = time.monotonic()-started
    token_events = [event for event in result["events"] if event.get("token_id", -1) >= 0]
    result["observed_output_tokens"] = max((event.get("token_index", -1)+1 for event in token_events), default=0)
    result["ttft_ms"] = token_events[0]["time_s"]*1000 if token_events else None
    visible = [event for event in result["events"] if event.get("text")]
    result["first_text_ms"] = visible[0]["time_s"]*1000 if visible else None
    result["event_gap_ms"] = [(b["time_s"]-a["time_s"])*1000 for a,b in zip(token_events, token_events[1:])]
    # SSE事件可能一次提交多个token，TPOT按累计token数摊销；不伪造token级时间戳。
    count = result["observed_output_tokens"]
    result["tpot_ms"] = ((token_events[-1]["time_s"]-token_events[0]["time_s"])*1000/(count-1)
                         if count > 1 else None)
    result["outcome"] = classify(result, job["kind"])
    if result["outcome"] == "UNEXPECTED" and result["request_id"] is not None:
        try:
            result["cleanup_cancel"] = request_json(base, f'/v1/requests/{result["request_id"]}/cancel', {}, timeout=3)
        except Exception as error:
            result["cleanup_cancel_error"] = str(error)
    raw.put(dict(record="request", **result))
    return result


def classify(record, kind):
    """Only the named edge phase can count its expected fault as exercised, not failed."""
    error_type = (record.get("service_error") or {}).get("type")
    if record.get("error"):
        return "UNEXPECTED"
    if kind == "capacity" and record.get("http_status") == 429 and error_type == "capacity_exceeded":
        return "EXPECTED_CAPACITY"
    if kind == "timeout" and error_type == "request_timeout":
        return "EXPECTED_TIMEOUT"
    if kind == "slow" and error_type == "slow_consumer":
        return "EXPECTED_SLOW_CONSUMER"
    if record.get("finish_reason") == "cancelled" and record.get("cancel_status") == 200:
        return "EXPECTED_CANCEL"
    if (record.get("finish_reason") == "length" and record.get("observed_output_tokens") == record["expected_tokens"]
            and record.get("cancel_status") != 200):
        return "COMPLETED_CANCEL_RACE" if kind == "cancel" else "COMPLETED"
    return "UNEXPECTED"


def phase_limits(kind, ttl):
    limits = dict(max_inflight_requests=128, request_timeout_seconds=120.0,
                  max_stream_events=128, finished_request_ttl_seconds=ttl, max_retained_requests=256)
    if kind == "edge_capacity":
        limits.update(max_inflight_requests=1, max_retained_requests=2)
    elif kind == "edge_timeout":
        limits.update(request_timeout_seconds=.001, max_retained_requests=2)
    elif kind == "edge_slow":
        limits.update(max_stream_events=2, max_retained_requests=2)
    return limits


def wait_ready(process, directory, base, abort, timeout):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if abort.is_set():
            raise RuntimeError("GPU/resource guard stopped initialization")
        if process.poll() is not None:
            raise RuntimeError(f"owned server exited during init: {process.returncode}")
        ready = directory / "ready.json"
        if ready.exists():
            try:
                data = json.loads(ready.read_text())
            except json.JSONDecodeError:
                time.sleep(.05)  # write-once producer may still be flushing its marker
                continue
            current = process_info(process.pid)
            if not current or data["pid"] != process.pid or data["birth"] != current["birth"]:
                raise RuntimeError("ready marker is not our server PID/birth")
            try:
                code, body = request_json(base, "/health", timeout=2)
                if code == 200 and body.get("status") == "ok":
                    return data
            except OSError:
                pass
        time.sleep(.1)
    raise TimeoutError("owned server readiness timeout")


def run_workload(base, phase, prompts, duration, concurrency, abort, raw, metrics_log):
    """Dynamic admission, real completion/cancellation, bounded aggregate memory."""
    counts, recent_ttft = Counter(), deque(maxlen=4096)
    started, next_metrics, next_arrival, number = time.monotonic(), 0.0, 0.0, 0
    futures, total_tokens, violations = set(), 0, []
    edge = phase.startswith("edge_")
    kind = phase.removeprefix("edge_") if edge else None
    target = 12 if kind == "capacity" else 3 if edge else None
    workers = 12 if kind == "capacity" else concurrency
    barrier = threading.Barrier(12) if kind == "capacity" else None
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        while futures or (number < target if edge else time.monotonic()-started < duration):
            if abort.is_set():
                raise RuntimeError("guard requested stop; no more requests admitted")
            elapsed = time.monotonic()-started
            allowed = number < target if edge else elapsed < duration
            while allowed and len(futures) < workers and (edge or elapsed >= next_arrival):
                request_kind = kind if edge else "cancel" if number % 4 == 3 else "complete"
                prompt = prompts["long" if number % 3 == 0 else "short"][number % len(prompts["short"])]
                job = dict(number=number, kind=request_kind, prompt=prompt,
                           max_tokens=256 if edge or request_kind == "cancel" else 48 if number % 2 else 96,
                           ngram=0 if edge else 3)
                futures.add(pool.submit(http_request, base, job, abort, raw, barrier))
                number += 1
                next_arrival = elapsed + (.02 if number % 5 else .12)
                allowed = number < target if edge else elapsed < duration
            done, futures = wait(futures, timeout=.03, return_when=FIRST_COMPLETED)
            if not futures and not done:
                time.sleep(.005)
            for future in done:
                record = future.result()
                counts[record["outcome"]] += 1
                total_tokens += record.get("observed_output_tokens", 0)
                if record.get("ttft_ms") is not None:
                    recent_ttft.append(record["ttft_ms"])
            if elapsed >= next_metrics:
                code, snapshot = request_json(base, "/metrics", timeout=3)
                problems = inspect_metrics(snapshot) if code == 200 else [f"HTTP_{code}"]
                metrics_log.put(dict(at=time.time(), elapsed_s=elapsed, snapshot=snapshot, violations=problems))
                violations.extend(problems)
                next_metrics = elapsed+1
                if problems:
                    raise RuntimeError(f"metrics contract violation: {problems}")
        elapsed = time.monotonic()-started
        required = {"capacity":"EXPECTED_CAPACITY", "timeout":"EXPECTED_TIMEOUT", "slow":"EXPECTED_SLOW_CONSUMER"}
        coverage = counts[required[kind]] > 0 if edge else counts["COMPLETED"] > 0 and counts["EXPECTED_CANCEL"] > 0
        return dict(status="FAIL" if counts["UNEXPECTED"] else "PASS" if coverage else "REVIEW_REQUIRED",
            seconds=elapsed, submission_duration_requested=duration, admitted=number, outcomes=dict(counts),
            coverage_satisfied=coverage, observed_output_tokens=total_tokens,
            observed_tokens_s=total_tokens/max(elapsed, 1e-9), metric_violations=violations,
            ttft_recent_samples_ms=list(recent_ttft), ttft_sample_policy="last4096; raw all events retained",
            metric_note="Client-delivered cumulative tokens, not GPU-generated throughput; TTFT is first token event, first_text separately records visible text; bursts have no invented per-token timestamps.")
    finally:
        # Main finally closes the server immediately after this returns/raises. Socket reads
        # are bounded; pending jobs check abort before submitting. No detached worker process.
        if abort.is_set():
            for future in futures:
                future.cancel()
        pool.shutdown(wait=True, cancel_futures=abort.is_set())


def drain(base, ttl, abort, log, timeout=30):
    """A completion record can remain briefly; engine records/KV must not remain."""
    deadline, last = time.monotonic()+timeout, None
    while time.monotonic() < deadline:
        if abort.is_set():
            raise RuntimeError("guard stopped drain")
        code, last = request_json(base, "/metrics", timeout=3)
        problems = inspect_metrics(last) if code == 200 else [f"HTTP_{code}"]
        log.put(dict(at=time.time(), stage="drain", snapshot=last, violations=problems))
        if problems:
            raise RuntimeError(f"drain metrics violation: {problems}")
        if is_idle(last):
            break
        time.sleep(.2)
    if not last or not is_idle(last):
        raise AssertionError("KV references/used blocks/engine request records did not return to zero")
    idle_snapshot = last
    # TTL is explicit test configuration, not a change to production defaults.
    end = time.monotonic()+ttl+3
    while time.monotonic() < end:
        if abort.is_set():
            raise RuntimeError("guard stopped TTL check")
        code, last = request_json(base, "/metrics", timeout=3)
        log.put(dict(at=time.time(), stage="ttl", snapshot=last))
        if code == 200 and is_idle(last) and last.get("retained_requests") == 0 and last.get("active_stream_events") == 0:
            return dict(status="PASS", idle_metrics_before_ttl=idle_snapshot, final_metrics=last,
                        kv_ref_total=0, engine_records=0, retained_records=0)
        time.sleep(.2)
    raise AssertionError("completed records/events did not expire after configured TTL")


def run_phase(manifest_path, phase, ready_timeout):
    manifest = json.loads(manifest_path.read_text())
    folder = manifest_path.parent / phase
    folder.mkdir(exist_ok=False)
    base = f'http://127.0.0.1:{manifest["port"]}'
    abort = threading.Event()
    monitor = Resources(folder, abort, os.getpid())
    raw, metrics = JsonLines(folder/"requests.jsonl"), JsonLines(folder/"metrics.jsonl")
    report = dict(phase=phase, status="RUNNING", at=time.time(), source_sha256=manifest["source_sha256"],
                  config_sha256=manifest["config_sha256"], limits=manifest["phases"][phase]["limits"])
    process = None
    log = (folder/"server.log").open("x", encoding="utf-8")
    try:
        if not port_free(manifest["port"]):
            raise RuntimeError("requested localhost port is already owned; not touching it")
        monitor.check_guard()
        monitor.sample()  # pre-Popen baseline includes GPU memory
        command = [sys.executable, "-m", "benchmarks.runtime_v2_soak", "server", "--manifest", str(manifest_path),
                   "--phase", phase, "--owner-pid", str(os.getpid())]
        before = time.time()
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONUNBUFFERED="1"))
        monitor.process = process
        row = process_info(process.pid)
        if row:
            monitor.known[process.pid] = row["birth"]
            monitor.leader_birth = row["birth"]
        write_new(folder/"process.json", dict(pid=process.pid, birth=row["birth"] if row else None,
            command=command, popen_started_at=before, popen_returned_at=time.time(), inherited_process_group=os.getpgrp()))
        monitor.sample()  # first observable instant, before waiting for readiness
        monitor.thread.start()
        ready = wait_ready(process, folder, base, abort, ready_timeout)
        report["init_seconds"] = time.time()-before
        report["ready"] = ready
        monitor.check_guard()
        write_new(folder/"phase_started.json", dict(at=time.time(), limits=report["limits"], real_http=True))
        if phase != "edge_timeout":
            # Warm compiler paths through REAL non-streaming HTTP. No fake model/event.
            warm_payload = dict(prompt=manifest["prompts"]["short"][0],
                sampling_params=dict(temperature=0, ignore_eos=True, max_tokens=8), stream=False)
            code, reply = request_json(base, "/v1/generate", warm_payload, timeout=30)
            write_new(folder/"http_warmup.json", dict(http_status=code, response=reply))
            if code != 200 or reply.get("finish_reason") != "length" or len(reply.get("token_ids", [])) != 8:
                raise AssertionError("real HTTP warmup did not complete exactly eight tokens")
            monitor.check_guard()
        report["workload"] = run_workload(base, phase, manifest["prompts"], manifest["phases"][phase]["duration"],
            manifest["concurrency"], abort, raw, metrics)
        report["drain"] = drain(base, report["limits"]["finished_request_ttl_seconds"], abort, metrics)
        report["status"] = report["workload"]["status"]
        if phase == "edge_timeout":
            # More completed faults than retained capacity, before any configured TTL.
            # This is separate evidence from admission-capacity HTTP 429.
            before_ttl = report["drain"]["idle_metrics_before_ttl"]
            cap = report["limits"]["max_retained_requests"]
            report["retention_capacity_observed"] = (
                report["workload"]["outcomes"].get("EXPECTED_TIMEOUT",0) > cap
                and report["workload"]["seconds"] < report["limits"]["finished_request_ttl_seconds"]
                and before_ttl.get("finished_records") == cap)
            if not report["retention_capacity_observed"] and report["status"] == "PASS":
                report["status"] = "REVIEW_REQUIRED"
    except BaseException as error:
        abort.set()
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    finally:
        abort.set()
        if process is not None:
            try:
                report["process_cleanup"] = stop_owned(process, monitor)
            except BaseException as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
                report["status"] = "FAIL"
        try:
            if monitor.thread.ident is not None:
                monitor.finish()
            else:
                monitor.log.close()
        except BaseException as error:
            report["monitor_cleanup_error"] = f"{type(error).__name__}: {error}"
            report["status"] = "FAIL"
        raw.close(); metrics.close(); log.close()
        owned_names = report.get("ready", {}).get("owned_shm_names", [])
        if process is not None:
            owned_names += [path.name for path in Path("/dev/shm").glob(f"nanovllm_{process.pid}_*")]
        report["remaining_owned_shm"] = sorted({name for name in owned_names if (Path("/dev/shm")/name).exists()})
        report["port_available_after_cleanup"] = port_free(manifest["port"])
        try:
            from benchmarks.runtime_v2_gpu_guard import _query
            selected = os.environ.get("NANOVLLM_BENCH_GPU_INDEX", "1")
            gpu = next(row for row in _query(("index","uuid","memory.used"),"gpu") if row["index"]==selected)
            remaining = [row for row in _query(("gpu_uuid","pid","used_memory"),"compute-apps") if row["gpu_uuid"]==gpu["uuid"]]
            own_gpu = []
            for row in remaining:
                pid = int(row["pid"])
                actual = process_info(pid)
                if pid in monitor.known and (actual is None or actual["birth"]==monitor.known[pid]):
                    own_gpu.append(row)
            report["gpu_after_cleanup"] = dict(gpu=gpu, processes=remaining, owned_remaining=own_gpu)
            if own_gpu:
                report["status"] = "FAIL"
        except BaseException as error:
            report["gpu_cleanup_check_error"] = f"{type(error).__name__}: {error}"
            report["status"] = "FAIL"
        report["resource_monitor"] = dict(samples=monitor.samples, max_interval_seconds=monitor.max_gap, errors=monitor.errors,
            note="1 second target cadence, actual timestamps retained; monitor covers initialization through owned cleanup")
        server_cleanup = folder/"server_cleanup.json"
        try:
            report["server_cleanup"] = json.loads(server_cleanup.read_text()) if server_cleanup.exists() else {"missing":True}
        except (OSError, ValueError) as error:
            report["server_cleanup"] = dict(errors=[f"unreadable/partial cleanup evidence: {error}"])
        cleanup = report.get("process_cleanup", {})
        if (not cleanup.get("leader_reaped") or cleanup.get("survivors") or report["remaining_owned_shm"]
                or not report["port_available_after_cleanup"] or monitor.errors
                or report["server_cleanup"].get("missing") or report["server_cleanup"].get("errors")
                or report["server_cleanup"].get("manager_worker_alive")):
            report["status"] = "FAIL"
        report["evidence_sha256"] = {path.name:sha(path) for path in sorted(folder.iterdir()) if path.is_file()}
        report["finished_at"] = time.time()
        write_new(folder/"summary.json", report)
    return report


def prepare(args):
    from transformers import AutoTokenizer
    from benchmarks.release_matrix import NATURAL_TASKS, CODE_TASKS
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(runtime_preset="baseline", batching_mode="mixed", speculative_policy="adaptive",
        metadata_backend="reuse", enable_mixed_graph=True, mixed_attention_backend="split",
        sampler_backend="torch", mlp_backend="torch", rmsnorm_backend="torch", kv_cache_dtype="model",
        tensor_parallel_size=1, max_model_len=2048, max_num_seqs=16, max_num_batched_tokens=512,
        prefill_chunk_size=128, max_prefill_steps=1, gpu_memory_utilization=.4, enable_prefix_caching=False,
        enforce_eager=False)
    if args.config:
        extra = json.loads(args.config.read_text())
        if not isinstance(extra, dict):
            raise ValueError("--config must contain engine keyword options")
        config.update(extra)
    if config.get("tensor_parallel_size") != 1 or config.get("sampler_backend") != "torch" or config.get("kv_cache_dtype") != "model":
        raise ValueError("this acceptance scope is TP1, Torch sampler, model-dtype KV")
    if config["max_model_len"] < 1024 or config["max_num_seqs"] < args.concurrency:
        raise ValueError("engine shape capacities are insufficient for frozen soak workload")
    model = Path(args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    sources = list(NATURAL_TASKS) + list(CODE_TASKS)
    prompts = {"short":[], "long":[]}
    for index in range(16):
        text = "\n\n".join(sources[index:]+sources[:index])
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) < 768:
            raise ValueError("authored soak material is shorter than the long fixture")
        prompts["short"].append(encoded[:96])
        prompts["long"].append(encoded[:768])
    phases = {name:dict(limits=phase_limits(name, args.ttl), duration=args.duration if name=="steady" else args.restart_seconds if name=="restart" else 0)
              for name in ("edge_capacity", "edge_timeout", "edge_slow", "steady", "restart")}
    phases["edge_slow"]["handler_pause_seconds"] = 3.0
    manifest = dict(schema=1, model=str(model), port=args.port, concurrency=args.concurrency,
        source_sha256=source_hashes(), engine_config=config, phases=phases,
        config_sha256=object_sha(dict(engine=config, phases=phases)), prompts=prompts, prompts_sha256=object_sha(prompts),
        model_sha256={path.name:sha(path) for path in sorted(model.iterdir()) if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".tiktoken", ".txt"}},
        environment={key:os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES","NANOVLLM_MASTER_PORT","HF_HUB_OFFLINE","NANOVLLM_BENCH_GPU_INDEX")},
        reporting_slo=dict(ttft_ms=1000, tpot_ms=30),
        notes=["No GPU execution has occurred merely by creating this manifest.",
               "Edge phases have deliberately altered limits; slow handler delays only queue consumption in the real HTTP/engine stack.",
               "Steady and restart use identical engine/service limits; TTL is explicitly 5s by default, not production 300s.",
               "Repeated authored fixtures are stress traffic, not independent quality samples or a performance tuning corpus.",
               "Clean restart tests a new process, not recovery from a wedged GPU/driver.",
               "Run under suite for outer group ownership; never signals external processes or removes unrelated SHM."])
    write_new(output/"manifest.json", manifest)
    return output/"manifest.json"


def run(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("real resource/ownership acceptance requires Linux /proc")
    manifest_path = prepare(args)
    reports, error = {}, None
    try:
        for phase in ("edge_capacity", "edge_timeout", "edge_slow", "steady", "restart"):
            report = run_phase(manifest_path, phase, args.ready_timeout)
            reports[phase] = dict(status=report["status"], summary_sha256=sha(manifest_path.parent/phase/"summary.json"))
            if report["status"] == "FAIL":
                raise RuntimeError(f"{phase} failed; no subsequent engine started")
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
    full = args.duration >= 1800
    status = "FAIL" if error else "PASS" if full and len(reports)==5 and all(row["status"]=="PASS" for row in reports.values()) else "REVIEW_REQUIRED"
    report = dict(status=status, phases=reports, error=error, release_duration_satisfied=full,
        duration_seconds=args.duration, manifest_sha256=sha(manifest_path),
        evidence_scope="Real HTTP only; CPU self-tests cannot satisfy these gates. Short duration is smoke, never full acceptance.")
    write_new(manifest_path.parent/"summary.json", report)
    print(json.dumps(report))
    return 0 if status=="PASS" else 2 if status=="REVIEW_REQUIRED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("run")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--config", type=Path)
    cmd.add_argument("--output", type=Path, required=True)
    cmd.add_argument("--port", type=int, default=18764)
    cmd.add_argument("--duration", type=float, default=1800)
    cmd.add_argument("--restart-seconds", type=float, default=15)
    cmd.add_argument("--ready-timeout", type=float, default=600)
    cmd.add_argument("--concurrency", type=int, default=8)
    cmd.add_argument("--ttl", type=float, default=5)
    cmd = sub.add_parser("server")
    cmd.add_argument("--manifest", type=Path, required=True)
    cmd.add_argument("--phase", choices=("edge_capacity","edge_timeout","edge_slow","steady","restart"), required=True)
    cmd.add_argument("--owner-pid", type=int, required=True)
    args = parser.parse_args()
    if args.command == "server":
        server(args)
    else:
        if (not 1 <= args.port <= 65535 or not 1 <= args.concurrency <= 16
                or not all(math.isfinite(value) and value > 0 for value in (args.duration,args.restart_seconds,args.ready_timeout,args.ttl))):
            parser.error("invalid finite-positive duration, port or concurrency")
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
