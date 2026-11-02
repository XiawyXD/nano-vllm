"""只读监测一次验收 session 的 server 进程，不启动、停止或修改被测服务。

PID 只读取 session.json 的 children.server.pid。每次采样前后验证 /proc
starttime，避免 PID 复用。输出从实际观测开始时计时，不能当作完整 soak RSS。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    # comm 允许空格/括号；最后一个 ')' 后才是从 state 开始的字段。
    fields = raw[raw.rfind(")") + 2:].split()
    return {"state": fields[0], "starttime_ticks": int(fields[19])}


def process_started_at(ticks):
    hz = os.sysconf("SC_CLK_TCK")
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            seconds = int(line.split()[1]) + ticks / hz
            return datetime.fromtimestamp(seconds, timezone.utc).isoformat()
    return None


def snapshot(pid, starttime):
    before = identity(pid)
    if before["starttime_ticks"] != starttime:
        return None, "pid_reused"
    if before["state"] in {"Z", "X"}:
        return None, "process_exited"
    status = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmSize", "Threads"}:
            status[key] = int(value.strip().split()[0])
    fd_count = len(list(Path(f"/proc/{pid}/fd").iterdir()))
    after = identity(pid)
    if after["starttime_ticks"] != starttime:
        return None, "pid_reused"
    if after["state"] in {"Z", "X"}:
        return None, "process_exited"
    return {
        "timestamp": utc_now(),
        "pid": pid, "starttime_ticks": starttime,
        "rss_kib": status.get("VmRSS"), "vmsize_kib": status.get("VmSize"),
        "threads": status.get("Threads"), "fd_count": fd_count,
    }, None


def summarize(samples, field):
    values = [sample[field] for sample in samples if sample.get(field) is not None]
    if not values:
        return None
    return {"first": values[0], "last": values[-1], "minimum": min(values),
            "maximum": max(values), "last_minus_first": values[-1] - values[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=1800.0)
    args = ap.parse_args()
    if args.interval <= 0 or not 0 < args.duration <= 1800:
        ap.error("interval must be positive and duration must be in (0, 1800]")
    session_path = Path(args.session).resolve()
    session = json.loads(session_path.read_text())
    pid = session["children"]["server"]["pid"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("session children.server.pid must be a positive integer")
    starttime = identity(pid)["starttime_ticks"]
    began = utc_now()
    monotonic_start = time.monotonic()
    output = session_path.parent / "process_memory.jsonl"
    summary_path = session_path.parent / "process_memory_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    metadata = {
        "type": "observation_metadata",
        "session_path": str(session_path), "pid": pid,
        "starttime_ticks": starttime,
        "process_started_at": process_started_at(starttime),
        "observation_started_at": began,
        "session_started_at": session.get("started_at"),
        "soak_started_at": session.get("stages", {}).get("soak", {}).get("started_at"),
        "interval_seconds": args.interval, "duration_limit_seconds": args.duration,
        "note": "RSS observation starts at observation_started_at; earlier soak time was not observed.",
    }
    samples, reason, error = [], "duration_limit", None
    # 独占创建，避免重复监测误覆盖同一次验收证据。
    with output.open("x") as log:
        log.write(json.dumps(metadata) + "\n")
        log.flush()
        print(json.dumps(metadata), flush=True)
        try:
            while True:
                try:
                    sample, reason_now = snapshot(pid, starttime)
                except (FileNotFoundError, ProcessLookupError):
                    reason_now, sample = "process_exited", None
                if reason_now:
                    reason = reason_now
                    break
                sample["type"] = "sample"
                sample["observed_elapsed_seconds"] = time.monotonic() - monotonic_start
                samples.append(sample)
                log.write(json.dumps(sample) + "\n")
                log.flush()
                remaining = args.duration - (time.monotonic() - monotonic_start)
                if remaining <= 0:
                    break
                time.sleep(min(args.interval, remaining))
        except KeyboardInterrupt:
            reason = "observer_interrupted"
        except Exception as exc:
            reason, error = "observer_error", f"{type(exc).__name__}: {exc}"
        finally:
            summary = {
                **metadata, "type": "observation_summary", "stop_reason": reason,
                "observation_finished_at": utc_now(),
                "observed_duration_seconds": time.monotonic() - monotonic_start,
                "sample_count": len(samples), "error": error,
                "rss_kib": summarize(samples, "rss_kib"),
                "vmsize_kib": summarize(samples, "vmsize_kib"),
                "threads": summarize(samples, "threads"), "fd_count": summarize(samples, "fd_count"),
                "first_sample_at": samples[0]["timestamp"] if samples else None,
                "last_sample_at": samples[-1]["timestamp"] if samples else None,
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

