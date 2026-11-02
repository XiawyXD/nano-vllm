"""仅面向 localhost 的 HTTP soak：持续完成/取消混合请求并保存原始证据。

本工具不启动或停止服务器。运行前应启动专用测试服务，确认没有其他请求。
默认运行30分钟；--duration 15 --concurrency 2 可用于 smoke。
SSE事件可以携带多个token，所以同时保存累计token计数和真实事件间隔。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import threading
import time
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from benchmarks.release_matrix import NATURAL_TASKS, CODE_TASKS, provenance, summarize


def request_json(base, path, payload=None, timeout=10):
    """Return status and parsed JSON, including HTTP failures as evidence."""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base+path, data=body, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
        except ValueError:
            data = {"error": text}
        return exc.code, data


def run_request(base, number, cancel, args, stop_event):
    """Read a real SSE stream and cancel a subset after its first generated event."""
    prompt = (NATURAL_TASKS if number % 2 == 0 else CODE_TASKS)[number % 8]
    params = dict(temperature=0, ignore_eos=True,
        max_tokens=args.cancel_tokens if cancel else args.max_tokens,
        speculative_ngram_size=args.ngram_size)
    body = json.dumps(dict(prompt=prompt, sampling_params=params, stream=True)).encode("utf-8")
    record = dict(number=number, intended_cancel=cancel, events=[], failed=False,
                  request_id=None, cancel_attempts=0)
    started = time.perf_counter()
    request = urllib.request.Request(base+"/v1/generate", data=body,
                                     headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
            record["http_status"] = response.status
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                text = line[5:].strip()
                if not text or text == b"[DONE]":
                    continue
                event = json.loads(text)
                now = time.perf_counter()-started
                record["request_id"] = event.get("request_id", record["request_id"])
                record["events"].append(dict(time_s=now, **event))
                if event.get("finished"):
                    record["finish_reason"] = event.get("finish_reason")
                    break
                if ((cancel or stop_event.is_set()) and record["request_id"] is not None
                        and not record["cancel_attempts"]):
                    record["cancel_attempts"] += 1
                    status, result = request_json(base,
                        f'/v1/requests/{record["request_id"]}/cancel', {}, args.request_timeout)
                    record["cancel_http_status"] = status
                    record["cancel_response"] = result
                    # 404 may be a legitimate race with completion, recorded separately.
                    if status not in (200, 404):
                        raise RuntimeError(f"cancel returned HTTP {status}: {result}")
            else:
                raise RuntimeError("SSE closed without a terminal event")
    except urllib.error.HTTPError as exc:
        record["http_status"] = exc.code
        record["error"] = exc.read().decode("utf-8", errors="replace")
        record["failed"] = True
    except Exception as exc:
        record["failed"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # If transport failed after obtaining an ID, request cancellation so the tool
        # does not intentionally leave background inference running.
        if record["failed"] and record["request_id"] is not None:
            try:
                record["cleanup_cancel"] = request_json(base,
                    f'/v1/requests/{record["request_id"]}/cancel', {}, args.request_timeout)
            except Exception as exc:
                record["cleanup_error"] = str(exc)
    record["seconds"] = time.perf_counter()-started
    token_events = [event for event in record["events"] if event.get("token_id", -1) >= 0]
    record["output_tokens"] = max((event.get("token_index", -1)+1 for event in token_events), default=0)
    record["observed_output_tokens"] = record["output_tokens"]
    record["ttft_ms"] = token_events[0]["time_s"]*1000 if token_events else None
    record["tpot_ms"] = ((token_events[-1]["time_s"]-token_events[0]["time_s"])*1000
                          /(record["output_tokens"]-1)) if record["output_tokens"] > 1 else None
    record["event_gap_ms"] = [(b["time_s"]-a["time_s"])*1000
                             for a,b in zip(token_events, token_events[1:])]
    if record.get("finish_reason") in {"error", "timeout"}:
        record["failed"] = True
    record["cancel_verified"] = (record.get("cancel_http_status")==200
                                  and record.get("finish_reason")=="cancelled")
    if record.get("cancel_http_status")==200 and not record["cancel_verified"]:
        record["failed"] = True
        record["error"] = "accepted cancellation did not produce a cancelled terminal event"
    return record


def inspect_metrics(snapshot):
    """Check reported limits without assuming zero retained *completed* records."""
    problems = []
    if not snapshot.get("healthy", False):
        problems.append("worker_unhealthy")
    limits = snapshot.get("limits", {})
    inflight = snapshot.get("inflight_requests")
    retained = snapshot.get("retained_requests")
    # Keep names explicit; server metrics are part of the report for auditing.
    max_inflight = limits.get("max_inflight_requests", limits.get("max_inflight"))
    max_finished = limits.get("max_finished_records", limits.get("max_retained_requests",
                        limits.get("max_completed_records", limits.get("max_completed"))))
    if max_inflight is not None and inflight is not None and inflight > max_inflight:
        problems.append("inflight_exceeds_limit")
    if max_finished is not None and retained is not None and inflight is not None:
        if retained > inflight + max_finished:
            problems.append("retained_records_exceed_limit")
    max_events = limits.get("max_stream_events")
    events = snapshot.get("active_stream_events")
    if max_events is not None and retained is not None and events is not None:
        if events > retained * max_events:
            problems.append("stream_events_exceed_limit")
    return problems


def is_idle(snapshot):
    engine = snapshot.get("engine", {})
    return (snapshot.get("inflight_requests")==0 and snapshot.get("queue_depth")==0
            and engine.get("used_blocks", engine.get("kv_used_blocks"))==0
            and engine.get("kv_ref_total")==0 and engine.get("request_records")==0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--output", required=True, help="New output directory")
    ap.add_argument("--duration", type=float, default=1800)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cancel-every", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--cancel-tokens", type=int, default=128)
    ap.add_argument("--ngram-size", type=int, default=0)
    ap.add_argument("--request-timeout", type=float, default=120)
    ap.add_argument("--metrics-interval", type=float, default=1)
    ap.add_argument("--drain-timeout", type=float, default=30)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        ap.error("only an explicitly local HTTP test server is supported")
    if (args.duration <= 0 or args.concurrency <= 0 or args.cancel_every < 2
            or args.max_tokens < 2 or args.cancel_tokens < 2 or args.metrics_interval <= 0):
        ap.error("duration/concurrency/token counts must be positive; cancel-every >= 2")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    status, first = request_json(base, "/metrics")
    (root/"startup.json").write_text(json.dumps(dict(http_status=status, metrics=first), indent=2), encoding="utf-8")
    if status != 200:
        raise RuntimeError(f"GET /metrics must be available, received {status}: {first}")
    if not is_idle(first):
        raise RuntimeError(f"use an idle dedicated test server; initial metrics: {first}")
    result = dict(provenance=provenance(), args=vars(args), initial_metrics=first,
        metric_note="HTTP client-observed cumulative token progress, not GPU-generated throughput. Cancellation may discard buffered events. ttft_ms is first token event even with empty text; first visible text can be later due to stop-prefix or incomplete UTF-8 buffering. Retained HTTP results may remain, but engine request records and KV references must reach zero.",
        status="RUNNING", requests=0, completed=0, cancelled=0, failed=0,
        cancel_verified=0, cancel_completion_races=0, http_status_counts={},
        metric_samples=0, metric_violations=[], output_tokens=0)
    # Raw events are streamed to disk. Keep only latency scalars for aggregate quantiles.
    ttft, tpot, gaps = [], [], []
    stop_event = threading.Event()
    started = time.perf_counter()
    next_metrics = 0.0
    next_request = 0
    futures = set()

    with (root/"requests.jsonl").open("w", encoding="utf-8") as raw, \
         (root/"metrics.jsonl").open("w", encoding="utf-8") as metrics_log:
        def poll_metrics():
            nonlocal next_metrics
            now = time.perf_counter()-started
            try:
                code, snapshot = request_json(base, "/metrics", timeout=10)
                violations = inspect_metrics(snapshot) if code == 200 else [f"metrics_http_{code}"]
            except Exception as exc:
                snapshot, violations = {"error":str(exc)}, ["metrics_unreachable"]
            row = dict(time_s=now, metrics=snapshot, violations=violations)
            metrics_log.write(json.dumps(row)+"\n"); metrics_log.flush()
            result["metric_samples"] += 1
            if violations:
                result["metric_violations"].append(dict(time_s=now, reasons=violations))
            next_metrics = now+args.metrics_interval
            return snapshot

        def consume(future):
            try:
                record = future.result()
            except Exception as exc:
                record = dict(failed=True, error=f"{type(exc).__name__}: {exc}",
                              output_tokens=0, event_gap_ms=[])
            raw.write(json.dumps(record)+"\n"); raw.flush()
            result["requests"] += 1
            result["failed"] += int(record["failed"])
            result["cancelled"] += int(record.get("finish_reason")=="cancelled")
            result["completed"] += int(record.get("finish_reason") in {"length","stop"})
            result["cancel_verified"] += int(record.get("cancel_verified", False))
            result["cancel_completion_races"] += int(record.get("cancel_http_status")==404)
            result["output_tokens"] += record.get("output_tokens",0)
            code = str(record.get("http_status", "transport_error"))
            result["http_status_counts"][code] = result["http_status_counts"].get(code,0)+1
            ttft.append(record.get("ttft_ms")); tpot.append(record.get("tpot_ms"))
            gaps.extend(record.get("event_gap_ms",[]))

        try:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                while time.perf_counter()-started < args.duration:
                    while len(futures) < args.concurrency and time.perf_counter()-started < args.duration:
                        cancel = next_request % args.cancel_every == args.cancel_every-1
                        futures.add(pool.submit(run_request, base, next_request, cancel, args, stop_event))
                        next_request += 1
                    done, futures = wait(futures, timeout=.05, return_when=FIRST_COMPLETED)
                    for future in done:
                        consume(future)
                    if time.perf_counter()-started >= next_metrics:
                        poll_metrics()
                    if result["metric_violations"]:
                        stop_event.set()
                        break
                stop_event.set()
                for future in futures:
                    consume(future)
        finally:
            stop_event.set()
            deadline = time.perf_counter()+args.drain_timeout
            final = poll_metrics()
            while not is_idle(final) and time.perf_counter() < deadline:
                time.sleep(min(args.metrics_interval, 1))
                final = poll_metrics()
            result["final_metrics"] = final
            result["kv_references_released"] = is_idle(final)
            result["seconds"] = time.perf_counter()-started
            result["tokens_per_second_including_cancelled"] = result["output_tokens"]/result["seconds"]
            result["observed_output_tokens"] = result["output_tokens"]
            result["observed_tokens_per_second"] = result["tokens_per_second_including_cancelled"]
            result["final_kv_ref_total"] = final.get("engine",{}).get("kv_ref_total")
            result["final_engine_request_records"] = final.get("engine",{}).get("request_records")
            result["ttft_ms"] = summarize(ttft)
            result["tpot_ms"] = summarize(tpot)
            result["event_gap_ms"] = summarize(gaps)
            result["status"] = ("PASS" if result["requests"] and not result["failed"]
                and not result["metric_violations"] and result["kv_references_released"]
                and result["cancel_verified"] else "FAIL")
            (root/"summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
