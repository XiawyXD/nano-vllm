"""投递版引擎评测：相同调度配置，分开记录冷启动和交错稳态。

输入是本仓库固定的、人工编写的自然语言/代码任务，不是线上流量数据。
重复输入单独成组。一个 speculative 事件可以包含多个 token，不能把事件
间隔当成逐 token 延迟。输出 JSONL 保存每次试验及每条请求的真实事件。
运行方式：PYTHONPATH=. python -m benchmarks.release_matrix --output artifacts/release_matrix
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

# 故意使用不同主题的完整段落；长输入拼接不同材料，不用 text * N 凑长度。
NATURAL_TASKS = (
    """A city library is deciding how to digitize its handwritten catalogue. Volunteers can scan
pages in batches, but the catalogue includes crossed-out entries and inconsistent dates.
Design a review workflow that preserves the original image, records uncertain transcriptions,
allows corrections without losing history, and makes it possible to resume interrupted work.
Explain which checks can be automated and where a librarian should make the final decision.""",
    """A research team measures lake temperatures with sensors that sometimes lose network access.
Readings have a device timestamp, a server arrival timestamp, a location, and a calibration version.
Explain how to store delayed measurements, distinguish missing data from invalid measurements,
and calculate a daily average without counting retransmitted records twice. Consider daylight
saving changes, sensor replacement, and a correction to a calibration coefficient made next week.""",
    """You maintain an online bookstore. Inventory is held in two warehouses, payments are handled
by another service, and customers may cancel an order while fulfillment is starting. Describe
a state machine for reservation, payment, shipment, cancellation, and refund. State the invariants
that prevent overselling or issuing two refunds. Explain how idempotency keys and reconciliation
help when a response is lost after the underlying operation actually succeeded.""",
    """A university course offers weekly programming assignments to students with varied experience.
Propose a grading rubric that separates correctness, clarity, testing, and efficiency. A submission
that times out on large inputs should still receive useful feedback on small cases. Discuss how
to report flaky tests, accommodate inaccessible development environments, and give students enough
information to debug without publishing a complete solution before the deadline.""",
    """A small observatory receives a stream of images from several telescopes. Most frames contain
only background noise, but occasional transient events require rapid investigation. Propose a
pipeline with an inexpensive first filter and a more detailed second analysis. Explain how to
keep false alarms measurable, save evidence for rejected candidates, and avoid claiming improved
sensitivity when a new version merely changes the threshold used to label an event.""",
    """An engineering team wants to compare two database indexes. The old index serves mostly warm
queries while the new index is tested immediately after a restart. Query length and selectivity
also vary throughout the day. Design a fair experiment with a fixed workload, independent warmup,
interleaved runs, and raw per-query records. Explain why an average alone can hide regressions and
why a handful of requests cannot support a reliable claim about the ninety-ninth percentile.""",
    """An editor must organize a history exhibit using letters, photographs, oral interviews, and
shipping records. Some accounts disagree about the date of an event. Explain how to construct
a timeline that distinguishes direct evidence, later recollection, and inference. Propose clear
labels for uncertainty, links between related artifacts, and a revision process when a newly
discovered document changes the interpretation of an existing exhibit.""",
    """A mobile application helps hikers record routes in areas without internet access. Users can
edit a route on two devices before either reconnects. Discuss conflict resolution for names,
waypoints, and notes, while retaining enough information to undo a mistaken merge. Consider
battery use, storage limits, privacy of precise locations, and the difference between a locally
saved change and a change that has been acknowledged by the synchronization service.""",
    """A hospital logistics office schedules deliveries of supplies, not medical treatment.
Each vehicle has limited space, some packages need refrigeration, and receiving desks operate
during different hours. Explain how to model constraints and what the planner should do when
an urgent request makes the existing schedule infeasible. Describe an auditable override process
and metrics that measure late deliveries without rewarding the exclusion of difficult requests.""",
    """A software team is retiring an old file format. Some users still have archives written ten
years ago and cannot upgrade all readers at once. Propose a migration strategy with a version
field, compatibility tests, reversible conversion, and validation of unknown optional fields.
Explain how to detect a truncated file, report a checksum failure, and keep the source archive
safe when conversion stops because the output disk has become full.""",
    """A community garden shares water from a rain tank among several plots. Measurements are noisy,
rain forecasts are uncertain, and some plots use drip irrigation while others are watered by hand.
Design a transparent allocation policy that can be explained to volunteers. Separate measured
consumption from estimated demand, describe how to handle a broken meter, and suggest an experiment
that evaluates conservation without ignoring differences in weather and crop requirements.""",
    """A language-model service receives short interactive questions and long document summaries.
The team observes higher throughput after batching more requests, but users wait longer for the
first answer. Explain the distinction between queue time, time to first token, inter-event gaps,
and time per output token. Propose a workload matrix and service objectives that expose this
tradeoff. Include cancellation, admission control, overload, and a method for detecting leaked
request metadata after all active work has finished.""",
)

CODE_TASKS = (
    """Review this cache for thread safety and describe a bounded-memory replacement.
from collections import OrderedDict
class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()
    def get(self, key, loader):
        if key in self.items:
            value = self.items.pop(key)
            self.items[key] = value
            return value
        value = loader(key)
        self.items[key] = value
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return value
Explain duplicate loads, exceptions, recursive loaders, and capacity zero. Provide tests.""",
    """Find the boundary errors and propose a streaming parser with a clear memory limit.
def read_records(chunks):
    pending = b""
    for chunk in chunks:
        pending += chunk
        while b"\\n" in pending:
            row, pending = pending.split(b"\\n", 1)
            yield row.decode("utf-8")
    if pending:
        yield pending.decode("utf-8")
Consider a multibyte character split between chunks, CRLF input, invalid UTF-8, a very large
record, and the caller stopping iteration before the source has been fully consumed.""",
    """Analyze this retry loop and rewrite its contract before changing its implementation.
import time
def send_with_retry(client, payload):
    for attempt in range(5):
        try:
            return client.post(payload)
        except Exception:
            time.sleep(2 ** attempt)
    return None
Distinguish permanent validation errors from temporary connection failure. Explain deadlines,
jitter, cancellation, idempotency keys, and the case where the server commits the operation but
the response is lost. Show how to test retries with a fake clock rather than real sleeps.""",
    """Explain the lifetime of each object in this producer and consumer example.
from queue import Queue
from threading import Thread
def producer(q, files):
    for name in files:
        with open(name, "rb") as stream:
            q.put(stream.read())
    q.put(None)
def consumer(q):
    while True:
        item = q.get()
        if item is None:
            return
        process(item)
Propose bounded backpressure, error propagation, task accounting, cancellation, and deterministic
shutdown. Discuss why placing a sentinel in a full queue may block shutdown indefinitely.""",
    """Review this incremental statistics implementation for numerical stability.
class Moments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.squares = 0.0
    def add(self, value):
        self.count += 1
        self.total += value
        self.squares += value * value
    def variance(self):
        mean = self.total / self.count
        return self.squares / self.count - mean * mean
Discuss empty input, population versus sample variance, cancellation error for large offsets,
and merging independently computed partitions. Give a small adversarial examples.example and tests.""",
    """Design tests for a token bucket rate limiter with a monotonic clock.
class Bucket:
    def __init__(self, capacity, rate, now):
        self.capacity = capacity
        self.rate = rate
        self.tokens = capacity
        self.updated = now()
        self.now = now
    def acquire(self, amount=1):
        current = self.now()
        self.tokens = min(self.capacity, self.tokens + (current-self.updated)*self.rate)
        self.updated = current
        if self.tokens < amount:
            return False
        self.tokens -= amount
        return True
Explain clock rollback, invalid rates, requests larger than capacity, concurrent calls,
and whether rejected requests should change the last-update timestamp.""",
    """Find the indexing problem in this matrix multiplication outline.
def matmul(a, b, m, n, k):
    c = [0.0] * (m*n)
    for row in range(m):
        for col in range(n):
            value = 0.0
            for inner in range(k):
                value += a[row*k+inner] * b[inner*k+col]
            c[row*n+col] = value
    return c
Explain row-major strides, rectangular shapes, bounds checks, and how a reference implementation
should compare floating-point results. Describe a tiled GPU version without claiming a speedup
until the tile size, register pressure, memory traffic, and measured workload are known.""",
    """Specify a minimal asynchronous job service.
Each job has an integer identifier, a creation time, a state, and a result. Clients may submit,
poll, stream progress, or cancel. Write Python type definitions and pseudocode for the state
transitions. Include a maximum number of active jobs, a bounded event buffer, a result retention
deadline, and an explicit failure state. Explain ownership of the worker thread and what a
health endpoint should report after that thread exits unexpectedly.""",
)


def summarize(values):
    """Nearest-rank quantiles, always paired with the contributing sample count."""
    values = sorted(value for value in values if value is not None)
    quantile = lambda q: values[max(0, math.ceil(len(values) * q) - 1)] if values else None
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p50=quantile(.5), p95=quantile(.95), p99=quantile(.99))


def provenance():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    # 服务测试还会导入根目录serve/HTTP辅助脚本；一并冻结，不能只追踪模型子包。
    paths = sorted(set((root / "nanovllm").rglob("*.py"))
                   | set((root / "benchmarks").rglob("*.py")) | set(root.glob("*.py")))
    return dict(git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--short"),
                source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths}, python=platform.python_version())


def build_jobs(tokenizer, dataset, scenario, concurrency, args):
    """Build a fixed, reproducible arrival table shared by all three execution modes."""
    count = concurrency * (2 if scenario == "dynamic" else 1)
    paragraphs = NATURAL_TASKS if dataset == "natural" else CODE_TASKS
    jobs = []
    for index in range(count):
        length = args.long_tokens if scenario == "long" or (
            scenario == "dynamic" and index % concurrency == concurrency - 1) else args.short_tokens
        if dataset == "repeat":
            prompt = ([1, 2, 3] * ((length + 2) // 3))[:length]
        else:
            # Rotating distinct passages gives reproducible inputs without repeated padding.
            offset = index % len(paragraphs)
            text = "\n\n".join(paragraphs[offset:] + paragraphs[:offset])
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) < length:
                raise ValueError(f"{dataset} corpus has {len(encoded)} tokens; requested {length}")
            prompt = encoded[:length]
        arrival = 0.0 if index < concurrency else .05 + (index - concurrency) * .015
        jobs.append(dict(job=index, arrival_s=arrival, prompt_token_ids=prompt,
                         max_tokens=args.output_tokens))
    return jobs


def run_trial(llm, jobs, mode, concurrency, timeout):
    """Run one arrival trace and retain actual event times, counts, and output token IDs."""
    import torch
    from nanovllm import SamplingParams
    runner = llm.model_runner
    runner.enable_verification_graph = mode == "spec_graph"
    before_graph = dict(runner.verification_graphs.stats)
    llm._last_speculative_stats = {}
    llm.scheduler.prefill_steps_since_decode = 0
    records, active, pending = [], {}, 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    failure = None
    try:
        while pending < len(jobs) or active:
            now = time.perf_counter() - start
            if now > timeout:
                raise TimeoutError(f"trial exceeded {timeout}s")
            while (pending < len(jobs) and jobs[pending]["arrival_s"] <= now
                   and len(active) < concurrency):
                job = jobs[pending]
                seq = llm.add_request(job["prompt_token_ids"], SamplingParams(
                    temperature=0, max_tokens=job["max_tokens"], ignore_eos=True,
                    speculative_ngram_size=0 if mode == "ordinary" else 3,
                    speculative_max_draft_tokens=4))
                record = dict(job=job["job"], request_id=seq.seq_id,
                    arrival_s=job["arrival_s"], submit_s=time.perf_counter()-start,
                    input_tokens=len(job["prompt_token_ids"]), events=[], failed=False)
                records.append(record)
                active[seq.seq_id] = (seq, record)
                pending += 1
            if llm.is_finished():
                time.sleep(.001)
                continue
            llm.step()
            now = time.perf_counter() - start
            for event in llm.step_events():
                seq, record = active[event.request_id]
                record["events"].append(dict(time_s=now, cumulative_tokens=event.token_index+1,
                    text=event.text, finished=event.finished, finish_reason=event.finish_reason))
                if event.finished:
                    record["output_token_ids"] = list(seq.completion_token_ids)
                    record["output_tokens"] = seq.num_completion_tokens
                    record["finish_reason"] = event.finish_reason
                    llm.forget_request(seq.seq_id)
                    del active[seq.seq_id]
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        for rid, (seq, record) in list(active.items()):
            record["failed"] = True
            record["error"] = failure
            record["output_token_ids"] = list(seq.completion_token_ids)
            record["output_tokens"] = seq.num_completion_tokens
            llm.cancel_request(rid)
            llm.forget_request(rid)
        # A GPU failure may leave device state invalid; the caller aborts further trials.
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    event_gaps = []
    for record in records:
        events = record["events"]
        count = record.get("output_tokens", 0)
        record["event_gap_ms"] = [(b["time_s"]-a["time_s"])*1000 for a,b in zip(events, events[1:])]
        event_gaps.extend(record["event_gap_ms"])
        record["ttft_ms"] = (events[0]["time_s"]-record["arrival_s"])*1000 if events else None
        record["tpot_ms"] = (events[-1]["time_s"]-events[0]["time_s"])*1000/(count-1) if events and count > 1 else None
        record["latency_ms"] = (events[-1]["time_s"]-record["arrival_s"])*1000 if events else None
    graph = {key: value-before_graph.get(key, 0) for key,value in runner.verification_graphs.stats.items()}
    graph["replay_fraction"] = graph.get("replays", 0) / max(1, graph.get("replays", 0)+graph.get("fallbacks", 0))
    manager = llm.scheduler.block_manager
    output_tokens = sum(record.get("output_tokens", 0) for record in records)
    return dict(mode=mode, seconds=seconds, output_tokens=output_tokens,
        tokens_per_second=output_tokens/seconds, requests=records,
        ttft_ms=summarize([record["ttft_ms"] for record in records]),
        tpot_ms=summarize([record["tpot_ms"] for record in records]),
        event_gap_ms=summarize(event_gaps), spec=llm.speculative_stats(), graph=graph,
        memory=dict(allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()),
        scheduler=llm.scheduler.stats(), used_blocks_after=len(manager.used_block_ids),
        failures=sum(record["failed"] for record in records)+(len(jobs)-len(records)),
        error=failure, greedy_equal=None)


def aggregate(trials):
    groups = {}
    for trial in trials:
        key = f'{trial["dataset"]}/{trial["scenario"]}/c{trial["concurrency"]}/{trial["mode"]}'
        groups.setdefault(key, []).append(trial)
    result = {}
    for key, items in groups.items():
        requests = [record for item in items for record in item["requests"]]
        result[key] = dict(trials=len(items), requests=len(requests),
            throughput_tokens_s=summarize([item["tokens_per_second"] for item in items]),
            ttft_ms=summarize([record["ttft_ms"] for record in requests]),
            tpot_ms=summarize([record["tpot_ms"] for record in requests]),
            event_gap_ms=summarize([gap for record in requests for gap in record["event_gap_ms"]]),
            failures=sum(item["failures"] for item in items),
            greedy_mismatch_trials=sum(item["greedy_equal"] is False for item in items))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-1.7B")
    ap.add_argument("--output", required=True, help="New output directory (never overwrite an old run)")
    ap.add_argument("--concurrencies", default="1,4,8,16")
    ap.add_argument("--datasets", default="natural,code,repeat")
    ap.add_argument("--scenarios", default="short,long,dynamic")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--short-tokens", type=int, default=96)
    ap.add_argument("--long-tokens", type=int, default=768)
    ap.add_argument("--output-tokens", type=int, default=32)
    ap.add_argument("--token-budget", type=int, default=512)
    ap.add_argument("--prefill-chunk-size", type=int, default=256)
    ap.add_argument("--prefill-quota", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=.5)
    ap.add_argument("--trial-timeout", type=float, default=120)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.concurrencies, args.scenarios, args.repeats = "1,4", "short,dynamic", 1
        args.short_tokens, args.long_tokens, args.output_tokens = 48, 96, 8
    cs = [int(value) for value in args.concurrencies.split(",")]
    datasets, scenarios = args.datasets.split(","), args.scenarios.split(",")
    if (min(cs) <= 0 or args.repeats < 1 or args.output_tokens < 2
            or not set(datasets) <= {"natural", "code", "repeat"}
            or not set(scenarios) <= {"short", "long", "dynamic"}):
        ap.error("invalid workload configuration")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    import torch
    from nanovllm import LLM
    metadata = dict(provenance=provenance(), args=vars(args), torch=torch.__version__,
        device=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        dataset_note="fixed authored tasks, not production traces; repeat is an independent synthetic group",
        metric_note="engine-only; nearest-rank quantiles include n; event gaps are not per-token timestamps",
        cold_start_note="model initialization is measured once; verification graph entries are cleared at each independent case boundary; first_use still reuses process-wide compiled kernels; captures/replays distinguish graph cold from warm; bounded fallback remains within each case",
        smoke=args.smoke, cold_start=[], trials=[], status="RUNNING")
    (root/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    start = time.perf_counter()
    llm = LLM(args.model, max_model_len=max(args.long_tokens,args.short_tokens)+args.output_tokens,
              max_num_batched_tokens=args.token_budget, max_num_seqs=max(cs),
              prefill_chunk_size=args.prefill_chunk_size, max_prefill_steps=args.prefill_quota,
              speculative_max_batch_size=0, gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=False, enforce_eager=False, sampler_backend="torch")
    metadata["model_initialization_seconds"] = time.perf_counter()-start
    metadata["dtype"] = str(next(llm.model_runner.model.parameters()).dtype)
    metadata["actual_sampler_backend"] = getattr(llm.model_runner.sampler, "backend", None)
    if metadata["actual_sampler_backend"] != "torch":
        llm.exit()
        raise RuntimeError("release benchmark requires explicit Sampler.backend='torch'")
    modes = ["ordinary", "spec_eager", "spec_graph"]
    try:
        with (root/"trials.jsonl").open("w", encoding="utf-8") as log, (root/"workloads.jsonl").open("w", encoding="utf-8") as inputs:
            for dataset in datasets:
                for scenario in scenarios:
                    for concurrency in cs:
                        # Independent cases must not inherit a full four-entry graph cache
                        # from an unrelated workload. Within each case normal bounded
                        # admission/fallback remains visible in first-use and steady data.
                        llm.model_runner.verification_graphs.entries.clear()
                        jobs = build_jobs(llm.tokenizer, dataset, scenario, concurrency, args)
                        case = dict(dataset=dataset, scenario=scenario, concurrency=concurrency)
                        inputs.write(json.dumps(dict(**case, jobs=jobs))+"\n")
                        inputs.flush()
                        # Each case records its first use independently from measured steady state.
                        # Graph cache is case-cold; process-wide kernels can remain warm.
                        for mode in modes:
                            trial = run_trial(llm, jobs, mode, concurrency, args.trial_timeout)
                            trial.update(**case, phase="first_use", repeat=-1)
                            metadata["cold_start"].append(trial)
                            log.write(json.dumps(trial)+"\n"); log.flush()
                            if trial["error"] or trial["used_blocks_after"]:
                                raise RuntimeError(trial["error"] or "KV references remained after trial")
                        for repeat in range(args.repeats):
                            round_results = []
                            order = modes[repeat%3:]+modes[:repeat%3]
                            for mode in order:
                                trial = run_trial(llm, jobs, mode, concurrency, args.trial_timeout)
                                trial.update(**case, phase="steady", repeat=repeat)
                                round_results.append(trial)
                                if trial["error"] or trial["used_blocks_after"]:
                                    metadata["trials"].extend(round_results)
                                    raise RuntimeError(trial["error"] or "KV references remained after trial")
                            expected = next(item for item in round_results if item["mode"]=="ordinary")
                            expected_ids = [item["output_token_ids"] for item in expected["requests"]]
                            for trial in round_results:
                                trial["greedy_equal"] = [item["output_token_ids"] for item in trial["requests"]] == expected_ids
                                metadata["trials"].append(trial)
                                log.write(json.dumps(trial)+"\n")
                                print(json.dumps({key:trial[key] for key in ("dataset","scenario","concurrency","mode","repeat","tokens_per_second","greedy_equal")}), flush=True)
                            log.flush()
        metadata["execution_checks_passed"] = True
        mismatch_count = sum(item["greedy_equal"] is False for item in metadata["trials"])
        metadata["greedy_mismatch_trials"] = mismatch_count
        metadata["status"] = "REVIEW_REQUIRED" if mismatch_count else "PASS"
        if mismatch_count:
            metadata["review_reason"] = ("No execution or KV-release failure, but greedy token IDs differed. "
                "Run fixed-input numerical/logit comparisons for identical query/cache states; "
                "do not claim semantic equivalence or dismiss the difference as numerical noise without evidence.")
    except Exception as exc:
        metadata["status"], metadata["error"] = "FAIL", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["aggregate"] = aggregate(metadata["trials"])
        # Raw records stay in trials.jsonl; summary retains first-use records and aggregate only.
        metadata["steady_trial_count"] = len(metadata.pop("trials"))
        (root/"summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        llm.exit()
    if metadata["status"] == "REVIEW_REQUIRED":
        raise SystemExit(2)
    if metadata["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
