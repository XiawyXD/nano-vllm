"""用请求级结果计算有效吞吐；不把取消、失败、缺失请求从分母中藏掉。

输入为 runtime_v2_performance 的单个 trial JSON，或HTTP soak的phase目录。
阈值必须显式给定，
应在看验证集之前冻结；本程序不自动寻找能让结果最好看的阈值。
"""
import argparse
import json
import math
from pathlib import Path


def evaluate(trial, ttft_limit_ms, tpot_limit_ms):
    seconds = trial["seconds"]
    for value in (seconds, ttft_limit_ms, tpot_limit_ms):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("duration and SLO limits must be finite and positive")
    records = trial["requests"]
    missing = trial.get("missing_requests", 0)
    if type(missing) is not int or missing < 0:
        raise ValueError("invalid missing_requests")
    total = len(records) + missing
    good = completed = tokens = 0
    def meets(value, limit):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and 0 <= value <= limit)
    for row in records:
        count = row.get("output_tokens", 0)
        done = (not row.get("failed", False) and row.get("finish_reason") in ("stop", "length")
                and type(count) is int and count > 0)
        completed += bool(done)
        # 仅一个 token 时 TPOT 没有定义，而不是假装为 0ms；只考核 TTFT。
        passed = done and meets(row.get("ttft_ms"), ttft_limit_ms) and (
            count == 1 or meets(row.get("tpot_ms"), tpot_limit_ms))
        if passed:
            good += 1
            tokens += count
    return dict(total_requests=total, completed_requests=completed, slo_met_requests=good,
                slo_attainment=good / total if total else None,
                goodput_requests_s=good / seconds, goodput_tokens_s=tokens / seconds,
                ttft_limit_ms=ttft_limit_ms, tpot_limit_ms=tpot_limit_ms,
                scope="engine arrival-to-event, not HTTP; TPOT is amortized, not inter-event P99")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial", type=Path)
    parser.add_argument("--ttft-ms", type=float, required=True)
    parser.add_argument("--tpot-ms", type=float, required=True)
    args = parser.parse_args()
    if args.trial.is_dir():
        # HTTP soak phase目录：使用完整原始请求，不删除取消/失败行来美化分母。
        summary = json.loads((args.trial / "summary.json").read_text())
        rows = [json.loads(line) for line in (args.trial / "requests.jsonl").read_text().splitlines()]
        rows = [dict(row, output_tokens=row.get("observed_output_tokens", 0),
                     failed=row.get("outcome") == "UNEXPECTED") for row in rows if row.get("record") == "request"]
        workload = summary["workload"]
        missing = workload["admitted"] - len(rows)
        if missing < 0:
            raise ValueError("raw request count exceeds admitted count")
        result = evaluate(dict(seconds=workload["seconds"], requests=rows, missing_requests=missing),
                          args.ttft_ms, args.tpot_ms)
        from benchmarks.release_matrix import summarize
        result.update(scope="localhost HTTP request-to-SSE; includes intentional cancellations in denominator",
                      ttft_ms=summarize([r.get("ttft_ms") for r in rows]),
                      tpot_ms=summarize([r.get("tpot_ms") for r in rows]),
                      event_gap_ms=summarize([g for r in rows for g in r.get("event_gap_ms", [])]))
    else:
        result = evaluate(json.loads(args.trial.read_text()), args.ttft_ms, args.tpot_ms)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
