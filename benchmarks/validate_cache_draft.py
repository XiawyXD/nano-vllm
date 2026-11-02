"""Archive CPU contracts and optional guarded GPU acceptance without overwrites."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.cache_draft_experiment import sources, write_new

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-1.7B")
    parser.add_argument("--draft-model", default="/home/hrxu/huggingface/Qwen3-0.6B")
    parser.add_argument("--with-gpu", action="store_true")
    parser.add_argument("--full", action="store_true", help="GPU gates then five-round matrices, profile and 30-minute HTTP soak")
    args = parser.parse_args()
    if args.full and not args.with_gpu:
        parser.error("--full requires --with-gpu")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(cpu=[], gpu=[], gpu_status="NOT_RUN", performance_status="NOT_RUN",
                  soak_status="NOT_RUN", optimized_profile="PENDING_VALIDATION", source_sha256=sources())

    def save():
        # This report is owned by this fresh session; original logs are exclusive.
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def execute(name, command, *, cpu=False, timeout=180):
        started = time.perf_counter()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="") if cpu else dict(os.environ)
        # Local HTTP fixtures must reach their owned server even on a host with
        # HTTP_PROXY set and no loopback bypass. Preserve all existing entries.
        for key in ("NO_PROXY", "no_proxy"):
            entries = [item for item in env.get(key, "").split(",") if item]
            env[key] = ",".join(dict.fromkeys([*entries, "127.0.0.1", "localhost"]))
        with (output / f"{name}.log").open("x") as log:
            try:
                result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        row = dict(name=name, command=command, exit_code=code, seconds=time.perf_counter() - started,
                   log=f"{name}.log")
        report["cpu" if cpu else "gpu"].append(row)
        save()
        print(json.dumps(row), flush=True)
        return code == 0

    stages = json.loads((ROOT / "benchmarks/configs/runtime_v2_cpu_plan.json").read_text())["stages"]
    for stage in stages:
        execute(stage["name"], ["-m", "benchmarks", "run", stage["entry"], *stage["args"]],
                cpu=True, timeout=stage["timeout_seconds"])
    execute("tiered_cache", ["-m", "tests.regression.check_tiered_cache", "-v"], cpu=True)
    execute("cache_draft", ["-m", "tests.regression.check_cache_draft", "-v"], cpu=True)
    execute("cache_draft_harness", ["-m", "tests.regression.check_cache_draft_harness", "-v"], cpu=True)
    report["cpu_status"] = "PASS" if all(row["exit_code"] == 0 for row in report["cpu"]) else "FAIL"
    if args.with_gpu and report["cpu_status"] == "PASS":
        try:
            from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
            report["gpu_guard"] = ensure_gpu1_available(evidence_path=output / "gpu_guard.jsonl")
        except Exception as error:
            report.update(gpu_status="NOT_RUN_GPU_UNAVAILABLE", gpu_error=f"{type(error).__name__}: {error}")
        else:
            cache_correct = True
            for mode in ("separate", "mixed"):
                for asynchronous in (False, True):
                    name = f"gpu_cache_{mode}_{'async' if asynchronous else 'sync'}"
                    command = ["-m", "tests.regression.check_tiered_model", "--model", args.model,
                        "--batching-mode", mode, "--output", str(output / name)]
                    if asynchronous:
                        command.append("--async-prefetch")
                    cache_correct &= execute(name, command, timeout=900)
            draft_correct = execute("gpu_draft", ["-m", "tests.regression.check_draft_model", "--model", args.model,
                "--draft-model", args.draft_model, "--output", str(output / "gpu_draft")], timeout=1800)
            existing_correct = execute("existing_numerics", ["-m", "tests.regression.check_runtime_v2_engine",
                "--model", args.model, "--quick", "--output", str(output / "existing_numerics.json")], timeout=1800)
            correct = cache_correct and draft_correct and existing_correct
            report.update(gpu_status="PASS_SCOPED" if correct else "REVIEW_REQUIRED",
                          cache_roundtrip_status="PASS" if cache_correct else "REVIEW_REQUIRED",
                          draft_numeric_status="PASS_SCOPED" if draft_correct else "REVIEW_REQUIRED",
                          existing_numeric_status="PASS_SCOPED" if existing_correct else "REVIEW_REQUIRED")
            if args.full and cache_correct:
                # A rejected numerical candidate still deserves raw diagnostic
                # measurements, but can never become an accepted speedup claim.
                report["performance_numeric_gate"] = "PASS_SCOPED" if correct else "REVIEW_REQUIRED"
                frozen = output / "experiment"
                prepared = execute("freeze", ["-m", "benchmarks.cache_draft_experiment", "prepare",
                    "--model", args.model, "--draft-model", args.draft_model, "--output", str(frozen)], timeout=600)
                measured = prepared
                if prepared:
                    for split in ("calibration", "validation"):
                        measured &= execute(f"performance_{split}", ["-m", "benchmarks.cache_draft_experiment", "run",
                            "--manifest", str(frozen / "manifest.json"), "--split", split,
                            "--output", str(output / f"performance_{split}")], timeout=172800)
                    execute("profile", ["-m", "benchmarks.cache_draft_experiment", "worker", "--manifest",
                        str(frozen / "manifest.json"), "--split", "calibration", "--variant", "cache_mixed_async_12",
                        "--round", "0", "--profile", "--case", "cache_repeat_c4", "--output", str(output / "profile")], timeout=14400)
                report["performance_status"] = "COMPLETE_MEASUREMENTS" if measured and correct else "REVIEW_REQUIRED"
                soaked = execute("mixed_soak", ["-m", "benchmarks.runtime_v2_soak", "run", "--model", args.model,
                    "--config", str(ROOT / "benchmarks/configs/cache_draft_mixed_soak.json"), "--duration", "1800",
                    "--concurrency", "8", "--output", str(output / "mixed_soak")], timeout=5400)
                report["soak_status"] = "PASS" if soaked else "REVIEW_REQUIRED"
    report["status"] = "FAIL" if report["cpu_status"] != "PASS" else "CPU_PASS_GPU_NOT_RUN"
    if report["gpu_status"] == "REVIEW_REQUIRED":
        report["status"] = "REVIEW_REQUIRED"
    elif report["gpu_status"] == "PASS_SCOPED":
        report["status"] = "GPU_SCOPED_ONLY" if not args.full else (
            "COMPLETE_SCOPED" if report["performance_status"] == "COMPLETE_MEASUREMENTS" and report["soak_status"] == "PASS"
            else "REVIEW_REQUIRED")
    save()
    print(json.dumps({key: value for key, value in report.items() if key not in ("cpu", "gpu", "source_sha256")}))
    return 1 if report["status"] == "FAIL" else 2 if report["status"] == "REVIEW_REQUIRED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
