"""Freeze CPU fixtures, then run guarded independent-process GPU comparisons.

prepare does not load CUDA weights. run/worker require the existing UUID guard.
Five rounds, two warmup trials per case, raw outputs retained, no auto promotion.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sources():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("nanovllm", "benchmarks") for p in sorted((ROOT / folder).rglob("*.py"))}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def file_sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def file_record(path):
    before = path.stat()
    record = dict(bytes=before.st_size, mtime_ns=before.st_mtime_ns, sha256=file_sha(path))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("checkpoint changed during fingerprinting")
    return record


def variants(model, draft_model):
    common = dict(max_model_len=2048, max_num_batched_tokens=512, prefill_chunk_size=128,
                  max_prefill_steps=1, gpu_memory_utilization=.65, tensor_parallel_size=1,
                  kv_cache_dtype="model", metadata_backend="reuse", enable_mixed_graph=False,
                  enforce_eager=False, speculative_policy="off")
    ordinary = dict(temperature=0, ignore_eos=True, speculative_method="off")
    result = {}
    for capacity in (12, 64):
        for mode in ("separate", "mixed"):
            for tier in ("off", "sync", "async"):
                name = f"cache_{mode}_{tier}_{capacity}"
                result[name] = dict(family="cache", engine=dict(common, batching_mode=mode, max_num_seqs=8,
                    max_kvcache_blocks=capacity, enable_prefix_caching=True,
                    host_kv_cache_bytes=0 if tier == "off" else 2**30,
                    host_kv_async_prefetch=tier == "async"), sampling=ordinary,
                    reference=f"cache_{mode}_off_{capacity}")
    for method, length in (("off", 0), ("ngram", 4), *(('draft_model', k) for k in (1, 2, 4, 8))):
        name = f"spec_{method}_{length}"
        sampling = dict(temperature=0, ignore_eos=True, speculative_method=method,
                        speculative_max_draft_tokens=max(1, length), speculative_ngram_size=3 if method == "ngram" else 0)
        result[name] = dict(family="draft", engine=dict(common, batching_mode="mixed", max_num_seqs=1,
            max_kvcache_blocks=12, enable_prefix_caching=False, speculative_policy="fixed",
            draft_model=draft_model if method == "draft_model" else None), sampling=sampling, reference="spec_off_0")
    return result


def workloads(tokenizer, split):
    rng = random.Random(20261023 if split == "calibration" else 20261024)
    pool = tokenizer.encode("Memory caching attention requests queues keys values. ", add_special_tokens=False)
    prompts = [[rng.choice(pool) for _ in range(1025)] for _ in range(12)]
    result = []
    for recurring in (False, True):
        for concurrency in (1, 4, 8):
            jobs = [dict(job=i, arrival_s=0., prompt_token_ids=prompts[i % 4 if recurring else i], max_tokens=64)
                    for i in range(12)]
            result.append(dict(id=f"cache_{'repeat' if recurring else 'unique'}_c{concurrency}",
                family="cache", dataset="synthetic_recurring" if recurring else "synthetic_unique",
                scenario="long_prefix", concurrency=concurrency, jobs=jobs))
    texts = {
        "calibration": ("请解释为什么 KV 缓存可以减少重复计算。", "Explain the role of queues in inference serving.",
                        "def lower_bound(items, key):\n", "red blue green. " * 128),
        "validation": ("请比较大模型推理中的吞吐量与首字延迟。", "Describe memory bandwidth costs in token generation.",
                       "def merge_sorted_lists(left, right):\n", "river stone willow. " * 128),
    }[split]
    for category, text in zip(("chinese", "english", "code", "repeat"), texts):
        jobs = [dict(job=i, arrival_s=0., prompt_token_ids=tokenizer.encode(f"Example {i}:\n" + text), max_tokens=64)
                for i in range(3)]
        result.append(dict(id=f"draft_{category}", family="draft", dataset=category,
                           scenario="single_request", concurrency=1, jobs=jobs))
    return result


def prepare(args):
    from transformers import AutoTokenizer
    from benchmarks.runtime_v2_performance import environment
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    splits = {s: workloads(tokenizer, s) for s in ("calibration", "validation")}
    hashes = {s: {digest(job["prompt_token_ids"]) for case in cases for job in case["jobs"]}
              for s, cases in splits.items()}
    if hashes["calibration"] & hashes["validation"]:
        raise ValueError("calibration and validation share exact tokenized prompts")
    model_files = {}
    for name, directory in (("target", args.model), ("draft", args.draft_model)):
        folder = Path(directory)
        model_files[name] = {p.name: file_record(p)
            for p in sorted(folder.iterdir()) if p.is_file() and p.suffix in (".json", ".safetensors")}
    manifest = dict(schema="cache-draft-v1", model=str(Path(args.model).resolve()),
        draft_model=str(Path(args.draft_model).resolve()), rounds=5, warmup_trials=2,
        source_sha256=sources(), model_files=model_files, environment=environment(),
        variants=variants(args.model, str(Path(args.draft_model).resolve())), workloads=splits,
        split_overlap=0, note="Frozen comparisons; no automatic config selection or default promotion")
    manifest["sha256"] = digest(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "manifest.json", manifest)
    print(json.dumps(dict(status="PREPARED_CPU_ONLY", manifest=str(args.output / "manifest.json"),
                          variants=len(manifest["variants"]), rounds=5)))


def load_manifest(path, *, verify_weights=True):
    manifest = json.loads(path.read_text())
    expected = manifest.pop("sha256")
    if digest(manifest) != expected or manifest["source_sha256"] != sources():
        raise ValueError("frozen manifest or runtime source changed; prepare a new experiment")
    manifest["sha256"] = expected
    # Do not combine workers after a local checkpoint change.
    records = [(manifest["model"], manifest["model_files"]["target"]),
               (manifest["draft_model"], manifest["model_files"]["draft"])]
    records += [(directory, files) for directory, files in manifest.get("additional_models", {}).items()]
    for directory, files in records:
        for filename, record in files.items():
            p = Path(directory) / filename
            stat = p.stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise ValueError("checkpoint file identity changed")
            if verify_weights and file_sha(p) != record["sha256"]:
                raise ValueError("checkpoint fingerprint changed")
    return manifest


def reset_cache(engine):
    manager = engine.scheduler.block_manager
    manager.reap_host_loads(blocking=True)
    if manager.used_block_ids or manager._reservations or manager._pending_host_stores:
        raise AssertionError("trial leaked physical KV ownership")
    if manager.host_cache is not None:
        manager.host_cache.clear()
    for bid in list(manager.free_block_ids):
        manager._remove_prefix_index(bid)


def prepare_trial_cache(engine, case, repeat):
    # Unique prompts must stay a no-reuse control even when warmup replays the
    # same fixture. Recurring cases intentionally retain their warmed prefixes.
    empty = repeat == 0 or case["dataset"] == "synthetic_unique"
    if empty:
        reset_cache(engine)
    return "disabled" if case["family"] == "draft" else "empty" if empty else "warmed_by_previous_trial"


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="NOT_RUN", trials=[])
    engine = None
    try:
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        report["gpu"] = ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
        # The coordinator verifies full SHA256 once; workers recheck source,
        # exact size and mtime to avoid re-reading multi-GB weights every trial.
        manifest = load_manifest(args.manifest, verify_weights=False)
        import torch
        from nanovllm import LLM
        from benchmarks.runtime_v2_performance import run_trial, environment
        current_environment = environment()
        for key in ("python", "packages"):
            if current_environment[key] != manifest["environment"][key]:
                raise ValueError("benchmark software environment changed")
        for key, value in manifest["environment"]["variables"].items():
            if key not in ("CUDA_VISIBLE_DEVICES", "NANOVLLM_BENCH_GPU_INDEX") and current_environment["variables"][key] != value:
                raise ValueError(f"benchmark environment option changed: {key}")
        variant = manifest["variants"][args.variant]
        if args.case is not None and not any(case["id"] == args.case and case["family"] == variant["family"]
                                              for case in manifest["workloads"][args.split]):
            raise ValueError("profile case is absent from this variant's frozen workload")
        started = time.perf_counter()
        engine_options = dict(variant["engine"])
        engine_model = engine_options.pop("model", manifest["model"])
        if engine_model != manifest["model"] and engine_model not in manifest.get("additional_models", {}):
            raise ValueError("variant model is not in the frozen checkpoint inventory")
        engine = LLM(engine_model, **engine_options)
        if engine.config.num_kvcache_blocks != variant["engine"]["max_kvcache_blocks"]:
            raise ValueError("available memory cannot realize the frozen physical KV capacity")
        torch.cuda.synchronize()
        report.update(startup_seconds=time.perf_counter() - started, variant=args.variant,
                      split=args.split, round=args.round, manifest_sha256=manifest["sha256"],
                      effective=engine.config.runtime_summary())
        report["environment"] = current_environment
        for case in manifest["workloads"][args.split]:
            if case["family"] != variant["family"]:
                continue
            if args.case is not None and case["id"] != args.case:
                continue
            load_manifest(args.manifest, verify_weights=False)
            for repeat in range(manifest["warmup_trials"] + 1):
                cache_state = prepare_trial_cache(engine, case, repeat)
                ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
                profile = args.profile and repeat == manifest["warmup_trials"]
                context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) if profile else nullcontext()
                with context as profiler:
                    trial = run_trial(engine, case, variant["sampling"], timeout=600)
                if profile:
                    profiler.export_chrome_trace(str(args.output / f"{case['id']}_trace.json"))
                name = f"{case['id']}_{repeat}.json"
                trial.update(phase="profile" if profile else "steady" if repeat == manifest["warmup_trials"] else "warmup",
                             manifest_sha256=manifest["sha256"], initial_cache_state=cache_state)
                write_new(args.output / name, trial)
                report["trials"].append(name)
                if trial["status"] != "PASS":
                    raise RuntimeError(f"trial failed: {name}")
        report["status"] = "DIAGNOSTIC_ONLY" if args.profile else "PASS_EXECUTION_ONLY"
    except Exception as error:
        report.update(status="NOT_RUN_GPU_UNAVAILABLE" if type(error).__name__ == "GPUUnavailable" else "FAIL",
                      error=f"{type(error).__name__}: {error}")
    finally:
        if engine is not None:
            engine.exit()
        write_new(args.output / "report.json", report)
    return 0 if report["status"] in ("PASS_EXECUTION_ONLY", "DIAGNOSTIC_ONLY") else 2


def summarize(manifest, folder, split):
    trials = {}
    gpu_uuids = set()
    for round_id in range(manifest["rounds"]):
        for name, variant in manifest["variants"].items():
            path = folder / f"r{round_id}_{name}"
            report = json.loads((path / "report.json").read_text())
            if (report["status"] != "PASS_EXECUTION_ONLY" or report["manifest_sha256"] != manifest["sha256"]
                    or report["split"] != split or report["variant"] != name or report["round"] != round_id):
                raise ValueError("incomplete or mismatched worker cannot be summarized")
            gpu_uuids.add(report["gpu"]["gpu"]["uuid"])
            expected = {case["id"] for case in manifest["workloads"][split] if case["family"] == variant["family"]}
            steady = {}
            warmups = {}
            for filename in report["trials"]:
                trial = json.loads((path / filename).read_text())
                if trial["status"] != "PASS" or trial["manifest_sha256"] != manifest["sha256"]:
                    raise ValueError("failed trial cannot be summarized")
                if trial["phase"] == "steady":
                    if trial["case"] in steady:
                        raise ValueError("duplicate steady trial")
                    steady[trial["case"]] = trial
                elif trial["phase"] == "warmup":
                    warmups[trial["case"]] = warmups.get(trial["case"], 0) + 1
                else:
                    raise ValueError("profiler diagnostics cannot enter performance summary")
            if set(steady) != expected:
                raise ValueError("missing steady cases")
            if warmups != {case: manifest["warmup_trials"] for case in expected}:
                raise ValueError("warmup coverage differs from the frozen manifest")
            trials.update({(round_id, name, case): row for case, row in steady.items()})
    if len(gpu_uuids) != 1:
        raise ValueError("cannot combine performance rounds from different GPUs")
    comparisons = []
    for name, variant in manifest["variants"].items():
        references = [variant["reference"]]
        if name.startswith("cache_mixed_"):
            references.append(name.replace("cache_mixed_", "cache_separate_"))
        for reference in dict.fromkeys(references):
            if reference == name:
                continue
            for case in manifest["workloads"][split]:
                if case["family"] != variant["family"]:
                    continue
                ratios, equal = [], True
                for round_id in range(manifest["rounds"]):
                    left, right = (trials[round_id, n, case["id"]] for n in (reference, name))
                    equal &= ([r["output_token_ids"] for r in left["requests"]]
                              == [r["output_token_ids"] for r in right["requests"]]
                              and len(left["requests"]) == len(right["requests"]) == len(case["jobs"]))
                    ratios.append(right["tokens_per_second"] / left["tokens_per_second"])
                comparisons.append(dict(variant=name, reference=reference, case=case["id"],
                    rounds=len(ratios), per_round_speed_ratio=ratios,
                    median_speed_ratio=statistics.median(ratios), output_tokens_equal=equal,
                    correctness_criterion=variant.get("correctness_criterion", "exact_tokens")))
    # 随机版本不能拿同 seed 文本相等当分布验收；有损量化亦有独立 NLL 门槛。
    # 这里只完成测量，绝不自动把缺失的分布/质量证据记为通过。
    exact = [r for r in comparisons if r["correctness_criterion"] == "exact_tokens"]
    status = ("REVIEW_REQUIRED" if not all(r["output_tokens_equal"] for r in exact) else
              "MEASUREMENTS_ONLY_SEPARATE_CORRECTNESS_REQUIRED" if len(exact)!=len(comparisons) else
              "COMPLETE_MEASUREMENTS")
    return dict(status=status, split=split, comparisons=comparisons, manifest_sha256=manifest["sha256"],
                note="Raw TTFT/TPOT, memory, transfer counts and acceptance remain in each trial. No universal speedup claim.")


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
        manifest = load_manifest(args.manifest)
        for round_id in range(manifest["rounds"]):
            names = list(manifest["variants"])
            random.Random(round_id).shuffle(names)
            for name in names:
                ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
                command = [sys.executable, "-m", "benchmarks.cache_draft_experiment", "worker",
                    "--manifest", str(args.manifest.resolve()), "--split", args.split, "--variant", name,
                    "--round", str(round_id), "--output", str((args.output / f"r{round_id}_{name}").resolve())]
                with (args.output / f"r{round_id}_{name}.log").open("x") as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
                if result.returncode:
                    raise RuntimeError(f"worker {name} round {round_id} failed; see raw report")
        report = summarize(manifest, args.output, args.split)
    except Exception as error:
        report = dict(status="NOT_RUN_GPU_UNAVAILABLE" if type(error).__name__ == "GPUUnavailable" else "FAIL",
                      error=f"{type(error).__name__}: {error}", performance="NOT_ACCEPTED")
    write_new(args.output / "summary.json", report)
    print(json.dumps(report))
    return 0 if report["status"] in ("COMPLETE_MEASUREMENTS", "MEASUREMENTS_ONLY_SEPARATE_CORRECTNESS_REQUIRED") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--draft-model", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    for action in ("run", "worker"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--split", choices=("calibration", "validation"), required=True)
        if action == "worker":
            cmd.add_argument("--variant", required=True)
            cmd.add_argument("--round", type=int, required=True)
            cmd.add_argument("--profile", action="store_true", help="Diagnostic traces; excluded from formal timing")
            cmd.add_argument("--case", help="Profile one case; not allowed for formal measurements")
    args = parser.parse_args()
    if args.action == "worker" and args.case is not None and not args.profile:
        parser.error("--case is only allowed for independent profiler diagnostics")
    return {"prepare": prepare, "run": run, "worker": worker}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
