"""限定校准：缓存容量压力与纯 P+D Graph 分开消融，不借 speculative 混淆收益。

每次调用只运行一种配置；输出目录必须为新目录。所有请求事件和失败均落盘。
缓存实验默认只限制 scheduler 可见页数，不能据此声称实际显存节省。
显式 --physical-kv-cap 时，在分配前限制真实GPU KV tensor为12页并验证字节数。
两种实验均把缓存写回开销包含在正式请求耗时中；输入分区以split字段为准。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
from benchmarks.runtime_v2_performance import run_trial, environment
from benchmarks.slo_goodput import evaluate

VARIANTS = ("cache_off", "cache_on", "baseline", "mixed", "mixed_graph", "mixed_graph_budget256")


def summarize_experiment(root, output_filename="summary.json"):
    """只聚合 steady，按相同输入逐请求比较 token；执行通过不等于数值通过。"""
    rows, trials = [], {}
    variants = [v for v in VARIANTS if (root / v / "report.json").exists()]
    if not variants:
        raise ValueError("no completed experiment reports")
    for variant in variants:
        folder = root / variant
        report = json.loads((folder / "report.json").read_text())
        for filename in report["trials"]:
            trial = json.loads((folder / filename).read_text())
            if trial["phase"].startswith("steady"):
                trials[(variant, trial["case"], trial["phase"])] = trial
        for case in sorted({key[1] for key in trials if key[0] == variant}):
            group = [t for (v, c, _), t in trials.items() if v == variant and c == case]
            def median(key):
                return statistics.median(t[key] for t in group)
            host = [t["runtime_after"].get("host_cache", {}) or {} for t in group]
            rows.append(dict(variant=variant, case=case, trials=len(group),
                tokens_s=median("tokens_per_second"),
                ttft_ms=statistics.median(r["ttft_ms"] for t in group for r in t["requests"]),
                tpot_ms=statistics.median(r["tpot_ms"] for t in group for r in t["requests"]),
                goodput_tokens_s=statistics.median(t["slo"]["goodput_tokens_s"] for t in group),
                attainment=statistics.mean(t["slo"]["slo_attainment"] for t in group),
                steady_captures=sum(t["runtime_deltas"]["packed_graph"]["captures"] for t in group),
                steady_replays=sum(t["runtime_deltas"]["packed_graph"]["replays"] for t in group),
                steady_fallbacks=sum(t["runtime_deltas"]["packed_graph"]["fallbacks"] for t in group),
                pd_steps=sum(t["observed_paths"]["pd_steps"] for t in group),
                host_loads=[h.get("loads", 0) for h in host],
                host_d2h_ms=[h.get("d2h_wall_ms", 0) for h in host],
                host_h2d_ms=[h.get("h2d_wall_ms", 0) for h in host],
                pass_execution=report["status"] == "PASS_EXECUTION_ONLY" and all(t["status"] == "PASS" for t in group)))
    comparisons = []
    for variant, case, phase in trials:
        reference = "cache_off" if variant == "cache_on" else "baseline"
        if variant in ("baseline", "cache_off"):
            continue
        if (reference, case, phase) not in trials:
            raise ValueError(f"missing reference trial: {reference}/{case}/{phase}")
        left, right = trials[(reference, case, phase)], trials[(variant, case, phase)]
        matches = [a["output_token_ids"] == b["output_token_ids"]
                   for a, b in zip(left["requests"], right["requests"])]
        comparisons.append(dict(variant=variant, reference=reference, case=case, phase=phase,
                                matched_requests=sum(matches), total_requests=len(right["requests"]),
                                all_equal=len(matches) == len(left["requests"]) == len(right["requests"]) and all(matches)))
    output = dict(rows=rows, comparisons=comparisons, variants=variants,
                  status="PASS_CALIBRATION_ONLY" if all(r["pass_execution"] for r in rows)
                    and all(c["all_equal"] for c in comparisons) else "REVIEW_REQUIRED",
                  note="12 requests/trial, 3 same-process steady repeats; no independent rounds or production P99 guarantee")
    with (root / output_filename).open("x") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def make_case(tokenizer, dataset, cache_experiment, split="calibration"):
    # 只从普通文本的 token 池中确定性抽样，不使用 EOS/控制 token。
    # synthetic token 流用于容量/调度压力，不是自然语言质量或真实生产分布。
    pool = tokenizer.encode("Caching attention keys values improves inference latency. "
                            "Write a clear explanation of scheduling memory and queues.")
    rng = random.Random(20260921 if split == "calibration" else 20260922)
    prompts = [[100 + i] + [rng.choice(pool) for _ in range(1024)] for i in range(12)]
    jobs = []
    for i in range(12):
        if cache_experiment:
            prompt = prompts[i % 4] if dataset == "recurring" else prompts[i]
        else:
            prompt = prompts[i][:1025 if i % 4 == 0 else 129]
        jobs.append(dict(job=i, arrival_s=0 if cache_experiment else i * .015,
                         prompt_token_ids=prompt, max_tokens=32))
    return dict(id=dataset, dataset=dataset, scenario="capacity" if cache_experiment else "dynamic",
                concurrency=1 if cache_experiment else 4, jobs=jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--physical-kv-cap", action="store_true",
                        help="cache实验真实只分配12页；旧实验默认保留逻辑容量限制以便复现")
    parser.add_argument("--split", choices=("calibration", "validation"), default="calibration")
    parser.add_argument("--model")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.summary_root:
        summarize_experiment(args.summary_root, args.summary_name)
        return
    if not all((args.model, args.variant, args.output)):
        parser.error("execution requires --model, --variant and --output")
    args.output.mkdir(parents=True, exist_ok=False)
    cache_experiment = args.variant.startswith("cache_")
    graph = args.variant.startswith("mixed_graph")
    config = dict(max_model_len=2048, max_num_seqs=4,
                  max_num_batched_tokens=256 if args.variant.endswith("budget256") else 512,
                  prefill_chunk_size=128, max_prefill_steps=1,
                  gpu_memory_utilization=.4, enforce_eager=False,
                  batching_mode="separate" if cache_experiment or args.variant == "baseline" else "mixed",
                  metadata_backend="reuse", speculative_policy="off",
                  enable_mixed_graph=graph, enable_prefix_caching=cache_experiment,
                  host_kv_cache_bytes=2**30 if args.variant == "cache_on" else 0)
    if args.physical_kv_cap:
        if not cache_experiment:
            parser.error("--physical-kv-cap only applies to cache_on/cache_off")
        config["max_kvcache_blocks"] = 12
    report = dict(status="RUNNING", variant=args.variant, split=args.split, config=config, trials=[],
                  environment=environment(), gpu_index=os.environ.get("NANOVLLM_BENCH_GPU_INDEX"),
                  slo=dict(ttft_ms=1000, tpot_ms=30),
                  scope="synthetic calibration; three same-process repetitions, not independent process rounds",
                  sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for folder in ("nanovllm", "benchmarks") for p in Path(folder).rglob("*.py")})
    llm = None
    try:
        ensure_gpu1_available(evidence_path=args.output / "gpu.jsonl")
        from nanovllm import LLM
        from nanovllm.engine.block_manager import BlockManager
        import torch
        llm = LLM(args.model, **config)
        report["effective_config"] = llm.config.runtime_summary()
        report["physical_kv_blocks"] = llm.config.num_kvcache_blocks
        report["physical_kv_bytes"] = llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size()
        report["capacity_scope"] = "physical" if args.physical_kv_cap else "logical_only" if cache_experiment else "automatic"
        if args.physical_kv_cap:
            assert llm.model_runner.kv_cache.shape[2] == llm.config.num_kvcache_blocks == 12
        cache = llm.scheduler.block_manager.host_cache
        cases = [make_case(llm.tokenizer, dataset, cache_experiment, args.split) for dataset in
                 (("recurring", "unique") if cache_experiment else ("dynamic",))]
        other_cases = [make_case(llm.tokenizer, case["dataset"], cache_experiment,
                      "validation" if args.split == "calibration" else "calibration") for case in cases]
        own = {tuple(j["prompt_token_ids"]) for c in cases for j in c["jobs"]}
        other = {tuple(j["prompt_token_ids"]) for c in other_cases for j in c["jobs"]}
        assert not own & other, "validation input overlaps calibration"
        report["input_split_disjoint"] = True
        (args.output / "inputs.json").write_text(json.dumps(cases))
        # 先保存阈值/配置/源码 hash，再看结果；本轮不按测量结果改 SLO。
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2))
        for case in cases:
            for phase in ("warmup0", "warmup1", "steady0", "steady1", "steady2"):
                ensure_gpu1_available(evidence_path=args.output / "gpu.jsonl")
                if cache_experiment:
                    assert not llm.scheduler.block_manager.used_block_ids
                    assert llm.config.num_kvcache_blocks >= 12
                    # 每个 trial 从空缓存开始；随后 4 个前缀循环三次自然产生压力。
                    llm.scheduler.block_manager = BlockManager(12, llm.config.kvcache_block_size)
                    llm.scheduler.block_manager.host_cache = cache
                    if cache is not None:
                        cache.clear()
                        cache.counters.clear()
                llm.set_graph_capture_enabled(phase.startswith("warmup"))
                trial = run_trial(llm, case, dict(temperature=0, ignore_eos=True,
                    speculative_ngram_size=0), timeout=120)
                trial.update(phase=phase, variant=args.variant,
                             logical_kv_blocks=12 if cache_experiment else llm.config.num_kvcache_blocks)
                trial["slo"] = evaluate(trial, 1000, 30)
                filename = f"{case['id']}_{phase}.json"
                (args.output / filename).write_text(json.dumps(trial, ensure_ascii=False, indent=2))
                report["trials"].append(filename)
                print(json.dumps(dict(variant=args.variant, phase=phase, case=case['id'],
                    status=trial['status'], tokens_s=trial['tokens_per_second'],
                    slo=trial['slo'], graph=trial['runtime_deltas']['packed_graph'])), flush=True)
                if trial["status"] != "PASS":
                    raise RuntimeError(trial["error"] or "resource leak")
                if phase.startswith("steady") and trial["runtime_deltas"]["packed_graph"]["captures"]:
                    raise AssertionError("steady phase unexpectedly captured a graph")
        report["status"] = "PASS_EXECUTION_ONLY"
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if llm is not None:
                llm.exit()
        finally:
            (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
