"""Guarded, small learning cases for the current runtime; not a benchmark.

Each invocation loads one profile in a fresh process and writes a new report.
Use --show to inspect a case without initializing CUDA or creating an output.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
# Keep this checkout ahead of any separately installed nano-vllm package.
sys.path.insert(0, str(ROOT))

CASES = {
    "step": "ordinary_service.json",
    "ordinary": "ordinary_service.json",
    "ordinary-eager": "ordinary_service.json",
    "prefix": "ordinary_service.json",
    "cancel": "ordinary_service.json",
    "stop": "ordinary_service.json",
    "ngram": "ordinary_service.json",
    "tiered": "tiered_cache.json",
    "draft": "fp16_draft.json",
    "draft-prefill": "fp16_draft.json",
    "draft-device": "experiments/fp16_draft_device_verification.json",
    "draft-packed": "experiments/fp16_draft_packed_linear.json",
    "stochastic": "fp16_draft.json",
    "quant-target": "quantized_target.json",
}


def make_case(name: str, max_tokens: int):
    profile_path = ROOT / "configs" / CASES[name]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    config = dict(profile["config"])
    sampling = dict(profile.get("sampling", {}))
    sampling.update(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    if name == "step":
        config.update(prefill_chunk_size=3, max_num_batched_tokens=7,
                      max_num_seqs=1, enable_prefix_caching=False)
        sampling.update(max_tokens=3)
    elif name == "ordinary-eager":
        config["enforce_eager"] = True
    elif name == "ngram":
        config["speculative_policy"] = "fixed"
        sampling.update(speculative_method="ngram", speculative_ngram_size=3,
                        speculative_max_draft_tokens=4)
    elif name == "draft-prefill":
        config["draft_prefill_graph"] = True
    elif name == "stochastic":
        config["enable_stochastic_speculation"] = True
        sampling.update(temperature=0.8, speculative_sampling="rejection", seed=123)
    return profile_path, profile, config, sampling


def emit(record):
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="请用一句话解释什么是 KV Cache。")
    parser.add_argument("--output", type=Path, help="new evidence directory; must not exist")
    parser.add_argument("--show", action="store_true", help="only print effective config; no GPU")
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    profile_path, profile, config, sampling = make_case(args.case, args.max_tokens)
    info = dict(case=args.case, profile=str(profile_path), profile_status=profile.get("status"),
                config=config, sampling=sampling,
                scope="Small current-source learning demo; not acceptance or performance evidence")
    if args.show:
        emit(info)
        return
    if args.output is None:
        parser.error("--output is required unless --show is used")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or not os.environ.get("NANOVLLM_BENCH_GPU_INDEX"):
        parser.error("set a full CUDA_VISIBLE_DEVICES UUID and physical NANOVLLM_BENCH_GPU_INDEX")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(info, status="RUNNING", events=[], records=[])
    engine = None
    try:
        # The guard checks the UUID/index mapping and foreign compute processes
        # before importing the model or initializing CUDA. It is not a GPU lock.
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        report["gpu_admission"] = ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
        from nanovllm import LLM, SamplingParams
        from nanovllm.metrics import cuda_memory_snapshot

        engine_config = dict(config)
        model = engine_config.pop("model")
        started = time.perf_counter()
        engine = LLM(model, **engine_config)
        report["startup_seconds"] = time.perf_counter() - started
        ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
        sp = SamplingParams(**sampling)
        manager = engine.scheduler.block_manager

        if args.case == "step":
            # Fixed valid IDs make the Q=3,3,1 example independent of tokenizer text.
            prompt = list(range(100, 107))
            seq = engine.add_request(prompt, sp)
            report["prompt_token_ids"] = prompt
            while not seq.is_finished:
                outputs, stats = engine.step()
                row = dict(step=len(report["records"]) + 1, stats=asdict(stats),
                           cached_after=seq.num_cached_tokens,
                           scheduled_after=seq.num_scheduled_tokens,
                           completion_count=seq.num_completion_tokens,
                           status=seq.status.name, finished_outputs=outputs,
                           block_table=list(seq.block_table))
                report["records"].append(row)
                emit(row)
            engine.forget_request(seq.seq_id)
        elif args.case == "prefix":
            # One complete 256-token prefix block plus a nonempty suffix.
            prompt = [100 + i % 50 for i in range(300)]
            report["prompt_length"] = len(prompt)
            for number in (1, 2):
                ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
                before = engine.scheduler.stats()
                outputs = engine.generate([prompt], sp, use_tqdm=False)
                after = engine.scheduler.stats()
                row = dict(request=number, outputs=outputs,
                           prefill_tokens=after["prefill_tokens"] - before["prefill_tokens"],
                           prefix_hits=after["prefix_cache_hits"] - before["prefix_cache_hits"],
                           scheduler=after)
                report["records"].append(row)
                emit(row)
        else:
            prompt = [100, 101, 102] * 60 if args.case == "ngram" else args.prompt
            report["prompt"] = prompt
            if args.case == "stop":
                # Derive a known stop character from a first greedy run, so the
                # second run exercises stopping independent of model wording.
                first = engine.generate([prompt], sp, use_tqdm=False)[0]
                if not first["text"]:
                    raise RuntimeError("no decodable text from first run; try another prompt")
                sampling["stop"] = first["text"][0]
                report["stop_exercise"] = dict(reference_text=first["text"],
                                               stop=sampling["stop"])
                sp = SamplingParams(**sampling)
                ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
            generator = engine.generate_stream(prompt, sp)
            started = time.perf_counter()
            try:
                for event in generator:
                    row = dict(elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                               **event.to_dict())
                    report["events"].append(row)
                    # An event can represent several committed tokens; capture
                    # the full sequence while the request is still registered.
                    sequence = engine.get_request(event.request_id)
                    if sequence is not None:
                        report["completion_token_ids"] = list(sequence.completion_token_ids)
                    emit(row)
                    if args.case == "cancel":
                        break
            finally:
                generator.close()
        manager.reap_host_loads(blocking=True)
        report["resource_state"] = dict(used_blocks=len(manager.used_block_ids),
                                        reservations=len(getattr(manager, "_reservations", {})),
                                        host_load_leases=len(getattr(manager, "_host_load_leases", {})),
                                        engine_requests=len(engine._requests))
        report["scheduler"] = engine.scheduler.stats()
        report["runtime_metrics"] = engine.runtime_metrics()
        report["cuda_memory"] = cuda_memory_snapshot()
        report["status"] = "DEMO_COMPLETED_NOT_ACCEPTANCE"
        emit(dict(status=report["status"], resource_state=report["resource_state"],
                  report=str(args.output / "report.json")))
    except BaseException as error:
        report.update(status="DEMO_FAILED", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        finally:
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
