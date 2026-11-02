"""单模型端到端回载验收：先生成、强制淘汰，再重新生成比较 token。

这是正确性冒烟，不是吞吐 benchmark。只有 GPU guard 允许才加载模型。
通过填零验证确实依赖 CPU 副本，而不是 GPU 上侥幸保留的旧字节。
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batching-mode", choices=("separate", "mixed"), default="separate")
    parser.add_argument("--async-prefetch", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="NOT_RUN", performance="NOT_MEASURED",
                  batching_mode=args.batching_mode, async_prefetch=args.async_prefetch)
    engine = None
    try:
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
        from nanovllm import LLM, SamplingParams
        engine = LLM(args.model, max_model_len=2048, max_num_seqs=1,
                     max_num_batched_tokens=512, gpu_memory_utilization=0.85,
                     enforce_eager=True, speculative_policy="off",
                     batching_mode=args.batching_mode, prefill_chunk_size=128,
                     max_kvcache_blocks=8, host_kv_async_prefetch=args.async_prefetch,
                     host_kv_cache_bytes=256 * 2**20)
        # 513 tokens 正好覆盖两个完整的 256-token prefix 页。
        tokens = engine.tokenizer.encode("请解释缓存命中和缓存淘汰之间的区别。")
        prompt = (tokens * (513 // len(tokens) + 1))[:513]
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        first = engine.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
        manager = engine.scheduler.block_manager
        assert not manager.used_block_ids
        import torch
        expected, chain = [], -1
        for start in (0, 256):
            chain = manager.compute_hash(prompt[start:start + 256], chain)
            expected.append(engine.model_runner.kv_cache[:, :, manager.hash_to_block_id[chain]].cpu().clone())
        # 人工制造缓存压力：占用并归还所有空闲页，不在模型中计算无关长 prompt。
        # 仅用于测试 cache roundtrip，不计入任何吞吐结果。
        allocated = []
        try:
            for _ in range(len(manager.free_block_ids)):
                allocated.append(manager._allocate_block())
            # Mixed allocation defers D2H until the engine's pre-forward
            # boundary. This test writes KV directly, so honor that boundary.
            manager.flush_host_stores()
            engine.model_runner.kv_cache.zero_()
        finally:
            for bid in allocated:
                manager.blocks[bid].ref_count = 0
                manager._deallocate_block(bid)
        before = manager.host_cache.stats()
        second = engine.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
        manager.reap_host_loads(blocking=True)
        after = manager.host_cache.stats()
        loads = after.get("loads", 0) - before.get("loads", 0)
        clean = (not manager.used_block_ids and not manager._host_load_leases
                 and not manager._reservations and all(b.ref_count == 0 for b in manager.blocks))
        exact, chain = True, -1
        for index, start in enumerate((0, 256)):
            chain = manager.compute_hash(prompt[start:start + 256], chain)
            bid = manager.hash_to_block_id.get(chain)
            exact &= bid is not None and torch.equal(expected[index], engine.model_runner.kv_cache[:, :, bid].cpu())
        report.update(status="PASS" if first == second and loads >= 2 and clean and exact else "REVIEW_REQUIRED",
                      first_token_ids=first, restored_token_ids=second, host_before=before,
                      host_after=after, restored_blocks=loads, clean=clean, exact_prefix_kv=exact,
                      note="different prefill shapes can produce BF16 near-tie differences; inspect before accepting")
    except Exception as error:
        report.update(status="BLOCKED" if type(error).__name__ == "GPUUnavailable" else "FAIL",
                      error=f"{type(error).__name__}: {error}")
    finally:
        if engine is not None:
            engine.exit()
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
