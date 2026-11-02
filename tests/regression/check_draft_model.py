"""Guarded native draft GPU acceptance; NOT_RUN is never a numeric pass."""
import argparse
import json
from pathlib import Path
from time import perf_counter


def same_path_kv_check(engine, history, prefix_length):
    """重复同一执行路径并按逻辑 token 顺序提取 KV，不比较物理页号。"""
    import torch
    manager = engine.scheduler.block_manager
    original = manager.deallocate
    snapshots = []
    def capture(sequence):
        kv = engine.model_runner.kv_cache
        slots = [sequence.block_table[i // engine.config.kvcache_block_size]
                 * engine.config.kvcache_block_size + i % engine.config.kvcache_block_size
                 for i in range(len(sequence))]
        flat = kv.flatten(2, 3)
        indices = torch.tensor(slots, device=kv.device, dtype=torch.int64)
        snapshots.append(flat.index_select(2, indices).cpu())
        return original(sequence)
    manager.deallocate = capture
    try:
        first = common_history_logits(engine, history, prefix_length, 1)
        second = common_history_logits(engine, history, prefix_length, 1)
    finally:
        manager.deallocate = original
    return dict(kv_bitwise_equal=torch.equal(*snapshots),
                kv_max_abs=float((snapshots[0].float()-snapshots[1].float()).abs().max()),
                logits=compare_logits(first, second),
                logical_tokens=len(history)), snapshots


def common_history_logits(engine, history, prefix_length, query_chunk):
    """Teacher-forced target, same real history, Q=1 versus multi-token Q.

    All rows are saved, with the repo's existing .01/.01 strict tolerance. This
    isolates target batch-shape numerics from draft proposal quality.
    """
    import torch
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    manager = engine.scheduler.block_manager
    budget = engine.config.max_num_batched_tokens
    initial = min(prefix_length, budget)
    seq = Sequence(list(history[:initial]), SamplingParams(temperature=0, max_tokens=1))
    rows = []
    end = initial
    try:
        # Long teacher-forced prefixes obey the same per-forward token budget.
        # Both comparisons consume identical prefix chunks before differing in Q.
        while end < prefix_length:
            reservation = manager.reserve_to(seq, end)
            engine.model_runner.run_speculative([seq])
            manager.commit(reservation, end)
            next_end = min(prefix_length, end + budget)
            for token in history[end:next_end]:
                seq.append_token(token)
            end = next_end
        while True:
            reservation = manager.reserve_to(seq, end)
            if seq.num_cached_tokens == end - 1 and query_chunk == 1:
                from nanovllm.utils.context import reset_context
                try:
                    ids, positions = engine.model_runner.prepare_decode([seq])
                    logits = engine.model_runner.run_model(ids, positions, False)
                finally:
                    reset_context()
            else:
                logits = engine.model_runner.run_speculative([seq])
            rows.append(logits[-1:].detach().cpu() if end == prefix_length else logits.detach().cpu())
            manager.commit(reservation, end)
            if end == len(history):
                break
            next_end = min(len(history), end + query_chunk)
            for token in history[end:next_end]:
                seq.append_token(token)
            end = next_end
        return torch.cat(rows).float()
    finally:
        if seq.seq_id in manager._reservations:
            manager.rollback(manager._reservations[seq.seq_id])
        manager.deallocate(seq)


def compare_logits(reference, candidate):
    import torch
    if reference.shape != candidate.shape:
        raise ValueError("fixed-history logits shapes differ")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    return dict(strict_allclose=finite and bool(torch.allclose(reference, candidate, atol=.01, rtol=.01)),
                finite=finite, atol=.01, rtol=.01,
                max_abs=float((reference - candidate).abs().max()),
                argmax_equal=int((reference.argmax(-1) == candidate.argmax(-1)).sum()),
                positions=len(reference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-backend", choices=("torch", "triton_stable"), default="torch")
    parser.add_argument("--rmsnorm-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--draft-execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--verify-graph", action="store_true")
    parser.add_argument("--verify-metadata", action="store_true")
    parser.add_argument("--target-graphs", action="store_true")
    parser.add_argument("--numerical-backend", choices=("default", "consistent"), default="default")
    parser.add_argument("--extended", action="store_true", help="Add held-out text, page-boundary histories and lifecycle checks")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-kv-blocks", type=int, default=12)
    parser.add_argument("--kv-check", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="NOT_RUN", performance="NOT_MEASURED")
    engine = None
    try:
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        report["gpu"] = ensure_gpu1_available(evidence_path=args.output / "gpu_guard.jsonl")
        import torch
        from nanovllm import LLM, SamplingParams
        startup_begin = perf_counter()
        engine = LLM(args.model, draft_model=args.draft_model,
                     batching_mode="mixed", speculative_policy="fixed", max_num_seqs=1,
                     max_model_len=args.max_model_len, max_num_batched_tokens=256, prefill_chunk_size=128,
                     max_kvcache_blocks=args.max_kv_blocks, linear_backend=args.linear_backend,
                     rmsnorm_backend=args.rmsnorm_backend,
                     numerical_backend=args.numerical_backend,
                     draft_execution=args.draft_execution, draft_verify_graph=args.verify_graph,
                     draft_verify_metadata=args.verify_metadata,
                     enable_prefix_caching=False, gpu_memory_utilization=args.gpu_memory_utilization,
                     enforce_eager=not (args.target_graphs or args.verify_graph or args.draft_execution == "graph"))
        torch.cuda.synchronize()
        report['startup_seconds'] = perf_counter() - startup_begin
        report['startup_memory'] = dict(allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
            target_parameter_bytes=sum(p.numel()*p.element_size() for p in engine.model_runner.model.parameters()),
            target_kv_bytes=engine.model_runner.kv_cache.numel()*engine.model_runner.kv_cache.element_size(),
            draft=engine.model_runner.draft_runner.stats(),
            target_verification_graph=engine.model_runner.native_verification_graphs.stats()
                if engine.model_runner.native_verification_graphs is not None else {})
        texts = ("请解释大模型推理中的缓存与调度。", "Explain memory bandwidth in autoregressive decoding.",
                 "def binary_search(values, target):\n", "red blue green. " * 48)
        if args.extended:
            texts += ("请比较缓存命中与重新计算的成本。", "Explain why a high draft acceptance rate may not improve latency.",
                      "def merge_sorted_lists(left, right):\n", "river stone willow. " * 100)
        rows = []
        for index, text in enumerate(texts):
            prompt = engine.tokenizer.encode(text)
            ordinary = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=32,
                ignore_eos=True, speculative_method="off"), use_tqdm=False)[0]["token_ids"]
            for length in (1, 2, 4, 8):
                result = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=32,
                    ignore_eos=True, speculative_method="draft_model", speculative_max_draft_tokens=length),
                    use_tqdm=False)[0]["token_ids"]
                rows.append(dict(task=index, draft_length=length, baseline=ordinary, candidate=result,
                                 equal=ordinary == result, speculative=engine.speculative_stats()))
        report["cases"] = rows  # Keep completed generation evidence if a later check fails.
        # Deliberately crosses a 256-token physical page boundary.
        material = engine.tokenizer.encode("Cache scheduling and speculative verification. ")
        history = (material * (264 // len(material) + 1))[:264]
        if args.kv_check:
            report['same_path_kv'], snapshots = same_path_kv_check(engine, history, 255)
            torch.save(dict(history=history, kv=snapshots), args.output/'same_path_kv.pt')
            del snapshots
        reference = common_history_logits(engine, history, 255, 1)
        candidate = common_history_logits(engine, history, 255, 4)
        numeric = compare_logits(reference, candidate)
        torch.save(dict(history=history, reference=reference, candidate=candidate), args.output / "fixed_history.pt")
        extra_numeric = []
        if args.extended:
            for prefix, chunk in ((1, 2), (127, 3), (255, 8), (256, 2), (511, 4), (1019, 8)):
                history = (material * ((prefix + 17) // len(material) + 1))[:prefix + 17]
                reference = common_history_logits(engine, history, prefix, 1)
                candidate = common_history_logits(engine, history, prefix, chunk)
                extra_numeric.append(dict(prefix=prefix, chunk=chunk, **compare_logits(reference, candidate)))
            # Real GPU cancellation/reuse: cached speculative tails must not cross requests.
            prompt = engine.tokenizer.encode("A short explanation of KV cache: ")
            params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True,
                                    speculative_method="draft_model", speculative_max_draft_tokens=4)
            seq = engine.add_request(prompt, params)
            for _ in range(3):
                engine.step()
            engine.cancel_request(seq.seq_id)
            assert engine.model_runner.draft_runner.request_id is None
            assert not engine.model_runner.draft_runner.proposer.cached_tokens
            ordinary = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=16,
                ignore_eos=True, speculative_method="off"), use_tqdm=False)[0]["token_ids"]
            resumed = engine.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
            rows.append(dict(task="after_cancel", draft_length=4, baseline=ordinary, candidate=resumed, equal=ordinary == resumed))
            original_eos = engine.scheduler.eos
            try:
                # Force an actually predicted token to exercise native EOS cleanup,
                # without waiting for this short prompt to naturally produce EOS.
                engine.scheduler.eos = ordinary[2]
                for name, options in (("forced_eos", dict(max_tokens=16, ignore_eos=False)),
                                      ("length_one", dict(max_tokens=1, ignore_eos=True))):
                    expected = engine.generate([prompt], SamplingParams(temperature=0,
                        speculative_method="off", **options), use_tqdm=False)[0]["token_ids"]
                    actual = engine.generate([prompt], SamplingParams(temperature=0,
                        speculative_method="draft_model", speculative_max_draft_tokens=4,
                        **options), use_tqdm=False)[0]["token_ids"]
                    rows.append(dict(task=name, draft_length=4, baseline=expected, candidate=actual, equal=expected == actual))
                    assert not engine.model_runner.draft_runner.proposer.cached_tokens
                    assert engine.model_runner.draft_runner.request_id is None
            finally:
                engine.scheduler.eos = original_eos
        manager = engine.scheduler.block_manager
        clean = not manager.used_block_ids and not manager._reservations
        report.update(status="PASS_SCOPED" if all(r["equal"] for r in rows)
                      and numeric["strict_allclose"] and all(r["strict_allclose"] for r in extra_numeric)
                      and report.get('same_path_kv', {}).get('kv_bitwise_equal', True)
                      and clean else "REVIEW_REQUIRED",
                      cases=rows, numerical=numeric, extended_numerical=extra_numeric, clean=clean, runtime=engine.runtime_metrics(),
                      scope="Qwen3 TP1 greedy, single Python request; no stochastic or batched draft claim")
    except Exception as error:
        report.update(status="NOT_RUN_GPU_UNAVAILABLE" if type(error).__name__ == "GPUUnavailable" else "FAIL",
                      error=f"{type(error).__name__}: {error}")
    finally:
        if engine is not None:
            engine.exit()
        from benchmarks.cache_draft_experiment import sources
        report["source_sha256"] = sources()
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "source_sha256", "runtime")}, ensure_ascii=False))
    return 0 if report["status"] == "PASS_SCOPED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
