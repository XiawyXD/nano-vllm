"""分页 Attention 单一验收入口；所有 GPU 模式先检查用户指定 GPU 占用。

cpu 不依赖 FlashInfer；check 对同一 Q/K/V 比较 FP32 oracle 和两种后端；
micro 区分 kernel 与含 plan 的开销；model 存固定历史 logits 和实际生成计时。
输出目录必须不存在，失败也保存 traceback，不覆盖之前证据。
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
from unittest.mock import patch

import torch
from nanovllm.layers.flashinfer_attention import FlashInferDecode, build_paged_metadata


def cpu_check():
    from nanovllm.config import Config
    from nanovllm.utils.context import get_context, set_context, reset_context
    assert build_paged_metadata([[7, 2, 9], [5]], [257, 10], 256, 10, 3) == (
        [0, 2, 3, 4], [7, 2, 5, 0], [1, 10, 1])
    for length in (1, 255, 256, 257, 512):
        ptr, ids, last = build_paged_metadata([[2, 1]], [length], 256, 3)
        assert last == [(length-1) % 256+1] and ptr[-1] == (length+255)//256
    for tables, lengths, pages, batch in (([[0]], [257], 2, None),
            ([[-1]], [1], 2, None), ([[2]], [1], 2, None), ([[0]], [0], 2, None),
            ([[0]], [1, 2], 2, None), ([], [], 2, None), ([[0]], [1], 2, 0)):
        try:
            build_paged_metadata(tables, lengths, 256, pages, batch)
        except ValueError:
            pass
        else:
            raise AssertionError((tables, lengths))
    marker = object()
    set_context(False, flashinfer_wrapper=marker, flashinfer_metadata=([[1]], [1]))
    assert get_context().flashinfer_wrapper is marker
    reset_context()
    assert get_context().flashinfer_wrapper is None and get_context().flashinfer_metadata is None
    for changes in (dict(tensor_parallel_size=2), dict(batching_mode="mixed"), dict(kv_cache_dtype="fp8")):
        with patch("nanovllm.config.os.path.isdir", return_value=True):
            try:
                Config("unused", decode_attention_backend="flashinfer", **changes)
            except ValueError as e:
                assert "FlashInfer decode requires" in str(e)
            else:
                raise AssertionError(changes)
    return dict(status="PASS", checks=17)


def reference(q, k, v, tables, lengths):
    """慢速独立 FP32 oracle，显式还原逻辑 KV；仅用于验证，绝不用于计时。"""
    rows = []
    for i, (table, length) in enumerate(zip(tables, lengths)):
        key = k[table].flatten(0, 1)[:length].float().repeat_interleave(q.shape[1]//k.shape[2], 1)
        val = v[table].flatten(0, 1)[:length].float().repeat_interleave(q.shape[1]//v.shape[2], 1)
        scores = torch.einsum("hd,thd->ht", q[i].float(), key) * q.shape[-1]**-.5
        rows.append(torch.einsum("ht,thd->hd", scores.softmax(-1), val))
    return torch.stack(rows)


def check():
    from flash_attn import flash_attn_with_kvcache
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for heads, kv_heads in ((8, 8), (16, 8), (16, 2)):
        manager = FlashInferDecode(heads, kv_heads, 128, 256, 1024, torch.bfloat16, (4,))
        k = torch.randn(12, 256, kv_heads, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        # 物理页非顺序且多请求共享页；同一已捕获 Graph 连续重新规划。
        tables = [[7, 2, 9, 3], [7, 4, 5, 6], [11, 10, 1, 8]]
        q = torch.randn(4, heads, 128, device="cuda", dtype=torch.bfloat16)
        wrapper = manager.plan(tables, [1, 255, 256], 12, 4)
        out = wrapper.run(q, (k, v))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = wrapper.run(q, (k, v))
        for step, lengths in enumerate(([1, 255, 256], [257, 512, 513], [1024, 17, 300], [2, 256, 1])):
            active_tables = tables if step % 2 == 0 else tables[::-1]
            q.normal_()
            eager = manager.plan(active_tables, lengths, 12).run(q[:3], (k, v))
            manager.plan(active_tables, lengths, 12, 4)
            graph.replay()
            expected = reference(q[:3], k, v, active_tables, lengths)
            fa = flash_attn_with_kvcache(q[:3, None], k, v,
                cache_seqlens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
                block_table=torch.tensor(active_tables, dtype=torch.int32, device="cuda"), causal=True).squeeze(1)
            for name, actual in (("eager", eager), ("graph", out[:3]), ("flash", fa)):
                torch.testing.assert_close(actual.float(), expected, atol=.01, rtol=.01)
                rows.append(dict(heads=heads, kv_heads=kv_heads, step=step, backend=name,
                                 max_abs=(actual.float()-expected).abs().max().item()))
            torch.testing.assert_close(out[:3], eager, atol=0, rtol=0)
        del graph, wrapper, manager, out
    return dict(status="PASS", comparisons=rows, tolerance=dict(atol=.01, rtol=.01),
                graph_eager="bitwise equal", scope="BF16/head_dim128/page256/GQA+MHA/padding/replan")


def wall_ms(fn, iterations=40):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-begin)*1000/iterations


def micro():
    from flash_attn import flash_attn_with_kvcache
    rows = []
    for batch in (1, 8, 32):
        manager = FlashInferDecode(16, 8, 128, 256, 2048, torch.bfloat16)
        for length in (256, 1024, 2048):
            pages = (length+255)//256
            tables = [list(range(i*pages, (i+1)*pages)) for i in range(batch)]
            lengths = [length]*batch
            k = torch.randn(batch*pages, 256, 8, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            q = torch.randn(batch, 16, 128, device="cuda", dtype=torch.bfloat16)
            ptr = torch.tensor(tables, dtype=torch.int32, device="cuda")
            lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
            w = manager.plan(tables, lengths, batch*pages)
            operations = {
                "flash_run": lambda: flash_attn_with_kvcache(q[:, None], k, v, cache_seqlens=lens, block_table=ptr, causal=True),
                "flashinfer_run": lambda: w.run(q, (k, v)),
                "flashinfer_plan_run": lambda: manager.plan(tables, lengths, batch*pages).run(q, (k, v)),
            }
            for repeat in range(3):
                names = list(operations) if repeat % 2 == 0 else list(operations)[::-1]
                for name in names:
                    row = dict(batch=batch, context=length, repeat=repeat, path=name, wall_ms=wall_ms(operations[name]))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return dict(rows=rows, scope="single layer, host launch included; plan shared across layers in model")


def model(args, output):
    from nanovllm import LLM, SamplingParams
    llm = None
    try:
        llm = LLM(args.model, decode_attention_backend=args.backend,
            enforce_eager=not args.graph, max_num_seqs=max(4, args.batch),
            max_model_len=1024, max_num_batched_tokens=512, prefill_chunk_size=128,
            gpu_memory_utilization=.4, max_kvcache_blocks=max(32, args.batch*3),
            enable_prefix_caching=False, sampler_backend="torch", speculative_policy="off")
        runner = llm.model_runner
        prompts = [f"Explain GPU inference for examples.example {i}. " + "KV cache stores previous keys and values. "*40 for i in range(args.batch)]
        # 前三条固定到跨页边界附近；ragged batch 同时测试 Graph 补齐。
        ids = [llm.tokenizer.encode(p)[:250+(i % 3)*6] for i, p in enumerate(prompts)]
        assert list(map(len, ids)) == [250+(i % 3)*6 for i in range(args.batch)]
        tensors = []
        original = runner.sampler.forward
        def teacher(logits, seqs):
            tensors.append(logits.detach().cpu())
            # 固定同一历史，避免第一个不同 token 导致之后不可比较。
            return torch.tensor([100+(s.num_completion_tokens % 17) for s in seqs], device=logits.device)
        runner.sampler.forward = teacher
        llm.generate(ids, SamplingParams(temperature=0, max_tokens=24, ignore_eos=True), use_tqdm=False)
        runner.sampler.forward = original
        torch.save(tensors, output / "teacher_logits.pt")
        params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
        golden = llm.generate(ids, params, use_tqdm=False)
        rows = []
        for repeat in range(3):
            from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
            ensure_gpu1_available(evidence_path=output / "gpu.jsonl")
            torch.cuda.synchronize()
            started = time.perf_counter()
            results = llm.generate(ids, params, use_tqdm=False)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            assert [r["token_ids"] for r in results] == [r["token_ids"] for r in golden]
            rows.append(dict(repeat=repeat, seconds=seconds, tokens_per_second=args.batch*64/seconds))
        assert not llm.scheduler.block_manager.used_block_ids and not llm._requests
        if runner.flashinfer_decode:
            plans = runner.flashinfer_decode.stats["plans"]
            expected = sum(value for key, value in runner.graph_counters.items()
                           if key == "decode_replays" or (key.startswith("decode_fallback_")
                               and key != "decode_fallback_prefill")) + len(getattr(runner, "graphs", {}))
            assert plans == expected, (plans, expected, "plan must execute once per step, not per layer")
        torch.cuda.synchronize()
        return dict(status="PASS", rows=rows, median_tokens_per_second=statistics.median(r["tokens_per_second"] for r in rows),
            token_ids=[r["token_ids"] for r in golden], prompt_lengths=list(map(len, ids)),
            graph_counters=dict(runner.graph_counters),
            flashinfer_counters=dict(runner.flashinfer_decode.stats) if runner.flashinfer_decode else {},
            config=llm.config.runtime_summary(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            note="teacher logits need external cross-backend comparison; no quality claim from throughput")
    finally:
        if llm is not None:
            llm.exit()


def compare_runs(args):
    """固定预设容差，不因 BF16 差异而放宽；不同历史不能冒充数值对照。"""
    left, right = Path(args.left), Path(args.right)
    a = json.loads((left / "result.json").read_text())
    b = json.loads((right / "result.json").read_text())
    assert a["prompt_lengths"] == b["prompt_lengths"]
    aa = torch.load(left / "teacher_logits.pt", map_location="cpu", weights_only=True)
    bb = torch.load(right / "teacher_logits.pt", map_location="cpu", weights_only=True)
    assert len(aa) == len(bb)
    rows = []
    for i, (x, y) in enumerate(zip(aa, bb)):
        assert x.shape == y.shape
        delta = (x.float()-y.float()).abs()
        passed = torch.isclose(x.float(), y.float(), atol=.01, rtol=.01)
        rows.append(dict(step=i, allclose=bool(passed.all()), max_abs=delta.max().item(),
                         mismatch_fraction=(~passed).float().mean().item(),
                         argmax_equal=bool(torch.equal(x.argmax(-1), y.argmax(-1)))))
    return dict(status="PASS" if all(r["allclose"] for r in rows) else "REVIEW_REQUIRED",
                comparisons=rows, token_ids_equal=a["token_ids"] == b["token_ids"],
                throughput_ratio=b["median_tokens_per_second"]/a["median_tokens_per_second"],
                left=str(left), right=str(right), tolerance=dict(atol=.01, rtol=.01))


def audit(args):
    """用真实 Qwen 层的同一 Q/K/V 排除页表错误；诊断有 hook，不计时。"""
    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.attention import Attention
    from nanovllm.utils.context import get_context
    from flash_attn import flash_attn_with_kvcache
    engine, hooks, rows = None, [], []
    seen = set()
    try:
        engine = LLM(args.model, decode_attention_backend="flashinfer", enforce_eager=True,
                     max_num_seqs=3, max_model_len=1024, max_num_batched_tokens=512,
                     max_kvcache_blocks=32, gpu_memory_utilization=.4,
                     enable_prefix_caching=False, speculative_policy="off")
        def hook(name):
            def inspect(module, inputs, actual):
                context = get_context()
                if context.is_prefill or name in seen:
                    return
                seen.add(name)
                q = inputs[0]
                tables, lengths = context.flashinfer_metadata
                expected = reference(q, module.k_cache, module.v_cache, tables, lengths)
                fa = flash_attn_with_kvcache(q[:, None], module.k_cache, module.v_cache,
                    cache_seqlens=context.context_lens, block_table=context.block_tables,
                    softmax_scale=module.scale, causal=True).squeeze(1)
                row = dict(layer=name)
                for backend, value in (("flashinfer", actual), ("flash", fa)):
                    row[backend] = dict(max_abs=(value.float()-expected).abs().max().item(),
                        allclose=bool(torch.allclose(value.float(), expected, atol=.01, rtol=.01)))
                row["between_backends_max_abs"] = (actual.float()-fa.float()).abs().max().item()
                rows.append(row)
            return inspect
        for name, module in engine.model_runner.model.named_modules():
            if isinstance(module, Attention):
                hooks.append(module.register_forward_hook(hook(name)))
        tokens = engine.tokenizer.encode("KV cache stores previous keys and values. "*45)
        engine.generate([tokens[:250], tokens[:256], tokens[:262]],
                        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True), use_tqdm=False)
        assert len(rows) == engine.config.hf_config.num_hidden_layers
        return dict(status="PASS" if all(r[b]["allclose"] for r in rows for b in ("flashinfer", "flash"))
                    else "REVIEW_REQUIRED", layers=rows, tolerance=dict(atol=.01, rtol=.01),
                    scope="same real Q/K/V, first decode step, all layers; not end-to-end logits equivalence")
    finally:
        for handle in hooks:
            handle.remove()
        if engine is not None:
            engine.exit()


def compatibility(args):
    """Graph + CPU prefix 回载 + streaming 取消；不混入性能数据。"""
    from nanovllm import LLM, SamplingParams
    engine = None
    try:
        engine = LLM(args.model, decode_attention_backend="flashinfer", enforce_eager=False,
                     max_num_seqs=4, max_model_len=1024, max_num_batched_tokens=512,
                     max_kvcache_blocks=32, gpu_memory_utilization=.4,
                     host_kv_cache_bytes=256*2**20, speculative_policy="off")
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        tokens = engine.tokenizer.encode("请解释缓存命中和缓存淘汰之间的区别。")
        prompt = (tokens * (513 // len(tokens)+1))[:513]
        first = engine.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
        manager = engine.scheduler.block_manager
        assert not manager.used_block_ids
        allocated = []
        try:
            for _ in range(len(manager.free_block_ids)):
                allocated.append(manager._allocate_block())
            # 主动清除 GPU 旧副本；若未真实回载，结果便无法依赖旧字节碰巧通过。
            engine.model_runner.kv_cache.zero_()
        finally:
            for bid in allocated:
                manager.blocks[bid].ref_count = 0
                manager._deallocate_block(bid)
        before = manager.host_cache.stats()
        second = engine.generate([prompt], params, use_tqdm=False)[0]["token_ids"]
        after = manager.host_cache.stats()
        assert after.get("loads", 0)-before.get("loads", 0) >= 2
        assert first == second
        stream = engine.generate_stream(prompt, SamplingParams(temperature=0, max_tokens=32, ignore_eos=True))
        events = [str(next(stream)) for _ in range(4)]  # 包含真实 decode，而不只测 prefill 的首 token。
        stream.close()
        assert not manager.used_block_ids and not engine._requests
        stats = engine.runtime_metrics()
        assert stats["flashinfer_attention"]["graph_plans"] > 0
        return dict(status="PASS", restored_token_ids_equal=True, host_before=before, host_after=after,
                    stream_events=events, cancel_clean=True, runtime=stats,
                    scope="in-process event streaming; not HTTP/SSE performance or concurrent load test")
    finally:
        if engine is not None:
            engine.exit()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("cpu", "check", "micro", "model", "compare", "audit", "compat"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-1.7B")
    ap.add_argument("--backend", choices=("flash", "flashinfer"), default="flashinfer")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--left")
    ap.add_argument("--right")
    args = ap.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(args=vars(args), status="RUNNING")
    report["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path("nanovllm/layers/flashinfer_attention.py"), Path("nanovllm/engine/model_runner.py"),
        Path("nanovllm/layers/attention.py"), Path("nanovllm/utils/context.py"), Path("nanovllm/config.py"))}
    try:
        if args.mode not in ("cpu", "compare"):
            from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
            ensure_gpu1_available(evidence_path=output / "gpu.jsonl")
            report["torch"] = torch.__version__
            report["gpu"] = torch.cuda.get_device_name()
        with torch.inference_mode():
            report.update(compatibility(args) if args.mode == "compat" else audit(args) if args.mode == "audit" else compare_runs(args) if args.mode == "compare" else cpu_check() if args.mode == "cpu" else check() if args.mode == "check"
                          else micro() if args.mode == "micro" else model(args, output))
        if report["status"] == "RUNNING":
            report["status"] = "PASS"
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
