"""FlashInfer采样：独立GPU正确性、微基准与真实模型验收，原始结果不覆盖。

check验证过滤后的概率与参考分布；随机token不要求跨后端逐个相等。
micro计入metadata/过滤/回退/D2H开销，不把单kernel速度冒充整个采样器速度。
model保持同一模型/Graph/请求规模，交错顺序比较完整生成；不测HTTP。
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from nanovllm.layers.sampler import Sampler
from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available


def seq(**kwargs):
    fields = dict(temperature=0.8, top_k=20, top_p=0.9, repetition_penalty=1.,
                  seed=None, token_ids=[1, 2], num_completion_tokens=0)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def reference_probs(row, s):
    """独立写出的旧采样分布oracle；不调用被测FlashInfer实现。"""
    x = row.float().clone() / s.temperature
    if 0 < s.top_k < x.numel():
        x[x < x.topk(s.top_k).values[-1]] = -float("inf")
    values, indices = x.sort(descending=True)
    remove = values.softmax(0).cumsum(0) > s.top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    if s.top_p < 1:
        x[indices[remove]] = -float("inf")
    return x.softmax(0)


def check():
    fi, baseline = Sampler("flashinfer", flashinfer_min_batch_size=1), Sampler("torch")
    passed = []
    torch.manual_seed(1403)
    # FP32/BF16、非连续输入、不同k/p和词表大小；所有非回退行比较支持集与概率。
    checked = fallback = 0
    for dtype in (torch.float32, torch.bfloat16):
        for vocab in (7, 257, 4097, 151936):
            x = torch.randn(24, vocab*2, device="cuda", dtype=dtype)[:, ::2]
            params = [seq(top_k=(0, 1, 5, vocab+10)[i%4], top_p=(1., .9, .5)[i%3]) for i in range(24)]
            before = x.clone()
            probs, bad = fi.flashinfer.filtered_probs(x, params)
            for i, s in enumerate(params):
                if bad[i].item():
                    repaired = fi.flashinfer.reference_filter_batch(x[i:i+1], [s])[0]
                    torch.testing.assert_close(repaired, reference_probs(x[i], s), atol=2e-6, rtol=2e-5)
                    fallback += 1
                    continue
                expected = reference_probs(x[i], s)
                assert torch.equal(probs[i] > 0, expected > 0), (dtype, vocab, i, "support")
                torch.testing.assert_close(probs[i], expected, atol=2e-6, rtol=2e-5)
                checked += 1
            tokens = fi(x, params)
            for i, s in enumerate(params):
                assert reference_probs(x[i], s)[tokens[i]] > 0
            assert torch.equal(x, before), "input mutated"
    passed.append(dict(name="filter_oracle_and_input_immutability", checked=checked, boundary_fallback=fallback))
    # top-p边界并列、精确累计阈值都必须保持参考支持集，不静默扩宽分布。
    x = torch.zeros(64, 8, device="cuda")
    params = [seq(top_k=0, top_p=.25) for _ in range(64)]
    tokens = fi(x, params)
    allowed = reference_probs(x[0], params[0]) > 0
    assert allowed[tokens].all() and fi.flashinfer.stats["reference_filter_rows"] >= 64
    passed.append(dict(name="tied_top_p_reference_repair", count=64))
    # seeded请求重排、批量变动后仍与原Torch完全相等；负seed与64-bit上界也测试。
    x = torch.randn(5, 257, device="cuda")
    seeds = [17, -3, 2**64-100, 0, 100]
    params = [seq(seed=value, num_completion_tokens=3, repetition_penalty=1.1) for value in seeds]
    expected = baseline(x, params)
    assert torch.equal(fi(x, params), expected)
    order = [3, 0, 4, 1, 2]
    assert torch.equal(fi(x[order], [params[i] for i in order]), expected[order])
    for i in range(5):
        assert fi(x[i:i+1], params[i:i+1]).item() == expected[i].item()
    passed.append(dict(name="seed_reorder_and_subset", count=5))
    # 混合greedy/seed/penalty/fast行，确保输出被scatter回正确的请求位置。
    params = [seq(temperature=0), seq(seed=3), seq(repetition_penalty=1.3), seq(top_k=1), seq()]
    actual = fi(x, params)
    assert actual[0] == x[0].argmax()
    assert actual[1] == baseline(x[1:2], params[1:2])[0]
    assert actual[3] == x[3].argmax()
    passed.append(dict(name="mixed_parameters_and_order"))
    with torch.device("cuda"):
        fi(x, [seq(top_k=0, top_p=1)] * len(x))
    passed.append(dict(name="cuda_default_factory_rng_state"))
    # 分布检查：固定已知四类分布，N=20000，误差界在实验前固定。
    p = torch.tensor([.05, .15, .3, .5], device="cuda")
    x = p.log().repeat(20000, 1)
    params = [seq(temperature=1, top_k=0, top_p=1)] * len(x)
    samples = fi(x, params)
    empirical = torch.bincount(samples, minlength=4).float() / len(samples)
    error = (empirical-p).abs().max().item()
    assert error < .015, (empirical, error)
    # 重设全局RNG时同batch可复现，连续调用不应因内部新建generator而重复整批。
    torch.cuda.manual_seed(321)
    a = fi(x[:1024], params[:1024])
    b = fi(x[:1024], params[:1024])
    torch.cuda.manual_seed(321)
    c = fi(x[:1024], params[:1024])
    assert torch.equal(a, c) and not torch.equal(a, b)
    passed.append(dict(name="empirical_distribution_and_rng", n=20000, max_abs_error=error,
                       empirical=empirical.tolist()))
    return dict(tests=passed, counters=dict(fi.flashinfer.stats))


def micro(output, args_min_batch):
    rows = []
    torch.manual_seed(321)
    for batch in (1, 8, 32):
        ensure_gpu1_available(evidence_path=output / "gpu.jsonl")
        for case, params in (
            ("unseeded", [seq(top_k=50, top_p=.9)] * batch),
            ("seeded_reference", [seq(seed=50+i) for i in range(batch)])):
            x = torch.randn(batch, 151936, device="cuda", dtype=torch.bfloat16)
            samplers = {name: Sampler(name, flashinfer_min_batch_size=args_min_batch) for name in ("torch", "flashinfer")}
            for sampler in samplers.values():
                for _ in range(5):
                    sampler(x, params)
            for repeat in range(3):
                for name in (("torch", "flashinfer") if repeat % 2 == 0 else ("flashinfer", "torch")):
                    sampler = samplers[name]
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    for _ in range(30):
                        sampler(x, params)
                    torch.cuda.synchronize()
                    row = dict(batch=batch, case=case, repeat=repeat, backend=name,
                               wall_ms=(time.perf_counter()-begin)*1000/30)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            rows.append(dict(batch=batch, case=case, counters=dict(samplers["flashinfer"].flashinfer.stats)))
    return dict(rows=rows, scope="full sampler wall time including sync/metadata; 3 same-process repeats, not model throughput")


def model(args):
    from nanovllm import LLM, SamplingParams
    engine = None
    rows = []
    try:
        engine = LLM(args.model, max_model_len=1024, max_num_seqs=args.batch,
                     max_num_batched_tokens=512, prefill_chunk_size=128,
                     max_prefill_steps=1, gpu_memory_utilization=.4,
                     max_kvcache_blocks=max(32, args.batch*2), enforce_eager=False,
                     enable_prefix_caching=False, speculative_policy="off", sampler_backend="flashinfer",
                     flashinfer_min_batch_size=args.min_batch)
        # 自然文本prompt固定；两后端请求数/输出长度/模型/Graph均相同。
        prompts = [f"Explain GPU inference concept {i}: " + "Caching and scheduling improve inference. "*12 for i in range(args.batch)]
        if args.validation:
            # 阈值确定后使用另一组自然文本，不根据其结果继续调min_batch。
            prompts = [f"Describe distributed systems topic {i}: " + "Communication overlaps computation when dependencies permit. "*12 for i in range(args.batch)]
        params = SamplingParams(temperature=.8, top_k=50, top_p=.9, max_tokens=64, ignore_eos=True)
        samplers = {name: Sampler(name, flashinfer_min_batch_size=args.min_batch) for name in ("torch", "flashinfer")}
        for phase in ("warmup", "steady0", "steady1", "steady2"):
            order = ("flashinfer", "torch") if phase == "steady1" else ("torch", "flashinfer")
            for name in order:
                ensure_gpu1_available(evidence_path=args.output / "gpu.jsonl")
                engine.model_runner.sampler = samplers[name]
                torch.cuda.synchronize()
                begin = time.perf_counter()
                result = engine.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-begin
                ids = [item["token_ids"] for item in result]
                assert len(ids) == args.batch and all(len(row) == 64 for row in ids)
                clean = not engine.scheduler.block_manager.used_block_ids and not engine._requests
                assert clean
                row = dict(phase=phase, backend=name, seconds=seconds, tokens_s=args.batch*64/seconds,
                           output_token_ids=ids, clean=clean,
                           counters=dict(samplers[name].flashinfer.stats) if name == "flashinfer" else {})
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k != "output_token_ids"}), flush=True)
                (args.output / "trials.json").write_text(json.dumps(rows, indent=2))
        # 原参考seed语义在真实模型中也逐token对照，不能用随机无seed结果要求相同。
        params.seed, params.max_tokens = 13, 16
        seeded = []
        for name in ("torch", "flashinfer"):
            engine.model_runner.sampler = samplers[name]
            seeded.append([r["token_ids"] for r in engine.generate(prompts[:2], params, use_tqdm=False)])
        assert seeded[0] == seeded[1]
        # SSE所用engine事件流和取消，验证不是只测离线generate。
        stream = engine.generate_stream(prompts[0], SamplingParams(top_k=20, top_p=.9, max_tokens=12))
        first = next(stream)
        stream.close()
        assert not engine.scheduler.block_manager.used_block_ids
        # 临时localhost服务，完成一个真实随机采样SSE请求；结束后关闭，不对外发布。
        import threading
        import urllib.request
        from nanovllm.server import create_http_server
        server, manager = create_http_server(engine, "127.0.0.1", 0, model_name=args.model)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = dict(model=args.model, prompt="Explain KV cache briefly.", stream=True,
                           temperature=.8, top_k=50, top_p=.9, max_tokens=16)
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/completions",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                sse = response.read().decode()
            assert "[DONE]" in sse and "choices" in sse
            (args.output / "http_sse.txt").write_text(sse)
        finally:
            server.shutdown()
            manager.close()
            server.server_close()
            thread.join(5)
        assert not engine.scheduler.block_manager.used_block_ids
        return dict(rows=rows, seeded_equal=True, cancel_clean=True, http_sse_pass=True,
                    first_event=str(first), effective_config=engine.config.runtime_summary(),
                    medians={name: statistics.median(r["tokens_s"] for r in rows if r["backend"] == name and r["phase"].startswith("steady")) for name in samplers},
                    counters=dict(samplers["flashinfer"].flashinfer.stats),
                    scope="offline full generation, ordinary model Graph; per-trial sampler override named in rows; sampler outside Graph; HTTP smoke only, no HTTP latency claim")
    finally:
        if engine is not None:
            engine.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "micro", "model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-1.7B")
    parser.add_argument("--batch", type=int, choices=(8, 32), default=32)
    parser.add_argument("--min-batch", type=int, default=32)
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", mode=args.mode, model=args.model, batch=args.batch, min_batch=args.min_batch, validation=args.validation,
                  versions={p: importlib.metadata.version(p) for p in ("torch", "triton", "flashinfer-python")},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path("nanovllm/layers/sampler.py"), Path("nanovllm/layers/flashinfer_sampler.py"), Path(__file__)]})
    try:
        report["gpu"] = ensure_gpu1_available(evidence_path=args.output / "gpu.jsonl")
        report["result"] = check() if args.mode == "check" else micro(args.output, args.min_batch) if args.mode == "micro" else model(args)
        report["status"] = "PASS"
    except Exception as error:
        import traceback
        report.update(status="FAIL", error=str(error), traceback=traceback.format_exc())
    finally:
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "result"}, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
