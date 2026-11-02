"""同进程交错比较验证eager/graph；冷捕获单独计时，不混进稳态收益。"""
import argparse
import json
from pathlib import Path
import time
import torch
from nanovllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-1.7B")
    args = ap.parse_args()
    llm = LLM(args.model, max_model_len=512, max_num_batched_tokens=512,
              max_num_seqs=16, gpu_memory_utilization=.5, enable_prefix_caching=False)
    runner = llm.model_runner
    prompts = [[1, 2, 3]*64 for _ in range(8)]
    params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True, speculative_ngram_size=3)
    result = {"runs": [], "correctness": [], "model": args.model}

    def run(mode, inputs=prompts, sp=params):
        runner.enable_verification_graph = mode == "graph"
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(inputs, sp, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        for rid in list(llm._requests):
            llm.forget_request(rid)
        assert not llm.scheduler.block_manager.used_block_ids
        return elapsed, [x["token_ids"] for x in outputs]

    try:
        # 普通decode不涉及本轮新代码；作为同一机器同一时段的参考。
        plain = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
        run("eager")
        result["cold_graph_seconds"], cold = run("graph")
        run("graph")
        run("ordinary", sp=plain)
        reference = run("eager")[1]
        assert cold == reference
        for repeat in range(5):
            modes = ("ordinary", "eager", "graph") if repeat%2 == 0 else ("graph", "eager", "ordinary")
            for mode in modes:
                elapsed, ids = run(mode, sp=plain if mode == "ordinary" else params)
                if mode != "ordinary":
                    assert ids == reference, (mode, repeat)
                result["runs"].append(dict(mode=mode, repeat=repeat, seconds=elapsed, tokens_s=512/elapsed))
        # 不同历史、query长度和请求数量；含跨256-token物理页边界。
        for batch, length, draft in ((1,255,2), (2,260,4), (3,72,3), (8,192,4)):
            inputs = [([1,2,3]*(length//3+1))[:length-i] for i in range(batch)]
            sp = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True,
                                speculative_ngram_size=3, speculative_max_draft_tokens=draft)
            expected = run("eager", inputs, sp)[1]
            # 单独清空有限缓存，保证这些形状能测到真实capture而不全是fallback。
            runner.verification_graphs.entries.clear()
            before = dict(runner.verification_graphs.stats)
            actual = run("graph", inputs, sp)[1]
            assert actual == expected, (batch,length,draft)
            result["correctness"].append(dict(batch=batch, length=length, draft=draft, equal=True,
                captures=runner.verification_graphs.stats["captures"]-before["captures"]))
        # 限额为0时必须走eager：验证缓存容量回退，不只测理想命中场景。
        runner.verification_graphs.entries.clear()
        runner.verification_graphs.max_entries = 0
        fallback_before = runner.verification_graphs.stats["fallbacks"]
        assert run("graph")[1] == reference
        assert runner.verification_graphs.stats["fallbacks"] > fallback_before
        result["capacity_fallback_equal"] = True
        runner.verification_graphs.max_entries = 4

        # 逐验证batch对比完整logits，而不只看最终文本；重复forward不提交token。
        # 这部分额外工作不放入上面的性能计时。
        original = runner.run_speculative
        logit_checks = []
        def checked(seqs):
            runner.enable_verification_graph = True
            actual = original(seqs)
            runner.enable_verification_graph = False
            try:
                expected = original(seqs)
            finally:
                runner.enable_verification_graph = True
            error = (actual.float()-expected.float()).abs().max().item()
            torch.testing.assert_close(actual, expected, atol=.0625, rtol=.01)
            same = bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))
            assert same
            logit_checks.append(dict(rows=actual.shape[0], max_absolute_error=error, argmax_equal=same))
            return actual
        runner.run_speculative = checked
        try:
            inputs = [[1,2,3]*85, [4,5,6]*24, [2,3,4]*35]
            run("graph", inputs, SamplingParams(temperature=0, max_tokens=16,
                ignore_eos=True, speculative_ngram_size=3))
        finally:
            runner.run_speculative = original
        assert logit_checks
        result["logit_checks"] = logit_checks
        result["graph_stats"] = dict(runner.verification_graphs.stats)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "PASS"
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        (root/"summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        llm.exit()

if __name__ == "__main__":
    main()
