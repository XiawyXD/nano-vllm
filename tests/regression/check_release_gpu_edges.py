"""发布验收 GPU 边界检查：decode graph 非标准 batch，及可选 FP8 组合。

这是正确性/可达性回归，不输出加速比。所有 graph/eager 比较覆盖完整 logits，
报告数值误差和 argmax 差异。严格阈值失败会保留证据并标记 REVIEW_REQUIRED，
继续检查其他 case；执行错误仍为 FAIL。FP8 只检查有限数值、实际 prefix
命中、验证执行和 KV 回收，不把此 smoke 当作质量评估。
"""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, set_context


def compare_logits(actual, expected, args):
    """保留原公差、失败详情和 top-2 margin，不把近并列差异强行判成通过。"""
    actual_fp32, expected_fp32 = actual.float(), expected.float()
    absolute = (actual_fp32 - expected_fp32).abs()
    actual_ids, expected_ids = actual.argmax(-1), expected.argmax(-1)
    actual_top2 = actual_fp32.topk(2, dim=-1).values
    expected_top2 = expected_fp32.topk(2, dim=-1).values
    check = {
        "logits_shape": list(actual.shape),
        "max_absolute_error": absolute.max().item(),
        "mean_absolute_error": absolute.mean().item(),
        "actual_argmax": actual_ids.tolist(),
        "eager_argmax": expected_ids.tolist(),
        "argmax_mismatch_rows": (actual_ids != expected_ids).nonzero().flatten().tolist(),
        "actual_top2_margin": (actual_top2[:, 0] - actual_top2[:, 1]).tolist(),
        "eager_top2_margin": (expected_top2[:, 0] - expected_top2[:, 1]).tolist(),
        "atol": args.atol, "rtol": args.rtol,
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
    }
    tolerance = args.atol + args.rtol * expected_fp32.abs()
    check["elements_exceeding_tolerance"] = int((absolute > tolerance).sum())
    check["fraction_exceeding_tolerance"] = float((absolute > tolerance).float().mean())
    try:
        torch.testing.assert_close(actual_fp32, expected_fp32, atol=args.atol, rtol=args.rtol)
        check["within_tolerance"] = True
    except AssertionError as exc:
        check["within_tolerance"] = False
        check["strict_assert_close_error"] = str(exc)
    return check


@torch.inference_mode()
def padded_eager_logits(runner, batch, bucket, context):
    """匹配 graph 的 Transformer batch shape，隔离 padding 与 capture 两因素。

    LM head 仍只处理真实 batch，与 ModelRunner graph 路径完全一致。
    这是诊断参考，不会替代普通 eager 的严格验收结果。
    """
    buffers = runner.graph_vars
    set_context(False, slot_mapping=buffers["slot_mapping"][:bucket],
                context_lens=buffers["context_lens"][:bucket],
                block_tables=buffers["block_tables"][:bucket])
    try:
        hidden = runner.model(buffers["input_ids"][:bucket], buffers["positions"][:bucket])
        return runner.model.compute_logits(hidden[:batch])
    finally:
        set_context(False, slot_mapping=context.slot_mapping,
                    context_lens=context.context_lens, block_tables=context.block_tables)


def assert_released(llm):
    manager = llm.scheduler.block_manager
    if manager.used_block_ids:
        raise AssertionError(f"KV blocks leaked: {sorted(manager.used_block_ids)}")
    return manager.stats()


def check_decode_graphs(args, report):
    """同一个真实 forward 输入交错运行 graph/eager，避免生成轨迹分叉。"""
    for maximum, batches in ((20, (9, 10, 17, 20)), (10, (9, 10))):
        if args.graph_limit is not None and maximum != args.graph_limit:
            continue
        llm = LLM(
            args.model, max_model_len=128, max_num_batched_tokens=256,
            max_num_seqs=maximum, gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=False, sampler_backend="torch",
        )
        row = {"max_num_seqs": maximum, "graph_buckets": list(llm.model_runner.graph_bs),
               "batches": [], "logit_checks": []}
        report["decode_graphs"].append(row)
        runner = llm.model_runner
        original_run = runner.run_model
        current_batch = None
        diagnosed_batches = set()

        def checked(input_ids, positions, is_prefill):
            actual = original_run(input_ids, positions, is_prefill)
            if is_prefill:
                return actual
            batch = input_ids.numel()
            bucket = next(size for size in runner.graph_bs if size >= batch)
            context = get_context()
            buffers = runner.graph_vars
            columns = context.block_tables.size(1)
            copies = {
                "input_ids": torch.equal(buffers["input_ids"][:batch], input_ids),
                "positions": torch.equal(buffers["positions"][:batch], positions),
                "slot_mapping": torch.equal(buffers["slot_mapping"][:batch], context.slot_mapping),
                "context_lens": torch.equal(buffers["context_lens"][:batch], context.context_lens),
                "block_tables": torch.equal(buffers["block_tables"][:batch, :columns], context.block_tables),
                "padding_slots_disabled": bool((buffers["slot_mapping"][batch:bucket] == -1).all()),
                "padding_context_empty": bool((buffers["context_lens"][batch:bucket] == 0).all()),
            }
            if not all(copies.values()):
                raise AssertionError(f"graph input-buffer mismatch: {copies}")
            # actual logits 是 LM head 的独立输出，不是 graph 的静态 hidden buffer。
            saved = runner.enforce_eager
            runner.enforce_eager = True
            try:
                expected = original_run(input_ids, positions, is_prefill)
            finally:
                runner.enforce_eager = saved
            check = compare_logits(actual, expected, args)
            check.update(requested_batch=current_batch, graph_bucket=bucket,
                         graph_input_copy_checks=copies)
            row["logit_checks"].append(check)
            if not check["finite"]:
                raise AssertionError("non-finite logits in graph/eager comparison")
            if current_batch not in diagnosed_batches:
                padded = padded_eager_logits(runner, batch, bucket, context)
                check["same_bucket_eager_diagnostic"] = compare_logits(actual, padded, args)
                if not check["same_bucket_eager_diagnostic"]["finite"]:
                    raise AssertionError("non-finite logits in same-bucket diagnostic")
                diagnosed_batches.add(current_batch)
            return actual

        runner.run_model = checked
        try:
            for batch in batches:
                current_batch = batch
                before = len(row["logit_checks"])
                prompts = [([1, 2, 3, 4] * 12)[:32 + index % 8] for index in range(batch)]
                outputs = llm.generate(
                    prompts,
                    SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True),
                    use_tqdm=False,
                )
                checks = row["logit_checks"][before:]
                if not checks or not any(check["logits_shape"][0] == batch for check in checks):
                    raise AssertionError(f"expected actual decode batch {batch} was not exercised")
                row["batches"].append({
                    "batch": batch, "decode_checks": len(checks),
                    "strict_logit_checks_all_passed": all(check["within_tolerance"] for check in checks),
                    "argmax_checks_all_equal": all(not check["argmax_mismatch_rows"] for check in checks),
                    "output_token_ids": [output["token_ids"] for output in outputs],
                    "runtime_stats_after": assert_released(llm),
                })
        finally:
            # 把 bound method 写回 instance 会保留 self 引用环。删除测试覆盖，
            # 让类方法重新生效，再销毁所有局部 bound method / closure 引用。
            del runner.run_model
            llm.exit()
            del checked, original_run, runner, llm
            gc.collect()
            torch.cuda.empty_cache()
            row["cleanup_cuda_allocated_bytes"] = torch.cuda.memory_allocated()
            row["cleanup_cuda_reserved_bytes"] = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()


class ProbeDraft:
    """仅为触达 verifier 的测试 proposer；不代表 n-gram 命中率或模型质量。"""

    def propose(self, token_ids, limit):
        # 来自已有真实 token，必在词表范围；目标模型自行接受或拒绝。
        return [token_ids[-1]] * min(2, limit)


def check_fp8_combination(args, report):
    llm = LLM(
        args.model, max_model_len=512, max_num_batched_tokens=128,
        prefill_chunk_size=64, max_num_seqs=2, enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True, kv_cache_dtype="fp8",
        decode_attention_backend="triton", sampler_backend="torch",
    )
    runner = llm.model_runner
    original_run = runner.run_model
    result = {
        "experimental": True,
        "scope": "single-GPU FP8 prefix + chunk + eager verification reachability; no BF16 quality equivalence",
        "logit_checks": [], "generations": [],
    }
    report["fp8_combination"] = result

    def finite_run(input_ids, positions, is_prefill):
        logits = original_run(input_ids, positions, is_prefill)
        check = {
            "is_prefill_or_verification": is_prefill,
            "input_tokens": input_ids.numel(),
            "logits_shape": list(logits.shape),
            "finite": bool(torch.isfinite(logits).all()),
        }
        result["logit_checks"].append(check)
        if not check["finite"]:
            raise AssertionError("FP8 combination produced non-finite logits")
        return logits

    runner.run_model = finite_run
    prompt = [1, 2, 3] * 100  # 跨过 256-token 页，且 chunk=64 强制历史读取。
    params = SamplingParams(
        temperature=0, max_tokens=4, ignore_eos=True,
        speculative_ngram_size=3, speculative_max_draft_tokens=2,
    )
    try:
        for repeat in range(2):
            before = llm.scheduler.stats()
            output = llm.generate([prompt], params, use_tqdm=False)
            after = assert_released(llm)
            hits = after["prefix_cache_hits"] - before["prefix_cache_hits"]
            result["generations"].append({
                "repeat": repeat, "token_ids": output[0]["token_ids"],
                "prefix_hits": hits, "runtime_stats_after": after,
            })
            if repeat == 1 and hits <= 0:
                raise AssertionError("second FP8 request did not actually reuse a prefix")

        # 强制有效 draft 仅为覆盖真实 target verification GPU 路径。
        # 生成性能和正常 n-gram 接受率不能用这个 probe 的数据表述。
        seq = llm.add_request(prompt, params)
        seq._ngram_proposer = ProbeDraft()
        before_calls = llm.speculative_stats().get("target_verification_calls", 0)
        for _ in range(32):
            if seq.is_finished:
                break
            llm.step()
        if not seq.is_finished:
            raise AssertionError("FP8 probe request did not finish")
        calls = llm.speculative_stats().get("target_verification_calls", 0) - before_calls
        if calls <= 0:
            raise AssertionError("FP8 eager verification was not exercised")
        result["verification_probe"] = {
            "forced_test_proposer": True, "target_verification_calls": calls,
            "token_ids": list(seq.completion_token_ids),
            "runtime_stats_after": assert_released(llm),
        }
        llm.forget_request(seq.seq_id)

        # 一个没有共享前缀的新请求在第一块 chunk 后取消，必须归还所有引用。
        cancelled = llm.add_request([4, 5, 6] * 100, params)
        llm.step()
        if not llm.cancel_request(cancelled.seq_id):
            raise AssertionError("active FP8 chunk could not be cancelled")
        result["cancellation"] = {
            "cancelled": True, "runtime_stats_after": assert_released(llm),
        }
        llm.forget_request(cancelled.seq_id)
    finally:
        del runner.run_model
        llm.exit()
        del finite_run, original_run, runner, llm
        gc.collect()
        torch.cuda.empty_cache()
        result["cleanup_cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        result["cleanup_cuda_reserved_bytes"] = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-0.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.0625)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--fp8", action="store_true", help="also run experimental FP8 combination smoke")
    parser.add_argument("--graph-limit", type=int, choices=(10, 20),
                        help="run one graph-capacity case; use separate processes for isolated acceptance")
    parser.add_argument("--fp8-only", action="store_true", help="run only the FP8 smoke in its own process")
    args = parser.parse_args()
    if args.max_tokens < 2:
        parser.error("--max-tokens must be >= 2 to exercise decode")
    if args.fp8_only and args.graph_limit is not None:
        parser.error("--fp8-only and --graph-limit are mutually exclusive")
    if not torch.cuda.is_available():
        raise RuntimeError("this release-edge check requires a CUDA GPU")
    started = time.perf_counter()
    report = {
        "status": "RUNNING", "model": args.model,
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "decode_graphs": [], "fp8_requested": args.fp8 or args.fp8_only,
        "graph_limit": args.graph_limit, "fp8_only": args.fp8_only,
        "measurement": "correctness smoke, not a latency benchmark",
    }
    try:
        if not args.fp8_only:
            check_decode_graphs(args, report)
        if args.fp8 or args.fp8_only:
            check_fp8_combination(args, report)
        reviews = []
        for group in report["decode_graphs"]:
            for index, check in enumerate(group["logit_checks"]):
                if not check["within_tolerance"] or check["argmax_mismatch_rows"]:
                    reviews.append({"max_num_seqs": group["max_num_seqs"],
                                    "check_index": index, "batch": check["requested_batch"]})
                diagnostic = check.get("same_bucket_eager_diagnostic")
                if diagnostic and (not diagnostic["within_tolerance"] or diagnostic["argmax_mismatch_rows"]):
                    reviews.append({"max_num_seqs": group["max_num_seqs"],
                                    "check_index": index, "kind": "same_bucket_diagnostic"})
        report["numerical_reviews_required"] = reviews
        report["status"] = "REVIEW_REQUIRED" if reviews else "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds_including_model_init"] = time.perf_counter() - started
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": report["status"], "output": str(output),
                          "elapsed_seconds": report["elapsed_seconds_including_model_init"]}))


if __name__ == "__main__":
    main()
