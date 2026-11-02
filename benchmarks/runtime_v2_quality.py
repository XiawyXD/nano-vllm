"""Runtime V2 的固定历史质量代理：tokenize -> 两个隔离进程 -> NLL与严格logits对照。

不是生成质量榜单，也不覆盖 scheduler 的接受/拒绝逻辑。每篇原文不重复、不填充；
超长原文分成相邻且仅重叠一个token的上下文窗口，所有相邻token对恰好评分一次。
baseline 使用本项目 Torch 层/FlashAttention、eager、模型dtype KV，并非HF FP32金标准。
candidate 使用显式 JSON Config 覆盖；P段逐chunk、尾部逐token建立并读取真实paged KV。
例如 mixed/Graph 候选需显式提供 enforce_eager=false、batching_mode="mixed"、
enable_mixed_graph=true；只改 runtime_preset 不会覆盖这里固定的 Torch/eager 基线选项。
CPU日志复制会同步执行，因此时间仅作诊断，不用于性能结论。
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "benchmarks/workloads/runtime_v2_holdout.jsonl"


def sha256(path: Path) -> str:
    """流式散列文件，不把大模型权重整体读入CPU内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """冻结运行时代码与所有验收脚本；不同源码不能拼成同一质量结论。"""
    files = set((ROOT / "nanovllm").rglob("*.py")) | set((ROOT / "benchmarks").rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(files)}


def write_new(path: Path, value) -> None:
    """报告是不可覆盖的证据；RUNNING与最终结果用不同文件名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_corpus(path: Path) -> list[dict]:
    """校验独立文档ID与文本；明确拒绝整篇重复，不自动生成/填充测试文本。"""
    records, ids, texts = [], set(), set()
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
                raise ValueError(f"corpus line {number}: missing/duplicate id")
            if not isinstance(row.get("text"), str) or len(row["text"].strip()) < 100:
                raise ValueError(f"corpus line {number}: text must contain at least 100 characters")
            digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if digest in texts:
                raise ValueError(f"corpus line {number}: duplicate document is not holdout expansion")
            ids.add(row["id"])
            texts.add(digest)
            records.append(dict(row, text_sha256=digest))
    if not records:
        raise ValueError("corpus is empty")
    return records


def prepare(args) -> Path:
    """只运行tokenizer、不使用GPU；评分token不足20000立即失败，不复制语料凑数。"""
    from transformers import AutoTokenizer
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    corpus = args.corpus.resolve()
    rows = read_corpus(corpus)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = [dict(id=row["id"], category=row.get("category", "unspecified"),
                    text_sha256=row["text_sha256"],
                    token_ids=tokenizer.encode(row["text"], add_special_tokens=False)) for row in rows]
    if any(len(row["token_ids"]) < 2 for row in encoded):
        raise ValueError("every document must contain at least two tokenizer tokens")
    evaluated = sum(max(0, len(row["token_ids"]) - 1) for row in encoded)
    if evaluated < 20000:
        raise ValueError(f"holdout has only {evaluated} next-token targets; require >=20000; no padding/repetition allowed")
    overrides = load_json(args.candidate_config)
    if not isinstance(overrides, dict):
        raise ValueError("candidate config must be a JSON object")
    tree = ast.parse((ROOT / "nanovllm/config.py").read_text(encoding="utf-8"))
    known = {item.target.id for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Config"
             for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
    if overrides.keys() - known:
        raise ValueError(f"unknown candidate Config keys: {sorted(overrides.keys() - known)}")
    if "model" in overrides or overrides.get("tensor_parallel_size", 1) != 1:
        raise ValueError("quality proxy owns model path and currently supports TP1 only")
    locked = {"max_model_len", "max_num_batched_tokens", "prefill_chunk_size", "max_num_seqs", "enable_prefix_caching"}
    if locked & overrides.keys():
        raise ValueError(f"candidate may not override fixed-history controls: {sorted(locked & overrides.keys())}")
    base = dict(runtime_preset="baseline", tensor_parallel_size=1, max_model_len=args.context_length,
        max_num_batched_tokens=args.chunk_size, prefill_chunk_size=args.chunk_size, max_num_seqs=1,
        enable_prefix_caching=False, enforce_eager=True, sampler_backend="torch", mlp_backend="torch",
        rmsnorm_backend="torch", decode_attention_backend="flash", kv_cache_dtype="model",
        gpu_memory_utilization=args.gpu_memory_utilization)
    candidate = dict(base, **overrides)
    # 策略字段未显式固定，可由preset解析；算子/Graph选项须在overrides里明确选择。
    model = Path(args.model).resolve()
    names = sorted({*model.glob("*.json"), *model.glob("*.safetensors"), *model.glob("*.model"),
                    *model.glob("*.txt"), *model.glob("*.tiktoken")})
    if not names:
        raise ValueError("model directory has no local checkpoint/tokenizer files")
    model_hashes = {path.name: sha256(path) for path in names if path.is_file()}
    output.mkdir(parents=True)
    corpus_copy = output / "corpus.jsonl"
    with corpus_copy.open("xb") as file:
        file.write(corpus.read_bytes())
    tokenized = output / "tokenized.jsonl"
    with tokenized.open("x", encoding="utf-8") as file:
        for row in encoded:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = dict(schema=1, model=str(model), corpus_source=str(corpus), corpus_sha256=sha256(corpus_copy),
        tokenized_sha256=sha256(tokenized), source_sha256=source_hashes(), model_sha256=model_hashes,
        candidate_config_sha256=sha256(args.candidate_config), baseline_config=base, candidate_config=candidate,
        documents=len(rows), raw_tokens=sum(len(row["token_ids"]) for row in encoded), evaluated_tokens=evaluated,
        context_length=args.context_length, chunk_size=args.chunk_size, decode_tail=args.decode_tail,
        ce_rows=args.ce_rows, thresholds=dict(mean_nll_increase=0.01, per_document_nll_increase=0.05,
                                            strict_logit_atol=0.0625, strict_logit_rtol=0.01),
        corpus_provenance=("Bundled authored holdout; text IDs and full source frozen; no padding."
                           if corpus == DEFAULT_CORPUS.resolve() else
                           "User-specified local corpus; source file frozen, external provenance not independently verified."),
        scope="Teacher-forced model/KV quality proxy. Mixed candidate uses single-request prepare_plan/metadata/graph paths. Does not exercise adaptive acceptance, request scheduling, or concurrent P/D mixing.")
    write_new(output / "quality_manifest.json", manifest)
    print(json.dumps(dict(status="PREPARED", output=str(output), documents=len(rows), evaluated_tokens=evaluated)))
    return output


def verify_prepared(directory: Path) -> dict:
    """每个子进程独立核对冻结语料/源码/模型，避免跨版本组合参考与候选。"""
    manifest = load_json(directory / "quality_manifest.json")
    if manifest["source_sha256"] != source_hashes():
        raise RuntimeError("source hashes changed after prepare")
    for name, key in (("corpus.jsonl", "corpus_sha256"), ("tokenized.jsonl", "tokenized_sha256")):
        if sha256(directory / name) != manifest[key]:
            raise RuntimeError(f"frozen {name} changed")
    model = Path(manifest["model"])
    for name, digest in manifest["model_sha256"].items():
        if sha256(model / name) != digest:
            raise RuntimeError(f"model file changed: {name}")
    return manifest


def window_ranges(length: int, context_length: int):
    """相邻窗口重叠一个token，因此跨窗口边界的下一个token目标不会丢失或重复。"""
    for start in range(0, length - 1, context_length - 1):
        yield start, min(length, start + context_length)


def query_schedule(input_length: int, chunk_size: int, decode_tail: int):
    """前面按chunk建KV，末尾按单token读KV；yield(start,end,is_prefill)。"""
    prefill_end = max(1, input_length - decode_tail)
    for start in range(0, prefill_end, chunk_size):
        yield start, min(prefill_end, start + chunk_size), True
    for start in range(prefill_end, input_length):
        yield start, start + 1, False


def worker(directory: Path, role: str) -> None:
    """一个独立进程只加载一个模型；始终exit并释放自己的引擎，然后进程退出。"""
    manifest = verify_prepared(directory)
    work = directory / role
    work.mkdir()
    write_new(work / "started.json", dict(event="WORKER_STARTED", pid=os.getpid(), role=role))
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.execution_plan import ExecutionItem, ExecutionPlan, ExecutionRole
    from nanovllm.engine.packed_runner import prepare_plan
    from nanovllm.utils.context import get_context, reset_context
    configuration = manifest[f"{role}_config"]
    report = dict(status="RUNNING", role=role, config=configuration, documents=[], pid=os.getpid(),
                  manifest_sha256=sha256(directory / "quality_manifest.json"),
                  scope=manifest["scope"], timing_note="Debug CE/logit copies synchronize; not a performance benchmark.")
    engine = None
    started = time.perf_counter()
    try:
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        ensure_gpu1_available(evidence_path=work / "gpu_observations.jsonl")
        with torch.inference_mode():
            engine = LLM(manifest["model"], **configuration)
            runner, manager = engine.model_runner, engine.scheduler.block_manager
            report["effective_runtime"] = engine.config.runtime_summary()
            report["dtype"] = str(next(runner.model.parameters()).dtype)
            report["device"] = torch.cuda.get_device_name()
            report["versions"] = dict(torch=torch.__version__, cuda=torch.version.cuda)
            execution_paths = Counter()
            for number, line in enumerate((directory / "tokenized.jsonl").read_text(encoding="utf-8").splitlines()):
                ensure_gpu1_available(evidence_path=work / "gpu_observations.jsonl")
                row = json.loads(line)
                all_tokens = row["token_ids"]
                losses, logits_evidence, schedules = [], {}, []
                selected = {0, (len(all_tokens) - 2) // 2, len(all_tokens) - 2}
                for window_start, window_end in window_ranges(len(all_tokens), manifest["context_length"]):
                    tokens = all_tokens[window_start:window_end]
                    seq = None
                    try:
                        for start, end, prefill in query_schedule(len(tokens) - 1, manifest["chunk_size"], manifest["decode_tail"]):
                            if seq is None:
                                seq = Sequence(tokens[:end], SamplingParams(temperature=0, max_tokens=1))
                                manager.allocate(seq, 0)
                            else:
                                for token in tokens[len(seq):end]:
                                    seq.append_token(token)
                                manager.ensure_blocks(seq)
                            seq.num_scheduled_tokens, seq.is_prefill = end - start, prefill
                            schedules.append((window_start, start, end, prefill))
                            try:
                                hidden = None
                                if engine.config.batching_mode == "mixed":
                                    item = ExecutionItem(seq.seq_id, ExecutionRole.PREFILL if prefill else ExecutionRole.DECODE,
                                        tuple(tokens[start:end]), start, tuple(seq.block_table),
                                        logit_positions=tuple(range(end - start)))
                                    plan = ExecutionPlan((item,))
                                    prepared = prepare_plan(runner, plan)
                                    ids, positions, context = prepared.input_ids, prepared.positions, get_context()
                                    if context.batch_kind == "D_NORMAL" and context.attention_groups is None:
                                        pieces = ((0, runner.run_model(ids, positions, False)),)
                                        execution_paths["packed_decode_graph_or_eager"] += 1
                                    else:
                                        if runner.config.enable_mixed_graph and not runner.enforce_eager:
                                            signature = (runner.config.mixed_attention_backend, runner.config.decode_attention_backend,
                                                runner.config.rmsnorm_backend, runner.config.mlp_backend,
                                                runner.config.mlp_min_tokens, str(next(runner.model.parameters()).dtype))
                                            hidden = runner.packed_graphs.run(runner.model, ids, positions, context, signature)
                                        path = "packed_graph" if hidden is not None else "packed_eager"
                                        if hidden is None:
                                            hidden = runner.model(ids, positions)
                                        execution_paths[path] += 1
                                        # Transformer已使用真实plan元数据；CE要求每一行，不是采样压缩行。
                                        # 仅LM head分片时取消全chunk的index_select，避免小切片索引越界。
                                        # Graph可能用set_context替换了对象，必须取当前对象，不能修改旧引用。
                                        context = get_context()
                                        context.logits_indices, context.return_all_logits = None, True
                                        pieces = ((offset, runner.model.compute_logits(hidden[offset:offset + manifest["ce_rows"]]))
                                                  for offset in range(0, end - start, manifest["ce_rows"]))
                                elif prefill:
                                    ids, positions = runner.prepare_prefill([seq], return_all_logits=True)
                                    hidden = runner.model(ids, positions)
                                    execution_paths["legacy_prefill_eager"] += 1
                                    # 每次仅 materialize ce_rows × vocab 的 logits，避免整个文档的巨矩阵。
                                    pieces = ((offset, runner.model.compute_logits(hidden[offset:offset + manifest["ce_rows"]]))
                                              for offset in range(0, end - start, manifest["ce_rows"]))
                                else:
                                    ids, positions = runner.prepare_decode([seq])
                                    pieces = ((0, runner.run_model(ids, positions, False)),)
                                    execution_paths["legacy_decode_graph_or_eager"] += 1
                                for offset, logits in pieces:
                                    size = logits.shape[0]
                                    targets = torch.tensor(tokens[start + offset + 1:start + offset + size + 1], device=logits.device)
                                    local_losses = F.cross_entropy(logits.float(), targets, reduction="none")
                                    if not bool(torch.isfinite(logits).all() and torch.isfinite(local_losses).all()):
                                        raise RuntimeError(f"non-finite logits/NLL at {row['id']}:{window_start + start + offset}")
                                    losses.extend(local_losses.double().cpu().tolist())
                                    for local in range(size):
                                        absolute = window_start + start + offset + local
                                        if absolute in selected:
                                            logits_evidence[absolute] = logits[local].detach().cpu().clone()
                                    del logits, local_losses
                                if hidden is not None:
                                    del hidden
                                seq.num_cached_tokens = end
                            finally:
                                reset_context()
                    finally:
                        reset_context()
                        if seq is not None:
                            manager.deallocate(seq)
                if len(losses) != len(all_tokens) - 1 or set(logits_evidence) != selected:
                    raise AssertionError("teacher-forced target coverage or diagnostic row coverage is incomplete")
                refs = sum(block.ref_count for block in manager.blocks)
                if manager.used_block_ids or refs:
                    raise AssertionError("teacher-forced scoring leaked KV references")
                evidence = dict(id=row["id"], token_ids=all_tokens, losses=torch.tensor(losses, dtype=torch.float64),
                                selected_logits=logits_evidence, schedules=schedules)
                evidence_path = work / f"document_{number:03d}.pt"
                if evidence_path.exists():
                    raise FileExistsError(evidence_path)
                torch.save(evidence, evidence_path)
                report["documents"].append(dict(id=row["id"], token_count=len(losses), nll_sum=sum(losses),
                    mean_nll=sum(losses) / len(losses), tensor_file=evidence_path.name,
                    tensor_sha256=sha256(evidence_path), used_blocks_after=0, kv_ref_total_after=0))
                print(json.dumps(dict(role=role, document=row["id"], targets=len(losses))), flush=True)
            report["evaluated_tokens"] = sum(row["token_count"] for row in report["documents"])
            report["mean_nll"] = sum(row["nll_sum"] for row in report["documents"]) / report["evaluated_tokens"]
            report["execution_paths"] = dict(execution_paths)
            report["metadata_stats"] = runner.metadata_buffers.stats()
            report["packed_graph_stats"] = runner.packed_graphs.stats()
            report["status"] = "PASS_SCORING_ONLY_COMPARISON_PENDING"
    except Exception as error:
        report["status"], report["error"] = "FAIL", f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if engine is not None:
                engine.exit()
        except Exception as error:
            report["status"], report["cleanup_error"] = "FAIL", f"{type(error).__name__}: {error}"
            raise
        finally:
            report["elapsed_seconds_not_benchmark"] = time.perf_counter() - started
            write_new(work / "report.json", report)


def compare(directory: Path) -> dict:
    """按冻结阈值分别报告质量与严格数值；质量PASS不掩盖logits REVIEW_REQUIRED。"""
    import torch
    manifest = verify_prepared(directory)
    reports = [load_json(directory / role / "report.json") for role in ("baseline", "candidate")]
    if any(report["status"] != "PASS_SCORING_ONLY_COMPARISON_PENDING" for report in reports):
        raise RuntimeError("both isolated workers must finish scoring successfully")
    if any(report["manifest_sha256"] != sha256(directory / "quality_manifest.json") for report in reports):
        raise RuntimeError("frozen configuration manifest changed between worker processes")
    if any(report["evaluated_tokens"] != manifest["evaluated_tokens"] for report in reports):
        raise RuntimeError("worker token count differs from frozen fixture")
    if any(len(report["documents"]) != manifest["documents"] for report in reports):
        raise RuntimeError("document count differs from frozen fixture")
    rows, strict_pass = [], True
    thresholds = manifest["thresholds"]
    for left, right in zip(reports[0]["documents"], reports[1]["documents"]):
        tensors = []
        for role, row in zip(("baseline", "candidate"), (left, right)):
            path = directory / role / row["tensor_file"]
            if sha256(path) != row["tensor_sha256"]:
                raise RuntimeError("tensor evidence hash changed")
            tensors.append(torch.load(path, map_location="cpu", weights_only=True))
        a, b = tensors
        if (a["id"] != b["id"] or a["token_ids"] != b["token_ids"] or a["schedules"] != b["schedules"]):
            raise RuntimeError("history or cached chunk schedule differs; comparison is invalid")
        for evidence, summary in zip((a, b), (left, right)):
            if (evidence["losses"].numel() != summary["token_count"] or
                    not bool(torch.isfinite(evidence["losses"]).all()) or
                    not math.isclose(float(evidence["losses"].sum()), summary["nll_sum"], abs_tol=1e-8, rel_tol=1e-9) or
                    not math.isclose(float(evidence["losses"].mean()), summary["mean_nll"], abs_tol=1e-9, rel_tol=1e-9)):
                raise RuntimeError("NLL summary differs from per-token tensor evidence")
        diagnostics = []
        for position in a["selected_logits"]:
            actual, reference = b["selected_logits"][position].float(), a["selected_logits"][position].float()
            delta = actual - reference
            close = bool(torch.allclose(actual, reference, atol=thresholds["strict_logit_atol"], rtol=thresholds["strict_logit_rtol"]))
            argmax_equal = int(actual.argmax()) == int(reference.argmax())
            strict_pass &= close and argmax_equal
            diagnostics.append(dict(position=position, max_abs=float(delta.abs().max()),
                rmse=float(delta.square().mean().sqrt()), allclose=close, argmax_equal=argmax_equal))
        increase = right["mean_nll"] - left["mean_nll"]
        rows.append(dict(id=left["id"], targets=left["token_count"], baseline_nll=left["mean_nll"],
                         candidate_nll=right["mean_nll"], nll_increase=increase,
                         quality_pass=increase <= thresholds["per_document_nll_increase"], strict_logits=diagnostics))
    for report in reports:
        total = sum(row["token_count"] for row in report["documents"])
        weighted_mean = sum(row["nll_sum"] for row in report["documents"]) / total
        if total != manifest["evaluated_tokens"] or not math.isclose(weighted_mean, report["mean_nll"], abs_tol=1e-9, rel_tol=1e-9):
            raise RuntimeError("global NLL summary differs from validated per-document evidence")
    increase = reports[1]["mean_nll"] - reports[0]["mean_nll"]
    quality_pass = increase <= thresholds["mean_nll_increase"] and all(row["quality_pass"] for row in rows)
    result = dict(status="FAIL" if not quality_pass else "PASS" if strict_pass else "REVIEW_REQUIRED",
        quality_status="PASS" if quality_pass else "FAIL", numerical_status="PASS_SAMPLED_LOGITS" if strict_pass else "REVIEW_REQUIRED",
        thresholds=thresholds, evaluated_tokens=manifest["evaluated_tokens"], documents=rows,
        baseline_mean_nll=reports[0]["mean_nll"], candidate_mean_nll=reports[1]["mean_nll"], mean_nll_increase=increase,
        token_weighted=True, scope=manifest["scope"],
        limitation="NLL evaluates every frozen target; strict logits check only first/middle/last prediction per document. This is not comprehensive task quality, free generation, or scheduler equivalence.")
    write_new(directory / "comparison.json", result)
    print(json.dumps({key: result[key] for key in ("status", "quality_status", "numerical_status", "mean_nll_increase")}))
    return result


def main():
    """run按先参考后候选启动子进程；任何阶段失败停止，既有输出绝不覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("prepare", "run"):
        command = sub.add_parser(action)
        command.add_argument("--model", required=True)
        command.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
        command.add_argument("--candidate-config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--context-length", type=int, default=1024)
        command.add_argument("--chunk-size", type=int, default=128)
        command.add_argument("--decode-tail", type=int, default=8)
        command.add_argument("--ce-rows", type=int, default=16)
        command.add_argument("--gpu-memory-utilization", type=float, default=0.3)
        command.add_argument("--worker-timeout", type=float, default=1800)
    command = sub.add_parser("worker")
    command.add_argument("--prepared", type=Path, required=True)
    command.add_argument("--role", choices=("baseline", "candidate"), required=True)
    command = sub.add_parser("compare")
    command.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("prepare", "run"):
        if (args.context_length < 2 or not 1 <= args.chunk_size <= args.context_length or args.decode_tail < 1 or
                not 1 <= args.ce_rows <= args.chunk_size or not 0 < args.gpu_memory_utilization <= 1 or
                not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0):
            parser.error("invalid shape, memory, or timeout controls")
        output = prepare(args)
        if args.command == "run":
            for role in ("baseline", "candidate"):
                from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
                ensure_gpu1_available(evidence_path=output / "gpu_observations.jsonl")
                command = [sys.executable, "-m", "benchmarks.runtime_v2_quality", "worker", "--prepared", str(output), "--role", role]
                with (output / f"{role}.log").open("x", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.worker_timeout, check=True)
            result = compare(output)
            if result["status"] != "PASS":
                raise SystemExit(2 if result["quality_status"] == "PASS" else 1)
    elif args.command == "worker":
        worker(args.prepared.resolve(), args.role)
    else:
        result = compare(args.prepared.resolve())
        if result["status"] != "PASS":
            raise SystemExit(2 if result["quality_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
