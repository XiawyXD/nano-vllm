"""Single-GPU fixed-history packed-runtime checks; not a throughput benchmark.

Runs the real prepare_plan/run_plan path. The reference is the same model with
all-paged-varlen attention, eager execution and dynamic metadata, not HF/FP32.
Candidate split groups are tested as P+D and D+V in one transformer forward.
Graph A->B->A changes token values, positions, sequence lengths and physical pages
without changing the Graph key. Numerical review is distinct from structural FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOGIT_ATOL = 0.0625
LOGIT_RTOL = 0.01


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def tensor_digest(tensor):
    import torch
    value = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def compare(actual, reference, *, logits=False):
    """Keep strict model-logit tolerances fixed; KV exactness is separate."""
    import torch
    if actual.shape != reference.shape:
        return dict(shape_equal=False, actual_shape=list(actual.shape),
                    reference_shape=list(reference.shape), exact=False, finite=False)
    a, b = actual.detach().float().cpu(), reference.detach().float().cpu()
    delta = a - b
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = dict(shape_equal=True, shape=list(a.shape), elements=a.numel(), finite=finite,
        exact=bool(torch.equal(actual.cpu(), reference.cpu())),
        different=int((delta != 0).sum()),
        max_abs=float(delta.abs().max()) if finite and delta.numel() else 0.0 if finite else None,
        rms=float(delta.square().mean().sqrt()) if finite and delta.numel() else 0.0 if finite else None)
    if logits:
        result.update(atol=LOGIT_ATOL, rtol=LOGIT_RTOL,
            allclose=finite and bool(torch.allclose(a, b, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)),
            argmax_equal=finite and bool(torch.equal(a.argmax(-1), b.argmax(-1))),
            argmax_mismatches=int((a.argmax(-1) != b.argmax(-1)).sum()) if finite else None)
        if finite:
            result["elements_exceeding_tolerance"] = int((delta.abs() > LOGIT_ATOL + LOGIT_RTOL*b.abs()).sum())
    return result


def provenance(args):
    import torch
    def git(*tail):
        result = subprocess.run(["git", *tail], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    paths = list((ROOT / "nanovllm").rglob("*.py")) + [Path(__file__)]
    return dict(git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--short"),
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)},
        device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch=torch.__version__, cuda=torch.version.cuda,
        model=str(Path(args.model).resolve()),
        model_config_sha256=digest(Path(args.model) / "config.json"))


def make_specs(quick, candidate_backend="split"):
    """Build fixed-history cases while varying only the candidate attention route."""
    if candidate_backend not in {"split", "varlen"}:
        raise ValueError("candidate backend must be split or varlen")
    histories = ((17, 29), (19, 31)) if quick else ((253, 263), (255, 269))
    specs = [dict(name="pd", roles=("prefill", "decode"), queries=(5, 1), backend=candidate_backend, partial=False),
             dict(name="dv", roles=("verify", "decode"), queries=(5, 1), backend=candidate_backend, partial=False),
             dict(name="pure_d_varlen", roles=("decode", "decode"), queries=(1, 1), backend="varlen", partial=False)]
    if not quick:
        specs.append(dict(name="pd_partial", roles=("prefill", "decode"), queries=(5, 1), backend=candidate_backend, partial=True))
    return [dict(spec, histories=histories) for spec in specs]


def logical_kv(runner, seq, length):
    """Reconstruct chronological K/V using the request's physical page table."""
    import torch
    pages = torch.tensor(seq.block_table, device=runner.kv_cache.device, dtype=torch.long)
    return runner.kv_cache.index_select(2, pages).flatten(2, 3)[:, :, :length].cpu().clone()


def prepare_state(engine, spec, variant, seed):
    """Build true prefixes once, then restore the exact same prefix before each path."""
    import torch
    from nanovllm import SamplingParams
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.execution_plan import ExecutionItem, ExecutionPlan, ExecutionRole
    from nanovllm.utils.context import reset_context
    runner, manager = engine.model_runner, engine.scheduler.block_manager
    seqs, items, initial, tables = [], [], [], []
    generator = torch.Generator(device="cpu").manual_seed(seed + variant * 1009)
    vocab = engine.config.hf_config.vocab_size
    try:
        for index, (role, qlen, history) in enumerate(zip(spec["roles"], spec["queries"], spec["histories"][variant])):
            # Deterministic vocabulary-valid tokens are an operator fixture, not a quality corpus.
            tokens = torch.randint(0, vocab, (history + qlen,), generator=generator).tolist()
            seq = Sequence(tokens, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
            seqs.append(seq)
            manager.allocate(seq, 0)
            pages = torch.tensor(seq.block_table, device=runner.kv_cache.device, dtype=torch.long)
            runner.kv_cache.index_fill_(2, pages, 0)
            try:
                for start in range(0, history, engine.config.prefill_chunk_size):
                    end = min(history, start + engine.config.prefill_chunk_size)
                    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = start, end-start, True
                    ids, positions = runner.prepare_prefill([seq], return_all_logits=True)
                    runner.model(ids, positions)
                    reset_context()
            finally:
                reset_context()
            seq.num_cached_tokens, seq.num_scheduled_tokens = history, qlen
            seq.is_prefill = role != "decode"
            positions = tuple(range(qlen)) if role == "verify" else (qlen-1,)
            if spec["partial"] and role == "prefill":
                positions = ()
            items.append(ExecutionItem(seq.seq_id, ExecutionRole(role), tuple(tokens[history:]), history,
                tuple(seq.block_table), draft=tuple(tokens[history+1:]) if role == "verify" else (),
                logit_positions=positions))
            initial.append(logical_kv(runner, seq, history))
            tables.extend(seq.block_table)
        page_ids = torch.tensor(tables, device=runner.kv_cache.device, dtype=torch.long)
        baseline = runner.kv_cache.index_select(2, page_ids)
        return dict(plan=ExecutionPlan(tuple(items)), seqs=seqs, initial=initial,
                    page_ids=page_ids, baseline=baseline, variant="AB"[variant], spec=spec,
                    source_tokens=[list(seq.token_ids) for seq in seqs])
    except BaseException:
        for seq in seqs:
            if seq.block_table:
                manager.deallocate(seq)
        raise


def metadata_snapshot(prepared):
    """Read actual production metadata before forward; these copies are diagnostic synchronization."""
    from nanovllm.utils.context import get_context
    context = get_context()
    groups = []
    for group in context.attention_groups or ():
        groups.append(dict(kind=group.kind, start=group.start, end=group.end,
            block_tables=group.block_tables.cpu().tolist(),
            cu_q=None if group.cu_seqlens_q is None else group.cu_seqlens_q.cpu().tolist(),
            cu_k=None if group.cu_seqlens_k is None else group.cu_seqlens_k.cpu().tolist(),
            context_lens=None if group.context_lens is None else group.context_lens.cpu().tolist(),
            max_q=group.max_seqlen_q, max_k=group.max_seqlen_k))
    return dict(input_ids=prepared.input_ids.cpu().tolist(), positions=prepared.positions.cpu().tolist(),
        slots=context.slot_mapping.cpu().tolist(), owners=list(prepared.row_request_ids),
        logits_indices=None if context.logits_indices is None else context.logits_indices.cpu().tolist(),
        batch_kind=context.batch_kind, groups=groups)


def legacy_decode_snapshot(input_ids, positions, sequences):
    """Observe the real pure-Decode fast path without forcing packed metadata creation."""
    from nanovllm.utils.context import get_context
    context = get_context()
    if context.attention_groups is not None or context.slot_mapping is None:
        raise AssertionError("pure Decode did not prepare ordinary paged metadata")
    return dict(input_ids=input_ids.cpu().tolist(), positions=positions.cpu().tolist(),
        slots=context.slot_mapping.cpu().tolist(), owners=[seq.seq_id for seq in sequences],
        logits_indices=None, batch_kind="D_NORMAL", groups=[])


def assert_metadata(snapshot, state, block_size):
    """Check actual positions/slots/row ownership against immutable plan, not another runner output."""
    plan = state["plan"]
    expected_ids, expected_positions, expected_slots, owners, rows = [], [], [], [], []
    for item in plan.items:
        offset = len(expected_ids)
        expected_ids.extend(item.input_token_ids)
        expected_positions.extend(range(item.start_pos, item.end_pos))
        expected_slots.extend(item.block_table[p // block_size] * block_size + p % block_size
                              for p in range(item.start_pos, item.end_pos))
        owners.extend([item.request_id] * len(item.logit_positions))
        rows.extend(offset + p for p in item.logit_positions)
    for key, expected in (("input_ids", expected_ids), ("positions", expected_positions),
                          ("slots", expected_slots), ("owners", owners)):
        if snapshot[key] != expected:
            raise AssertionError(f"actual packed {key} differs from immutable execution plan")
    if snapshot["logits_indices"] is not None and snapshot["logits_indices"] != rows:
        raise AssertionError("packed logits row selection is wrong")


class Observer:
    """Capture real run_plan sampler inputs without replacing its predictions or metadata."""
    def __init__(self, runner):
        import nanovllm.engine.packed_runner as packed
        self.runner, self.module = runner, packed
        self.original_prepare, self.original_decode = packed.prepare_plan, runner.prepare_decode
        self.original_sampler = runner.sampler
        self.active = False
        self.calls = 0
        def prepared(*args, **kwargs):
            result = self.original_prepare(*args, **kwargs)
            if self.active:
                self.metadata = metadata_snapshot(result)
            return result
        def prepared_decode(seqs):
            ids, positions = self.original_decode(seqs)
            if self.active:
                self.metadata = legacy_decode_snapshot(ids, positions, seqs)
            return ids, positions
        def sampled(logits, owners):
            if self.active:
                self.logits = logits.detach().cpu().clone()
                self.sample_owners = [seq.seq_id for seq in owners]
            return self.original_sampler(logits, owners)
        def count_forward(_module, _inputs):
            if self.active:
                self.calls += 1
        packed.prepare_plan, runner.prepare_decode, runner.sampler = prepared, prepared_decode, sampled
        self.handle = runner.model.register_forward_pre_hook(count_forward)

    def close(self):
        self.module.prepare_plan = self.original_prepare
        self.runner.prepare_decode = self.original_decode
        self.runner.sampler = self.original_sampler
        self.handle.remove()

    def run(self, engine, state, *, backend, graph, pool, label):
        import torch
        from nanovllm.utils.context import get_context
        runner = self.runner
        runner.kv_cache.index_copy_(2, state["page_ids"], state["baseline"])
        protected_before = runner.kv_cache.index_select(2, state["protected_pages"])
        runner.config.mixed_attention_backend = backend
        runner.config.enable_mixed_graph = graph
        runner.enforce_eager = not graph
        runner.metadata_buffers = pool
        self.calls, self.metadata, self.logits, self.sample_owners = 0, None, None, None
        self.active = True
        try:
            predictions = self.module.run_plan(runner, state["plan"], state["seqs"])
        finally:
            self.active = False
        if self.logits is None or self.metadata is None:
            raise AssertionError("production run_plan did not prepare metadata and call sampler")
        assert_metadata(self.metadata, state, runner.block_size)
        if self.sample_owners != self.metadata["owners"]:
            raise AssertionError("sampler rows were assigned to wrong requests")
        if self.logits.shape[0] != len(self.sample_owners):
            raise AssertionError("LM head result has wrong compact row count")
        if not bool(torch.isfinite(self.logits).all()):
            raise AssertionError("non-finite model logits")
        if get_context().slot_mapping is not None:
            raise AssertionError("run_plan failed to reset Context")
        if not torch.equal(runner.kv_cache.index_select(2, state["protected_pages"]), protected_before):
            raise AssertionError("packed forward polluted another request state's physical KV pages")
        stats = dict(runner.last_plan_stats)
        if not graph and self.calls != 1:
            raise AssertionError(f"packed eager used {self.calls} transformer forwards, expected one")
        if stats.get("forward_calls") != 1:
            raise AssertionError("runner did not report one semantic forward")
        kv, history_checks = [], []
        for index, (seq, item) in enumerate(zip(state["seqs"], state["plan"].items)):
            if seq.token_ids != state["source_tokens"][index] or seq.num_cached_tokens != item.start_pos:
                raise AssertionError("run_plan mutated caller sequence history/cached length")
            values = logical_kv(runner, seq, item.end_pos)
            pristine = torch.equal(values[:, :, :item.start_pos], state["initial"][index])
            if not pristine:
                raise AssertionError("packed forward overwrote an already cached logical prefix")
            history_checks.append(dict(prefix_length=item.start_pos, exact=True,
                                       prefix_sha256=tensor_digest(state["initial"][index])))
            kv.append(values)
        metadata = self.metadata
        summary = dict(label=label, variant=state["variant"], backend=backend, graph_requested=graph,
            plan_kind=state["plan"].kind, roles=[item.role.value for item in state["plan"].items],
            query_lengths=[item.query_len for item in state["plan"].items],
            prefix_lengths=[item.start_pos for item in state["plan"].items],
            physical_pages=[list(item.block_table) for item in state["plan"].items],
            compact_rows=self.logits.shape[0], actual_python_forward_calls=self.calls,
            semantic_forward_calls=1, runner_stats=stats, metadata_contract="PASS",
            immutable_history=history_checks, context_reset=True,
            inactive_state_pages_unchanged=True, inactive_pages=state["protected_pages"].numel(),
            capture_note="Cold Graph capture intentionally performs warmups/capture; Python hooks do not execute during replay.")
        return dict(summary=summary, logits=self.logits, kv=kv, metadata=metadata, predictions=predictions)


def comparison(actual, reference, *, require_exact):
    """Separate structural validity, strict numerical agreement, and greedy agreement."""
    same_metadata = all(actual["metadata"][key] == reference["metadata"][key]
                        for key in ("input_ids", "positions", "slots", "owners"))
    same_history = actual["summary"]["immutable_history"] == reference["summary"]["immutable_history"]
    if not same_metadata or not same_history:
        raise AssertionError("numerical comparison does not have the same history/input/output rows")
    logits = compare(actual["logits"], reference["logits"], logits=True)
    kv = [compare(a, b) for a, b in zip(actual["kv"], reference["kv"])]
    finite = logits["finite"] and all(item["finite"] for item in kv)
    exact = logits["exact"] and all(item["exact"] for item in kv)
    passes = (logits.get("allclose", False) and logits.get("argmax_equal", False) and
              all(item["exact"] for item in kv) and (exact or not require_exact))
    return dict(status="FAIL" if not finite else "PASS" if passes else "REVIEW_REQUIRED",
        logits=logits, logical_kv=kv, same_history=True, same_input_and_rows=True,
        exact_same_path_required=require_exact,
        note="Cross-backend KV differences are reported as numerical REVIEW, not relaxed by a newly invented tolerance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3-0.6B for quick, Qwen3-1.7B for primary acceptance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--metadata-backend", choices=("dynamic", "reuse"), default="reuse")
    parser.add_argument("--candidate-attention-backend", choices=("split", "varlen"), default="split",
        help="Use role-split P/V-varlen+D-decode, or route every query in the mixed candidate through varlen")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--linear-backend", choices=("torch", "triton_stable"), default="torch")
    parser.add_argument("--rmsnorm-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.suffix.lower() != ".json":
        parser.error("--output must name a new .json report")
    for path in (out, out.with_suffix(".pt"), out.with_suffix(".events.jsonl"), out.with_suffix(".started.json")):
        if path.exists():
            parser.error(f"evidence already exists; choose a new output path: {path}")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu memory utilization must be in (0,1]")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_new(out.with_suffix(".started.json"), dict(event="STARTED", args={**vars(args), "output": str(out)}, pid=os.getpid()))
    report = dict(status="RUNNING", args={**vars(args), "output": str(out)},
        strict_logit_thresholds=dict(atol=LOGIT_ATOL, rtol=LOGIT_RTOL), cases=[],
        scope="Fixed-history single-GPU real run_plan P+D/D+V and Graph/metadata checks; not scheduler acceptance, quality, TP2, or performance.",
        reference="Same model + pure paged-varlen eager + dynamic metadata; neither HF nor full FP32.")
    engine = observer = None
    tensors = {}
    started = time.perf_counter()
    log = out.with_suffix(".events.jsonl").open("x", encoding="utf-8")
    def event(value):
        log.write(json.dumps(value, ensure_ascii=False) + "\n")
        log.flush()
    try:
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        ensure_gpu1_available(evidence_path=out.with_suffix(".gpu.jsonl"))
        import torch
        from nanovllm import LLM
        from nanovllm.engine.metadata_buffers import MetadataBuffers
        from nanovllm.engine.packed_graph import PackedGraphs
        report["provenance"] = provenance(args)
        with torch.inference_mode():
            engine = LLM(args.model, tensor_parallel_size=1, max_model_len=512,
                max_num_batched_tokens=128, prefill_chunk_size=128, max_num_seqs=4,
                gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
                enforce_eager=False, sampler_backend="torch", mlp_backend="torch",
                rmsnorm_backend=args.rmsnorm_backend, linear_backend=args.linear_backend,
                kv_cache_dtype="model", batching_mode="mixed", metadata_backend=args.metadata_backend,
                enable_mixed_graph=True, mixed_attention_backend="split", speculative_policy="off")
            runner, manager = engine.model_runner, engine.scheduler.block_manager
            report["dtype"] = str(next(runner.model.parameters()).dtype)
            observer = Observer(runner)
            reference_pool = MetadataBuffers("dynamic", engine.config.metadata_max_bytes)
            candidate_pool = MetadataBuffers(args.metadata_backend, engine.config.metadata_max_bytes)
            for case_index, spec in enumerate(make_specs(args.quick, args.candidate_attention_backend)):
                ensure_gpu1_available(evidence_path=out.with_suffix(".gpu.jsonl"))
                states, results = [], {}
                case = dict(name=spec["name"], runs=[], comparisons={})
                report["cases"].append(case)
                event(dict(event="CASE_STARTED", name=spec["name"]))
                try:
                    for variant in (0, 1):
                        states.append(prepare_state(engine, spec, variant, args.seed + case_index*313))
                    if set(states[0]["page_ids"].tolist()) & set(states[1]["page_ids"].tolist()):
                        raise AssertionError("independent metadata variants share physical KV pages")
                    states[0]["protected_pages"], states[1]["protected_pages"] = states[1]["page_ids"], states[0]["page_ids"]
                    runner.packed_graphs = PackedGraphs(max_entries=4, min_observations=3)
                    for state in states:
                        variant = state["variant"]
                        reference = observer.run(engine, state, backend="varlen", graph=False,
                            pool=reference_pool, label=f"reference_{variant}")
                        eager = observer.run(engine, state, backend=spec["backend"], graph=False,
                            pool=candidate_pool, label=f"candidate_eager_{variant}")
                        case["runs"].extend((reference["summary"], eager["summary"]))
                        case["comparisons"][f"candidate_{variant}_vs_varlen_reference"] = comparison(eager, reference, require_exact=False)
                        results[f"ref_{variant}"], results[f"eager_{variant}"] = reference, eager
                    if all(role == "decode" for role in spec["roles"]):
                        # 纯 Decode 已恢复到普通 Decode/Graph 路径，不属于 packed Graph。
                        case["packed_graph_status"] = "NOT_APPLICABLE_PURE_DECODE_USES_DECODE_PATH"
                    else:
                        # Observe/capture A, then replay A->B->A. B changes all tensor values,
                        # but page-table dimensions and max-K scheduling bounds remain fixed.
                        for index in range(3):
                            warmed = observer.run(engine, states[0], backend=spec["backend"], graph=True,
                                pool=candidate_pool, label=f"graph_observation_{index+1}")
                            case["runs"].append(warmed["summary"])
                            case["comparisons"][f"graph_observation_{index+1}_vs_eager"] = comparison(warmed, results["eager_A"], require_exact=True)
                        entries_after_warmup = len(runner.packed_graphs.entries)
                        if entries_after_warmup != 1:
                            raise AssertionError(f"expected exactly one captured packed Graph, got {entries_after_warmup}")
                        for index, state in enumerate((states[0], states[1], states[0])):
                            label = f"graph_replay_{index+1}_{state['variant']}"
                            actual = observer.run(engine, state, backend=spec["backend"], graph=True,
                                pool=candidate_pool, label=label)
                            stats = actual["summary"]["runner_stats"]
                            if stats["path"] != "packed_graph" or stats["captured"] or observer.calls != 0:
                                raise AssertionError("same-key replay unexpectedly fell back, re-captured, or entered eager model")
                            if len(runner.packed_graphs.entries) != entries_after_warmup:
                                raise AssertionError("dynamic metadata values created a new Graph key")
                            case["runs"].append(actual["summary"])
                            case["comparisons"][f"{label}_vs_eager"] = comparison(actual, results[f"eager_{state['variant']}"], require_exact=True)
                            tensors[f"{spec['name']}/{label}"] = dict(logits=actual["logits"], metadata=actual["metadata"],
                                query_kv=[values[:, :, item.start_pos:].clone() for values, item in zip(actual["kv"], state["plan"].items)])
                        case["graph_stats"] = runner.packed_graphs.stats()
                    for name, result in results.items():
                        state = states[0 if name.endswith("A") else 1]
                        tensors[f"{spec['name']}/{name}"] = dict(logits=result["logits"], metadata=result["metadata"],
                            query_kv=[values[:, :, item.start_pos:].clone() for values, item in zip(result["kv"], state["plan"].items)])
                    statuses = [value["status"] for value in case["comparisons"].values()]
                    case["status"] = "FAIL" if "FAIL" in statuses else "REVIEW_REQUIRED" if "REVIEW_REQUIRED" in statuses else "PASS"
                    event(dict(event="CASE_FINISHED", name=spec["name"], status=case["status"]))
                finally:
                    for state in states:
                        for seq in state["seqs"]:
                            if seq.block_table:
                                manager.deallocate(seq)
                if manager.used_block_ids or sum(block.ref_count for block in manager.blocks):
                    raise AssertionError("case leaked KV references")
                case["resources_after"] = dict(used_blocks=0, kv_ref_total=0)
            report["metadata_stats"] = dict(reference=reference_pool.stats(), candidate=candidate_pool.stats())
            report["resources_after"] = dict(used_blocks=len(manager.used_block_ids),
                kv_ref_total=sum(block.ref_count for block in manager.blocks), request_records=len(engine._requests))
            statuses = [case["status"] for case in report["cases"]]
            report["status"] = "FAIL" if "FAIL" in statuses else "REVIEW_REQUIRED" if "REVIEW_REQUIRED" in statuses else "PASS"
    except BaseException as error:
        report["status"], report["error"] = "FAIL", f"{type(error).__name__}: {error}"
        event(dict(event="ERROR", error=report["error"]))
        raise
    finally:
        try:
            if observer is not None:
                observer.close()
            if engine is not None:
                engine.exit()
        except BaseException as error:
            report["status"], report["cleanup_error"] = "FAIL", f"{type(error).__name__}: {error}"
            raise
        finally:
            if tensors:
                torch.save(tensors, out.with_suffix(".pt"))
                report["tensor_evidence"] = dict(path=str(out.with_suffix(".pt")), sha256=digest(out.with_suffix(".pt")))
            report["elapsed_seconds_not_benchmark"] = time.perf_counter()-started
            event(dict(event="FINISHED", status=report["status"]))
            log.close()
            report["event_log_sha256"] = digest(out.with_suffix(".events.jsonl"))
            write_new(out, report)
            print(json.dumps(dict(status=report["status"], output=str(out))))
    return 0 if report["status"] == "PASS" else 2 if report["status"] == "REVIEW_REQUIRED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
