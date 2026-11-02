"""Native Qwen3 draft, borrowing the target's TP1 process group.

This is not a second LLMEngine. It never initializes/destroys distributed state,
and owns an independent bounded KV tensor and optional single-token CUDA graph.
"""
from collections import Counter
from time import perf_counter

import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.engine.proposer import DraftProposer
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model


class DraftRunner:
    def __init__(self, config):
        if not dist.is_initialized() or dist.get_world_size() != 1:
            raise ValueError("draft runner borrows an initialized TP1 process group")
        self.capacity = config.max_model_len
        self.block_size = config.kvcache_block_size
        self.chunk_size = min(config.max_num_batched_tokens, self.capacity)
        self.request_id = None
        self.counters = Counter()
        self.graph = None
        self.stochastic = None
        self.suffix_graphs = {}
        self.prefill_graphs = {}
        self.prefill_replay_stream = None
        self._pending_device_proposal = None
        self._device_work_stream = None
        self.prefill_graph_enabled = getattr(config, "draft_prefill_graph", False)
        if type(self.prefill_graph_enabled) is not bool:
            raise ValueError("draft_prefill_graph must be bool")
        self.prefill_tail_graph_enabled = getattr(config, "draft_prefill_tail_graph", False)
        if type(self.prefill_tail_graph_enabled) is not bool:
            raise ValueError("draft_prefill_tail_graph must be bool")
        if self.prefill_tail_graph_enabled and (not self.prefill_graph_enabled or getattr(config, "enable_stochastic_speculation", False)):
            raise ValueError("draft_prefill_tail_graph requires greedy draft_prefill_graph")
        self.requested_execution = getattr(config, "draft_execution", "eager")
        if self.requested_execution not in {"eager", "graph"}:
            raise ValueError("draft_execution must be eager or graph")
        self.execution = "eager" if getattr(config, "enforce_eager", False) else self.requested_execution
        if self.prefill_graph_enabled and self.execution != "graph":
            raise ValueError("draft_prefill_graph requires active draft graphs")
        self.async_metadata = getattr(config, "draft_async_metadata", False)
        if type(self.async_metadata) is not bool:
            raise ValueError("draft_async_metadata must be bool")
        if self.async_metadata and self.execution != "graph":
            raise ValueError("draft_async_metadata requires active draft graphs")
        self.graph_max_tokens = getattr(config, "draft_graph_max_tokens", 8)
        if type(self.graph_max_tokens) is not int or not 1 <= self.graph_max_tokens <= 16:
            raise ValueError("draft_graph_max_tokens must be an integer in [1, 16]")
        self.graph_resident_bytes = 0
        self.graph_reserved_bytes = 0
        self.model = self.kv = self.proposer = self.block_table = None
        hf = AutoConfig.from_pretrained(config.draft_model)
        from nanovllm.utils.quant_config import configure_weight_format
        configure_weight_format(hf, getattr(config, "draft_quantization", None),
                                getattr(config, "draft_dtype", "checkpoint"))
        # A second hidden size otherwise competes for the global torch.compile
        # RMSNorm cache and can force target kernels into eager execution.
        hf.nanovllm_rmsnorm_backend = "triton"
        dtype_before = torch.get_default_dtype()
        device_before = torch.get_default_device()
        try:
            torch.set_default_dtype(hf.dtype)
            torch.set_default_device("cuda")
            self.model = Qwen3ForCausalLM(hf).eval()
            loading_started = perf_counter()
            load_model(self.model, config.draft_model,
                quant_backend=getattr(config, "draft_quant_backend", "triton"),
                marlin_library=getattr(config, "marlin_library", None))
            torch.cuda.synchronize()
            self.weight_load_seconds = perf_counter() - loading_started
            blocks = (self.capacity + self.block_size - 1) // self.block_size
            head_dim = hf.head_dim or hf.hidden_size // hf.num_attention_heads
            self.kv = torch.empty(2, hf.num_hidden_layers, blocks, self.block_size,
                                  hf.num_key_value_heads, head_dim, dtype=hf.dtype)
            layer = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv[0, layer]
                    module.v_cache = self.kv[1, layer]
                    layer += 1
            self.block_table = torch.arange(blocks, dtype=torch.int32).unsqueeze(0)
            self.device = self.kv.device
            self.proposer = DraftProposer(self._forward, self.capacity, self.chunk_size)
        except BaseException:
            self.close()
            raise
        finally:
            torch.set_default_device(device_before)
            torch.set_default_dtype(dtype_before)
            reset_context()
        # Both weights and draft KV are resident before target warmup/budgeting.
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._forward([0] * self.chunk_size, 0)
            memory = torch.cuda.memory_stats(self.device)
            self.workspace_bytes = max(0, memory["allocated_bytes.all.peak"]
                                       - memory["allocated_bytes.all.current"])
            if self.execution == "graph":
                from nanovllm.engine.draft_graph import DraftTokenGraph, DraftSuffixGraph
                before = torch.cuda.memory_allocated(self.device)
                reserved_before = torch.cuda.memory_reserved(self.device)
                self.graph = DraftTokenGraph(self.model, self.block_table, self.capacity,
                                             self.graph_max_tokens, self.device,
                                             async_metadata=self.async_metadata)
                # After full acceptance plus a target bonus, two committed
                # inputs lack draft KV: the last candidate and the bonus.
                # Preserve eager Q2 shape; chunk_size=1 must retain Q1 semantics.
                if min(self.chunk_size, self.capacity) >= 2:
                    self.suffix_graphs[2] = DraftSuffixGraph(
                        self.model, self.block_table, self.capacity, 2, self.device,
                        async_metadata=self.async_metadata)
                if self.prefill_graph_enabled:
                    # No padding: each entry preserves the real eager Q width and K bound.
                    # Two static variants omit/keep the LM head for intermediate/final chunks.
                    # Bound startup work to four full chunks plus the common Q32 prompt.
                    specs = [(self.chunk_size, end, prediction)
                             for end in range(self.chunk_size, min(self.capacity, 4*self.chunk_size)+1, self.chunk_size)
                             for prediction in (False, True)]
                    # A 32-token prompt reaches draft after target's first token: Q33.
                    for width in (32, 33):
                        if width <= min(self.capacity, self.chunk_size):
                            specs.append((width, width, True))
                    if self.prefill_tail_graph_enabled:
                        # 例如 Q769 = 256+256+256+1。最后 Q1 仍须保持真实
                        # prefill Q/K；不改成 padding 或借用其他 K 上界的 decode 图。
                        specs.extend((1, count*self.chunk_size+1, True)
                                     for count in range(1,5)
                                     if count*self.chunk_size+1 <= self.capacity)
                    pool = None
                    for width, end, prediction in dict.fromkeys(specs):
                        graph = DraftSuffixGraph(self.model, self.block_table, self.capacity, width,
                            self.device, async_metadata=self.async_metadata, max_key_length=end,
                            pool=pool, needs_prediction=prediction)
                        if pool is None:
                            pool = graph.graph.pool()
                        self.prefill_graphs[width, end, prediction] = graph
                self.graph_resident_bytes = max(0, torch.cuda.memory_allocated(self.device) - before)
                self.graph_reserved_bytes = max(0, torch.cuda.memory_reserved(self.device) - reserved_before)
                self.proposer.rollout = self._rollout
                memory = torch.cuda.memory_stats(self.device)
                self.workspace_bytes = max(self.workspace_bytes,
                    memory["allocated_bytes.all.peak"] - memory["allocated_bytes.all.current"])
        except BaseException:
            self.close()
            raise
        self.counters.clear()  # Startup warmup is excluded from request counters.
        self.proposer.clear()
        if getattr(config, "enable_stochastic_speculation", False):
            from nanovllm.engine.stochastic_draft import StochasticDraft
            try:
                self.stochastic = StochasticDraft(self, use_graph=self.execution == "graph",
                                                 max_tokens=min(self.graph_max_tokens, 8))
            except BaseException:
                self.close()
                raise

    @torch.inference_mode()
    def _forward(self, tokens, start):
        started = perf_counter()
        try:
            result = int(self._forward_device(tokens, start).item())
            self.counters["host_result_syncs"] += 1
            return result
        finally:
            self.counters["forward_wall_ms"] += (perf_counter() - started) * 1000

    @torch.inference_mode()
    def _forward_device(self, tokens, start, *, needs_prediction=True):
        end = start + len(tokens)
        if not tokens or start < 0 or end > self.capacity:
            raise ValueError("draft forward is outside its KV capacity")
        try:
            ids = torch.tensor(tokens, device=self.device, dtype=torch.int64)
            positions = torch.arange(start, end, device=self.device, dtype=torch.int64)
            cu_q = torch.tensor([0, len(tokens)], device=self.device, dtype=torch.int32)
            cu_k = torch.tensor([0, end], device=self.device, dtype=torch.int32)
            set_context(True, cu_q, cu_k, len(tokens), end,
                        positions.to(torch.int32), block_tables=self.block_table)
            hidden = self.model(ids, positions)
            # 中间前缀块只需写入 KV；其 next-token 预测会被下一块覆盖。
            # 最后一块保留完全相同的 LM head/argmax 与 query 形状。
            if needs_prediction:
                logits = self.model.compute_logits(hidden)
                result = logits[-1].argmax()
                self.counters["eager_prediction_calls"] += 1
            else:
                result = None
                self.counters["skipped_prefix_prediction_calls"] += 1
            self.counters["forward_calls"] += 1
            self.counters["forwarded_tokens"] += len(tokens)
            self.counters["eager_forward_calls"] += 1
            return result
        finally:
            reset_context()

    @torch.inference_mode()
    def _rollout(self, suffix, start, limit, *, defer_host_result=False):
        started = perf_counter()
        try:
            if getattr(self, "prefill_graphs", None):
                # Shared prefill graph pool + private KV have a single execution stream.
                from nanovllm.layers.quant_graph import assert_graph_replay_stream
                assert_graph_replay_stream(self, self.device)
                if self.device.type == "cuda":
                    self.prefill_replay_stream = torch.cuda.current_stream(self.device)
            suffix_bucket = str(len(suffix)) if len(suffix) <= 2 else "long"
            self.counters[f"suffix_q{suffix_bucket}_rollouts"] += 1
            if limit > self.graph_max_tokens:
                self.counters["graph_limit_fallbacks"] += 1
                for offset in range(0, len(suffix), self.chunk_size):
                    prediction = self._forward(suffix[offset:offset + self.chunk_size], start + offset)
                result = [prediction]
                for i in range(limit - 1):
                    result.append(self._forward([result[-1]], start + len(suffix) + i))
                return result
            if len(suffix) == 1:
                result = self.graph.run(suffix[0], start, limit)
                replays = limit
            else:
                suffix_graph = self.suffix_graphs.get(len(suffix))
                if suffix_graph is not None:
                    prediction = suffix_graph.run(suffix, start)
                    self.counters["graph_prefill_replays"] += 1
                    self.counters["graph_replays"] += 1
                    self.counters["graph_metadata_h2d_copies"] += 1
                    self.counters["forward_calls"] += 1
                    self.counters["forwarded_tokens"] += len(suffix)
                else:
                    for offset in range(0, len(suffix), self.chunk_size):
                        chunk = suffix[offset:offset + self.chunk_size]
                        final = offset + self.chunk_size >= len(suffix)
                        graph = getattr(self, "prefill_graphs", {}).get((len(chunk), start+offset+len(chunk), final))
                        if graph is None:
                            prediction = self._forward_device(chunk, start+offset, needs_prediction=final)
                            self.counters["eager_suffix_forward_calls"] += 1
                            self.counters["prefill_graph_fallbacks"] += 1
                        else:
                            prediction = graph.run(chunk, start+offset)
                            self.counters["prefill_graph_replays"] += 1
                            if final and len(chunk)==1:
                                self.counters["prefill_tail_graph_replays"] += 1
                            self.counters["graph_replays"] += 1
                            self.counters["graph_metadata_h2d_copies"] += 1
                            self.counters["forward_calls"] += 1
                            self.counters["forwarded_tokens"] += len(chunk)
                            if not final:
                                self.counters["skipped_prefix_prediction_calls"] += 1
                replays = limit - 1
                if replays:
                    # graph.run mutates its input buffer, so preserve the first
                    # prediction tensor separately until the final concatenation.
                    tail = self.graph.run(prediction, start + len(suffix), replays)
                    result = torch.cat((prediction.reshape(1), tail))
                else:
                    result = prediction.reshape(1)
            self.counters["graph_replays"] += replays
            self.counters["graph_token_replays"] += replays
            self.counters["graph_metadata_h2d_copies"] += bool(replays)
            self.counters["forward_calls"] += replays
            self.counters["forwarded_tokens"] += replays
            if defer_host_result:
                self.counters["device_proposal_enqueues"] += 1
                return result
            candidates = result.cpu().tolist()
            self.counters["host_result_syncs"] += 1
            self.counters["graph_rollouts"] += 1
            return candidates
        finally:
            # Legacy includes final D2H; deferred mode records enqueue only, with
            # the completion wait explicitly recorded by device verification.
            self.counters["rollout_wall_ms"] += (perf_counter() - started) * 1000
            reset_context()

    @torch.inference_mode()
    def begin_device_proposal(self, request_id, tokens, limit):
        from nanovllm.engine.device_draft import PendingDeviceDraft
        if self._pending_device_proposal is not None or self._device_work_stream is not None:
            raise RuntimeError("a device proposal is already pending")
        if self.execution!="graph" or not self.async_metadata or type(limit) is not int or not 1<=limit<=self.graph_max_tokens:
            raise ValueError("device proposal requires supported graph K")
        if not tokens or any(type(t) is not int or t<0 for t in tokens):
            raise ValueError("device proposal requires integer committed history")
        if getattr(self,"stochastic",None) is not None:
            raise ValueError("device proposal is greedy only")
        if len(tokens)+limit-1>self.capacity:
            raise ValueError("device proposal exceeds private KV capacity")
        if self.request_id!=request_id:
            self.proposer.clear();self.request_id=request_id
        self.proposer.synchronize(tokens)
        prefix=len(self.proposer.cached_tokens)
        self._device_work_stream=torch.cuda.current_stream(self.device)
        try:
            ids=self._rollout(tokens[prefix:],prefix,limit,defer_host_result=True)
            result=PendingDeviceDraft(request_id,tuple(tokens),prefix,limit,ids,self._device_work_stream)
            self._pending_device_proposal=result
            return result
        except BaseException:
            self.abort_device_proposal()
            raise

    def finish_device_proposal(self, pending, tokens):
        if self._pending_device_proposal is not pending or pending.request_id!=self.request_id:
            raise RuntimeError("GPU proposal provenance differs from owner")
        if len(tokens)!=pending.count or any(type(t) is not int or t<0 for t in tokens):
            raise ValueError("materialized proposal has wrong tokens")
        # The final D2H in verification has completed before this publication.
        self.proposer.cached_tokens[:]=list(pending.history)+list(tokens[:-1])
        suffix=len(pending.history)-pending.prefix
        self.proposer.counters["forward_calls"]+=(suffix+self.chunk_size-1)//self.chunk_size+pending.count-1
        self.proposer.counters["forwarded_tokens"]+=suffix+pending.count-1
        self.proposer.counters["proposed_tokens"]+=pending.count
        self.proposer.counters["rollout_calls"]+=1
        self.counters["device_proposal_completions"]+=1
        self._pending_device_proposal=None
        self._device_work_stream=None

    def abort_device_proposal(self):
        stream=getattr(self,"_device_work_stream",None)
        try:
            if stream is not None:stream.synchronize()
        finally:
            self._pending_device_proposal=None
            self._device_work_stream=None
            if self.proposer is not None:self.proposer.clear()

    def propose(self, request_id, tokens, limit):
        if getattr(self,"_pending_device_proposal",None) is not None:
            raise RuntimeError("device proposal must finish before another proposal")
        if getattr(self, "stochastic", None) is not None and self.stochastic.request_id is not None:
            if self.stochastic.request_id == request_id:
                raise ValueError("cannot switch a live request from rejection sampling to greedy")
            self.stochastic.clear()
        if self.request_id != request_id:
            self.proposer.clear()
            self.request_id = request_id
        return self.proposer.propose(tokens, limit)

    def propose_probabilities(self, request_id, tokens, limit, temperature, seed):
        if self.stochastic is None:
            raise ValueError("stochastic draft was not enabled at engine construction")
        started = perf_counter()
        before = self.stochastic.counters.copy()
        try:
            return self.stochastic.propose(tokens, min(limit, self.stochastic.max_tokens),
                                           temperature, seed, request_id)
        finally:
            self.counters["probability_rollouts"] += 1
            self.counters["probability_rollout_wall_ms"] += (perf_counter() - started) * 1000
            for name in ("forward_calls", "forwarded_tokens"):
                self.counters[name] += self.stochastic.counters[name] - before[name]
            self.counters["host_result_syncs"] += (self.stochastic.counters["candidate_host_transfers"]
                                                   - before["candidate_host_transfers"])

    def synchronize(self, request_id, tokens):
        if self.request_id == request_id:
            self.proposer.synchronize(tokens)

    def forget(self, request_id):
        if self.request_id == request_id:
            if getattr(self,"_device_work_stream",None) is not None:self.abort_device_proposal()
            if getattr(self, "stochastic", None) is not None:
                self.stochastic.clear()
            self.proposer.clear()
            self.request_id = None

    def stats(self):
        from nanovllm.utils.weight_memory import weight_storage_bytes, quantization_stats
        return dict(self.counters, proposer=dict(self.proposer.counters),
                    stochastic=self.stochastic.stats() if self.stochastic is not None else {},
                    cached_tokens=len(self.proposer.cached_tokens),
                    kv_bytes=self.kv.numel() * self.kv.element_size(),
                    parameter_bytes=weight_storage_bytes(self.model), workspace_bytes=self.workspace_bytes,
                    quantization=quantization_stats(self.model),
                    weight_load_seconds=self.weight_load_seconds,
                    rmsnorm_backend="triton",
                    graph_resident_bytes=self.graph_resident_bytes,
                    graph_reserved_bytes=self.graph_reserved_bytes,
                    graph_entries=int(self.graph is not None) + len(self.suffix_graphs) + len(getattr(self, "prefill_graphs", {})),
                    draft_prefill_graph=getattr(self, "prefill_graph_enabled", False),
                    draft_prefill_tail_graph=getattr(self, "prefill_tail_graph_enabled", False),
                    prefill_graph_keys=sorted(getattr(self, "prefill_graphs", {})),
                    graph_suffix_widths=sorted(self.suffix_graphs),
                    graph_max_tokens=self.graph_max_tokens,
                    draft_async_metadata=getattr(self, "async_metadata", False),
                    async_metadata_host_waits=sum(getattr(graph, "metadata_host_waits", 0)
                        for graph in (self.graph, *self.suffix_graphs.values()) if graph is not None),
                    requested_execution=self.requested_execution,
                    rollout_timing_scope=("device enqueue only; final GPU wait is in verification" if self.counters.get("device_proposal_enqueues",0)
                        else "legacy rollout includes final D2H wait"),
                    device_proposal_pending=getattr(self,"_pending_device_proposal",None) is not None,
                    execution=self.execution + "_tp1_native_qwen3")

    def close(self):
        if getattr(self,"_device_work_stream",None) is not None:self.abort_device_proposal()
        stream = getattr(self, "prefill_replay_stream", None)
        if stream is not None:
            # Also protects exceptional cleanup before the rollout's final D2H.
            stream.synchronize()
            self.prefill_replay_stream = None
        for graph in getattr(self, "prefill_graphs", {}).values():
            graph.close()
        self.prefill_graphs = {}
        if getattr(self, "stochastic", None) is not None:
            self.stochastic.close()
            self.stochastic = None
        for graph in getattr(self, "suffix_graphs", {}).values():
            graph.close()
        self.suffix_graphs = {}
        if getattr(self, "graph", None) is not None:
            self.graph.close()
            self.graph = None
        if self.proposer is not None:
            self.proposer.clear()
        self.request_id = None
        self.model = None
        self.kv = None
        self.block_table = None
        reset_context()
