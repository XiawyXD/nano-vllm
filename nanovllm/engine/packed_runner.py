"""把显式执行计划打包为一次 transformer forward 的输入。

CPU 计划是不可变快照，不要求 TP worker 拥有完整 Sequence 历史。
P/V 放前、D 放后形成连续 Attention 子区间，logits 行映射随同一次重排更新。
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from nanovllm.utils.context import AttentionGroup, get_context, set_context, reset_context


@dataclass(frozen=True)
class PreparedPlan:
    """GPU 输入以及 logits 行对应的请求 ID；这里只保存有界的本轮元数据。"""
    input_ids: torch.Tensor
    positions: torch.Tensor
    row_request_ids: tuple[int, ...]
    ordered_items: tuple
    native_query: int | None = None


@dataclass(frozen=True)
class StochasticVerification:
    """已校正的真实输出，不能伪装成每行一个 greedy argmax。"""
    tokens: tuple[int, ...]
    accepted: int


def role_name(item):
    """兼容 str Enum，避免序列化后依赖枚举对象 identity。"""
    return getattr(item.role, "value", item.role).upper()


def prepare_plan(runner, plan):
    """按真实 query 打包 token、位置、KV slot 和 Attention 子组。"""
    config = runner.config
    items = tuple(sorted(plan.items, key=lambda item: role_name(item) == "DECODE"))
    if not items:
        raise ValueError("cannot execute an empty plan")
    total = sum(len(item.input_token_ids) for item in items)
    if total > config.max_num_batched_tokens or len(items) > config.max_num_seqs:
        raise ValueError("execution plan exceeds configured token/request budget")
    page_capacity = (config.max_model_len + runner.block_size - 1) // runner.block_size
    pool = runner.metadata_buffers
    pool.begin()
    try:
        ids, positions, slots, logit_rows, owners = [], [], [], [], []
        offsets = [0]
        for item in items:
            query = len(item.input_token_ids)
            if query <= 0 or item.start_pos < 0 or item.start_pos + query > config.max_model_len:
                raise ValueError("invalid query range in execution plan")
            end = item.start_pos + query
            if len(item.block_table) * runner.block_size < end:
                raise ValueError("execution plan has insufficient resident KV pages")
            resident_pages = (end + runner.block_size - 1) // runner.block_size
            # Attention还会读取历史KV；只检查本轮写入slot不能防住坏的历史页。
            if any(isinstance(page, bool) or not isinstance(page, int)
                   or not 0 <= page < config.num_kvcache_blocks
                   for page in item.block_table[:resident_pages]):
                raise ValueError("execution plan contains an invalid resident physical page")
            ids.extend(item.input_token_ids)
            positions.extend(range(item.start_pos, end))
            for position in range(item.start_pos, end):
                page = item.block_table[position // runner.block_size]
                slots.append(page * runner.block_size + position % runner.block_size)
            for relative in item.logit_positions:
                if not 0 <= relative < query:
                    raise ValueError("logits row is outside request query")
                logit_rows.append(offsets[-1] + relative)
                owners.append(item.request_id)
            offsets.append(offsets[-1] + query)
        native = getattr(runner, "native_verification_graphs", None)
        if (native is not None and getattr(native, "packed_metadata", False)
                and len(items)==1 and role_name(items[0])=="VERIFY"
                and tuple(items[0].logit_positions)==tuple(range(total))):
            item = items[0]
            if len(item.block_table)>page_capacity:
                raise ValueError("page table exceeds configured max_model_len")
            if native.prepare_cpu(total, ids, positions, slots, item.block_table, item.start_pos+total):
                _, device_ids, device_pos, static, _ = native.entries[total]
                native._context(static)
                return PreparedPlan(device_ids, device_pos, tuple(owners), items, native_query=total)
        token_cap = config.max_num_batched_tokens
        input_ids = pool.tensor("input_ids", ids, torch.int64, max_elements=token_cap)
        pos = pool.tensor("positions", positions, torch.int64, max_elements=token_cap)
        slot_mapping = pool.tensor("slot_mapping", slots, torch.int32, max_elements=token_cap)
        indices = pool.tensor("logits_indices", logit_rows, torch.int64, max_elements=token_cap)
        all_decode = all(role_name(item) == "DECODE" for item in items)
        # 即使 mixed backend 选择 varlen，纯 Decode 也使用专用 paged-decode
        # kernel。只有一轮里真的同时出现 Prefill/Verify 与 Decode 时，
        # varlen 模式才把所有 query 放在同一 Attention 组中，保持统一数值路径。
        if all_decode:
            partitions = [(0, len(items), "decode")]
        elif config.mixed_attention_backend == "varlen":
            partitions = [(0, len(items), "varlen")]
        else:
            split = next((i for i, item in enumerate(items) if role_name(item) == "DECODE"), len(items))
            partitions = []
            if split:
                partitions.append((0, split, "varlen"))
            if split < len(items):
                partitions.append((split, len(items), "decode"))
        groups = []
        for group_id, (begin, end, kind) in enumerate(partitions):
            group_items = items[begin:end]
            # Fresh full prefills have no historical KV: attend directly over this
            # call's contiguous Q/K/V, while store_kvcache still commits the same
            # K/V tensors to pages before Attention. This matches legacy Prefill's
            # numerical path. Partial/prefix-cached prefills must read paged history.
            current_prefill = (kind == "varlen" and group_items
                and all(role_name(item) == "PREFILL" and item.start_pos == 0
                        for item in group_items))
            if current_prefill:
                kind = "varlen_current"
            columns = max(len(item.block_table) for item in group_items)
            if columns > page_capacity:
                raise ValueError("page table exceeds configured max_model_len")
            tables = [list(item.block_table) + [-1] * (columns - len(item.block_table))
                      for item in group_items]
            table = pool.tensor(f"pages_{group_id}", tables, torch.int32,
                shape=(len(group_items), columns), max_elements=config.max_num_seqs * page_capacity)
            lengths = [item.start_pos + len(item.input_token_ids) for item in group_items]
            if kind == "decode":
                if any(len(item.input_token_ids) != 1 for item in group_items):
                    raise ValueError("decode attention requires one query per request")
                lens = pool.tensor(f"lengths_{group_id}", lengths, torch.int32,
                                   max_elements=config.max_num_seqs)
                group = AttentionGroup(kind, offsets[begin], offsets[end], table, context_lens=lens)
            else:
                cu_q, cu_k = [0], [0]
                for item, length in zip(group_items, lengths):
                    cu_q.append(cu_q[-1] + len(item.input_token_ids))
                    cu_k.append(cu_k[-1] + length)
                q = pool.tensor(f"cu_q_{group_id}", cu_q, torch.int32, max_elements=config.max_num_seqs + 1)
                k = pool.tensor(f"cu_k_{group_id}", cu_k, torch.int32, max_elements=config.max_num_seqs + 1)
                # Current-KV groups mirror legacy varlen metadata exactly; paged
                # groups retain a rounded scheduling bound, with cu_k the true length.
                max_k = (max(lengths) if kind == "varlen_current"
                         else ((max(lengths) + 127) // 128) * 128)
                group = AttentionGroup(kind, offsets[begin], offsets[end], table, q, k,
                                       max(len(item.input_token_ids) for item in group_items), max_k)
            groups.append(group)
        if all_decode:
            # 纯 D 无论候选混合 backend 如何配置，都复用原 Decode 路径和
            # CUDA Graph；LM head 的所有输入行都有效，不额外 index_select。
            group = groups[0]
            set_context(False, slot_mapping=slot_mapping, context_lens=group.context_lens,
                        block_tables=group.block_tables, batch_kind="D_NORMAL")
        else:
            set_context(True, slot_mapping=slot_mapping, logits_indices=indices,
                        attention_groups=tuple(groups), batch_kind=plan.kind)
        return PreparedPlan(input_ids, pos, tuple(owners), items)
    finally:
        pool.end()


@torch.inference_mode()
def run_plan(runner, plan, sequences, stochastic_proposals=None):
    """执行并一次性回传所需 token ID；返回 request_id -> 有序预测 token 列表。"""
    try:
        started = perf_counter()
        if plan.kind == "D_NORMAL":
            # 纯 Decode 不需要构建 packed AttentionGroup、logits 行索引与
            # 多 query 页表。仍以不可变 plan 校验传入的 Sequence 快照，
            # 然后复用已经验证过的普通 Decode 准备和 CUDA Graph 路径。
            if not plan.items or len(sequences) != len(plan.items):
                raise ValueError("decode sequence count differs from execution plan")
            if (len(plan.items) > runner.config.max_num_seqs
                    or len(plan.items) > runner.config.max_num_batched_tokens):
                raise ValueError("decode plan exceeds configured request budget")
            for item, seq in zip(plan.items, sequences):
                if (item.end_pos > runner.config.max_model_len
                        or len(item.block_table) * runner.block_size < item.end_pos
                        or any(type(page) is not int or not 0 <= page < runner.config.num_kvcache_blocks
                               for page in item.block_table[:(item.end_pos + runner.block_size - 1)
                                                            // runner.block_size])):
                    raise ValueError("decode plan has invalid resident KV pages or length")
                if (role_name(item) != "DECODE" or item.logit_positions != (0,)
                        or (runner.rank == 0 and seq.seq_id != item.request_id)
                        or seq.last_token != item.input_token_ids[0]
                        or len(seq) - 1 != item.start_pos
                        or tuple(seq.block_table) != item.block_table):
                    raise ValueError("decode sequence changed after execution plan reservation")
            input_ids, positions = runner.prepare_decode(sequences)
            prepared_at = perf_counter()
            logits = runner.run_model(input_ids, positions, False)
            enqueued_at = perf_counter()
            runner.last_plan_stats = dict(
                path=runner.last_model_path, captured=False,
                metadata_grew=runner.metadata_buffers.grew,
                query_tokens=len(sequences), forward_calls=1,
                host_prepare_ms=(prepared_at - started) * 1000,
                model_enqueue_ms=(enqueued_at - prepared_at) * 1000,
            )
            if runner.rank != 0:
                return None
            tokens = runner.sampler(logits, sequences).tolist()
            if len(tokens) != len(plan.items):
                raise RuntimeError("decode sampler returned wrong token count")
            runner.last_plan_stats["sample_and_wait_ms"] = (perf_counter() - enqueued_at) * 1000
            return {item.request_id: [int(token)] for item, token in zip(plan.items, tokens)}
        prepared = prepare_plan(runner, plan)
        prepared_at = perf_counter()
        context = get_context()
        captured = False
        native_graphs = getattr(runner, "native_verification_graphs", None)
        native_logits = None
        if prepared.native_query is not None:
            native_logits = native_graphs.replay_prepared(prepared.native_query)
        elif (native_graphs is not None and len(prepared.ordered_items) == 1
                and role_name(prepared.ordered_items[0]) == "VERIFY"
                and len(prepared.row_request_ids) == prepared.input_ids.numel()):
            native_logits = native_graphs.run(prepared.input_ids, prepared.positions, context)
        if native_logits is not None:
            logits = native_logits
            path = "native_verification_graph"
        elif context.batch_kind == "D_NORMAL" and context.attention_groups is None:
            logits = runner.run_model(prepared.input_ids, prepared.positions, False)
            path = runner.last_model_path
        else:
            hidden = None
            if runner.config.enable_mixed_graph and not runner.enforce_eager:
                signature = (runner.config.mixed_attention_backend, runner.config.decode_attention_backend,
                             runner.config.rmsnorm_backend, runner.config.mlp_backend,
                             runner.config.mlp_min_tokens, str(next(runner.model.parameters()).dtype))
                hidden = runner.packed_graphs.run(runner.model, prepared.input_ids,
                                                prepared.positions, context, signature)
                captured = runner.packed_graphs.captured_last
            if hidden is None:
                hidden = runner.model(prepared.input_ids, prepared.positions)
                path = "packed_eager"
            else:
                path = "packed_graph"
            logits = runner.model.compute_logits(hidden) if prepared.row_request_ids else None
        runner.last_plan_stats = dict(path=path, captured=captured,
                                     metadata_grew=runner.metadata_buffers.grew,
                                     query_tokens=prepared.input_ids.numel(), forward_calls=1,
                                     host_prepare_ms=(prepared_at - started) * 1000)
        enqueued_at = perf_counter()
        runner.last_plan_stats["model_enqueue_ms"] = (enqueued_at - prepared_at) * 1000
        if runner.rank != 0:
            return None
        results = {item.request_id: [] for item in prepared.ordered_items}
        if prepared.row_request_ids:
            seq_map = {seq.seq_id: seq for seq in sequences}
            owners = [seq_map[request_id] for request_id in prepared.row_request_ids]
            stochastic = [item for item in prepared.ordered_items if role_name(item) == "VERIFY"
                          and getattr(seq_map[item.request_id], "speculative_sampling", "greedy") == "rejection"]
            if stochastic:
                if len(prepared.ordered_items) != 1 or runner.world_size != 1:
                    raise ValueError("probabilistic verification requires exactly one TP1 request")
                item = stochastic[0]
                seq = seq_map[item.request_id]
                proposal = (stochastic_proposals or {}).get(item.request_id)
                k = len(item.draft)
                if (proposal is None or tuple(proposal.token_ids[:k]) != item.draft
                        or logits.shape[0] != k + 1 or len(prepared.row_request_ids) != k + 1):
                    raise ValueError("proposal probability provenance or target row layout differs from plan")
                from nanovllm.engine.speculative_sampling import temperature_probabilities, speculative_correction
                p = temperature_probabilities(logits, seq.temperature, top_k=seq.top_k,
                    top_p=seq.top_p, repetition_penalty=seq.repetition_penalty)
                tokens, accepted = speculative_correction(proposal.ids[:k], proposal.probabilities[:k], p,
                    generators=runner.draft_runner.stochastic.generators).to_host()
                results[item.request_id] = StochasticVerification(tuple(tokens), accepted)
                runner.graph_counters["probability_corrections"] += 1
                runner.graph_counters["probability_result_transfers"] += 1
            else:
                predictions = runner.sampler(logits, owners).tolist()
                for request_id, token in zip(prepared.row_request_ids, predictions):
                    results[request_id].append(token)
        # enqueue时间并不是kernel耗时；tolist通常包含GPU完成等待，分别记录。
        runner.last_plan_stats["sample_and_wait_ms"] = (perf_counter() - enqueued_at) * 1000
        return results
    finally:
        reset_context()
