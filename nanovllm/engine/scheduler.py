"""请求调度：在 waiting/prefill 和 running/decode 之间分配 GPU 资源。"""

from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """决定每轮模型 forward 处理哪些 Sequence。

    `waiting` 保存还没有完成 prompt prefill 的请求；`running` 保存已经
    可以进行 decode 的请求。返回的 `scheduled_seqs` 只是本轮 batch，
    并不等于整个 waiting 或 running 队列。
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        # prefill 时限制一个 batch 的 token 总数，避免显存峰值过大。
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 可选的单请求 chunk 上限，用于降低长 prefill 对 decode 的阻塞。
        self.prefill_chunk_size = config.prefill_chunk_size
        # 可选的 prefill/decode 轮转策略；0 表示兼容旧策略。
        self.max_prefill_steps = config.max_prefill_steps
        self.prefill_steps_since_decode = 0
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            enable_prefix_caching=config.enable_prefix_caching,
        )
        # deque 便于从队首取出最早请求、从队尾抢占低优先级请求。
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # 轻量调度统计，供 benchmark 和 README 报告使用。
        self.total_prefill_steps = 0
        self.total_decode_steps = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0
        self.total_requests_submitted = 0
        self.total_finished_requests = 0
        self.total_preemptions = 0
        self.latency_policy = None
        if getattr(config, "latency_aware_scheduling", False):
            from nanovllm.engine.latency_policy import LatencyPolicy
            import json
            from pathlib import Path
            path=getattr(config,"scheduler_cost_model_path",None)
            model=json.loads(Path(path).read_text()) if path else None
            self.latency_policy = LatencyPolicy(config.scheduler_ttft_target_ms,
                config.scheduler_tpot_target_ms, config.scheduler_min_prefill_tokens, model)

    def is_finished(self):
        # 两个队列都为空才表示所有请求都结束。
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 新请求先进入 waiting，必须完成 prompt prefill 才能 decode。
        self.waiting.append(seq)
        if self.latency_policy is not None:
            self.latency_policy.add(seq.seq_id, perf_counter())
        self.total_requests_submitted += 1

    def cancel(self, seq: Sequence) -> bool:
        """从 waiting/running 中移除一条请求并释放它的 KV block。

        HTTP 取消请求发生在两个调度 step 之间，因此不会和
        ``schedule``/``postprocess`` 同时修改队列。返回值表示请求是否仍
        在调度器中；已经完成的请求不会重复释放 cache。
        """
        removed = False
        if getattr(getattr(self.block_manager,"host_cache",None),"asynchronous",False):
            self.block_manager.reap_host_loads(blocking=True)
        if self.latency_policy is not None:
            self.latency_policy.forget(seq.seq_id)
        for queue in (self.waiting, self.running):
            try:
                queue.remove(seq)
                removed = True
            except ValueError:
                pass
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        seq.finish_reason = "cancelled"
        seq.status = SequenceStatus.CANCELLED
        return removed

    def schedule(self) -> tuple[list[Sequence], bool]:
        """选择本轮 batch，并返回 `(scheduled_seqs, is_prefill)`。"""
        scheduled_seqs = []
        cache=getattr(self.block_manager,'host_cache',None)
        if getattr(cache,'asynchronous',False) and (not getattr(cache,'idle_fastpath',False)
                or cache.needs_reap or self.block_manager._host_load_leases):
            self.block_manager.reap_host_loads()
        num_batched_tokens = 0
        token_budget = self.max_num_batched_tokens
        latency_force_decode = False
        if self.latency_policy is not None:
            latency_force_decode, token_budget = self.latency_policy.choose(
                self.waiting, self.running, token_budget, perf_counter())

        # ------------------------------ prefill ------------------------------
        # 当两类请求都存在且达到 prefill 配额时，跳过本轮 prefill，先服务 decode。
        force_decode = (
            self.waiting
            and self.running
            and self.max_prefill_steps > 0
            and self.prefill_steps_since_decode >= self.max_prefill_steps
        )
        force_decode = force_decode or latency_force_decode
        if not force_decode:
            while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
                # FIFO：优先处理 waiting 队首请求。
                seq = self.waiting[0]
                remaining = token_budget - num_batched_tokens
                if remaining == 0:
                    break
                if not seq.block_table:
                    # 第一次看到该 Sequence：先尝试 prefix cache 命中并计算资源需求。
                    self.block_manager.restore_host_prefix(seq)
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        # KV Cache 不够，暂时不能继续 prefill。
                        break
                    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                else:
                    # chunked prefill：block_table 已存在，只处理尚未缓存的 token。
                    num_tokens = seq.num_tokens - seq.num_cached_tokens
                # 单条请求的 chunk 上限只限制本轮工作量，不改变逻辑序列长度。
                request_limit = self.prefill_chunk_size or num_tokens
                request_tokens = min(num_tokens, request_limit)
                if remaining < request_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                    # 为避免 batch 内出现多个不同 chunk，只有本轮第一个请求允许被切块。
                    break
                if not seq.block_table:
                    # 真正分配/复用物理 block，并写入 seq.block_table。
                    self.block_manager.allocate(seq, num_cached_blocks)
                # 本轮实际交给模型的 prompt token 数。
                seq.num_scheduled_tokens = min(request_tokens, remaining)
                num_batched_tokens += seq.num_scheduled_tokens
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    # 本轮会处理完整个 prompt：后续可以进入 decode 队列。
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                scheduled_seqs.append(seq)
                # 一个 Sequence 在同一个 step 中只能出现一次；未完成的 chunk
                # 留到下一轮，避免重复计算或突破 token budget。
                if seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens:
                    break

        if scheduled_seqs:
            # 只要成功安排了 prefill，就先执行 prefill，不混入 decode。
            self.prefill_steps_since_decode += 1
            self.total_prefill_steps += 1
            self.total_prefill_tokens += num_batched_tokens
            return scheduled_seqs, True

        # ------------------------------- decode -------------------------------
        while self.running and len(scheduled_seqs) < min(self.max_num_seqs, self.max_num_batched_tokens):
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 没有新 block 时优先抢占 running 队尾请求。
                    self.preempt(self.running.pop())
                else:
                    # 连当前请求都无法容纳，只能把它自己抢占回 waiting。
                    self.preempt(seq)
                    break
            else:
                # decode 每条 Sequence 每轮只输入/生成一个 token。
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 本轮请求放到队尾；max_num_seqs/token budget 较小时避免队首长期独占。
        self.running.extend(scheduled_seqs)
        self.prefill_steps_since_decode = 0
        self.total_decode_steps += 1
        self.total_decode_tokens += len(scheduled_seqs)
        return scheduled_seqs, False

    def stats(self) -> dict[str, int]:
        """返回可序列化的调度统计，便于 benchmark 记录。"""
        stats = {
            "prefill_steps": self.total_prefill_steps,
            "decode_steps": self.total_decode_steps,
            "prefill_tokens": self.total_prefill_tokens,
            "decode_tokens": self.total_decode_tokens,
            "requests_submitted": self.total_requests_submitted,
            "finished_requests": self.total_finished_requests,
            "preemptions": self.total_preemptions,
        }
        stats.update(self.block_manager.stats())
        if self.latency_policy is not None:
            stats["latency_policy"] = self.latency_policy.stats()
        return stats

    def preempt(self, seq: Sequence):
        """释放一个 running 请求的 block，并让它稍后重新 prefill。"""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.total_preemptions += 1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """消费模型输出并提交本轮调度结果。

        `token_ids[i]` 是 `seqs[i]` 本轮采样出的下一个 token；这里不再
        做 Transformer 计算，而是更新 cache 计数、序列内容和生命周期。
        """
        for seq, token_id in zip(seqs, token_ids):
            # 先把本轮新写满的完整 block 放入 prefix cache 索引。
            self.block_manager.hash_blocks(seq)
            # scheduler 之前计划的 token 现在已经真正进入 KV Cache。
            seq.num_cached_tokens += seq.num_scheduled_tokens
            # 本轮计划已结算，下一轮重新计算。
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill 尚未覆盖完整 prompt；当前 logits 不能用来
                # 生成 completion，因此不追加 token，下一轮继续处理 prompt。
                continue
            # prompt 完整后（或 decode 阶段）提交本轮生成的 token。
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # EOS 或达到上限：标记完成并释放它占用的所有物理 block。
                seq.finish_reason = "stop" if (not seq.ignore_eos and token_id == self.eos) else "length"
                seq.status = SequenceStatus.FINISHED
                self.total_finished_requests += 1
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
