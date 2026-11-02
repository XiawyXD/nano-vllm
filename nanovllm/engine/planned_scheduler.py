"""可选混合执行 planner：渐进 KV、decode 优先、有限 prefill 饥饿保护。

旧 Scheduler 保持不变。显式 mixed 模式使用此类；一个 plan 必须全部 commit
或 rollback 后才能生成下一轮。所有方法只处理 CPU 元数据，不启动 GPU。
"""
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.execution_plan import ExecutionItem, ExecutionPlan, ExecutionRole, StepStats


class KVProgressBlocked(RuntimeError):
    """No CPU reservation can proceed; engine may first finish pending copies."""


class PlannedScheduler(Scheduler):
    """P+D 同 forward；没有 P 时才允许 D+V，防止 prompt 成本污染验证控制。"""

    def __init__(self, config):
        super().__init__(config)
        # The engine flushes backups before any GPU write. Allocation while
        # constructing a plan must never submit a device copy itself.
        self.block_manager.defer_host_stores = True
        self.max_model_len = config.max_model_len
        self.prefill_max_wait_steps = getattr(config, "prefill_max_wait_steps", 8)
        self.mixed_min_active_decodes = getattr(config, "mixed_min_active_decodes", 0)
        self.mixed_low_load_mode = getattr(config, "mixed_low_load_mode", "separate")
        self._step_id = 0
        self._prefill_wait_rounds = 0
        self._waiting_ages = {}
        self._active_plan = None
        self._active_items = {}
        self._plan_sequences = {}
        self._reservations = {}
        self.last_step_stats = StepStats()
        self.total_mixed_steps = 0
        self.total_verification_tokens = 0
        self.last_preempted_ids = []

    def add(self, seq: Sequence):
        """即使渐进分配，也拒绝独占所有物理页仍装不下的请求。"""
        required = (len(seq) + seq.max_tokens - 1 + self.block_size - 1) // self.block_size
        if required > len(self.block_manager.blocks):
            raise ValueError("request exceeds total physical KV capacity")
        super().add(seq)

    def get_sequence(self, request_id: int) -> Sequence:
        """取得 CPU 请求对象；ExecutionItem 本身不引用该可变对象。"""
        if request_id in self._plan_sequences:
            return self._plan_sequences[request_id]
        for seq in (*self.waiting, *self.running):
            if seq.seq_id == request_id:
                return seq
        raise KeyError(request_id)

    def _preempt_candidate(self, protected: set[int]) -> bool:
        """释放最新的未选请求，给较老请求前进机会；绝不抢占本轮 pin。"""
        candidates = [seq for seq in (*self.waiting, *self.running)
                      if seq.block_table and seq.seq_id not in protected]
        if not candidates:
            return False
        victim = max(candidates, key=lambda seq: seq.seq_id)
        for queue in (self.waiting, self.running):
            if victim in queue:
                queue.remove(victim)
        self.block_manager.deallocate(victim)
        victim.num_scheduled_tokens = 0
        victim.is_prefill = True
        victim.status = SequenceStatus.WAITING
        self.waiting.append(victim)
        self.total_preemptions += 1
        self.last_preempted_ids.append(victim.seq_id)
        return True

    def _reserve(self, seq, end, prefix, pinned):
        """只从尚未进入本轮计划的请求回收页；失败不留下半次 reservation。"""
        # 低优先级新请求不能反复把更老请求的部分 prefill 抢掉再重算。
        protected = pinned | {seq.seq_id} | {
            other.seq_id for other in (*self.waiting, *self.running) if other.seq_id < seq.seq_id}
        while True:
            try:
                return self.block_manager.reserve_to(seq, end, prefix)
            except MemoryError:
                if not self._preempt_candidate(protected):
                    return None

    def _record(self, seq, reservation, role, inputs, *, draft=(), sample=False):
        self._reservations[seq.seq_id] = reservation
        self._plan_sequences[seq.seq_id] = seq
        seq.is_prefill = role is ExecutionRole.PREFILL
        positions = tuple(range(len(inputs))) if role is ExecutionRole.VERIFY else (
            (len(inputs) - 1,) if sample else ())
        item = ExecutionItem(seq.seq_id, role, tuple(inputs), reservation.start_pos,
                             tuple(seq.block_table), tuple(draft), positions)
        self._active_items[seq.seq_id] = item
        return item

    def _take_prefills(self, budget, slots, pinned, *, max_items=None, preserve_fifo=False):
        """waiting 默认轮转；实验 FIFO 让未完成的队首 chunk 阻止后续入队。"""
        selected = []
        candidates = tuple(self.waiting) if preserve_fifo else range(len(self.waiting))
        for candidate in candidates:
            if budget <= 0 or slots <= 0:
                break
            if preserve_fifo:
                seq = candidate
            else:
                seq = self.waiting.popleft()
                self.waiting.append(seq)
            if seq.seq_id in pinned or seq.is_finished:
                continue
            match = self.block_manager.probe_prefix(seq)
            start = seq.num_cached_tokens if seq.block_table else match.cached_tokens
            request_tokens = min(len(seq) - start, self.prefill_chunk_size or (len(seq) - start))
            if preserve_fifo and selected and request_tokens > budget:
                break
            count = min(request_tokens, budget)
            if count <= 0:
                raise RuntimeError("waiting request has no uncached input")
            # 若一个较小 chunk 能直接容纳，先缩小，不为批量大小去抢占其他请求。
            if self.block_manager.required_new_blocks(seq, start + count, match) > len(self.block_manager.free_block_ids):
                low, high, fitting = 1, count, 0
                while low <= high:
                    mid = (low + high) // 2
                    if self.block_manager.required_new_blocks(seq, start + mid, match) <= len(self.block_manager.free_block_ids):
                        fitting, low = mid, mid + 1
                    else:
                        high = mid - 1
                count = fitting or 1
            reservation = self._reserve(seq, start + count, match, pinned)
            if reservation is None:
                if preserve_fifo:
                    break
                continue
            inputs = seq.token_ids[start:start + count]
            item = self._record(seq, reservation, ExecutionRole.PREFILL, inputs,
                                sample=start + count == len(seq))
            selected.append(item)
            pinned.add(seq.seq_id)
            budget -= count
            slots -= 1
            if max_items is not None and len(selected) >= max_items:
                break
            if preserve_fifo and start + count < len(seq):
                # 对齐 legacy：长 prompt 这一轮未完成，后面的短 prompt
                # 不能抢先进入 Decode。它仍可在之后的调度轮中推进。
                break
        return selected

    def _take_decodes(self, budget, slots, pinned):
        """所有普通 D 基础槽位先占好，再允许 draft 使用剩余预算。"""
        selected = []
        attempts = len(self.running)
        while self.running and attempts and budget > 0 and slots > 0:
            attempts -= 1
            seq = self.running.popleft()
            self.running.append(seq)
            if seq.seq_id in pinned or seq.is_finished:
                continue
            if seq.num_cached_tokens != len(seq) - 1:
                raise RuntimeError("running request must have exactly one uncached token")
            reservation = self._reserve(seq, len(seq), None, pinned)
            if reservation is None:
                continue
            item = self._record(seq, reservation, ExecutionRole.DECODE,
                                (seq.last_token,), sample=True)
            selected.append(item)
            pinned.add(seq.seq_id)
            budget -= 1
            slots -= 1
        return selected

    def schedule_plan(self, draft_provider=None, allow_verification=True, *, verification_budget=128):
        """返回一个非空计划；全部 commit/rollback 前禁止开始下一轮。

        通常 decode-first。waiting 连续等 8 轮后，先 pin 一个最多 budget/4
        （至少 1 token）的 prefill chunk，再安排 D，避免仅保留数字预算却被
        已 pin 的 D 占满所有物理页。没有 P 时，剩余预算才可扩展为 V。
        """
        if self._active_items:
            raise RuntimeError("previous execution plan is still active")
        self.last_preempted_ids = []
        if self.is_finished():
            return ExecutionPlan((), self._step_id)
        self._plan_sequences = {}
        self._reservations = {}
        budget, slots, pinned = self.max_num_batched_tokens, self.max_num_seqs, set()
        items = []
        try:
            active_decodes = sum(not seq.is_finished for seq in self.running)
            separate_low_load = bool(
                self.waiting and self.running and self.mixed_min_active_decodes
                and active_decodes < self.mixed_min_active_decodes)
            fifo_low_load = bool(
                self.waiting and self.mixed_min_active_decodes
                and active_decodes < self.mixed_min_active_decodes
                and self.mixed_low_load_mode == "legacy_fifo")
            if fifo_low_load:
                # 只在显式候选中模仿 legacy 的 Prefill admission 次序。
                # 当已有 D 时仍遵守 max_prefill_steps，不能无限饿死 D。
                quota = max(1, self.max_prefill_steps)
                force_decode = bool(self.running and self.prefill_steps_since_decode >= quota)
                chosen = [] if force_decode else self._take_prefills(
                    budget, slots, pinned, preserve_fifo=True)
                items.extend(chosen)
                budget -= sum(item.query_len for item in chosen)
                slots -= len(chosen)
                if not chosen:
                    chosen = self._take_decodes(budget, slots, pinned)
                    items.extend(chosen)
                    budget -= len(chosen)
                    slots -= len(chosen)
                    if not chosen and force_decode:
                        chosen = self._take_prefills(budget, slots, pinned, preserve_fifo=True)
                        items.extend(chosen)
                        budget -= sum(item.query_len for item in chosen)
                        slots -= len(chosen)
            elif separate_low_load:
                if self.mixed_low_load_mode == "micro":
                    # 每轮优先推进 D；waiting 满公平等待轮数时先 pin 最老
                    # 请求的一个 token，再与 D 同 forward。一次 P 的最小
                    # 进展被保留，避免大 chunk 阻塞当前流式输出。
                    if self._prefill_wait_rounds >= self.prefill_max_wait_steps:
                        oldest_waiter = max(self.waiting, key=lambda seq: (
                            self._waiting_ages.get(seq.seq_id, 0), -seq.seq_id))
                        self.waiting.remove(oldest_waiter)
                        self.waiting.appendleft(oldest_waiter)
                        chosen = self._take_prefills(1, slots, pinned, max_items=1)
                        items.extend(chosen)
                        budget -= sum(item.query_len for item in chosen)
                        slots -= len(chosen)
                    chosen = self._take_decodes(budget, slots, pinned)
                    items.extend(chosen)
                    budget -= len(chosen)
                    slots -= len(chosen)
                    if not items:
                        # 没有可预留的 D 时允许 P 正常推进；不能空转。
                        chosen = self._take_prefills(budget, slots, pinned)
                        items.extend(chosen)
                        budget -= sum(item.query_len for item in chosen)
                        slots -= len(chosen)
                else:
                    # 纯 P/D 轮次；原配置 0 表示无限 Prefill，门控启用
                    # 时至少每一轮 P 后给 D 一轮机会。
                    prefill_quota = max(1, self.max_prefill_steps)
                    force_decode = self.prefill_steps_since_decode >= prefill_quota
                    chosen = [] if force_decode else self._take_prefills(budget, slots, pinned)
                    items.extend(chosen)
                    budget -= sum(item.query_len for item in chosen)
                    slots -= len(chosen)
                    if not chosen:
                        chosen = self._take_decodes(budget, slots, pinned)
                        items.extend(chosen)
                        budget -= len(chosen)
                        slots -= len(chosen)
                        if not chosen and force_decode:
                            # Decode 受 KV 压力无法预留时允许 P 推进。
                            chosen = self._take_prefills(budget, slots, pinned)
                            items.extend(chosen)
                            budget -= sum(item.query_len for item in chosen)
                            slots -= len(chosen)
            else:
                if self.waiting and self._prefill_wait_rounds >= self.prefill_max_wait_steps:
                    oldest_waiter = max(self.waiting, key=lambda seq: (self._waiting_ages.get(seq.seq_id, 0), -seq.seq_id))
                    self.waiting.remove(oldest_waiter)
                    self.waiting.appendleft(oldest_waiter)
                    fair_budget = min(budget, max(1, budget // 4))
                    chosen = self._take_prefills(fair_budget, slots, pinned, max_items=1)
                    items.extend(chosen)
                    budget -= sum(item.query_len for item in chosen)
                    slots -= len(chosen)
                chosen = self._take_decodes(budget, slots, pinned)
                items.extend(chosen)
                budget -= len(chosen)
                slots -= len(chosen)
                chosen = self._take_prefills(budget, slots, pinned)
                items.extend(chosen)
                budget -= sum(item.query_len for item in chosen)
            if not items:
                # Admission 已保证单请求可装入；如果仍无进展，不允许空转无限重试。
                raise KVProgressBlocked("no execution progress despite admission; check KV ownership")
            has_prefill = any(item.role is ExecutionRole.PREFILL for item in items)
            if not has_prefill and draft_provider is not None and allow_verification:
                # 128 限制整个 D+V forward，不仅限制 V 分区；基础 D 不得被 draft 挤走。
                verify_left = min(128, max(0, int(verification_budget))) - sum(item.query_len for item in items)
                for index, item in enumerate(items):
                    seq = self._plan_sequences[item.request_id]
                    limit = min(seq.speculative_max_draft_tokens,
                                seq.max_tokens - seq.num_completion_tokens - 1,
                                self.max_model_len - len(seq), budget, verify_left)
                    if not seq.use_speculative or limit <= 0:
                        continue
                    draft = tuple(draft_provider(seq, limit))[:limit]
                    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in draft):
                        raise ValueError("draft token ids must be non-negative integers")
                    reservation = self._reservations[seq.seq_id]
                    while draft:
                        try:
                            self.block_manager.extend_reservation(reservation, len(seq) + len(draft))
                            break
                        except MemoryError:
                            draft = draft[:-1]
                    if not draft:
                        continue
                    upgraded = self._record(seq, reservation, ExecutionRole.VERIFY,
                                            (seq.last_token,) + draft, draft=draft, sample=True)
                    items[index] = upgraded
                    budget -= len(draft)
                    verify_left -= len(draft)
            # 各线性层共同处理 packed token；Attention 的多 query 分区始终在前。
            items.sort(key=lambda item: item.role is ExecutionRole.DECODE)
            self._step_id += 1
            plan = ExecutionPlan(tuple(items), self._step_id)
            assert plan.num_tokens <= self.max_num_batched_tokens
            assert len(plan.items) <= self.max_num_seqs
            assert plan.verification_tokens <= 128
            assert not plan.has_verification or plan.num_tokens <= 128
            self._active_plan = plan
            self.last_step_stats = StepStats.from_plan(plan)
            selected_prefills = {item.request_id for item in items if item.role is ExecutionRole.PREFILL}
            self._waiting_ages = {
                seq.seq_id: (0 if seq.seq_id in selected_prefills else self._waiting_ages.get(seq.seq_id, 0) + 1)
                for seq in self.waiting}
            self._prefill_wait_rounds = max(self._waiting_ages.values(), default=0)
            self.total_prefill_steps += int(has_prefill)
            self.total_decode_steps += int(plan.decode_tokens > 0 or plan.has_verification)
            if plan.decode_tokens > 0 or plan.has_verification:
                self.prefill_steps_since_decode = 0
            elif has_prefill:
                self.prefill_steps_since_decode += 1
            self.total_prefill_tokens += plan.prefill_tokens
            self.total_decode_tokens += plan.decode_tokens
            self.total_verification_tokens += plan.verification_tokens
            self.total_mixed_steps += self.last_step_stats.mixed
            return plan
        except BaseException:
            self.rollback_plan()
            raise

    def upgrade_device_verification(self, plan, count):
        """Reserve metadata before GPU work; never invent CPU proposal IDs."""
        from nanovllm.engine.device_draft import DeviceVerificationItem
        if (self._active_plan is not plan or len(plan.items)!=1
                or plan.items[0].role is not ExecutionRole.DECODE
                or type(count) is not int or count<1):
            raise ValueError("device upgrade requires one active decode plan")
        old=plan.items[0];seq=self._plan_sequences[old.request_id]
        if (not seq.use_speculative or seq.resolved_speculative_method!="draft_model"
                or getattr(seq,"speculative_sampling","greedy")!="greedy"
                or count>seq.speculative_max_draft_tokens
                or count>seq.max_tokens-seq.num_completion_tokens-1
                or old.start_pos!=len(seq)-1
                or old.start_pos+count+1>self.max_model_len
                or count+1>min(self.max_num_batched_tokens,128)):
            raise ValueError("device upgrade exceeds request or query budget")
        reservation=self._reservations[old.request_id]
        self.block_manager.extend_reservation(reservation,len(seq)+count)
        item=DeviceVerificationItem(old.request_id,seq.last_token,count,old.start_pos,tuple(seq.block_table))
        upgraded=ExecutionPlan((item,),plan.step_id)
        self._active_items[item.request_id]=item;self._active_plan=upgraded
        self.last_step_stats=StepStats.from_plan(upgraded)
        self.total_decode_tokens-=1
        self.total_verification_tokens+=item.query_len
        return upgraded

    def materialize_device_verification(self, plan, result):
        from nanovllm.engine.device_draft import DeviceVerificationItem,DeviceVerificationResult
        if (self._active_plan is not plan or len(plan.items)!=1
                or not isinstance(plan.items[0],DeviceVerificationItem)
                or not isinstance(result,DeviceVerificationResult)):
            raise ValueError("invalid pending device plan/result")
        ticket=plan.items[0];seq=self._plan_sequences[ticket.request_id]
        if (result.request_id!=ticket.request_id or len(result.draft)!=ticket.count
                or seq.last_token!=ticket.last_token or len(seq)-1!=ticket.start_pos
                or tuple(seq.block_table)!=ticket.block_table):
            raise ValueError("device result differs from reserved request metadata")
        item=ExecutionItem(ticket.request_id,ExecutionRole.VERIFY,(ticket.last_token,)+result.draft,
            ticket.start_pos,ticket.block_table,result.draft,ticket.logit_positions)
        materialized=ExecutionPlan((item,),plan.step_id)
        self._active_items[item.request_id]=item;self._active_plan=materialized
        return materialized

    def commit_item(self, item, token_ids=(), *, stop_checker=None):
        """提交真实输出；V 只接受 draft 前缀+纠正/bonus，支持逐 token EOS/stop。

        stop_checker(seq)->bool 在每个真实 token 追加后调用。它只能判断停止，
        不能自行释放页；停止与取消的资源释放由本方法统一完成。
        """
        if self._active_items.get(item.request_id) is not item:
            raise RuntimeError("item is not active in this planner")
        seq = self._plan_sequences[item.request_id]
        outputs = list(token_ids)
        if item.role is ExecutionRole.PREFILL:
            expected = 1 if item.sample_eligible else 0
            if len(outputs) != expected:
                raise ValueError("prefill output count does not match sampling eligibility")
        elif item.role is ExecutionRole.DECODE and len(outputs) != 1:
            raise ValueError("decode must produce one token")
        elif item.role is ExecutionRole.VERIFY:
            if not 1 <= len(outputs) <= item.query_len or tuple(outputs[:-1]) != item.draft[:len(outputs) - 1]:
                raise ValueError("verification output is not accepted draft prefix plus next token")
        if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in outputs):
            raise ValueError("generated token ids must be non-negative integers")
        generated, finish_reason = [], None
        old_tokens, old_last, old_length = list(seq.token_ids), seq.last_token, len(seq)
        try:
            for token in outputs:
                seq.append_token(token)
                generated.append(token)
                if (not seq.ignore_eos and token == self.eos) or (stop_checker is not None and stop_checker(seq)):
                    finish_reason = "stop"
                    break
                if seq.num_completion_tokens >= seq.max_tokens:
                    finish_reason = "length"
                    break
            committed_end = item.end_pos if item.role is ExecutionRole.PREFILL else item.start_pos + len(generated)
            self.block_manager.commit(self._reservations[item.request_id], committed_end)
        except BaseException:
            seq.token_ids[:] = old_tokens
            seq.last_token, seq.num_tokens = old_last, old_length
            raise
        self._reservations.pop(item.request_id)
        seq.is_prefill = seq.num_cached_tokens < len(seq) - (1 if generated else 0)
        if item.role is ExecutionRole.PREFILL and item.sample_eligible:
            self._waiting_ages.pop(seq.seq_id, None)
            if seq in self.waiting:
                self.waiting.remove(seq)
            if seq not in self.running:
                self.running.append(seq)
            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
        if finish_reason:
            self._waiting_ages.pop(seq.seq_id, None)
            seq.finish_reason = finish_reason
            seq.status = SequenceStatus.FINISHED
            self.total_finished_requests += 1
            self.block_manager.deallocate(seq)
            for queue in (self.waiting, self.running):
                if seq in queue:
                    queue.remove(seq)
        self.last_step_stats.generated_tokens += len(generated)
        if item.role is ExecutionRole.VERIFY:
            self.last_step_stats.accepted_draft_tokens += min(len(generated), len(outputs) - 1)
        self._active_items.pop(item.request_id)
        if not self._active_items:
            self._active_plan = None
            self._plan_sequences.clear()
            self._prefill_wait_rounds = max(self._waiting_ages.values(), default=0)
        return generated

    def rollback_plan(self):
        """撤销尚未提交条目的 reservation；已经提交的条目不做第二次回滚。"""
        for reservation in list(self._reservations.values()):
            self.block_manager.rollback(reservation)
        self._reservations.clear()
        self._active_items.clear()
        self._active_plan = None
        self._plan_sequences.clear()

    def cancel(self, seq):
        """取消只在 step 边界处理，避免 GPU 仍使用页时提前归还。"""
        if seq.seq_id in self._active_items:
            raise RuntimeError("rollback or commit the active plan before cancellation")
        cancelled = super().cancel(seq)
        self._waiting_ages.pop(seq.seq_id, None)
        self._prefill_wait_rounds = max(self._waiting_ages.values(), default=0)
        return cancelled

    def stats(self):
        result = super().stats()
        result.update(mixed_steps=self.total_mixed_steps,
                      verification_query_tokens=self.total_verification_tokens,
                      prefill_wait_rounds=self._prefill_wait_rounds)
        return result
