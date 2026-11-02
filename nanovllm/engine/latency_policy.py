"""默认关闭的SLO启发式：限制prefill阻塞，不保证deadline、不丢弃请求。

观测完整step墙钟，不误称GPU纯计算时间；状态只保留活跃请求。
"""
from collections import Counter


class LatencyPolicy:
    def __init__(self, ttft_ms=1000., tpot_ms=30., min_prefill_tokens=32, cost_model=None):
        self.ttft_ms, self.tpot_ms, self.min_tokens = ttft_ms, tpot_ms, min_prefill_tokens
        self.arrivals, self.last_output = {}, {}
        self.prefill_ms_per_token, self.decode_ms = .05, 5.
        self.decode_streak = 0
        self.counters = Counter()
        # 冻结的离线模型 t=a+b*N。a表示launch、权重读取、采样等固定开销；
        # 不能把一个token尾块的全部时间都当成每token成本，再形成越切越碎的反馈。
        # 没有显式模型时仍保留旧实验路径，以便复现上一轮负结果。
        self.cost_model = dict(cost_model) if cost_model is not None else None
        if cost_model is not None:
            import math
            for key in ("prefill_fixed_ms", "prefill_ms_per_token", "decode_ms"):
                value=cost_model.get(key)
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError("invalid frozen scheduler cost: "+key)
            if cost_model["prefill_ms_per_token"]<=0:
                raise ValueError("frozen prefill slope must be positive")
            self.prefill_ms_per_token=cost_model["prefill_ms_per_token"]
            self.decode_ms=cost_model["decode_ms"]

    def estimate_prefill_ms(self, tokens):
        fixed=self.cost_model["prefill_fixed_ms"] if self.cost_model else 0.
        return fixed+self.prefill_ms_per_token*tokens

    def add(self, request_id, now):
        self.arrivals[request_id] = now

    def forget(self, request_id):
        self.arrivals.pop(request_id, None)
        self.last_output.pop(request_id, None)

    def choose(self, waiting, running, max_tokens, now):
        if not waiting or not running:
            return False, max_tokens
        if self.cost_model and self.estimate_prefill_ms(self.min_tokens)+self.decode_ms>=self.tpot_ms:
            # 这里只判定该模型下“一次prefill+decode”的单间隔预算不可行，
            # 不是证明请求平均TPOT、所有工作负载或其他调度策略都不可能达标。
            # 校准已证明连最小prefill都无法在目标间隔内完成：继续缩块或
            # 永远decode都无法满足该SLO，只会饿死新请求。显式报告不可行，
            # 返回原budget，让既有max_prefill_steps轮转规则保持进展。
            self.counters["infeasible_slo_baseline_fallbacks"]+=1
            return False,max_tokens
        deadline = min(self.last_output.get(s.seq_id, now)+self.tpot_ms/1000 for s in running)
        slack_ms = (deadline-now)*1000-self.decode_ms
        oldest = min(self.arrivals.get(s.seq_id, now) for s in waiting)
        if (now-oldest)*1000 >= self.ttft_ms and self.decode_streak >= 4:
            # 有限公平性救援，避免长prompt被持续decode饿死。
            self.counters["fairness_prefills"] += 1
            # 冻结模型下公平性救援用一个正常budget，不再强行缩到32token，
            # 避免已超TTFT的长请求还被固定开销反复惩罚。旧路径保持可复现。
            return False, max_tokens if self.cost_model else min(max_tokens, self.min_tokens)
        if slack_ms <= self.estimate_prefill_ms(self.min_tokens):
            self.counters["deadline_decode_choices"] += 1
            return True, max_tokens
        fixed=self.cost_model["prefill_fixed_ms"] if self.cost_model else 0.
        budget = min(max_tokens, max(self.min_tokens, int((slack_ms-fixed)/self.prefill_ms_per_token)))
        if self.cost_model:
            # 固定粒度减少近似预算产生的形状抖动；不能超出原token预算。
            budget=max(min(max_tokens,self.min_tokens),(budget//self.min_tokens)*self.min_tokens)
        self.counters["budget_limited_steps"] += int(budget < max_tokens)
        return False, budget

    def observe(self, seqs, generated_ids, is_prefill, query_tokens, elapsed_ms, now):
        if is_prefill and query_tokens:
            if self.cost_model is None:
                self.prefill_ms_per_token = .8*self.prefill_ms_per_token+.2*max(.001,elapsed_ms/query_tokens)
            self.decode_streak = 0
        elif not is_prefill:
            if self.cost_model is None:
                self.decode_ms = .8*self.decode_ms+.2*max(.01,elapsed_ms)
            self.decode_streak += 1
        for seq in seqs:
            if seq.is_finished:
                self.forget(seq.seq_id)
            elif seq.seq_id in generated_ids:
                self.last_output[seq.seq_id] = now

    def stats(self):
        return dict(self.counters, tracked_requests=len(self.arrivals),
                    prefill_ms_per_token=self.prefill_ms_per_token, decode_ms=self.decode_ms,
                    frozen_cost_model=dict(self.cost_model) if self.cost_model is not None else None)
