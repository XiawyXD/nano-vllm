"""纯 CPU 自适应推测控制器；不执行模型、不导入 torch、不改变采样语义。

引擎调用顺序：begin_round -> decide(每条 eligible 请求) -> 模型与提交 ->
observe_batch(整个 D 调度轮一次) -> observe_request(各验证请求)。
下一轮 decide 消耗上一轮反馈；重复读取 decision 不会重复累计升降档 streak。

bucket 应为可哈希的 (batch_size, context_bucket, backend_signature)，normal 与
spec 使用相同分桶。不要把 query 长度或 draft 长度放到主 bucket 中，否则成本永远
无法匹配。成本预测是同 bucket 的整轮 ms/实际提交 token 的 EMA，不是单请求延迟，
也不是某档位的因果成本模型；它允许包含普通 decode + verification 的异构轮。
请求级 prefix EMA 仅预测候选长度的接受量，不能冒充整个 batch 的精确耗时。
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import asdict, dataclass, field
import math
from time import monotonic
from typing import Callable, Hashable


LEVELS = (0, 2, 4, 8)
_EXCLUSIONS = {"capture", "jit", "grow", "error", "prefill"}


@dataclass(frozen=True)
class AdaptiveSpeculationConfig:
    """策略常数集中保存；默认值对应批准方案，测试可以缩小窗口但不能使用非法值。"""

    ema_alpha: float = 0.2
    max_cost_buckets: int = 256
    interval_window: int = 128
    min_normal_samples: int = 8
    calibration_rounds: int = 16
    beneficial_ratio: float = 0.9
    beneficial_streak: int = 3
    unhelpful_streak: int = 2
    low_acceptance_ratio: float = 0.25
    low_acceptance_streak: int = 2
    request_cooldown_rounds: int = 16
    global_cooldown_rounds: int = 32
    min_interval_samples: int = 32
    interval_evaluation_stride: int = 8
    tail_ratio: float = 1.2
    tail_bad_windows: int = 2
    catastrophic_tail_ratio: float = 3.0
    catastrophic_cooldown_rounds: int = 128
    catastrophic_cooldown_max_rounds: int = 1024

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        float_fields = {"ema_alpha", "beneficial_ratio", "low_acceptance_ratio", "tail_ratio",
                        "catastrophic_tail_ratio"}
        for name, value in asdict(self).items():
            if name not in float_fields and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.ema_alpha > 1 or self.beneficial_ratio > 1 or self.low_acceptance_ratio > 1:
            raise ValueError("EMA/benefit/acceptance ratios must not exceed 1")
        if (self.tail_ratio <= 1 or self.catastrophic_tail_ratio <= self.tail_ratio
                or self.catastrophic_cooldown_max_rounds < self.catastrophic_cooldown_rounds
                or self.min_interval_samples > self.interval_window):
            raise ValueError("tail ratios must increase above 1 and interval window must fit its minimum")


@dataclass(frozen=True)
class Decision:
    """某请求本轮的策略选择；引擎仍须按预算/KV/剩余长度进一步裁剪实际草稿。"""

    draft_tokens: int
    reason: str
    probe: bool
    calibration: bool
    expected_committed_tokens: float
    batch_cost_ratio: float | None
    round_id: int
    timestamp_s: float


@dataclass
class _RequestState:
    level: int = 2
    eligible_rounds: int = 0
    prefix_probabilities: list[float | None] = field(default_factory=lambda: [None] * 8)
    observed_lengths: set[int] = field(default_factory=set)
    observations: int = 0
    consumed_observations: int = 0
    observed_bucket: Hashable | None = None
    observed_proposed_tokens: int = 0
    last_bucket: Hashable | None = None
    good_streak: int = 0
    bad_streak: int = 0
    low_streak: int = 0
    cooldown_remaining: int = 0
    recovery_pending: bool = False
    seen_global_epoch: int = 0
    last_decision: Decision | None = None


@dataclass
class _CostBucket:
    normal_samples: int = 0
    spec_samples: int = 0
    normal_ms_per_token: float | None = None
    spec_ms_per_token: float | None = None
    normal_elapsed_ms: float | None = None
    spec_elapsed_ms: float | None = None
    last_batch_size: int = 0
    last_committed_tokens: int = 0
    requested_lengths: Counter = field(default_factory=Counter)
    mixed_length_batches: int = 0
    normal_intervals: deque = field(default_factory=deque)
    spec_intervals: deque = field(default_factory=deque)
    intervals_since_evaluation: int = 0
    bad_tail_windows: int = 0
    catastrophic_cooldown_until: int = 0
    catastrophic_violations: int = 0
    catastrophic_backoff_rounds: int = 0


class AdaptiveSpeculationController:
    """单 worker 使用的有界成本控制器；请求状态由 engine 在完成/取消时 forget。

    普通请求/非 greedy 请求不应传入本控制器；eligible=False 是额外保险。
    成本和输出间隔都用毫秒。excluded_reason 非空时只计数，不学习、不累计 streak。
    本类不做 GPU synchronize；引擎必须提供有明确测量边界的整轮时间。
    """

    def __init__(self, config: AdaptiveSpeculationConfig | None = None, *,
                 clock: Callable[[], float] = monotonic):
        self.config = config or AdaptiveSpeculationConfig()
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        """清空策略/成本/计数器；配置与注入的时钟保留，适合独立实验重新开始。"""
        self._requests: dict[Hashable, _RequestState] = {}
        self._buckets: OrderedDict[Hashable, _CostBucket] = OrderedDict()
        self._counts: Counter = Counter()
        self._reasons: Counter = Counter()
        self._round = 0
        self._global_cooldown_until = 0
        self._global_epoch = 0
        self._recovery_probe_owner: Hashable | None = None
        self._started_at = self._clock()

    def begin_round(self) -> int:
        """每个 eligible D 调度轮只调用一次；P 轮不调用，故冷却单位不是墙钟时间。"""
        self._round += 1
        self._counts["decode_rounds"] += 1
        return self._round

    def needs_normal_samples(self, bucket: Hashable) -> bool:
        """引擎用整批普通 D 收集基线，不能把异构轮的耗时分摊成普通样本。"""
        value = self._buckets.get(bucket)
        return value is None or value.normal_samples < self.config.min_normal_samples

    @staticmethod
    def _integer(value, name: str, minimum: int = 0) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    @staticmethod
    def _positive(value, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    def _excluded(self, reason: str | None, kind: str) -> bool:
        if reason is None:
            return False
        if not isinstance(reason, str) or not reason:
            raise ValueError("excluded_reason must be a nonempty string or None")
        # 未知标签汇总到 other，避免外部错误文本制造无限增长的 Counter key。
        normalized = reason.lower()
        if normalized not in _EXCLUSIONS:
            normalized = "other"
        self._counts[f"excluded_{kind}_{normalized}"] += 1
        return True

    def _bucket(self, key: Hashable) -> _CostBucket:
        hash(key)
        if key in self._buckets:
            self._buckets.move_to_end(key)
            return self._buckets[key]
        while len(self._buckets) >= self.config.max_cost_buckets:
            self._buckets.popitem(last=False)
            self._counts["cost_bucket_evictions"] += 1
        bucket = _CostBucket(normal_intervals=deque(maxlen=self.config.interval_window),
                             spec_intervals=deque(maxlen=self.config.interval_window))
        self._buckets[key] = bucket
        return bucket

    def _ema(self, old: float | None, value: float) -> float:
        return value if old is None else (1 - self.config.ema_alpha) * old + self.config.ema_alpha * value

    def _cost_ratio(self, bucket: _CostBucket) -> float | None:
        if (bucket.normal_samples < self.config.min_normal_samples or
                bucket.normal_ms_per_token is None or bucket.spec_ms_per_token is None):
            return None
        return bucket.spec_ms_per_token / bucket.normal_ms_per_token

    @staticmethod
    def _expected(state: _RequestState, length: int) -> float:
        """E[提交数]=1+ΣP(接受前 k 个)。未知尾部按每步 0.5 衰减，仅作为探测先验。"""
        total, previous = 1.0, 1.0
        for probability in state.prefix_probabilities[:length]:
            probability = previous * 0.5 if probability is None else min(previous, probability)
            total += probability
            previous = probability
        return total

    def _apply_feedback(self, state: _RequestState, bucket_key: Hashable, ratio: float | None) -> str | None:
        if state.observations == state.consumed_observations:
            return None
        state.consumed_observations = state.observations
        if state.observed_bucket != bucket_key or ratio is None or state.cooldown_remaining:
            state.good_streak = state.bad_streak = 0
            return None
        if state.observed_proposed_tokens < state.level:
            # 预算/KV/提案器可能把实际草稿裁短。L2的收益不能当成L4已验证，
            # 否则长期cap=2也会把内部档位升到8，解除限制后跳过L4探测。
            state.good_streak = state.bad_streak = 0
            return "short_draft_observation"
        if ratio <= self.config.beneficial_ratio:
            state.good_streak += 1
            state.bad_streak = 0
            if state.good_streak >= self.config.beneficial_streak and state.level < 8:
                next_level = LEVELS[LEVELS.index(state.level) + 1]
                if self._expected(state, next_level) > self._expected(state, state.level):
                    state.level = next_level
                    state.good_streak = 0
                    self._counts["promotions"] += 1
                    return "promote_probe"
        else:
            state.bad_streak += 1
            state.good_streak = 0
            if state.bad_streak >= self.config.unhelpful_streak:
                state.bad_streak = 0
                self._counts["demotions"] += 1
                if state.level > 2:
                    state.level = LEVELS[LEVELS.index(state.level) - 1]
                    return "demote_cost"
                state.cooldown_remaining = self.config.request_cooldown_rounds
                state.recovery_pending = True
                return "cost_cooldown"
        return None

    def decide(self, request_id: Hashable, bucket: Hashable, max_draft_tokens: int = 8, *,
               eligible: bool = True) -> Decision:
        """决定 0/2/4/8 档位。每 request/round 最多一项决策，重复读取返回同一结果。"""
        if self._round == 0:
            raise RuntimeError("call begin_round before decide")
        self._integer(max_draft_tokens, "max_draft_tokens")
        hash(request_id)
        hash(bucket)
        if request_id not in self._requests:
            # 冷却结束后新到达的请求不是“恢复中的旧请求”，应照常从 L2 探测。
            # 冷却期间到达的请求仍需全局恢复许可，不能绕开延迟保护。
            state = _RequestState(seen_global_epoch=self._global_epoch)
            state.recovery_pending = bool(self._global_cooldown_until and
                                          self._round <= self._global_cooldown_until)
            self._requests[request_id] = state
        else:
            state = self._requests[request_id]
        if state.last_decision is not None and state.last_decision.round_id == self._round:
            return state.last_decision
        cost = self._bucket(bucket)
        ratio = self._cost_ratio(cost)
        state.last_bucket = bucket
        length, reason, probe, calibration = 0, "ineligible", False, False
        if eligible:
            state.eligible_rounds += 1
            feedback = self._apply_feedback(state, bucket, ratio)
            if state.seen_global_epoch != self._global_epoch:
                state.seen_global_epoch = self._global_epoch
                state.recovery_pending = True
                state.good_streak = state.bad_streak = 0
            if self._global_cooldown_until and self._round <= self._global_cooldown_until:
                # 全局保护中的这一轮同样是纯 D，已满足一次请求级冷却，不能叠成48轮。
                if state.cooldown_remaining:
                    state.cooldown_remaining -= 1
                reason = "global_latency_cooldown"
            elif state.cooldown_remaining:
                state.cooldown_remaining -= 1
                reason = "request_cooldown"
            elif self._round <= cost.catastrophic_cooldown_until:
                reason = "catastrophic_bucket_cooldown"
            elif max_draft_tokens < 2:
                reason = "draft_budget_too_small"
            elif state.eligible_rounds % self.config.calibration_rounds == 0:
                reason, calibration = "normal_calibration", True
            elif state.recovery_pending:
                if self._recovery_probe_owner not in (None, request_id):
                    reason = "recovery_probe_busy"
                else:
                    self._recovery_probe_owner = request_id
                    state.recovery_pending = False
                    state.level = 2
                    length, reason, probe = 2, "recovery_probe", True
            else:
                length = min(state.level, max(v for v in LEVELS if v <= min(max_draft_tokens, 8)))
                probe = length not in state.observed_lengths
                reason = feedback or ("initial_probe" if state.observations == 0 else
                                      "normal_cost_warmup" if ratio is None else
                                      "beneficial" if ratio <= self.config.beneficial_ratio else "cost_hysteresis")
        decision = Decision(length, reason, probe, calibration, self._expected(state, length),
                            ratio, self._round, self._clock())
        state.last_decision = decision
        self._reasons[reason] += 1
        self._counts[f"decisions_length_{length}"] += 1
        return decision

    def observe_request(self, request_id: Hashable, proposed_tokens: int, accepted_tokens: int, *,
                        excluded_reason: str | None = None) -> None:
        """学习连续接受前缀，不是每位置独立命中率；accepted 必须处于 [0, proposed]。

        例如 proposed=4、accepted=2，四个观测值为 1/1/0/0。因 EOS/stop 提前结束
        的 censored 样本应由引擎 excluded，不能把未观察位置当拒绝。
        """
        self._integer(proposed_tokens, "proposed_tokens")
        self._integer(accepted_tokens, "accepted_tokens")
        if proposed_tokens > 8 or accepted_tokens > proposed_tokens:
            raise ValueError("require 0 <= accepted_tokens <= proposed_tokens <= 8")
        if request_id not in self._requests:
            raise KeyError("observe_request requires a previous decide")
        state = self._requests[request_id]
        if self._recovery_probe_owner == request_id:
            self._recovery_probe_owner = None
        if self._excluded(excluded_reason, "request"):
            if state.last_decision and state.last_decision.reason == "recovery_probe":
                state.recovery_pending = True
            return
        if proposed_tokens == 0:
            self._counts["no_draft_samples"] += 1
            return
        for index in range(proposed_tokens):
            state.prefix_probabilities[index] = self._ema(state.prefix_probabilities[index], float(index < accepted_tokens))
        state.observed_lengths.add(proposed_tokens)
        state.observations += 1
        state.observed_bucket = state.last_bucket
        state.observed_proposed_tokens = proposed_tokens
        self._counts["request_samples"] += 1
        self._counts["proposed_tokens"] += proposed_tokens
        self._counts["accepted_tokens"] += accepted_tokens
        state.low_streak = state.low_streak + 1 if accepted_tokens / proposed_tokens < self.config.low_acceptance_ratio else 0
        if state.low_streak >= self.config.low_acceptance_streak:
            state.cooldown_remaining = self.config.request_cooldown_rounds
            state.recovery_pending = True
            state.low_streak = state.good_streak = state.bad_streak = 0
            self._counts["low_acceptance_cooldowns"] += 1

    def observe_batch(self, bucket: Hashable, elapsed_ms: float, committed_tokens: int,
                      batch_size: int, mode: str = "normal", requested_lengths: tuple[int, ...] = (), *,
                      excluded_reason: str | None = None) -> None:
        """整轮样本一次：elapsed 包含本轮所有 ordinary/verification/proposer/commit。

        主指标 elapsed_ms/committed_tokens，绝不除以 batch_size 冒充请求计时。
        mode=spec 允许 (0,2,4,...) 异构草稿长度；requested_lengths 只作子统计。
        capture/JIT/grow/error/P 轮或没有提交 token 的轮不能进入成本 EMA。
        """
        if mode not in ("normal", "spec"):
            raise ValueError("mode must be normal or spec")
        self._integer(batch_size, "batch_size", 1)
        self._integer(committed_tokens, "committed_tokens")
        if requested_lengths and len(requested_lengths) != batch_size:
            raise ValueError("requested_lengths must have batch_size entries")
        if any(isinstance(v, bool) or v not in LEVELS for v in requested_lengths):
            raise ValueError("requested lengths must be 0/2/4/8")
        if mode == "normal" and any(requested_lengths):
            raise ValueError("a normal calibration batch cannot contain speculative requests")
        if self._excluded(excluded_reason, "batch"):
            return
        self._positive(elapsed_ms, "elapsed_ms")
        if committed_tokens == 0:
            self._counts["zero_commit_batches"] += 1
            return
        cost = self._bucket(bucket)
        per_token = elapsed_ms / committed_tokens
        count_field, rate_field, elapsed_field = f"{mode}_samples", f"{mode}_ms_per_token", f"{mode}_elapsed_ms"
        setattr(cost, count_field, getattr(cost, count_field) + 1)
        setattr(cost, rate_field, self._ema(getattr(cost, rate_field), per_token))
        setattr(cost, elapsed_field, self._ema(getattr(cost, elapsed_field), elapsed_ms))
        cost.last_batch_size, cost.last_committed_tokens = batch_size, committed_tokens
        cost.requested_lengths.update(requested_lengths)
        cost.mixed_length_batches += int(len(set(requested_lengths)) > 1)
        self._counts[f"{mode}_batch_samples"] += 1

    @staticmethod
    def _p95(values) -> float:
        ordered = sorted(values)
        return ordered[math.ceil(0.95 * len(ordered)) - 1]

    def observe_interval(self, bucket: Hashable, interval_ms: float, *, normal: bool = False,
                         excluded_reason: str | None = None) -> None:
        """记录真实连续输出事件间隔；勿将含多个 token 的一个事件拆成伪 token 时间。

        同桶 normal/spec 各至少32点，之后每8个新 spec 点比较128点窗口的P95；
        连续2次 >1.2× 基线触发全局32个D轮冷却。若已有8个普通间隔基线，
        单次间隔 >3× 基线P95会立即触发冷却，防止严重burst在慢窗口内持续出现。
        恢复需要新窗口，旧坏点不反复触发。
        """
        if self._excluded(excluded_reason, "interval"):
            return
        self._positive(interval_ms, "interval_ms")
        cost = self._bucket(bucket)
        if normal:
            cost.normal_intervals.append(interval_ms)
            self._counts["normal_interval_samples"] += 1
            return
        # 同一执行轮可能同时为多条请求产出事件。首个极端间隔已经触发保护后，
        # 不重复计数/延长同一轮的全局冷却，也不把这轮其余burst继续加入窗口。
        if self._global_cooldown_until and self._round <= self._global_cooldown_until:
            self._counts["spec_interval_samples"] += 1
            self._counts["tail_ignored_during_global_cooldown"] += 1
            return
        cost.spec_intervals.append(interval_ms)
        self._counts["spec_interval_samples"] += 1
        if len(cost.normal_intervals) >= self.config.min_normal_samples:
            normal_p95 = self._p95(cost.normal_intervals)
            if interval_ms > self.config.catastrophic_tail_ratio * normal_p95:
                cost.catastrophic_violations += 1
                previous = cost.catastrophic_backoff_rounds
                cost.catastrophic_backoff_rounds = min(
                    self.config.catastrophic_cooldown_max_rounds,
                    self.config.catastrophic_cooldown_rounds if previous == 0 else previous * 2)
                cost.catastrophic_cooldown_until = max(
                    cost.catastrophic_cooldown_until,
                    self._round + cost.catastrophic_backoff_rounds)
                self._trigger_global_cooldown(cost, "global_catastrophic_latency_cooldowns")
                return
            if interval_ms <= self.config.tail_ratio * normal_p95:
                # 一次安全恢复探针说明前次极端间隔可能是瞬时异常，撤销指数退避。
                cost.catastrophic_violations = 0
                cost.catastrophic_backoff_rounds = 0
        cost.intervals_since_evaluation += 1
        if min(len(cost.normal_intervals), len(cost.spec_intervals)) < self.config.min_interval_samples:
            self._counts["tail_skipped_insufficient_samples"] += 1
            return
        if cost.intervals_since_evaluation < self.config.interval_evaluation_stride:
            self._counts["tail_waiting_evaluation_stride"] += 1
            return
        cost.intervals_since_evaluation = 0
        self._counts["tail_window_evaluations"] += 1
        bad = self._p95(cost.spec_intervals) > self.config.tail_ratio * self._p95(cost.normal_intervals)
        cost.bad_tail_windows = cost.bad_tail_windows + 1 if bad else 0
        if cost.bad_tail_windows >= self.config.tail_bad_windows:
            self._trigger_global_cooldown(cost, "global_sustained_latency_cooldowns")

    def _trigger_global_cooldown(self, cost: _CostBucket, reason: str) -> None:
        """一次坏窗口统一执行全局冷却；避免严重与持续尾延迟路径行为分叉。"""
        if self._global_cooldown_until and self._round <= self._global_cooldown_until:
            return
        self._global_cooldown_until = self._round + self.config.global_cooldown_rounds
        self._global_epoch += 1
        self._recovery_probe_owner = None
        self._counts["global_latency_cooldowns"] += 1
        self._counts[reason] += 1
        cost.spec_intervals.clear()
        cost.intervals_since_evaluation = 0
        cost.bad_tail_windows = 0

    def forget(self, request_id: Hashable) -> None:
        """完成/取消/失败时释放请求级EMA和probe占位；不会删除可复用的有界成本桶。"""
        if self._requests.pop(request_id, None) is not None:
            self._counts["forgotten_requests"] += 1
        if self._recovery_probe_owner == request_id:
            self._recovery_probe_owner = None

    def reasons(self) -> dict[str, int]:
        """返回决策原因快照，外部修改不会污染内部计数。"""
        return dict(self._reasons)

    def stats(self, *, include_buckets: bool = True, include_requests: bool = False) -> dict:
        """JSON友好的快照；默认不暴露请求ID，最多256成本桶、每桶两条128点延迟窗。"""
        result = dict(config=asdict(self.config), round_id=self._round,
                      request_count=len(self._requests), cost_bucket_count=len(self._buckets),
                      global_cooldown_remaining=max(0, self._global_cooldown_until - self._round),
                      recovery_probe_active=self._recovery_probe_owner is not None,
                      counters=dict(self._counts), reasons=self.reasons(),
                      elapsed_s=self._clock() - self._started_at,
                      cost_estimator="EMA(full_decode_round_ms / actual_committed_tokens)")
        if include_buckets:
            result["buckets"] = [dict(key=repr(key), normal_samples=value.normal_samples,
                spec_samples=value.spec_samples, normal_ms_per_token=value.normal_ms_per_token,
                spec_ms_per_token=value.spec_ms_per_token, batch_cost_ratio=self._cost_ratio(value),
                normal_elapsed_ms=value.normal_elapsed_ms, spec_elapsed_ms=value.spec_elapsed_ms,
                mixed_length_batches=value.mixed_length_batches, requested_lengths=dict(value.requested_lengths),
                normal_interval_count=len(value.normal_intervals), spec_interval_count=len(value.spec_intervals),
                catastrophic_cooldown_remaining=max(0, value.catastrophic_cooldown_until - self._round),
                catastrophic_violations=value.catastrophic_violations,
                catastrophic_backoff_rounds=value.catastrophic_backoff_rounds,
                tail_sample_ready=min(len(value.normal_intervals), len(value.spec_intervals)) >= self.config.min_interval_samples,
                normal_p95_ms=self._p95(value.normal_intervals) if value.normal_intervals else None,
                spec_p95_ms=self._p95(value.spec_intervals) if value.spec_intervals else None)
                for key, value in self._buckets.items()]
        if include_requests:
            result["requests"] = {str(key): dict(level=value.level, eligible_rounds=value.eligible_rounds,
                prefix_probabilities=list(value.prefix_probabilities), observations=value.observations,
                cooldown_remaining=value.cooldown_remaining, recovery_pending=value.recovery_pending,
                good_streak=value.good_streak, bad_streak=value.bad_streak)
                for key, value in self._requests.items()}
        return result
