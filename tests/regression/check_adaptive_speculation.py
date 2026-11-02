"""不需要 torch/CUDA 的自适应策略回归：fake clock、成本噪声、边界与资源上限。

运行：python -B tests/regression/check_adaptive_speculation.py
可选 --output 新文件.json 保存测试状态；不运行模型，也不是性能 benchmark。
使用文件路径加载控制器，避免 nanovllm/__init__.py 间接导入 torch。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest


def load_controller():
    """只载入纯 Python 策略文件，确保无 GPU 环境也能运行这套状态机测试。"""
    source = Path(__file__).resolve().parents[2] / "nanovllm/engine/adaptive_speculation.py"
    spec = importlib.util.spec_from_file_location("_nanovllm_adaptive_speculation_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_controller()
Controller = MODULE.AdaptiveSpeculationController
Config = MODULE.AdaptiveSpeculationConfig


class FakeClock:
    """用手动推进的秒数替代 sleep，验证冷却单位是 decode 轮而非墙钟时间。"""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class AdaptiveTests(unittest.TestCase):
    """每个测试从全新控制器开始，不依赖测试执行顺序或设备速度。"""

    def setUp(self):
        self.clock = FakeClock()
        self.controller = Controller(clock=self.clock)
        self.bucket = (4, 512, "torch-bf16")

    def normal_samples(self, count=8, bucket=None, per_token=1.0):
        """建立同一批次形状的纯 D 基线，batch4 一次真实提交4个 token。"""
        for _ in range(count):
            self.controller.observe_batch(bucket or self.bucket, 4 * per_token, 4, 4,
                                          "normal", (0, 0, 0, 0))

    def next_decision(self, request="a", **kwargs):
        """启动一次 D 调度轮，产生一条请求的决策。"""
        self.controller.begin_round()
        return self.controller.decide(request, self.bucket, **kwargs)

    def sample(self, decision, request="a", per_token=0.5, accepted=None):
        """模拟完整混合 D/V 轮：三个普通请求 + 一条接受若干草稿的请求。"""
        length = decision.draft_tokens
        accepted = length if accepted is None else accepted
        committed = 4 + accepted
        self.controller.observe_batch(self.bucket, committed * per_token, committed, 4,
                                      "spec", (length, 0, 0, 0))
        self.controller.observe_request(request, length, accepted)

    def test_defaults_and_initial_probe_fake_clock(self):
        cfg = self.controller.config
        self.assertEqual((cfg.ema_alpha, cfg.max_cost_buckets, cfg.interval_window), (0.2, 256, 128))
        self.assertEqual((cfg.min_normal_samples, cfg.calibration_rounds), (8, 16))
        self.assertEqual(cfg.catastrophic_cooldown_rounds, 128)
        self.assertEqual(cfg.catastrophic_cooldown_max_rounds, 1024)
        with self.assertRaises(RuntimeError):
            self.controller.decide("a", self.bucket)
        decision = self.next_decision()
        self.assertEqual((decision.draft_tokens, decision.reason, decision.probe), (2, "initial_probe", True))
        self.assertEqual(decision.timestamp_s, 100.0)
        self.assertAlmostEqual(decision.expected_committed_tokens, 1.75)
        self.clock.advance(7200)
        self.assertEqual(self.controller.stats()["round_id"], 1)
        self.assertEqual(self.controller.stats()["elapsed_s"], 7200)

    def test_repeated_decision_is_idempotent(self):
        decision = self.next_decision()
        before = self.controller.stats(include_requests=True)
        for _ in range(10):
            self.assertIs(self.controller.decide("a", self.bucket), decision)
        self.assertEqual(self.controller.stats(include_requests=True), before)

    def test_prefix_probability_not_flat_acceptance_rate(self):
        decision = self.next_decision()
        self.sample(decision, accepted=1)
        probabilities = self.controller.stats(include_requests=True)["requests"]["a"]["prefix_probabilities"]
        self.assertEqual(probabilities[:2], [1.0, 0.0])
        decision = self.next_decision()
        self.sample(decision, accepted=2)
        decision = self.next_decision()
        self.assertAlmostEqual(decision.expected_committed_tokens, 2.2)
        probabilities = self.controller.stats(include_requests=True)["requests"]["a"]["prefix_probabilities"]
        self.assertEqual(probabilities[:2], [1.0, 0.2])

    def test_need_eight_normal_samples_before_cost_decision(self):
        self.normal_samples(7)
        for _ in range(6):
            decision = self.next_decision()
            self.assertEqual(decision.draft_tokens, 2)
            self.assertIsNone(decision.batch_cost_ratio)
            self.sample(decision)
        self.normal_samples(1)
        self.assertIsNotNone(self.next_decision().batch_cost_ratio)

    def test_three_beneficial_observations_promote_one_level_at_a_time(self):
        self.normal_samples()
        lengths = []
        for _ in range(7):
            decision = self.next_decision()
            lengths.append(decision.draft_tokens)
            self.sample(decision)
        self.assertEqual(lengths, [2, 2, 2, 4, 4, 4, 8])
        self.assertEqual(self.controller.stats()["counters"]["promotions"], 2)

    def test_budget_limited_short_drafts_cannot_skip_an_unobserved_level(self):
        """L2被预算裁剪时不能积累假L4证据，放开预算先实际探测4而非跳8。"""
        self.normal_samples()
        actual_lengths = []
        for _ in range(10):
            decision = self.next_decision(max_draft_tokens=2)
            actual_lengths.append(decision.draft_tokens)
            self.sample(decision)
        self.assertEqual(actual_lengths, [2] * 10)
        self.assertEqual(self.controller._requests["a"].level, 4)
        # 持续执行短草稿不会抹掉L2已提供的升级依据，但不能升级从未执行的L4。
        released = []
        for _ in range(4):
            decision = self.next_decision(max_draft_tokens=8)
            released.append(decision.draft_tokens)
            self.sample(decision)
        self.assertEqual(released, [4, 4, 4, 8])

    def test_short_draft_low_acceptance_still_enters_cooldown(self):
        """跳过短草稿的档位成本判断，不应禁用它的连续低接受率保护。"""
        self.normal_samples()
        for _ in range(3):
            decision = self.next_decision()
            self.sample(decision)
        self.assertEqual(self.next_decision().draft_tokens, 4)
        # 模拟planner因KV容量把L4实际缩短为1；未观察的尾部不记失败。
        self.controller.observe_request("a", 1, 0)
        self.next_decision(max_draft_tokens=2)
        self.controller.observe_request("a", 1, 0)
        self.assertEqual(self.next_decision().reason, "request_cooldown")

    def test_cost_hysteresis_requires_two_unhelpful_observations(self):
        self.normal_samples()
        first = self.next_decision()
        self.sample(first, per_token=1.1)
        second = self.next_decision()
        self.assertEqual(second.draft_tokens, 2)
        self.sample(second, per_token=1.1)
        third = self.next_decision()
        self.assertEqual((third.draft_tokens, third.reason), (0, "request_cooldown"))

    def test_single_noisy_cost_sample_does_not_immediately_disable(self):
        self.normal_samples()
        first = self.next_decision()
        self.sample(first, per_token=0.5)
        second = self.next_decision()
        self.sample(second, per_token=2.0)
        third = self.next_decision()
        self.assertEqual(third.draft_tokens, 2)
        self.assertAlmostEqual(third.batch_cost_ratio, 0.8)
        self.assertNotIn("demotions", self.controller.stats()["counters"])

    def test_batch_cost_uses_actual_tokens_and_mixed_batches_learn(self):
        self.normal_samples()
        self.controller.observe_batch(self.bucket, 6.0, 12, 4, "spec", (0, 2, 4, 8))
        bucket = self.controller.stats()["buckets"][0]
        self.assertAlmostEqual(bucket["spec_ms_per_token"], 0.5)
        self.assertAlmostEqual(bucket["batch_cost_ratio"], 0.5)
        self.assertEqual(bucket["mixed_length_batches"], 1)
        self.assertEqual(bucket["requested_lengths"][8], 1)
        with self.assertRaises(ValueError):
            self.controller.observe_batch(self.bucket, 6.0, 12, 4, "normal", (0, 2, 4, 8))

    def test_low_acceptance_cools_exactly_sixteen_decode_rounds(self):
        for _ in range(2):
            decision = self.next_decision()
            self.sample(decision, accepted=0)
        self.clock.advance(99999)
        for _ in range(16):
            decision = self.next_decision()
            self.assertEqual((decision.draft_tokens, decision.reason), (0, "request_cooldown"))
        recovered = self.next_decision()
        self.assertEqual((recovered.draft_tokens, recovered.reason, recovered.probe), (2, "recovery_probe", True))

    def test_only_one_recovery_probe_outstanding_globally(self):
        for _ in range(2):
            self.controller.begin_round()
            for request in ("a", "b"):
                self.controller.decide(request, self.bucket)
                self.controller.observe_request(request, 2, 0)
        for _ in range(16):
            self.controller.begin_round()
            for request in ("a", "b"):
                self.controller.decide(request, self.bucket)
        self.controller.begin_round()
        self.assertEqual(self.controller.decide("a", self.bucket).reason, "recovery_probe")
        self.assertEqual(self.controller.decide("b", self.bucket).reason, "recovery_probe_busy")
        self.controller.begin_round()
        self.assertEqual(self.controller.decide("b", self.bucket).reason, "recovery_probe_busy")
        self.controller.observe_request("a", 2, 2)
        self.controller.begin_round()
        self.assertEqual(self.controller.decide("b", self.bucket).reason, "recovery_probe")
        self.controller.forget("b")
        self.assertFalse(self.controller.stats()["recovery_probe_active"])

    def test_calibration_every_sixteen_eligible_rounds(self):
        for index in range(1, 33):
            decision = self.next_decision()
            self.assertEqual(decision.calibration, index % 16 == 0)
            self.assertEqual(decision.draft_tokens, 0 if index % 16 == 0 else 2)
        self.assertEqual(self.controller.reasons()["normal_calibration"], 2)

    def test_global_p95_guard_requires_samples_and_sustained_bad_windows(self):
        self.next_decision()
        for _ in range(32):
            self.controller.observe_interval(self.bucket, 10.0, normal=True)
        for _ in range(31):
            self.controller.observe_interval(self.bucket, 13.0)
        self.assertNotIn("tail_window_evaluations", self.controller.stats()["counters"])
        self.controller.observe_interval(self.bucket, 13.0)
        self.assertEqual(self.controller.stats()["counters"]["tail_window_evaluations"], 1)
        self.assertNotIn("global_latency_cooldowns", self.controller.stats()["counters"])
        for _ in range(8):
            self.controller.observe_interval(self.bucket, 13.0)
        self.assertEqual(self.controller.stats()["counters"]["global_latency_cooldowns"], 1)
        for _ in range(32):
            decision = self.next_decision()
            self.assertEqual((decision.draft_tokens, decision.reason), (0, "global_latency_cooldown"))
        self.assertEqual(self.next_decision().reason, "recovery_probe")

    def test_catastrophic_gap_trips_fast_guard_after_eight_normal_intervals(self):
        """严重burst立即熔断；普通1.2x尾差仍走原有的P95持续窗口逻辑。"""
        self.next_decision()
        for _ in range(8):
            self.controller.observe_interval(self.bucket, 10.0, normal=True)
        # 严格大于3x才算灾难性间隔，恰好3x不触发。
        self.controller.observe_interval(self.bucket, 30.0)
        self.assertNotIn("global_catastrophic_latency_cooldowns", self.controller.stats()["counters"])
        self.controller.observe_interval(self.bucket, 30.1)
        counters = self.controller.stats()["counters"]
        self.assertEqual(counters["global_latency_cooldowns"], 1)
        self.assertEqual(counters["global_catastrophic_latency_cooldowns"], 1)
        self.assertEqual(self.controller.stats()["global_cooldown_remaining"], 32)
        self.assertEqual(self.controller.stats()["buckets"][0]["catastrophic_cooldown_remaining"], 128)
        # 同一执行轮的其他请求不重复延长冷却，也不污染下一轮的采样窗口。
        self.controller.observe_interval(self.bucket, 100.0)
        counters = self.controller.stats()["counters"]
        self.assertEqual(counters["global_latency_cooldowns"], 1)
        self.assertEqual(counters["tail_ignored_during_global_cooldown"], 1)
        for _ in range(32):
            self.assertEqual(self.next_decision().reason, "global_latency_cooldown")
        for _ in range(96):
            self.assertEqual(self.next_decision().reason, "catastrophic_bucket_cooldown")
        self.assertEqual(self.next_decision().reason, "recovery_probe")
        # 同一形状恢复探针再次造成灾难性间隔时，桶级退避翻倍而全局冷却仍短。
        self.controller.observe_interval(self.bucket, 30.1)
        bucket = self.controller.stats()["buckets"][0]
        self.assertEqual(bucket["catastrophic_violations"], 2)
        self.assertEqual(bucket["catastrophic_backoff_rounds"], 256)
        self.assertEqual(bucket["catastrophic_cooldown_remaining"], 256)
        self.assertEqual(self.controller.stats()["global_cooldown_remaining"], 32)

    def test_catastrophic_guard_needs_an_eight_sample_normal_baseline(self):
        self.next_decision()
        for _ in range(7):
            self.controller.observe_interval(self.bucket, 10.0, normal=True)
        for _ in range(32):
            self.controller.observe_interval(self.bucket, 100.0)
        self.assertNotIn("global_catastrophic_latency_cooldowns", self.controller.stats()["counters"])
        self.controller.observe_interval(self.bucket, 10.0, normal=True)
        self.controller.observe_interval(self.bucket, 100.0)
        self.assertEqual(self.controller.stats()["counters"]["global_catastrophic_latency_cooldowns"], 1)

    def test_tail_threshold_is_strict_and_insufficient_baseline_is_ignored(self):
        self.next_decision()
        for _ in range(32):
            self.controller.observe_interval(self.bucket, 10.0, normal=True)
        for _ in range(128):
            self.controller.observe_interval(self.bucket, 12.0)
        self.assertNotIn("global_latency_cooldowns", self.controller.stats()["counters"])
        for _ in range(128):
            self.controller.observe_interval("no-normal", 999.0)
        self.assertNotIn("global_latency_cooldowns", self.controller.stats()["counters"])

    def test_new_requests_after_global_recovery_are_not_old_recovery_probes(self):
        self.next_decision()
        for _ in range(32):
            self.controller.observe_interval(self.bucket, 10.0, normal=True)
        for _ in range(40):
            self.controller.observe_interval(self.bucket, 13.0)
        for _ in range(32):
            self.controller.begin_round()
        self.controller.begin_round()
        old = self.controller.decide("a", self.bucket)
        new = self.controller.decide("new-arrival", self.bucket)
        self.assertEqual(old.reason, "recovery_probe")
        self.assertEqual((new.reason, new.draft_tokens), ("initial_probe", 2))

    def test_overlapping_global_and_request_cooldowns_do_not_add(self):
        for _ in range(2):
            self.controller.begin_round()
            self.controller.decide("a", self.bucket)
            self.controller.observe_request("a", 2, 0)
        for _ in range(32):
            self.controller.observe_interval(self.bucket, 10.0, normal=True)
        for _ in range(40):
            self.controller.observe_interval(self.bucket, 13.0)
        for _ in range(32):
            self.assertEqual(self.next_decision().reason, "global_latency_cooldown")
        self.assertEqual(self.next_decision().reason, "recovery_probe")

    def test_excluded_samples_never_learn_but_are_counted(self):
        self.next_decision()
        for reason in ("capture", "jit", "grow", "error", "prefill"):
            self.controller.observe_batch(self.bucket, 999.0, 1, 1, "spec", (2,), excluded_reason=reason)
            self.controller.observe_request("a", 2, 0, excluded_reason=reason)
            self.controller.observe_interval(self.bucket, 999.0, excluded_reason=reason)
        stats = self.controller.stats(include_requests=True)
        self.assertEqual(stats["buckets"][0]["spec_samples"], 0)
        self.assertEqual(stats["requests"]["a"]["observations"], 0)
        self.assertEqual(stats["buckets"][0]["spec_interval_count"], 0)
        self.assertEqual(sum(value for key, value in stats["counters"].items() if key.startswith("excluded_")), 15)

    def test_lru_bucket_bound_and_interval_window_bound(self):
        for index in range(256):
            self.controller.observe_batch(index, 1.0, 1, 1)
        self.controller.observe_batch(0, 1.0, 1, 1)
        self.controller.observe_batch(256, 1.0, 1, 1)
        stats = self.controller.stats()
        keys = [row["key"] for row in stats["buckets"]]
        self.assertEqual(stats["cost_bucket_count"], 256)
        self.assertIn("0", keys)
        self.assertNotIn("1", keys)
        for _ in range(300):
            self.controller.observe_interval(256, 1.0, normal=True)
            self.controller.observe_interval(256, 1.0)
        bucket = self.controller.stats()["buckets"][-1]
        self.assertEqual((bucket["normal_interval_count"], bucket["spec_interval_count"]), (128, 128))

    def test_ineligible_budget_forget_reset_and_snapshot(self):
        self.assertEqual(self.next_decision(eligible=False).draft_tokens, 0)
        self.assertEqual(self.next_decision(max_draft_tokens=1).reason, "draft_budget_too_small")
        self.assertEqual(self.next_decision(max_draft_tokens=3).draft_tokens, 2)
        reasons = self.controller.reasons()
        reasons.clear()
        self.assertTrue(self.controller.reasons())
        self.controller.forget("a")
        self.controller.forget("a")
        self.assertEqual(self.controller.stats()["request_count"], 0)
        self.controller.reset()
        stats = self.controller.stats()
        self.assertEqual((stats["round_id"], stats["request_count"], stats["cost_bucket_count"]), (0, 0, 0))
        self.assertEqual(stats["counters"], {})

    def test_invalid_inputs_and_zero_commit_sample(self):
        self.next_decision()
        for value in (float("nan"), float("inf"), -1, 0, True):
            with self.assertRaises(ValueError):
                self.controller.observe_batch(self.bucket, value, 1, 1)
        with self.assertRaises(ValueError):
            self.controller.observe_request("a", 2, 3)
        with self.assertRaises(ValueError):
            self.controller.observe_request("a", 9, 0)
        with self.assertRaises(ValueError):
            self.controller.observe_batch(self.bucket, 1.0, 1, 2, "spec", (2,))
        with self.assertRaises(ValueError):
            self.controller.observe_batch(self.bucket, 1.0, 1, 1, "spec", (3,))
        with self.assertRaises(ValueError):
            Config(ema_alpha=float("nan"))
        with self.assertRaises(ValueError):
            Config(max_cost_buckets=1.5)
        self.controller.observe_batch(self.bucket, 1.0, 0, 1)
        self.assertEqual(self.controller.stats()["counters"]["zero_commit_batches"], 1)


def main():
    """输出独立 CPU 测试摘要；失败返回非零，JSON 拒绝覆盖历史结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new evidence file")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdaptiveTests))
    report = dict(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), device="CPU only",
                  note="Synthetic state-machine tests; not a GPU correctness or performance claim.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
