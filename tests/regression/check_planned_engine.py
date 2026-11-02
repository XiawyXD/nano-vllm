"""混合引擎 CPU 状态机回归：真实 planner，假 forward/物理 KV，不加载模型。

运行：CUDA_VISIBLE_DEVICES='' python -m tests.regression.check_planned_engine
仅检查协议：追加 --protocol-only。结果不能代替 BF16 数值验证或真实 TP2。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

# 必须在 import torch/nanovllm 前屏蔽 GPU，防止测试意外启动实际模型。
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from nanovllm.engine.execution_plan import ExecutionItem, ExecutionPlan, ExecutionRole, StepStats
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.planned_scheduler import PlannedScheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.tp_runtime import encode_call, decode_call
from nanovllm.sampling_params import SamplingParams


class DigitTokenizer:
    """每个 ID 解码成可见数字，便于检查跨 token 的 stop 字符串。"""

    def decode(self, ids, **kwargs):
        return "".join(str(token) for token in ids)


class FixedDraft:
    """只用于反例测试，明确控制接受/拒绝位置，不模拟真实 proposer 质量。"""

    def __init__(self, tokens):
        self.tokens = tuple(tokens)
        self.calls = 0

    def propose(self, token_ids, max_tokens=None):
        self.calls += 1
        return list(self.tokens[:max_tokens])


class FakePlanRunner:
    """把物理 KV 位置保存到字典，并根据当前输入确定性地产生预测。

    访问已缓存历史时逐项检查真实 token，因而能发现前缀复用、回滚和 trim
    的 slot 映射错误；未接受 draft 可以残留在物理字典，但不应再成为有效 KV。
    """

    def __init__(self, block_size, rule=None):
        self.block_size = block_size
        self.rule = rule or (lambda token: token % 3 + 1)
        self.cache = {}
        self.plans = []
        self.calls = 0
        self.queries = 0
        self.sampled_rows = 0
        self.fail_after_write = False
        self.last_plan_stats = {}

    def call(self, method, plan, sequences):
        assert method == "run_plan", ("mixed mode used a legacy forward", method)
        assert isinstance(plan, ExecutionPlan)
        by_id = {seq.seq_id: seq for seq in sequences}
        assert set(by_id) == set(plan.request_ids)
        self.calls += 1
        self.plans.append(plan)
        result = {}
        for item in plan.items:
            seq = by_id[item.request_id]
            assert tuple(seq.block_table) == item.block_table

            def slot(position):
                return item.block_table[position // self.block_size], position % self.block_size

            for position in range(item.start_pos):
                assert self.cache.get(slot(position)) == seq.token_ids[position], (
                    "stale or missing committed KV", item.request_id, position)
            if item.role is ExecutionRole.VERIFY:
                # 草稿只在 plan 里，绝不能暂时污染真实 Sequence.token_ids。
                assert item.start_pos == len(seq) - 1
                assert item.input_token_ids == (seq.last_token,) + item.draft
            else:
                assert item.input_token_ids == tuple(seq.token_ids[item.start_pos:item.end_pos])
            for relative, token in enumerate(item.input_token_ids):
                self.cache[slot(item.start_pos + relative)] = token
            result[item.request_id] = [self.rule(item.input_token_ids[row])
                                       for row in item.logit_positions]
        self.queries += plan.num_tokens
        self.sampled_rows += len(plan.logits_indices)
        self.last_plan_stats = dict(path="cpu_fake", captured=False, metadata_grew=False,
                                    query_tokens=plan.num_tokens, forward_calls=1)
        if self.fail_after_write:
            raise RuntimeError("injected forward failure after temporary KV writes")
        return result


def make_engine(*, blocks=64, block_size=4, budget=8, chunk=2, seqs=8,
                eos=-1, rule=None, policy="fixed"):
    """绕开模型构造，只初始化真实引擎 CPU 状态及真实 PlannedScheduler。"""
    Sequence.block_size = block_size
    config = SimpleNamespace(
        max_model_len=256, max_num_seqs=seqs, max_num_batched_tokens=budget,
        prefill_chunk_size=chunk, max_prefill_steps=0, eos=eos,
        kvcache_block_size=block_size, num_kvcache_blocks=blocks,
        enable_prefix_caching=True, prefill_max_wait_steps=2,
        speculative_max_batch_size=0, speculative_policy=policy,
        adaptive_max_batch_size=16, batching_mode="mixed", runtime_preset="optimized",
        tensor_parallel_size=1, kv_cache_dtype="model", enforce_eager=True,
        metadata_backend="reuse", metadata_max_bytes=2**20,
        enable_mixed_graph=False, mixed_attention_backend="split",
        hf_config=SimpleNamespace(vocab_size=128),
    )
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = config
    engine.scheduler = PlannedScheduler(config)
    engine.model_runner = FakePlanRunner(block_size, rule)
    engine.tokenizer = DigitTokenizer()
    engine._requests = {}
    engine._stream_texts = {}
    engine._last_speculative_stats = {}
    engine._last_step_events = []
    engine._packed_enabled = True
    engine._init_runtime_state()
    return engine


def params(spec=False, *, maximum=8, **kwargs):
    return SamplingParams(temperature=0, max_tokens=maximum,
                          speculative_ngram_size=2 if spec else 0,
                          speculative_max_draft_tokens=4, **kwargs)


def take_step(engine):
    """每个非空计划只调用一次 runner；结构化统计与实际增量互相核对。"""
    previous_calls = engine.model_runner.calls
    previous = {key: seq.num_completion_tokens for key, seq in engine._requests.items()}
    outputs, stats = engine.step()
    assert isinstance(stats, StepStats), type(stats)
    assert isinstance(engine._last_step_stats, StepStats)
    assert asdict(stats) == asdict(engine._last_step_stats)
    assert asdict(stats) == asdict(engine.scheduler.last_step_stats)
    assert engine.model_runner.calls == previous_calls + 1
    plan = engine.model_runner.plans[-1]
    assert stats.scheduled_prefill == plan.prefill_tokens
    assert stats.query_decode == plan.decode_tokens
    assert stats.query_verify == plan.verification_tokens
    assert stats.forward_calls == 1
    assert stats.generated_tokens == sum(
        seq.num_completion_tokens - previous[key] for key, seq in engine._requests.items())
    assert stats.mixed == int(plan.decode_tokens > 0 and (plan.has_prefill or plan.has_verification))
    assert not engine.scheduler._active_items and not engine.scheduler._reservations
    assert not engine.scheduler._plan_sequences
    for seq in engine.scheduler.running:
        assert seq.num_cached_tokens == len(seq) - 1
    return outputs, stats, plan


def drain(engine, maximum=100):
    texts = []
    for _ in range(maximum):
        if engine.is_finished():
            assert not engine.scheduler.block_manager.used_block_ids
            return texts
        take_step(engine)
        texts.extend(event.text for event in engine.step_events())
    raise AssertionError("planned engine did not finish within the CPU step limit")


def sequence_snapshot(seq):
    return tuple(seq.token_ids), tuple(seq.block_table), seq.num_cached_tokens, seq.num_scheduled_tokens


def check_partial_prefill():
    engine = make_engine(budget=2, chunk=2)
    seq = engine.add_request([1, 2, 3, 1, 2], params(maximum=3))
    for expected_cached in (2, 4):
        outputs, stats, plan = take_step(engine)
        assert plan.kind == "P_PREFILL" and not plan.logits_indices
        assert stats.generated_tokens == stats.query_decode == stats.query_verify == 0
        assert seq.num_cached_tokens == expected_cached
        assert not seq.completion_token_ids and not outputs and not engine.step_events()
        assert engine.model_runner.sampled_rows == 0
    _, stats, plan = take_step(engine)
    assert plan.prefill_tokens == 1 and stats.generated_tokens == 1
    assert seq.completion_token_ids == [3]
    drain(engine)
    assert seq.completion_token_ids == [3, 1, 2]


def check_prefill_decode_single_forward():
    engine = make_engine(budget=4, chunk=2)
    running = engine.add_request([1], params(maximum=6))
    take_step(engine)
    waiting = engine.add_request([11, 12, 13, 14, 15], params(maximum=3))
    before = engine.model_runner.calls
    _, stats, plan = take_step(engine)
    assert engine.model_runner.calls == before + 1
    assert plan.kind == "PD_PACKED"
    assert (stats.scheduled_prefill, stats.query_decode, stats.query_verify) == (2, 1, 0)
    assert stats.generated_tokens == 1 and stats.mixed == 1
    assert waiting.num_cached_tokens == 2 and not waiting.completion_token_ids
    assert running.completion_token_ids == [2, 3]
    assert [event.request_id for event in engine.step_events()] == [running.seq_id]
    drain(engine)


def check_decode_verify_accept_reject():
    cases = [([2, 3], [2, 3, 1], 2), ([2, 9, 9], [2, 3], 1), ([9, 9], [2], 0)]
    for draft, expected, accepted in cases:
        engine = make_engine(chunk=None)
        spec = engine.add_request([1, 2, 3], params(True, maximum=10))
        ordinary = engine.add_request([3], params(maximum=10))
        take_step(engine)
        proposer = FixedDraft(draft)
        spec._ngram_proposer = proposer
        old_history = list(spec.token_ids)
        _, stats, plan = take_step(engine)
        assert plan.kind == "DV_PACKED" and plan.decode_tokens == 1
        assert plan.verification_tokens == len(draft) + 1 and proposer.calls == 1
        assert spec.token_ids == old_history + expected
        assert ordinary.completion_token_ids == [1, 2]
        assert stats.generated_tokens == len(expected) + 1
        assert stats.accepted_draft_tokens == accepted
        assert stats.mixed == 1
        assert len(engine.step_events()) == 2
        assert all(event.token_index == engine.get_request(event.request_id).num_completion_tokens - 1
                   for event in engine.step_events())
        # draft 拒绝后可立即继续普通 D；错误的临时 KV 不得变成真实前缀。
        spec.speculative_ngram_size = 0
        drain(engine)


def check_speculative_stopping():
    cases = [dict(eos=2, stop=None, maximum=8, expected=[1, 2], reason="stop", text="12"),
             dict(eos=-1, stop="23", maximum=8, expected=[1, 2, 3], reason="stop", text="1"),
             dict(eos=-1, stop=None, maximum=3, expected=[1, 2, 3], reason="length", text="123"),
             dict(eos=-1, stop="23", maximum=3, expected=[1, 2, 3], reason="stop", text="1")]
    for case in cases:
        engine = make_engine(chunk=None, eos=case["eos"])
        seq = engine.add_request([1, 2, 3], params(True, maximum=case["maximum"], stop=case["stop"]))
        take_step(engine)
        deltas = [event.text for event in engine.step_events()]
        seq._ngram_proposer = FixedDraft([2, 3, 1, 2])
        _, stats, plan = take_step(engine)
        deltas.extend(event.text for event in engine.step_events())
        assert plan.has_verification and seq.is_finished
        assert seq.completion_token_ids == case["expected"]
        assert seq.finish_reason == case["reason"]
        assert engine.completion_text(seq) == "".join(deltas) == case["text"]
        assert stats.generated_tokens == len(case["expected"]) - 1
        assert not engine.scheduler.block_manager.used_block_ids
        assert seq._ngram_proposer is None
    # 未命中的 stop 前缀在长度结束时必须最终 flush。
    engine = make_engine(chunk=None)
    seq = engine.add_request([3], params(maximum=2, stop="23"))
    assert "".join(drain(engine)) == "12" and engine.completion_text(seq) == "12"


def check_unicode_streaming():
    class ByteTokenizer:
        def decode(self, ids, **kwargs):
            values = {1: b"A", 2: b"\xe4", 3: b"\xbd", 4: b"\xa0", 5: b"!"}
            return b"".join(values[token] for token in ids).decode("utf-8", errors="replace")

    rule = lambda token: 1 if token == 8 else token + 1
    engine = make_engine(chunk=None, rule=rule)
    engine.tokenizer = ByteTokenizer()
    seq = engine.add_request([8], params(True, maximum=8, stop="你!"))
    take_step(engine)
    deltas = [event.text for event in engine.step_events()]
    seq._ngram_proposer = FixedDraft([2, 3, 4, 5])
    _, stats, plan = take_step(engine)
    deltas.extend(event.text for event in engine.step_events())
    assert seq.completion_token_ids == [1, 2, 3, 4, 5]
    assert "".join(deltas) == engine.completion_text(seq) == "A"
    assert stats.generated_tokens == 4 and plan.has_verification
    assert seq.is_finished and not engine.scheduler.block_manager.used_block_ids
    engine = make_engine(chunk=None, rule=rule)
    engine.tokenizer = ByteTokenizer()
    seq = engine.add_request([8], params(maximum=5))
    assert "".join(drain(engine)) == engine.completion_text(seq) == "A你!"


def check_failure_rollback_and_cancel():
    engine = make_engine(chunk=None, block_size=2)
    spec = engine.add_request([1, 2, 3], params(True, maximum=12))
    ordinary = engine.add_request([3], params(maximum=8))
    take_step(engine)
    spec._ngram_proposer = FixedDraft([2, 3, 1, 2])
    seqs = [spec, ordinary]
    before = [sequence_snapshot(seq) for seq in seqs]
    used = set(engine.scheduler.block_manager.used_block_ids)
    engine.model_runner.fail_after_write = True
    try:
        engine.step()
    except RuntimeError as exc:
        assert "injected forward failure" in str(exc)
    else:
        raise AssertionError("forward failure was swallowed")
    assert [sequence_snapshot(seq) for seq in seqs] == before
    assert engine.scheduler.block_manager.used_block_ids == used
    assert not engine.scheduler._active_items and not engine.scheduler._reservations
    assert not engine.scheduler._plan_sequences
    engine.model_runner.fail_after_write = False
    take_step(engine)
    for seq in seqs:
        assert engine.cancel_request(seq.seq_id)
        assert not engine.cancel_request(seq.seq_id)
        assert getattr(seq, "_ngram_proposer", None) is None
    assert engine.is_finished() and not engine.scheduler.block_manager.used_block_ids
    # 只有部分 P 已提交的 waiting 请求，也可以释放逐步分配的页。
    engine = make_engine(chunk=2)
    partial = engine.add_request([10, 11, 12, 13, 14], params())
    take_step(engine)
    assert partial.num_cached_tokens == 2 and partial.block_table
    assert engine.cancel_request(partial.seq_id)
    waiting = engine.add_request([8], params())
    assert engine.cancel_request(waiting.seq_id)
    assert engine.is_finished() and not engine.scheduler.block_manager.used_block_ids


def check_generator_close_and_prefix_reuse():
    engine = make_engine(chunk=2)
    generator = engine.generate_stream([1, 2, 3], params(maximum=8))
    event = next(generator)
    assert event.text == "1"
    generator.close()
    assert engine.is_finished() and not engine._requests
    assert not engine.scheduler.block_manager.used_block_ids
    # 第一请求完成后，第二请求复用完整已提交页；fake KV 逐 token 校验值。
    engine = make_engine(chunk=None, block_size=2)
    first = engine.add_request([1, 2, 3, 1, 2], params(maximum=1))
    take_step(engine)
    assert first.is_finished
    second = engine.add_request([1, 2, 3, 1, 2], params(maximum=2))
    _, stats, plan = take_step(engine)
    assert plan.items[0].start_pos == 4 and stats.scheduled_prefill == 1
    drain(engine)
    assert second.completion_token_ids == [3, 1]


def check_stream_external_cancellation():
    """yield 后取消请求，再 next 必须退出；不能在空 scheduler 上无限 step。"""
    engine = make_engine(chunk=None)
    stream = engine.generate_stream([1, 2, 3], params(maximum=8))
    first = next(stream)
    assert not first.finished
    seq = engine.get_request(first.request_id)
    assert engine.cancel_request(first.request_id) and seq.is_finished
    # 取消发生在 step 边界；取消后的 next 不应再执行一次 target forward。
    with patch.object(engine, "step", side_effect=AssertionError("cancelled stream kept stepping")):
        try:
            next(stream)
        except StopIteration:
            pass
        else:
            raise AssertionError("cancelled stream produced another event")
    assert engine.is_finished() and not engine._requests
    assert not engine.scheduler.block_manager.used_block_ids
    assert not engine._stream_texts and not engine._event_times


def check_tp_execution_plan_protocol():
    """TP 只依赖 plan 的本轮完整显式输入，不依赖 legacy Sequence 的压缩历史。"""
    items = (
        ExecutionItem(101, ExecutionRole.VERIFY, (17, 18, 19), 7,
                      (3, 4, 8, 10, 11), (18, 19), (0, 1, 2)),
        ExecutionItem(102, ExecutionRole.DECODE, (23,), 5, (12, 13, 14), (), (0,)),
    )
    plan = ExecutionPlan(items, step_id=42)
    # legacy decode 序列仍按旧协议只发送 last_token；worker 使用 plan 的页表和输入。
    seq = Sequence([7, 8, 9, 10], params())
    seq.is_prefill = False
    seq.num_cached_tokens = 3
    seq.num_scheduled_tokens = 1
    seq.block_table = [1]
    original = tuple(seq.token_ids)
    packet = encode_call("run_plan", (plan, [seq]), 65536)
    method, arguments = decode_call(packet + b"\0" * 32)
    restored, worker_sequences = arguments
    assert method == "run_plan" and restored == plan
    assert restored.items[0].input_token_ids == (17, 18, 19)
    assert restored.items[0].start_pos == 7 and restored.items[0].end_pos == 10
    assert restored.items[0].block_table == (3, 4, 8, 10, 11)
    assert restored.logits_layout == ((101, 0, 3), (102, 3, 4))
    assert restored.logits_indices == (0, 1, 2, 3)
    assert tuple(seq.token_ids) == original
    assert worker_sequences[0].token_ids == [] and worker_sequences[0].last_token == 10
    partial = ExecutionPlan((ExecutionItem(103, ExecutionRole.PREFILL, (1, 2), 4,
                            (1, 2, 3), (), ()),), step_id=43)
    assert decode_call(encode_call("run_plan", (partial, []), 65536))[1][0] == partial
    assert not partial.logits_indices
    try:
        encode_call("run_plan", (plan, []), 8)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized plan payload was accepted")


class TickClock:
    """每次读钟推进固定2ms；测试策略使用确定性整轮成本，不依赖CPU机器快慢。"""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        self.now += 0.002
        return self.now


def adaptive_snapshot(controller):
    """忽略真实wall-clock观测，只比较应该在P轮保持不变的控制/成本状态。"""
    result = controller.stats(include_requests=True)
    result.pop("elapsed_s", None)
    return result


def warm_normal_cost(engine, seqs):
    """为KV边界反例准备8个合格整批样本；不伪造被测轮的请求级接受反馈。"""
    bucket = engine._adaptive_bucket(seqs)
    for _ in range(8):
        engine.adaptive_controller.observe_batch(bucket, 2.0, len(seqs), len(seqs),
                                                  "normal", (0,) * len(seqs))
    return bucket


def mark_shape_warm(engine, kind, roles_and_queries, page_counts):
    """边界反例显式排除首形状JIT预热，避免测试只走excluded而没执行学习分支。"""
    engine._seen_plan_shapes[(kind, tuple(roles_and_queries), tuple(page_counts))] = None


def check_adaptive_calibration_and_prefill():
    engine = make_engine(policy="adaptive", block_size=128, chunk=None, budget=16)
    seqs = [engine.add_request([1, 2, 3], params(True, maximum=120)) for _ in range(2)]
    with patch("nanovllm.engine.llm_engine.perf_counter", TickClock()):
        take_step(engine)
        assert engine.adaptive_controller.stats()["round_id"] == 0
        assert engine.adaptive_controller.stats()["cost_bucket_count"] == 0
        verification_rounds = []
        for round_id in range(1, 34):
            before = engine.adaptive_controller.stats()
            qualified = sum(row["normal_samples"] for row in before["buckets"])
            _, stats, plan = take_step(engine)
            assert engine.adaptive_controller.stats()["round_id"] == round_id
            if qualified < 8:
                assert plan.kind == "D_NORMAL" and not plan.has_verification
                assert stats.generated_tokens == len(seqs)
            if round_id % 16 == 0:
                # 必须整批校准，而非一条D一条V混在一起仍当normal成本。
                assert plan.kind == "D_NORMAL" and stats.query_verify == 0
                assert stats.generated_tokens == len(seqs)
            if plan.has_verification:
                verification_rounds.append(round_id)
            if round_id == 9:
                # 首个D形状被记作JIT排除，随后才积累8个合格普通整批样本。
                snapshot = engine.adaptive_controller.stats()
                assert sum(row["normal_samples"] for row in snapshot["buckets"]) == 8
                assert snapshot["counters"]["excluded_batch_jit"] >= 1
        assert verification_rounds and min(verification_rounds) == 10
        assert 16 not in verification_rounds and 32 not in verification_rounds
        # 插入一个新的partial P；P+D仍可单forward，但不推进控制器轮次或成本。
        waiting = engine.add_request([11] * 20, params(maximum=4))
        engine.scheduler.prefill_chunk_size = 2
        before = adaptive_snapshot(engine.adaptive_controller)
        _, stats, plan = take_step(engine)
        assert plan.kind == "PD_PACKED" and not plan.has_verification
        assert waiting.num_cached_tokens == 2 and stats.query_decode == 2
        assert adaptive_snapshot(engine.adaptive_controller) == before
    for seq in list(engine._requests.values()):
        if not seq.is_finished:
            engine.cancel_request(seq.seq_id)
    assert not engine.scheduler.block_manager.used_block_ids


def check_adaptive_opt_in_preserves_ordinary_params():
    engine = make_engine(policy="adaptive", block_size=128, chunk=None, budget=16)
    spec = engine.add_request([1, 2, 3] * 3, params(True, maximum=80))
    ordinary_params = SamplingParams(temperature=0.8, max_tokens=80, top_k=5,
                                     top_p=0.9, repetition_penalty=1.2, seed=11)
    ordinary = engine.add_request([3], ordinary_params)
    preserved = (ordinary.temperature, ordinary.top_k, ordinary.top_p,
                 ordinary.repetition_penalty, ordinary.seed)
    saw_verification = False
    with patch("nanovllm.engine.llm_engine.perf_counter", TickClock()):
        take_step(engine)
        for _ in range(18):
            _, _, plan = take_step(engine)
            by_id = {item.request_id: item for item in plan.items}
            assert by_id[ordinary.seq_id].role is ExecutionRole.DECODE
            assert by_id[ordinary.seq_id].draft == ()
            assert ordinary.seq_id not in engine.adaptive_controller._requests
            assert (ordinary.temperature, ordinary.top_k, ordinary.top_p,
                    ordinary.repetition_penalty, ordinary.seed) == preserved
            saw_verification |= by_id[spec.seq_id].role is ExecutionRole.VERIFY
    assert saw_verification
    # 本测试验证策略/采样参数路由，不声称fake target验证了随机采样分布。
    for seq in (spec, ordinary):
        engine.cancel_request(seq.seq_id)
    assert not engine.adaptive_controller._requests


def check_adaptive_full_promotion_with_cold_exclusions():
    """真实engine采样闭环达到实际0/2/4/8；模拟冷耗时不能进入成本EMA。"""
    engine = make_engine(policy="adaptive", block_size=128, chunk=None, budget=16, seqs=1)
    seq = engine.add_request([1, 2, 3], SamplingParams(temperature=0, max_tokens=200,
        speculative_ngram_size=2, speculative_max_draft_tokens=8))

    class PerfectDraft:
        # 隔离proposer质量变量；只验证控制器与planner之间实际长度/成本反馈。
        def propose(self, token_ids, max_tokens):
            token, result = token_ids[-1], []
            for _ in range(max_tokens):
                token = token % 3 + 1
                result.append(token)
            return result

    class ManualClock:
        now = 100.0

        def __call__(self):
            return self.now

    clock, observations = ManualClock(), {}
    original = engine.model_runner.call

    def measured_call(method, plan, sequences):
        result = original(method, plan, sequences)
        length = len(plan.items[0].draft)
        key = plan.kind, length
        count = observations[key] = observations.get(key, 0) + 1
        # 新L4首形状仅靠engine的JIT排除；其余新形状模拟双帧扩容。
        grow = count <= 2 and not (length == 4 and count == 1)
        capture = bool(length and count == 3)
        cold = grow or capture or (length == 4 and count == 1)
        clock.now += 1.0 if cold else 0.002
        engine.model_runner.last_plan_stats = dict(metadata_grew=grow, captured=capture)
        return result

    with patch.object(engine.model_runner, "call", measured_call), patch(
            "nanovllm.engine.llm_engine.perf_counter", clock):
        take_step(engine)
        seq._ngram_proposer = PerfectDraft()
        actual = []
        for _ in range(40):
            before = engine.adaptive_controller.stats()
            _, _, plan = take_step(engine)
            length = len(plan.items[0].draft)
            actual.append(length)
            after = engine.adaptive_controller.stats()
            if not before["buckets"] or before["buckets"][0]["normal_samples"] < 8:
                assert length == 0, "normal warmup was bypassed"
            if after["round_id"] % 16 == 0:
                assert length == 0, "whole-batch normal calibration was bypassed"
            if length == 8:
                break
        assert set(actual) == {0, 2, 4, 8}, actual
        assert actual.index(2) < actual.index(4) < actual.index(8)
        snapshot = engine.adaptive_controller.stats(include_requests=True)
        assert snapshot["cost_bucket_count"] == 1
        cost = snapshot["buckets"][0]
        assert abs(cost["normal_ms_per_token"] - 2.0) < 1e-8, cost
        assert 0 < cost["spec_ms_per_token"] <= 2.0 / 3 + 1e-8, cost
        for reason in ("grow", "capture", "jit"):
            assert snapshot["counters"].get(f"excluded_batch_{reason}", 0) > 0
            assert snapshot["counters"].get(f"excluded_request_{reason}", 0) > 0
        assert snapshot["counters"]["promotions"] == 2
    engine.cancel_request(seq.seq_id)
    assert not engine.scheduler.block_manager.used_block_ids
    assert not engine.adaptive_controller._requests


def check_adaptive_kv_and_empty_feedback():
    # 总共两页，两条请求各占一页：长度3时还剩一格，L2因KV不足缩短成L1。
    engine = make_engine(policy="adaptive", blocks=2, block_size=4, chunk=None, budget=8)
    spec = engine.add_request([1, 2], params(True, maximum=6))
    ordinary = engine.add_request([10, 11], params(maximum=6))
    take_step(engine)
    spec._ngram_proposer = FixedDraft([1, 2])
    warm_normal_cost(engine, [spec, ordinary])
    mark_shape_warm(engine, "DV_PACKED", [(ExecutionRole.VERIFY, 2), (ExecutionRole.DECODE, 1)], [1, 1])
    original = engine.adaptive_controller.observe_request
    with patch.object(engine.adaptive_controller, "observe_request", wraps=original) as observed:
        with patch("nanovllm.engine.llm_engine.perf_counter", TickClock()):
            _, stats, plan = take_step(engine)
    assert plan.items[0].draft == (1,) and stats.accepted_draft_tokens == 1
    observed.assert_called_once_with(spec.seq_id, 1, 1, excluded_reason=None)
    request = engine.adaptive_controller._requests[spec.seq_id]
    assert request.observed_lengths == {1} and request.prefix_probabilities[:2] == [1.0, None]
    assert request.low_streak == 0
    for seq in (spec, ordinary):
        engine.cancel_request(seq.seq_id)

    # 长度4时已无空格；整批退回D，必须反馈0/0，而不是把没执行的L2算成拒绝。
    for empty_source in ("kv_capacity", "empty_proposer"):
        engine = make_engine(policy="adaptive", blocks=2 if empty_source == "kv_capacity" else 8,
                             block_size=4, chunk=None, budget=8)
        spec = engine.add_request([1, 2, 3], params(True, maximum=6))
        ordinary = engine.add_request([10, 11, 12], params(maximum=6))
        take_step(engine)
        spec._ngram_proposer = FixedDraft([2, 3] if empty_source == "kv_capacity" else [])
        warm_normal_cost(engine, [spec, ordinary])
        mark_shape_warm(engine, "D_NORMAL", [(ExecutionRole.DECODE, 1)] * 2, [1, 1])
        original = engine.adaptive_controller.observe_request
        with patch.object(engine.adaptive_controller, "observe_request", wraps=original) as observed:
            with patch("nanovllm.engine.llm_engine.perf_counter", TickClock()):
                _, stats, plan = take_step(engine)
        assert plan.kind == "D_NORMAL" and stats.generated_tokens == 2
        observed.assert_called_once_with(spec.seq_id, 0, 0, excluded_reason=None)
        request = engine.adaptive_controller._requests[spec.seq_id]
        assert request.observations == request.low_streak == 0
        assert request.prefix_probabilities == [None] * 8
        assert engine.adaptive_controller.stats()["counters"]["no_draft_samples"] == 1
        for seq in (spec, ordinary):
            engine.cancel_request(seq.seq_id)

    # query预算只剩1个额外token时，0/2/4/8控制器不会擅自把L2塞进去。
    engine = make_engine(policy="adaptive", chunk=None, budget=3)
    spec = engine.add_request([3], params(True, maximum=8))
    ordinary = engine.add_request([3], params(maximum=8))
    take_step(engine)
    spec._ngram_proposer = FixedDraft([2, 3])
    warm_normal_cost(engine, [spec, ordinary])
    with patch("nanovllm.engine.llm_engine.perf_counter", TickClock()):
        _, _, plan = take_step(engine)
    assert plan.kind == "D_NORMAL" and spec._ngram_proposer.calls == 0
    assert engine.adaptive_controller._requests[spec.seq_id].last_decision.reason == "draft_budget_too_small"
    for seq in (spec, ordinary):
        engine.cancel_request(seq.seq_id)


def check_adaptive_request_cleanup_and_preempt():
    for finish in (False, True):
        engine = make_engine(policy="adaptive", chunk=None, block_size=128)
        seq = engine.add_request([1, 2, 3], params(True, maximum=3))
        take_step(engine)
        with patch("nanovllm.engine.llm_engine.perf_counter", TickClock()):
            take_step(engine)
            assert seq.seq_id in engine.adaptive_controller._requests
            assert seq.seq_id in engine._event_times
            if finish:
                take_step(engine)
                assert seq.is_finished
            else:
                assert engine.cancel_request(seq.seq_id)
        assert seq.seq_id not in engine.adaptive_controller._requests
        assert seq.seq_id not in engine._event_times
        assert not engine.scheduler.block_manager.used_block_ids

    # 小池迫使最老请求抢占已partial-P的年轻请求；旧控制器/事件窗口立即forget。
    engine = make_engine(policy="adaptive", blocks=3, block_size=2, budget=2, chunk=2)
    oldest = engine.add_request([1, 2, 3], params(maximum=3))
    younger = engine.add_request([10, 11, 12], params(maximum=3))
    take_step(engine)
    take_step(engine)
    assert younger.block_table and younger.num_cached_tokens == 2
    controller = engine.adaptive_controller
    controller.begin_round()
    controller.decide(younger.seq_id, (1, 1, "seeded-before-preemption"))
    engine._event_times[younger.seq_id] = (0.0, True, "old-bucket")
    saw_preempt = False
    with patch("nanovllm.engine.llm_engine.perf_counter", TickClock()):
        for _ in range(8):
            take_step(engine)
            if younger.seq_id in engine.scheduler.last_preempted_ids:
                saw_preempt = True
                assert younger.seq_id not in controller._requests
                assert younger.seq_id not in engine._event_times
                break
    assert saw_preempt and oldest.is_finished
    drain(engine)
    assert not controller._requests and not engine._event_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    checks = [check_tp_execution_plan_protocol]
    if not args.protocol_only:
        if not hasattr(LLMEngine, "_step_planned") or not hasattr(LLMEngine, "_init_runtime_state"):
            raise RuntimeError("planned engine integration is not present; use --protocol-only during integration")
        checks += [check_partial_prefill, check_prefill_decode_single_forward,
                   check_decode_verify_accept_reject, check_speculative_stopping,
                   check_unicode_streaming, check_failure_rollback_and_cancel,
                   check_generator_close_and_prefix_reuse,
                   check_stream_external_cancellation,
                   check_adaptive_calibration_and_prefill,
                   check_adaptive_opt_in_preserves_ordinary_params,
                   check_adaptive_full_promotion_with_cold_exclusions,
                   check_adaptive_kv_and_empty_feedback,
                   check_adaptive_request_cleanup_and_preempt]
    for check in checks:
        check()
    import torch
    assert not torch.cuda.is_initialized(), "CPU regression unexpectedly initialized CUDA"
    print(json.dumps(dict(status="PASS", device="CPU", checks=[check.__name__ for check in checks],
                          scope="fake target; no GPU numeric accuracy or actual TP2 claim"), indent=2))


if __name__ == "__main__":
    main()
