"""CPU 上的确定性状态机测试：不受 BF16/GPU batch 数值变化影响。

FakeRunner 同时模拟物理 KV slot，并逐项检查被复用的 prefix 是否正确。
目标规则固定为 1→2→3→1，所以可构造全接受、拒绝和跨 block 情况。
"""
from types import SimpleNamespace
import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class Tokenizer:
    def decode(self, ids):
        return "".join(str(i) for i in ids)


class Runner:
    def __init__(self):
        self.cache = {}
        self.queries = 0

    def call(self, method, seqs, is_prefill=False):
        rows = []
        for seq in seqs:
            def slot(i):
                return seq.block_table[i // seq.block_size], i % seq.block_size
            for i in range(seq.num_cached_tokens):
                assert self.cache[slot(i)] == seq.token_ids[i], ("stale KV", i)
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            for i in range(start, end):
                self.cache[slot(i)] = seq.token_ids[i]
            self.queries += end - start
            ids = [seq.token_ids[i] % 3 + 1 for i in range(start, end)]
            rows.extend(ids if method == "run_speculative" else [ids[-1]])
        if method == "run_speculative":
            return torch.nn.functional.one_hot(torch.tensor(rows), 4).float()
        return rows


def engine(blocks=16, budget=8, eos=-1):
    Sequence.block_size = 4
    e = LLMEngine.__new__(LLMEngine)
    e.config = SimpleNamespace(max_model_len=128)
    e.scheduler = Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=budget,
        prefill_chunk_size=4, max_prefill_steps=1, eos=eos, kvcache_block_size=4,
        num_kvcache_blocks=blocks, enable_prefix_caching=True))
    e.model_runner, e.tokenizer = Runner(), Tokenizer()
    e._requests, e._stream_texts, e._last_speculative_stats = {}, {}, {}
    e._last_step_events = []
    return e


def sp(spec=True, **kw):
    return SamplingParams(temperature=0, max_tokens=kw.pop("max_tokens", 12),
                          speculative_ngram_size=2 if spec else 0, **kw)


def drain(e):
    for _ in range(200):
        if e.is_finished():
            assert not e.scheduler.block_manager.used_block_ids
            return
        before = e.model_runner.queries
        e.step()
        assert e.model_runner.queries - before <= e.scheduler.max_num_batched_tokens
        for seq in e.scheduler.running:
            assert seq.num_cached_tokens == len(seq) - 1
            assert len(seq.block_table) == seq.num_blocks or len(seq.block_table) == seq.num_blocks - 1
    raise AssertionError("scheduler did not finish")


def main():
    for budget in (1, 4, 8):
        e = engine(budget=budget)
        seqs = [e.add_request(p, sp(s)) for p, s in [([1,2,3]*3, True), ([1,2]*4, True), ([3], False)]]
        drain(e)
        for seq in seqs:
            current = seq.prompt_token_ids[-1]
            expected = []
            for _ in range(12):
                current = current % 3 + 1
                expected.append(current)
            assert seq.completion_token_ids == expected
    # stop 在一轮接受的多 token 中间出现；后面的已验证 token 也不得提交。
    e = engine()
    seq = e.add_request([1,2,3]*3, sp(stop="23"))
    drain(e)
    assert seq.completion_token_ids == [1,2,3] and seq.finish_reason == "stop"
    e = engine(eos=2)
    seq = e.add_request([1,2,3]*3, sp())
    drain(e)
    assert seq.completion_token_ids == [1,2] and seq.finish_reason == "stop"
    # KV 空间不足以容纳 draft 时，应降级/缩短，而非耗尽 free deque。
    e = engine(blocks=2)
    seq = e.add_request([1,2,3], sp(max_tokens=5))
    drain(e)
    assert seq.completion_token_ids == [1,2,3,1,2]
    # running 与 waiting 两种取消都必须释放引用，重复取消应返回 False。
    e = engine()
    seq = e.add_request([1,2,3], sp())
    e.step()
    assert e.cancel_request(seq.seq_id) and not e.cancel_request(seq.seq_id)
    seq = e.add_request([1], sp())
    assert e.cancel_request(seq.seq_id)
    drain(e)
    # forward 失败后 draft 历史与临时 block 必须恢复；随后可取消或重试。
    e = engine()
    seq = e.add_request([1,2,3]*3, sp())
    while not seq.num_completion_tokens:
        e.step()
    saved = list(seq.token_ids)
    original_call = e.model_runner.call
    def fail(method, *args):
        if method == "run_speculative":
            raise RuntimeError("injected forward failure")
        return original_call(method, *args)
    e.model_runner.call = fail
    try:
        e.step()
    except RuntimeError as exc:
        assert "injected" in str(exc)
    else:
        raise AssertionError("expected injected failure")
    assert seq.token_ids == saved
    assert seq.num_cached_tokens == len(saved) - 1
    assert len(seq.block_table) == seq.num_blocks
    e.model_runner.call = original_call
    drain(e)
    print("PASS: deterministic tokens, rejection, budgets 1/4/8, stop, EOS, KV pressure, cancellation")
    # 门控不得改变 plain-greedy 目标；高并发应跳过额外 verification。
    e = engine()
    e.config.speculative_max_batch_size = 1
    first = e.add_request([1,2,3]*3, sp())
    second = e.add_request([1,2,3]*3, sp())
    drain(e)
    assert first.completion_token_ids == second.completion_token_ids == [1,2,3]*4
    assert e.speculative_stats()["batch_gate_skips"] > 0
    print("PASS: batch-size gate preserves deterministic greedy tokens")


if __name__ == "__main__":
    main()
