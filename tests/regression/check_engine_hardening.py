"""无需 GPU 的 stop/UTF-8/admission 回归；用确定性 runner 隔离浮点差异。"""
from types import SimpleNamespace
from tests.regression.test_unified_scheduler import engine, sp, drain


def collect(e, seq):
    deltas = []
    while not e.is_finished():
        e.step()
        deltas.extend(event.text for event in e.step_events())
    assert not e.scheduler.block_manager.used_block_ids
    return deltas


def main():
    for speculative in (False, True):
        e = engine()
        seq = e.add_request([3] if not speculative else [1, 2, 3] * 3,
                            sp(speculative, stop=["23"], max_tokens=8))
        deltas = collect(e, seq)
        assert "".join(deltas) == "1", deltas
        assert e.completion_text(seq) == "1"
        assert seq.finish_reason == "stop"
    # stop 列表顺序不应优先于文本中的实际位置。
    e = engine()
    seq = SimpleNamespace(seq_id=900, completion_token_ids=[1, 2, 3, 4],
                          stop=["34", "2"], is_finished=True)
    assert e.completion_text(seq) == "1"
    # 最终长度停止时，尚未匹配的 stop 前缀必须补发。
    seq = e.add_request([3], sp(False, stop=["23"], max_tokens=2))
    assert "".join(collect(e, seq)) == "12"
    e = engine()
    seq = e.add_request([3], sp(False, stop=["23"], max_tokens=3))
    assert "".join(collect(e, seq)) == "1" and seq.finish_reason == "stop"

    # 模拟一个汉字由多个 byte token 构成，前两个前缀只能解码出替换字符。
    class ByteTokenizer:
        def decode(self, ids, **kw):
            values = {1: b"A", 2: b"\xe4", 3: b"\xbd", 4: b"\xa0", 5: b"!"}
            return b"".join(values[i] for i in ids).decode("utf-8", errors="replace")
    e = engine()
    e.tokenizer = ByteTokenizer()
    seq = SimpleNamespace(seq_id=901, completion_token_ids=[], stop=["你!"], is_finished=False)
    deltas = []
    for token in [1, 2, 3, 4, 5]:
        seq.completion_token_ids.append(token)
        deltas.append(e._stream_delta(seq))
    assert "".join(deltas) == "A", deltas
    seq = SimpleNamespace(seq_id=902, completion_token_ids=[], stop=[], is_finished=False)
    deltas = []
    for token in [1, 2, 3, 4, 5]:
        seq.completion_token_ids.append(token)
        deltas.append(e._stream_delta(seq))
    assert "".join(deltas) == "A你!", deltas

    # 永久不可行在入队前拒绝；正好装得下以及暂时拥塞仍可排队。
    e = engine(blocks=1)
    for prompt, max_tokens in [([1] * 5, 1), ([1] * 4, 2)]:
        try:
            e.add_request(prompt, sp(False, max_tokens=max_tokens))
        except ValueError as exc:
            assert "physical capacity" in str(exc)
        else:
            raise AssertionError("impossible request admitted")
    assert not e._requests and e.is_finished()
    first = e.add_request([1] * 4, sp(False, max_tokens=1))
    second = e.add_request([2] * 4, sp(False, max_tokens=1))
    drain(e)
    assert first.is_finished and second.is_finished
    e = engine()
    results = e.generate([[3]], sp(False, stop=["23"], max_tokens=4), use_tqdm=False)
    assert results[0]["text"] == "1" and not e._requests
    e = engine()
    e.config.hf_config = SimpleNamespace(vocab_size=4)
    for prompt in ([4], [-1], [1.5], [True], {"bad": 1}, 23):
        try:
            e.add_request(prompt, sp(False, max_tokens=1))
        except ValueError:
            pass
        else:
            raise AssertionError(("invalid prompt admitted", prompt))
    stream = e.generate_stream([3], sp(False, max_tokens=8))
    next(stream)
    stream.close()
    assert e.is_finished() and not e._requests and not e.scheduler.block_manager.used_block_ids
    print("PASS: cross-token stop, Unicode tails, earliest stop, terminal flush, KV admission, result release")


if __name__ == "__main__":
    main()
