"""用真实本地 tokenizer 测逐 token 解码（无 GPU），包括中英文、emoji 和 stop。"""
import argparse
from types import SimpleNamespace
from transformers import AutoTokenizer
from nanovllm.engine.llm_engine import LLMEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    engine = LLMEngine.__new__(LLMEngine)
    engine.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    engine._stream_texts = {}
    cases = [("你好，CUDA！🦊\n下一行", []), ("a你!后面", ["你!"]),
             ("first 23 after", ["23"]), ("a  b\n\n c\t d", []),
             ("你好世界结束", ["世界"]), ("ababa", ["ba", "aba"]),
             ("Hello, I'm testing.", []), ("literal \ufffd tail", []),
             ("prefix 2", ["23"]), ("emoji 🦊🐱 stop", ["🦊🐱"])]
    for index, (text, stops) in enumerate(cases):
        seq = SimpleNamespace(seq_id=index, completion_token_ids=[], stop=stops, is_finished=False)
        deltas = []
        for token in engine.tokenizer.encode(text, add_special_tokens=False):
            seq.completion_token_ids.append(token)
            deltas.append(engine._stream_delta(seq))
            if any(stop in engine._decode_completion(seq) for stop in stops):
                break
        seq.is_finished = True
        deltas.append(engine._stream_delta(seq))
        expected = engine.completion_text(seq)
        assert "".join(deltas) == expected, (text, deltas, expected)
    print(f"PASS: {len(cases)} real-tokenizer streaming cases")


if __name__ == "__main__":
    main()
