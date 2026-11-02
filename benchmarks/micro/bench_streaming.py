"""验证请求级 streaming 事件。"""

import argparse
import json
import os

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    llm = LLM(args.model, max_model_len=256, max_num_batched_tokens=256, enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    events = []
    for event in llm.generate_stream("Explain CUDA memory in one short sentence.", params):
        events.append(event.to_dict())
        print(json.dumps(event.to_dict(), ensure_ascii=False))
    assert events
    assert events[-1]["finished"] is True
    assert [event["token_index"] for event in events] == list(range(len(events)))
    print(json.dumps({
        "status": "PASS",
        "event_count": len(events),
        "first_event": events[0],
        "last_event": events[-1],
        "text": "".join(event["text"] for event in events),
    }, ensure_ascii=False))
    llm.exit()


if __name__ == "__main__":
    main()
