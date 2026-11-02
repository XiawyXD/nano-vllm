"""不加载 GPU 模型的 HTTP/SSE、取消和并发连接回归测试。"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

from nanovllm.sampling_params import SamplingParams
from nanovllm.server import create_http_server
from nanovllm.streaming import GenerationEvent


class _Tokenizer:
    def decode(self, token_ids):
        return "".join(chr(64 + token_id) for token_id in token_ids)


class _Sequence:
    def __init__(self, seq_id, params):
        self.seq_id = seq_id
        self.params = params
        self.stop = list(params.stop)
        self.num_prompt_tokens = 0
        self.num_completion_tokens = 0
        self.completion_token_ids = []


class _FakeEngine:
    """只模拟 scheduler/step 接口，验证 HTTP 层而非模型数值。"""

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.next_id = 1
        self.active = []
        self.requests = {}
        self._events = []

    def add_request(self, prompt, params):
        seq = _Sequence(self.next_id, params)
        self.next_id += 1
        self.active.append(seq)
        self.requests[seq.seq_id] = seq
        return seq

    def is_finished(self):
        return not self.active

    def step(self):
        time.sleep(0.005)
        self._events = []
        for seq in list(self.active):
            token_id = (seq.num_completion_tokens % 3) + 1
            seq.completion_token_ids.append(token_id)
            seq.num_completion_tokens += 1
            generated_text = self.tokenizer.decode(seq.completion_token_ids)
            matched_stop = next((stop for stop in seq.stop if stop in generated_text), None)
            finished = seq.num_completion_tokens >= seq.params.max_tokens or matched_stop is not None
            self._events.append(GenerationEvent(
                seq.seq_id,
                token_id,
                self.tokenizer.decode([token_id]),
                seq.num_completion_tokens - 1,
                finished,
                "stop" if matched_stop is not None else ("length" if finished else None),
            ))
            if finished:
                self.active.remove(seq)
        return [], -len(self._events)

    def step_events(self):
        events, self._events = self._events, []
        return events

    def get_request(self, request_id):
        return self.requests.get(request_id)

    def forget_request(self, request_id):
        self.requests.pop(request_id, None)

    def cancel_request(self, request_id):
        seq = self.requests.get(request_id)
        if seq is None or seq not in self.active:
            return False
        self.active.remove(seq)
        return True


def _post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode())


def main():
    engine = _FakeEngine()
    server, manager = create_http_server(engine, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        result = _post(base + "/v1/generate", {
            "prompt": "hello",
            "sampling_params": {"temperature": 0, "max_tokens": 3},
        })
        assert result["token_ids"] == [1, 2, 3]

        # urllib 会把 SSE body 当作普通 bytes 读完；逐行解析 data: 事件。
        stream_request = urllib.request.Request(
            base + "/v1/generate",
            data=json.dumps({
                "prompt": "hello",
                "stream": True,
                "sampling_params": {"temperature": 0, "max_tokens": 4},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(stream_request, timeout=5) as response:
            body = response.read().decode()
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        assert len(events) == 4
        assert events[-1]["finished"] and events[-1]["finish_reason"] == "length"

        cancel_body = {}

        def read_long_stream():
            request = urllib.request.Request(
                base + "/v1/generate",
                data=json.dumps({
                    "prompt": "long",
                    "stream": True,
                    "sampling_params": {"temperature": 0, "max_tokens": 1000},
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                cancel_body["raw"] = response.read().decode()

        reader = threading.Thread(target=read_long_stream, daemon=True)
        reader.start()
        request_id = None
        for _ in range(100):
            with manager.lock:
                if manager.records:
                    request_id = max(manager.records)
            if request_id is not None and manager.get(request_id).sequence.params.max_tokens == 1000:
                break
            time.sleep(0.005)
        assert request_id is not None
        cancel_result = _post(base + f"/v1/requests/{request_id}/cancel", {})
        assert cancel_result["cancelled"] is True
        reader.join(timeout=5)
        cancelled_events = [
            json.loads(line[6:])
            for line in cancel_body["raw"].splitlines()
            if line.startswith("data: ")
        ]
        assert cancelled_events[-1]["finished"]
        assert cancelled_events[-1]["finish_reason"] == "cancelled"
        print({
            "status": "PASS",
            "sync_token_ids": result["token_ids"],
            "stream_event_count": len(events),
            "cancel_request_id": request_id,
            "cancel_finish_reason": cancelled_events[-1]["finish_reason"],
        })
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


if __name__ == "__main__":
    main()
