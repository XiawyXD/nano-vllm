"""OpenAI-compatible completions/chat completions 的 HTTP 回归测试。"""

from __future__ import annotations

import json
import threading
import urllib.request

from tests.regression.check_http_sse import _FakeEngine
from nanovllm.server import create_http_server


def request_json(url, payload=None):
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode())


def read_sse(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read().decode()
    return [
        line[6:] if line[6:] == "[DONE]" else json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def main():
    engine = _FakeEngine()
    server, manager = create_http_server(engine, "127.0.0.1", 0, model_name="qwen3-demo")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        models = request_json(base + "/v1/models")
        completion = request_json(base + "/v1/completions", {
            "model": "qwen3-demo",
            "prompt": "hello",
            "temperature": 0,
            "max_tokens": 3,
        })
        chat = request_json(base + "/v1/chat/completions", {
            "model": "qwen3-demo",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_tokens": 3,
        })
        stopped = request_json(base + "/v1/completions", {
            "model": "qwen3-demo",
            "prompt": "hello",
            "temperature": 0,
            "max_tokens": 10,
            "stop": "B",
        })
        streamed = read_sse(base + "/v1/chat/completions", {
            "model": "qwen3-demo",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_tokens": 3,
            "stream": True,
        })
        assert models["data"][0]["id"] == "qwen3-demo"
        assert completion["object"] == "text_completion"
        assert completion["choices"][0]["text"] == "ABC"
        assert completion["usage"]["completion_tokens"] == 3
        assert chat["object"] == "chat.completion"
        assert chat["choices"][0]["message"]["role"] == "assistant"
        assert stopped["choices"][0]["text"] == "A"
        assert stopped["choices"][0]["finish_reason"] == "stop"
        assert streamed[-1] == "[DONE]"
        assert streamed[0]["object"] == "chat.completion.chunk"
        assert streamed[-2]["choices"][0]["finish_reason"] == "length"
        print({
            "status": "PASS",
            "models": models["data"][0]["id"],
            "completion_finish_reason": completion["choices"][0]["finish_reason"],
            "chat_finish_reason": chat["choices"][0]["finish_reason"],
            "stop_text": stopped["choices"][0]["text"],
            "stream_chunk_count": len(streamed) - 1,
        })
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


if __name__ == "__main__":
    main()
