"""启动真实 Qwen3 服务并验证 OpenAI-compatible API。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode())


def post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def read_sse(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode()
    return [
        line[6:] if line[6:] == "[DONE]" else json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--port", type=int, default=18125)
    parser.add_argument("--master-port", type=int, default=2530)
    args = parser.parse_args()
    env = os.environ.copy()
    env.update({"NANOVLLM_MASTER_PORT": str(args.master_port), "HF_HUB_OFFLINE": "1"})
    command = [
        sys.executable, "serve_nano_vllm.py", "--model", args.model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-model-len", "128", "--max-num-seqs", "4",
        "--max-num-batched-tokens", "128", "--enforce-eager",
    ]
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(120):
            try:
                if get_json(base + "/health")["status"] == "ok":
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("real server did not become healthy")

        models = get_json(base + "/v1/models")
        completion = post_json(base + "/v1/completions", {
            "model": "qwen3-0.6b",
            "prompt": "hello",
            "temperature": 0,
            "max_tokens": 3,
        })
        chat = post_json(base + "/v1/chat/completions", {
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Say hello briefly."}],
            "temperature": 0,
            "max_tokens": 3,
        })
        streamed = read_sse(base + "/v1/chat/completions", {
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Say hi."}],
            "temperature": 0,
            "max_tokens": 3,
            "stream": True,
        })
        assert models["object"] == "list" and models["data"]
        assert completion["object"] == "text_completion"
        assert chat["object"] == "chat.completion"
        assert streamed[-1] == "[DONE]"
        assert streamed[-2]["object"] == "chat.completion.chunk"
        print(json.dumps({
            "status": "PASS",
            "model_id": models["data"][0]["id"],
            "completion_finish_reason": completion["choices"][0]["finish_reason"],
            "chat_finish_reason": chat["choices"][0]["finish_reason"],
            "stream_chunk_count": len(streamed) - 1,
            "completion_usage": completion["usage"],
        }, ensure_ascii=False))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
