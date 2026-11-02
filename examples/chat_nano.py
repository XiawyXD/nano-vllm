#!/usr/bin/env python3
"""Tiny interactive chat client for the nano-vLLM OpenAI-compatible API."""

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def chat_once(url: str, model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Send the complete conversation history and return the assistant reply."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat with the nano-vLLM server")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="Chat Completions API URL",
    )
    parser.add_argument(
        "--model",
        default="/home/hrxu/huggingface/Qwen3-1.7B",
        help="Model name expected by the server",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    messages: list[dict] = []
    print("nano-vLLM 连续对话已就绪。输入 /reset 清空历史，输入 /exit 退出。")

    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if user_text == "/exit":
            print("已退出。")
            break
        if user_text == "/reset":
            messages.clear()
            print("对话历史已清空。")
            continue
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            answer = chat_once(
                args.url,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
            )
        except HTTPError as error:
            messages.pop()
            detail = error.read().decode("utf-8", errors="replace")
            print(f"\nHTTP {error.code}: {detail}")
            continue
        except (URLError, TimeoutError, KeyError, ValueError) as error:
            messages.pop()
            print(f"\n请求失败：{error}")
            continue

        print(f"\n模型> {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
