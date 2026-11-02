"""标准库 HTTP/SSE serving 层和 OpenAI-compatible API。

一个后台 worker 独占 LLMEngine 的 scheduler/GPU；HTTP handler 只负责入队、
读取事件和取消请求，因此多个 HTTP 连接不会并发调用 CUDA 模型。
"""

from __future__ import annotations

import json
import logging
import math
import select
import socket
from collections import deque
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Full, Queue
from typing import Any
from urllib.parse import urlparse

from nanovllm.sampling_params import SamplingParams
from nanovllm.streaming import GenerationEvent


@dataclass(frozen=True)
class ServiceLimits:
    """服务资源上限。运行时调度限制由引擎 Config 单独负责。"""

    max_inflight_requests: int = 128
    request_timeout_seconds: float = 120.0
    max_stream_events: int = 128
    finished_request_ttl_seconds: float = 300.0
    max_retained_requests: int = 256

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_inflight_requests", "max_stream_events", "max_retained_requests"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")


class ServiceError(RuntimeError):
    """只包含可对客户端公开的信息，内部 traceback 仅进入服务日志。"""

    def __init__(self, message: str, status: int = 503, error_type: str = "service_unavailable"):
        super().__init__(message)
        self.status = status
        self.error_type = error_type


@dataclass
class RequestRecord:
    """队列仅为流式请求创建；history 也使用有界 deque。"""

    request_id: int
    sequence: Any
    public_id: str = ""
    model: str = "nano-vllm"
    endpoint: str = "generate"
    chat: bool = False
    events: Queue | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=0))
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str | None = None
    error_status: int = 503
    error_type: str = "server_error"
    cancelled: bool = False
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    stream_attached: bool = False

    def push(self, event: GenerationEvent) -> bool:
        if self.events is None:
            return True
        try:
            self.events.put_nowait(event)
        except Full:
            return False
        self.history.append(event)
        return True


class RequestManager:
    """单 worker 推进引擎；HTTP 线程通过锁保护提交/取消和元数据。"""

    def __init__(self, engine, model_name: str = "nano-vllm", poll_interval: float = 0.001,
                 limits: ServiceLimits | None = None):
        if getattr(getattr(engine, "config", None), "draft_model", None) is not None:
            raise ValueError("draft_model v1 is Python-only; start HTTP with draft_model disabled")
        self.engine = engine
        self.model_name = model_name
        self.poll_interval = poll_interval
        self.limits = limits or ServiceLimits()
        self.records: dict[int, RequestRecord] = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.healthy = True
        self.counters = dict(requests_total=0, completed=0, cancelled=0, failed=0, rejected=0)
        self.worker = threading.Thread(target=self._run, name="nanovllm-worker", daemon=True)
        self.worker.start()

    def submit(self, prompt: str | list[int], params: SamplingParams, *,
               public_id: str | None = None, model: str | None = None,
               endpoint: str = "generate", chat: bool = False, stream: bool = False) -> RequestRecord:
        with self.lock:
            self._cleanup()
            if not self.healthy or self.stop_event.is_set():
                self.counters["rejected"] += 1
                raise ServiceError("inference service is unavailable")
            if sum(not r.done.is_set() for r in self.records.values()) >= self.limits.max_inflight_requests:
                self.counters["rejected"] += 1
                raise ServiceError("too many unfinished requests", 429, "capacity_exceeded")
            sequence = self.engine.add_request(prompt, params)
            record = RequestRecord(
                request_id=sequence.seq_id, sequence=sequence,
                public_id=public_id or f"req-{sequence.seq_id}", model=model or self.model_name,
                endpoint=endpoint, chat=chat,
                events=Queue(maxsize=self.limits.max_stream_events) if stream else None,
                history=deque(maxlen=self.limits.max_stream_events if stream else 0),
            )
            self.records[sequence.seq_id] = record
            self.counters["requests_total"] += 1
            return record

    def get(self, request_id: int) -> RequestRecord | None:
        with self.lock:
            self._cleanup()
            return self.records.get(request_id)

    def attach_stream(self, record: RequestRecord) -> None:
        """允许事后订阅：先发当前文本快照，再发后续 delta；只允许一个消费者。"""
        with self.lock:
            if record.stream_attached:
                raise ServiceError("a stream consumer is already attached", 409, "stream_conflict")
            record.stream_attached = True
            if record.events is None:
                record.events = Queue(maxsize=self.limits.max_stream_events)
                record.history = deque(maxlen=self.limits.max_stream_events)
                # 活跃请求快照不能泄露仍可能匹配 stop 的文本后缀。
                text = self._text(record.sequence, streaming=not record.done.is_set())
                record.push(GenerationEvent(record.request_id, -1, text,
                                           record.sequence.num_completion_tokens,
                                           record.done.is_set(),
                                           (record.result or {}).get("finish_reason")))

    def _text(self, sequence, *, streaming=False) -> str:
        if hasattr(self.engine, "completion_text"):
            return self.engine.completion_text(sequence, streaming=streaming)
        text = self.engine.tokenizer.decode(sequence.completion_token_ids)
        indexes = [text.find(stop) for stop in getattr(sequence, "stop", []) if stop in text]
        return text[:min(indexes)] if indexes else text

    @staticmethod
    def _usage(sequence) -> dict[str, int]:
        prompt_tokens = int(getattr(sequence, "num_prompt_tokens", 0))
        completion_tokens = int(getattr(sequence, "num_completion_tokens", 0))
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens}

    def _finish(self, record: RequestRecord, reason: str) -> None:
        record.result = {"request_id": record.request_id, "text": self._text(record.sequence),
                         "token_ids": list(record.sequence.completion_token_ids),
                         "finish_reason": reason, "usage": self._usage(record.sequence)}
        record.finished_at = time.monotonic()
        # 完成请求已经释放 KV；立即移除引擎副本，HTTP 结果按 TTL 有界保留。
        self.engine.forget_request(record.request_id)
        record.done.set()

    def _terminate(self, record: RequestRecord, reason: str, message: str | None = None,
                   status: int = 503, error_type: str = "server_error", safe_cancel: bool = True) -> None:
        if record.done.is_set():
            return
        if safe_cancel:
            self.engine.cancel_request(record.request_id)
        record.cancelled = True
        record.error, record.error_status, record.error_type = message, status, error_type
        self.counters["failed" if message else "cancelled"] += 1
        terminal = GenerationEvent(record.request_id, -1, "", record.sequence.num_completion_tokens,
                                   True, reason)
        # 满队列必须腾出终止事件的位置，绝不等待慢客户端消费。
        if record.events is not None:
            while True:
                try:
                    record.events.get_nowait()
                except Empty:
                    break
            record.push(terminal)
        self._finish(record, reason)

    def cancel(self, request_id: int) -> bool:
        with self.lock:
            record = self.records.get(request_id)
            if record is None or record.done.is_set():
                return False
            self._terminate(record, "cancelled")
            return True

    def _complete_event(self, event: GenerationEvent) -> None:
        record = self.records.get(event.request_id)
        if record is None or record.done.is_set():
            return
        if not record.push(event):
            self._terminate(record, "error", "stream consumer is too slow", 429, "slow_consumer")
            return
        if event.finished:
            self.counters["completed"] += 1
            self._finish(record, event.finish_reason or "length")

    def _cleanup(self) -> None:
        """删除过期完成记录，并按完成时间约束最大保留数量。"""
        now = time.monotonic()
        finished = sorted((r for r in self.records.values() if r.done.is_set()),
                          key=lambda r: r.finished_at or r.created_at)
        excess = max(0, len(finished) - self.limits.max_retained_requests)
        for index, record in enumerate(finished):
            if index < excess or now - (record.finished_at or record.created_at) >= self.limits.finished_request_ttl_seconds:
                self.records.pop(record.request_id, None)
                self.engine.forget_request(record.request_id)

    def _expire(self) -> None:
        now = time.monotonic()
        for record in list(self.records.values()):
            if not record.done.is_set() and now - record.created_at >= self.limits.request_timeout_seconds:
                self._terminate(record, "timeout", "request deadline exceeded", 504, "request_timeout")
        self._cleanup()

    def _fail_all(self) -> None:
        """失败后停止 forward；取消仅触碰 CPU scheduler 元数据。"""
        self.healthy = False
        self.stop_event.set()
        for record in list(self.records.values()):
            if record.done.is_set():
                continue
            try:
                self._terminate(record, "error", "inference worker failed", 503, "worker_failed")
            except Exception:
                logging.exception("request cleanup failed after worker failure")
                record.error, record.error_status, record.error_type = "inference worker failed", 503, "worker_failed"
                record.finished_at = time.monotonic()
                record.done.set()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    self._expire()
                    if not self.engine.is_finished():
                        self.engine.step()
                        self._expire()
                        for event in self.engine.step_events():
                            self._complete_event(event)
                    self._cleanup()
                self.stop_event.wait(self.poll_interval)
        except Exception:
            logging.exception("inference worker stopped; service is unhealthy")
            with self.lock:
                self._fail_all()

    def metrics(self) -> dict[str, Any]:
        with self.lock:
            self._cleanup()
            scheduler = getattr(self.engine, "scheduler", None)
            engine_stats = scheduler.stats() if scheduler is not None and hasattr(scheduler, "stats") else {}
            block_manager = getattr(scheduler, "block_manager", None)
            engine_stats = dict(engine_stats)
            engine_stats["kv_ref_total"] = sum(getattr(b, "ref_count", 0) for b in getattr(block_manager, "blocks", []))
            engine_stats["request_records"] = len(getattr(self.engine, "_requests", getattr(self.engine, "requests", {})))
            return {"healthy": self.healthy and not self.stop_event.is_set(),
                    "queue_depth": len(getattr(scheduler, "waiting", [])),
                    "inflight_requests": sum(not r.done.is_set() for r in self.records.values()),
                    "retained_requests": len(self.records),
                    "finished_records": sum(r.done.is_set() for r in self.records.values()),
                    "active_stream_events": sum(r.events.qsize() for r in self.records.values() if r.events is not None),
                    "limits": asdict(self.limits), "engine": engine_stats,
                    "runtime": self.engine.runtime_metrics() if hasattr(self.engine, "runtime_metrics") else {},
                    **self.counters}

    def close(self) -> None:
        self.stop_event.set()
        self.healthy = False
        self.worker.join(timeout=5.0)
        if self.lock.acquire(timeout=1.0):
            try:
                for record in list(self.records.values()):
                    if not record.done.is_set():
                        self._terminate(record, "shutdown", "service is shutting down", 503, "service_shutdown")
                self.records.clear()
            finally:
                self.lock.release()
        else:
            # GPU 调用卡死时不能安全操作引擎，但仍唤醒 HTTP 等待者。
            for record in list(self.records.values()):
                if not record.done.is_set():
                    record.error = "service is shutting down"
                    record.error_type = "service_shutdown"
                    record.done.set()


def _sampling_params(payload: dict[str, Any]) -> SamplingParams:
    if not isinstance(payload, dict):
        raise ValueError("sampling_params must be an object")
    if payload.get("speculative_method") == "draft_model":
        raise ValueError("draft_model v1 is supported only by the Python API")
    allowed = {
        "temperature", "max_tokens", "ignore_eos", "top_k", "top_p",
        "repetition_penalty", "seed", "stop", "speculative_ngram_size",
        "speculative_max_draft_tokens", "speculative_method",
    }
    values = {key: value for key, value in payload.items() if key in allowed and value is not None}
    return SamplingParams(**values)


def _openai_params(payload: dict[str, Any]) -> SamplingParams:
    """把 OpenAI 请求字段映射成 nano-vLLM SamplingParams。"""
    values = {
        "temperature": payload.get("temperature", 1.0),
        "max_tokens": payload.get("max_tokens", payload.get("max_completion_tokens", 64)),
        "top_p": payload.get("top_p", 1.0),
        "top_k": payload.get("top_k", 0),
        "repetition_penalty": payload.get("repetition_penalty", 1.0),
        "seed": payload.get("seed"),
        "stop": payload.get("stop"),
        "speculative_ngram_size": payload.get("speculative_ngram_size", 0),
        "speculative_max_draft_tokens": payload.get("speculative_max_draft_tokens", 4),
        "speculative_method": payload.get("speculative_method"),
    }
    return _sampling_params(values)


def _messages_to_prompt(messages: Any, tokenizer) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if not all(isinstance(item, dict) and "role" in item and "content" in item for item in messages):
        raise ValueError("each message must contain role and content")
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"


class _Handler(BaseHTTPRequestHandler):
    """HTTP JSON + raw SSE + OpenAI-compatible SSE handler。"""

    manager: RequestManager

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, error_type: str = "invalid_request_error") -> None:
        self._json(status, {"error": {"message": message, "type": error_type}})

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size)
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _model_name(self, payload: dict[str, Any]) -> str:
        model = payload.get("model")
        return model if isinstance(model, str) and model else self.manager.model_name

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in {"/health", "/v1/health"}:
                healthy = self.manager.healthy and not self.manager.stop_event.is_set()
                self._json(200 if healthy else 503, {"status": "ok" if healthy else "unhealthy"})
                return
            if path in {"/metrics", "/v1/metrics"}:
                self._json(200, self.manager.metrics())
                return
            if path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{"id": self.manager.model_name, "object": "model", "owned_by": "nano-vllm"}],
                })
                return
            if path.startswith("/v1/requests/") and path.endswith("/events"):
                request_id = int(path.split("/")[3])
                record = self.manager.get(request_id)
                if record is None:
                    self._error(404, "unknown request_id", "not_found_error")
                    return
                self._stream_raw(record)
                return
            if path.startswith("/v1/requests/"):
                request_id = int(path.split("/")[3])
                record = self.manager.get(request_id)
                if record is None:
                    self._error(404, "unknown request_id", "not_found_error")
                    return
                self._json(200, {
                    "request_id": request_id,
                    "public_id": record.public_id,
                    "done": record.done.is_set(),
                    "cancelled": record.cancelled,
                    "result": record.result,
                    "error": ({"message": record.error, "type": record.error_type} if record.error else None),
                })
                return
            self._error(404, "not found", "not_found_error")
        except ServiceError as exc:
            self._error(exc.status, str(exc), exc.error_type)
        except (TypeError, ValueError) as exc:
            self._error(400, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._body()
            if path in {"/generate", "/v1/generate"}:
                self._handle_raw_generate(payload)
                return
            if path == "/v1/completions":
                self._handle_openai(payload, chat=False)
                return
            if path == "/v1/chat/completions":
                self._handle_openai(payload, chat=True)
                return
            if path.startswith("/v1/requests/") and path.endswith("/cancel"):
                request_id = int(path.split("/")[3])
                if self.manager.cancel(request_id):
                    self._json(200, {"request_id": request_id, "cancelled": True})
                else:
                    self._error(404, "request is missing, finished, or already cancelled", "not_found_error")
                return
            self._error(404, "not found", "not_found_error")
        except ServiceError as exc:
            self._error(exc.status, str(exc), exc.error_type)
        except (TypeError, ValueError, KeyError) as exc:
            self._error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return
        except Exception:
            logging.exception("HTTP request failed")
            self._error(500, "internal server error", "server_error")

    def _handle_raw_generate(self, payload: dict[str, Any]) -> None:
        prompt = payload.get("prompt")
        if not isinstance(prompt, (str, list)):
            raise ValueError("prompt must be a string or token-id list")
        params = _sampling_params(payload.get("sampling_params", {}))
        record = self.manager.submit(prompt, params, stream=bool(payload.get("stream", False)))
        if payload.get("stream", False):
            self._stream_raw(record)
        else:
            if self._wait_result(record):
                self._json(200, record.result or {})

    def _handle_openai(self, payload: dict[str, Any], *, chat: bool) -> None:
        if payload.get("n", 1) != 1:
            raise ValueError("only n=1 is currently supported")
        if chat:
            prompt = _messages_to_prompt(payload.get("messages"), self.manager.engine.tokenizer)
            prefix, endpoint = "chatcmpl", "chat.completions"
        else:
            prompt = payload.get("prompt")
            if not isinstance(prompt, (str, list)):
                raise ValueError("prompt must be a string or token-id list")
            prefix, endpoint = "cmpl", "completions"
        params = _openai_params(payload)
        record = self.manager.submit(
            prompt,
            params,
            public_id=f"{prefix}-{uuid.uuid4().hex}",
            model=self._model_name(payload),
            endpoint=endpoint,
            chat=chat,
            stream=bool(payload.get("stream", False)),
        )
        if payload.get("stream", False):
            self._stream_openai(record)
        else:
            if self._wait_result(record):
                self._json(200, self._openai_result(record))

    def _client_disconnected(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable and not self.connection.recv(1, socket.MSG_PEEK))
        except OSError:
            return True

    def _wait_result(self, record: RequestRecord) -> bool:
        """等待结果时检测断连；worker 的 deadline/失败同样会唤醒这里。"""
        while not record.done.wait(0.1):
            if self._client_disconnected():
                self.manager.cancel(record.request_id)
                return False
        if record.error:
            self._error(record.error_status, record.error, record.error_type)
            return False
        return True

    @staticmethod
    def _created() -> int:
        return int(time.time())

    def _openai_result(self, record: RequestRecord) -> dict[str, Any]:
        result = record.result or {}
        finish_reason = result.get("finish_reason", "length")
        usage = result.get("usage", {})
        if record.chat:
            return {
                "id": record.public_id,
                "object": "chat.completion",
                "created": self._created(),
                "model": record.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.get("text", "")},
                    "finish_reason": finish_reason,
                }],
                "usage": usage,
            }
        return {
            "id": record.public_id,
            "object": "text_completion",
            "created": self._created(),
            "model": record.model,
            "choices": [{
                "index": 0,
                "text": result.get("text", ""),
                "logprobs": None,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    def _stream_raw(self, record: RequestRecord) -> None:
        self.manager.attach_stream(record)
        try:
            self._start_sse()
            while True:
                if record.error:
                    self._write_sse({"error": {"message": record.error, "type": record.error_type},
                                     "request_id": record.request_id})
                    break
                try:
                    event = record.events.get(timeout=0.2)
                except Empty:
                    if self._client_disconnected():
                        self.manager.cancel(record.request_id)
                        break
                    if record.done.is_set():
                        break
                    continue
                if record.error:
                    continue
                self._write_sse(event.to_dict())
                if event.finished:
                    break
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.manager.cancel(record.request_id)
        finally:
            record.stream_attached = False

    def _stream_openai(self, record: RequestRecord) -> None:
        self.manager.attach_stream(record)
        role_sent = False
        try:
            self._start_sse()
            while True:
                if record.error:
                    self._write_sse({"error": {"message": record.error, "type": record.error_type},
                                     "request_id": record.request_id})
                    self._write_sse("[DONE]")
                    break
                try:
                    event = record.events.get(timeout=0.2)
                except Empty:
                    if self._client_disconnected():
                        self.manager.cancel(record.request_id)
                        break
                    if record.done.is_set():
                        break
                    continue
                if record.error:
                    continue
                finish_reason = event.finish_reason if event.finished else None
                if record.chat:
                    delta: dict[str, str] = {}
                    if not role_sent:
                        delta["role"] = "assistant"
                        role_sent = True
                    if event.text:
                        delta["content"] = event.text
                    payload = {
                        "id": record.public_id,
                        "object": "chat.completion.chunk",
                        "created": self._created(),
                        "model": record.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                    }
                else:
                    payload = {
                        "id": record.public_id,
                        "object": "text_completion",
                        "created": self._created(),
                        "model": record.model,
                        "choices": [{"index": 0, "text": event.text, "logprobs": None, "finish_reason": finish_reason}],
                    }
                self._write_sse(payload)
                if event.finished:
                    self._write_sse("[DONE]")
                    break
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.manager.cancel(record.request_id)
        finally:
            record.stream_attached = False

    def _start_sse(self) -> None:
        self.connection.settimeout(5.0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    def _write_sse(self, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False) if payload != "[DONE]" else payload
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()


def create_http_server(
    engine,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = "nano-vllm",
    limits: ServiceLimits | None = None,
):
    """创建 `(ThreadingHTTPServer, RequestManager)`，便于测试或嵌入。"""
    manager = RequestManager(engine, model_name=model_name, limits=limits)
    handler = type("NanoVLLMHandler", (_Handler,), {"manager": manager})
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except Exception:
        manager.close()
        raise
    server.request_manager = manager
    return server, manager


def serve(engine, host: str = "0.0.0.0", port: int = 8000, model_name: str = "nano-vllm",
          limits: ServiceLimits | None = None) -> None:
    """阻塞运行 HTTP/SSE 服务，进程退出时释放 worker。"""
    server, manager = create_http_server(engine, host, port, model_name=model_name, limits=limits)
    try:
        print(f"nano-vLLM HTTP/SSE listening on http://{host}:{port}", flush=True)
        server.serve_forever()
    finally:
        manager.close()
        server.server_close()
