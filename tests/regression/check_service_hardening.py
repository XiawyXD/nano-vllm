"""CPU/fake-engine regressions for bounded HTTP request lifecycles.

Run: PYTHONPATH=. python tests/regression/check_service_hardening.py
"""
from __future__ import annotations

import json
import logging
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.regression.check_http_sse import _FakeEngine
from nanovllm.sampling_params import SamplingParams
from nanovllm.server import RequestManager, ServiceError, ServiceLimits, create_http_server


class ControlledEngine(_FakeEngine):
    def __init__(self, paused=False, fail=False):
        super().__init__()
        self.paused, self.fail = paused, fail

    def is_finished(self):
        return self.paused or super().is_finished()

    def step(self):
        if self.fail:
            raise RuntimeError('private CUDA failure details must not reach clients')
        return super().step()


def eventually(predicate, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError('condition did not become true')


class LifecycleTests(unittest.TestCase):
    def manager(self, **kwargs):
        engine = kwargs.pop('engine', ControlledEngine())
        manager = RequestManager(engine, poll_interval=0.001, limits=ServiceLimits(**kwargs))
        self.addCleanup(manager.close)
        return manager, engine

    def test_limits_and_capacity(self):
        defaults = ServiceLimits()
        self.assertEqual(defaults.max_inflight_requests, 128)
        self.assertEqual(defaults.max_retained_requests, 256)
        for bad in (0, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                ServiceLimits(request_timeout_seconds=bad)
        manager, engine = self.manager(engine=ControlledEngine(paused=True), max_inflight_requests=1)
        first = manager.submit('held', SamplingParams(max_tokens=20))
        with self.assertRaises(ServiceError) as error:
            manager.submit('excess', SamplingParams(max_tokens=20))
        self.assertEqual(error.exception.status, 429)
        self.assertTrue(manager.cancel(first.request_id))
        self.assertEqual(engine.requests, {})

    def test_nonstream_allocates_no_events_or_history(self):
        manager, engine = self.manager()
        record = manager.submit('ordinary', SamplingParams(max_tokens=16))
        self.assertTrue(record.done.wait(3))
        self.assertIsNone(record.events)
        self.assertEqual(len(record.history), 0)
        self.assertEqual(len(record.result['token_ids']), 16)
        self.assertEqual(engine.requests, {})

    def test_slow_stream_does_not_block_other_requests(self):
        manager, engine = self.manager(max_stream_events=2)
        slow = manager.submit('slow', SamplingParams(max_tokens=100), stream=True)
        fast = manager.submit('fast', SamplingParams(max_tokens=8))
        self.assertTrue(slow.done.wait(3))
        self.assertTrue(fast.done.wait(3))
        self.assertEqual(slow.error_type, 'slow_consumer')
        self.assertLessEqual(slow.events.qsize(), 2)
        self.assertLessEqual(len(slow.history), 2)
        self.assertIsNone(fast.error)
        self.assertEqual(engine.requests, {})
        self.assertTrue(manager.healthy)

    def test_retention_bound_and_expiry(self):
        manager, engine = self.manager(max_retained_requests=2, finished_request_ttl_seconds=0.05)
        for _ in range(6):
            record = manager.submit('one', SamplingParams(max_tokens=1))
            self.assertTrue(record.done.wait(3))
        self.assertLessEqual(manager.metrics()['retained_requests'], 2)
        self.assertEqual(engine.requests, {})
        eventually(lambda: manager.metrics()['retained_requests'] == 0)

    def test_deadline_releases_paused_request(self):
        manager, engine = self.manager(engine=ControlledEngine(paused=True), request_timeout_seconds=0.025)
        record = manager.submit('held', SamplingParams(max_tokens=10), stream=True)
        self.assertTrue(record.done.wait(3))
        self.assertEqual(record.error_status, 504)
        self.assertEqual(record.error_type, 'request_timeout')
        self.assertFalse(engine.active)
        self.assertEqual(engine.requests, {})

    def test_worker_failure_wakes_all_and_rejects_new(self):
        engine = ControlledEngine(paused=True, fail=True)
        manager, _ = self.manager(engine=engine)
        records = [manager.submit('fail', SamplingParams(max_tokens=10), stream=True) for _ in range(3)]
        with self.assertLogs(level='ERROR'):
            engine.paused = False
            for record in records:
                self.assertTrue(record.done.wait(3))
        self.assertFalse(manager.healthy)
        self.assertFalse(engine.active)
        self.assertFalse(engine.requests)
        for record in records:
            self.assertEqual(record.error, 'inference worker failed')
        with self.assertRaises(ServiceError) as error:
            manager.submit('new', SamplingParams())
        self.assertEqual(error.exception.status, 503)

    def test_close_cancels_and_forgets_pending(self):
        manager, engine = self.manager(engine=ControlledEngine(paused=True))
        record = manager.submit('held', SamplingParams(max_tokens=5))
        manager.close()
        self.assertTrue(record.done.is_set())
        self.assertEqual(record.error_type, 'service_shutdown')
        self.assertFalse(engine.active)
        self.assertFalse(engine.requests)
        self.assertFalse(manager.records)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.engine = ControlledEngine()
        self.server, self.manager = create_http_server(
            self.engine, port=0, limits=ServiceLimits(max_inflight_requests=1))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.manager.close()
        self.server.server_close()

    def request(self, path, payload=None):
        request = urllib.request.Request(self.base + path, data=(json.dumps(payload).encode() if payload is not None else None),
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_http_capacity_health_metrics(self):
        self.engine.paused = True
        record = self.manager.submit('held', SamplingParams(max_tokens=100))
        status, body = self.request('/v1/generate', {'prompt': 'overload'})
        self.assertEqual(status, 429)
        self.assertEqual(body['error']['type'], 'capacity_exceeded')
        status, body = self.request('/metrics')
        self.assertEqual(status, 200)
        self.assertEqual(body['inflight_requests'], 1)
        self.assertEqual(body['limits']['max_stream_events'], 128)
        self.manager.cancel(record.request_id)
        with self.manager.lock:
            self.manager._fail_all()
        self.assertEqual(self.request('/health')[0], 503)
        self.assertEqual(self.request('/v1/generate', {'prompt': 'after_failure'})[0], 503)

    def test_nonstream_disconnect_cancels_request(self):
        self.engine.paused = True
        client = socket.create_connection(self.server.server_address, timeout=3)
        body = json.dumps({'prompt': 'held', 'sampling_params': {'max_tokens': 100}}).encode()
        headers = f'POST /v1/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\nContent-Type: application/json\r\n\r\n'.encode()
        client.sendall(headers + body)
        eventually(lambda: self.manager.metrics()['inflight_requests'] == 1)
        client.close()
        eventually(lambda: self.manager.metrics()['inflight_requests'] == 0)
        self.assertFalse(self.engine.active)
        self.assertFalse(self.engine.requests)

    def test_worker_error_is_sanitized_http(self):
        self.engine.fail = True
        with self.assertLogs(level='ERROR'):
            status, body = self.request('/v1/generate', {'prompt': 'fail'})
        self.assertEqual(status, 503)
        self.assertEqual(body['error']['message'], 'inference worker failed')
        self.assertNotIn('private', json.dumps(body))

    def test_bad_sampling_is_400_and_service_stays_healthy(self):
        for params in ({'max_tokens': 1.5}, {'temperature': float('nan')}, {'stop': ''}, 'not an object'):
            status, _ = self.request('/v1/generate', {'prompt': 'invalid', 'sampling_params': params})
            self.assertEqual(status, 400)
        self.assertTrue(self.manager.healthy)
        self.assertFalse(self.engine.requests)

    def test_sse_disconnect_before_first_token_cancels(self):
        self.engine.paused = True
        client = socket.create_connection(self.server.server_address, timeout=3)
        body = json.dumps({'prompt': 'held', 'stream': True}).encode()
        headers = f'POST /v1/generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\nContent-Type: application/json\r\n\r\n'.encode()
        client.sendall(headers + body)
        self.assertIn(b'200 OK', client.recv(4096))
        client.close()
        eventually(lambda: self.manager.metrics()['inflight_requests'] == 0)
        self.assertFalse(self.engine.active)
        self.assertFalse(self.engine.requests)

    def test_openai_sse_worker_error_has_terminal_marker(self):
        self.engine.fail = True
        payload = {'prompt': 'fail', 'stream': True, 'max_tokens': 5}
        request = urllib.request.Request(self.base + '/v1/completions',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with self.assertLogs(level='ERROR'):
            with urllib.request.urlopen(request, timeout=3) as response:
                body = response.read().decode()
        chunks = [line[6:] for line in body.splitlines() if line.startswith('data: ')]
        self.assertEqual(chunks[-1], '[DONE]')
        self.assertEqual(json.loads(chunks[0])['error']['type'], 'worker_failed')
        self.assertNotIn('private', body)


if __name__ == '__main__':
    unittest.main(verbosity=2)
