"""CPU-only checks of the soak harness; never evidence of actual model/GPU health."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from benchmarks import runtime_v2_soak as soak


class SoakHarnessChecks(unittest.TestCase):
    def test_process_identity_does_not_require_fd_access(self):
        stat = "123 (test name) " + " ".join(["S", "12"] + ["0"]*17 + ["9999"])
        with patch.object(Path, "read_text", return_value=stat), \
             patch.object(Path, "iterdir", side_effect=PermissionError("fd closed")) as descriptor:
            row = soak.process_info(123, resources=False)
        self.assertEqual(row["birth"], "9999")
        self.assertEqual(row["ppid"], 12)
        descriptor.assert_not_called()

    def test_fd_permission_failure_is_unknown_not_zero(self):
        stat = "123 (test name) " + " ".join(["S", "12"] + ["0"]*17 + ["9999"])
        with patch.object(Path, "read_text", side_effect=[stat, "VmRSS: 100 kB\nThreads: 2\n"]), \
             patch.object(Path, "iterdir", side_effect=PermissionError("fd closing")):
            row = soak.process_info(123)
        self.assertEqual(row["birth"], "9999")
        self.assertEqual(row["rss_kib"], 100)
        self.assertIsNone(row["fd_count"])
        self.assertEqual(row["fd_error"], "PermissionError")

    def test_write_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"result.json"
            soak.write_new(path, {"status":"FAIL"})
            with self.assertRaises(FileExistsError):
                soak.write_new(path, {"status":"PASS"})
            self.assertEqual(json.loads(path.read_text())["status"], "FAIL")

    def test_classify_expected_fault_only_in_edge(self):
        for kind, name, expected in (("capacity","capacity_exceeded","EXPECTED_CAPACITY"),
                                     ("timeout","request_timeout","EXPECTED_TIMEOUT"),
                                     ("slow","slow_consumer","EXPECTED_SLOW_CONSUMER")):
            item = dict(http_status=429, service_error={"type":name}, expected_tokens=8)
            self.assertEqual(soak.classify(item, kind), expected)
            self.assertEqual(soak.classify(item, "complete"), "UNEXPECTED")

    def test_cancel_requires_real_accepted_id_result(self):
        item = dict(expected_tokens=8, finish_reason="cancelled")
        self.assertEqual(soak.classify(item, "cancel"), "UNEXPECTED")
        item["cancel_status"] = 200
        self.assertEqual(soak.classify(item, "cancel"), "EXPECTED_CANCEL")
        item["error"] = "transport failure"
        self.assertEqual(soak.classify(item, "cancel"), "UNEXPECTED")

    def test_no_early_completion_pass(self):
        item = dict(expected_tokens=8, finish_reason="length", observed_output_tokens=7)
        self.assertEqual(soak.classify(item, "complete"), "UNEXPECTED")
        item["observed_output_tokens"] = 8
        self.assertEqual(soak.classify(item, "complete"), "COMPLETED")
        item["cancel_status"] = 200
        self.assertEqual(soak.classify(item, "cancel"), "UNEXPECTED")

    def test_limits_are_explicit_and_restart_equal(self):
        self.assertEqual(soak.phase_limits("steady",5), soak.phase_limits("restart",5))
        self.assertEqual(soak.phase_limits("steady",5)["request_timeout_seconds"],120)
        self.assertEqual(soak.phase_limits("edge_capacity",5)["max_inflight_requests"],1)
        self.assertEqual(soak.phase_limits("edge_timeout",5)["request_timeout_seconds"],.001)
        self.assertEqual(soak.phase_limits("edge_slow",5)["max_stream_events"],2)

    def test_frozen_config_and_workload(self):
        config, phases, prompts = {"x":1}, {"steady":{}}, {"short":[[1]]}
        manifest = dict(engine_config=config, phases=phases,prompts=prompts,source_sha256={"a":"b"},
            config_sha256=soak.object_sha(dict(engine=config,phases=phases)),prompts_sha256=soak.object_sha(prompts),environment={})
        with patch.object(soak,"source_hashes",return_value={"a":"b"}):
            soak.validate_manifest(manifest, verify_model=False)
            config["x"] = 2
            with self.assertRaisesRegex(RuntimeError,"configuration changed"):
                soak.validate_manifest(manifest, verify_model=False)

    def test_guard_error_stops_new_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            abort = threading.Event()
            monitor = soak.Resources(Path(folder), abort, 1234)
            with patch.object(monitor,"check_guard",side_effect=RuntimeError("foreign GPU process")):
                monitor.thread.start()
                self.assertTrue(abort.wait(2))
                monitor.finish()
            self.assertTrue(monitor.errors)
            self.assertIn("foreign",monitor.errors[0]["error"])

    def test_idle_requires_kv_refs_and_engine_records_zero(self):
        item = dict(inflight_requests=0,queue_depth=0,engine=dict(used_blocks=0,kv_ref_total=0,request_records=0))
        self.assertTrue(soak.is_idle(item))
        item["engine"]["kv_ref_total"] = 1
        self.assertFalse(soak.is_idle(item))
        item["engine"].update(kv_ref_total=0,request_records=1)
        self.assertFalse(soak.is_idle(item))

    def test_reused_leader_cannot_reclaim_unrelated_subtree(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor = soak.Resources(Path(folder),threading.Event(),1234)
            monitor.process = SimpleNamespace(pid=100)
            monitor.leader_birth = "old"
            monitor.known = {100:"old",101:"child-old"}
            foreign = [dict(pid=100,birth="new"),dict(pid=102,birth="foreign-child")]
            current = {100:foreign[0],101:dict(pid=101,birth="child-old"),102:foreign[1]}
            with patch.object(soak,"descendants",return_value=foreign), patch.object(soak,"process_info",side_effect=lambda pid, **kwargs: current.get(pid)):
                rows = monitor.owned_snapshot()
            self.assertEqual(rows,[current[101]])
            self.assertNotIn(102,monitor.known)
            self.assertEqual(monitor.known[100],"old")
            monitor.log.close()

    def test_real_local_http_protocol_uses_received_request_id(self):
        """A CPU HTTP fixture exercises parser/cancel protocol, NOT the production LLM."""
        cancelled, paths = threading.Event(), []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length",0)))
                paths.append(self.path)
                if self.path.endswith("/cancel"):
                    self.send_response(200); self.end_headers()
                    self.wfile.write(b'{"request_id":731,"cancelled":true}')
                    cancelled.set()
                    return
                self.send_response(200)
                self.send_header("Content-Type","text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"request_id":731,"token_id":10,"token_index":0,"text":"","finished":false}\n\n')
                self.wfile.flush()
                if not cancelled.wait(4): return
                self.wfile.write(b'data: {"request_id":731,"token_id":-1,"token_index":1,"text":"","finished":true,"finish_reason":"cancelled"}\n\n')
                self.wfile.flush()
        server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                raw = soak.JsonLines(Path(folder)/"raw.jsonl")
                result = soak.http_request(f"http://127.0.0.1:{server.server_port}",
                    dict(number=0,kind="cancel",prompt=[1,2],max_tokens=8),threading.Event(),raw)
                raw.close()
            self.assertEqual(result["outcome"],"EXPECTED_CANCEL")
            self.assertIn("/v1/requests/731/cancel",paths)
            self.assertEqual(result["observed_output_tokens"],1)
            self.assertIsNotNone(result["ttft_ms"])
            self.assertIsNone(result["first_text_ms"])
        finally:
            server.shutdown(); server.server_close(); thread.join(2)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SoakHarnessChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
        scope="CPU utility and local fake-HTTP protocol only; no actual model/GPU/server-resource acceptance")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
