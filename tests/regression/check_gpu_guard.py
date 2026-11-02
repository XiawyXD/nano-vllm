"""GPU1 guard的纯CPU mock回归：不执行nvidia-smi，不读取真实进程，不启动GPU。

模拟进程树、CUDA_VISIBLE_DEVICES和nvidia-smi CSV，验证fail-closed及JSONL
追加证据。此测试不是设备锁，也不能消除实际检查与启动之间的竞争窗口。
"""
from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout, redirect_stderr
import builtins
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load_guard():
    """只加载标准库guard文件，不导入nanovllm模型或torch。"""
    path = (Path(__file__).resolve().parents[2] / "benchmarks/runtime_v2_gpu_guard.py")
    spec = importlib.util.spec_from_file_location("_runtime_v2_gpu_guard_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_benchmark(name):
    """读取入口模块供CPU mock检查，不执行模块main或启动任何子进程。"""
    path = (Path(__file__).resolve().parents[2] / ("tests/regression" if name.startswith("check_") else "benchmarks") / (name + ".py"))
    spec = importlib.util.spec_from_file_location("_entry_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARD = load_guard()
GPU1_UUID = "GPU-11111111-2222-3333-4444-555555555555"
GPU0_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class GuardTests(unittest.TestCase):
    """每个测试使用虚构PID；所有外部命令均由setUp中的mock拦截。"""

    def setUp(self):
        self.parents = {4242: 4000, 4000: 3000, 3000: 1,
                        5000: 4242, 5001: 5000, 6000: 9000, 9000: 1}
        self.gpu_rows = f"0, {GPU0_UUID}, 500, 30\n1, {GPU1_UUID}, 0, 0\n"
        self.app_rows = ""
        self.env = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU1_UUID,
                                         "NANOVLLM_BENCH_GPU_INDEX": "1"})
        self.pid = patch.object(GUARD.os, "getpid", return_value=4242)
        self.parent = patch.object(GUARD, "_parent", side_effect=lambda pid: self.parents.get(pid))
        self.runner = patch.object(GUARD.subprocess, "run", side_effect=self.mock_command)
        self.env.start()
        self.pid.start()
        self.parent.start()
        self.command = self.runner.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.pid.stop)
        self.addCleanup(self.parent.stop)
        self.addCleanup(self.runner.stop)

    def mock_command(self, args, **kwargs):
        """只允许guard约定的两种查询，并核对查询有超时与错误检查。"""
        self.assertEqual(args[0], "nvidia-smi")
        self.assertTrue(kwargs["check"] and kwargs["capture_output"] and kwargs["text"])
        self.assertEqual(kwargs["timeout"], 10)
        if args[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout=self.gpu_rows)
        if args[1].startswith("--query-compute-apps="):
            return SimpleNamespace(stdout=self.app_rows)
        self.fail(f"unexpected external command: {args}")

    def test_only_exact_physical_gpu1_uuid(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU1_UUID}):
            result = GUARD.ensure_gpu1_available()
            self.assertEqual(result["status"], "AVAILABLE")
            self.assertEqual(result["gpu"]["index"], "1")
        for visible in ("0", "1", "2", "1,2", "0,1", "", "01", "GPU-111", GPU0_UUID):
            self.command.reset_mock()
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}):
                with self.assertRaises(GUARD.GPUUnavailable) as caught:
                    GUARD.ensure_gpu1_available()
                self.assertIn(GPU1_UUID, str(caught.exception))
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], visible,
                                 "guard must not silently rewrite the frozen environment")
            self.assertEqual(self.command.call_count, 1, "wrong env must fail before process query")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GUARD.GPUUnavailable):
                GUARD.ensure_gpu1_available()

    def test_this_task_descendants_and_verified_ancestor_owner(self):
        self.app_rows = f"{GPU1_UUID}, 4242, 400\n{GPU1_UUID}, 5001, 800\n"
        for owner in (None, 4242, 4000):
            result = GUARD.ensure_gpu1_available(owner_pid=owner)
            self.assertEqual(result["status"], "AVAILABLE")
            self.assertEqual(len(result["compute_processes"]), 2)
            self.assertEqual(result["foreign_processes"], [])
        # 即使9000确实拥有GPU进程，它并非当前任务祖先，不能借其PID绕过边界。
        self.command.reset_mock()
        with self.assertRaises(GUARD.GPUUnavailable):
            GUARD.ensure_gpu1_available(owner_pid=9000)
        self.assertEqual(self.command.call_count, 0)

    def test_init_owner_is_not_a_task_boundary(self):
        # init是几乎所有进程的祖先；允许owner=1会把别人的GPU进程都放行。
        self.app_rows = f"{GPU1_UUID}, 6000, 900\n"
        self.command.reset_mock()
        for owner in (1, 0, -1, True, 4000.0, "4000"):
            with self.assertRaises(GUARD.GPUUnavailable):
                GUARD.ensure_gpu1_available(owner_pid=owner)
        self.assertEqual(self.command.call_count, 0)

    def test_foreign_or_unverifiable_process_denied(self):
        for process in (6000, 7000):
            self.app_rows = f"{GPU1_UUID}, 5001, 20\n{GPU1_UUID}, {process}, 500\n"
            with self.assertRaises(GUARD.GPUUnavailable):
                GUARD.ensure_gpu1_available()
        # 他人的GPU0进程不被误认成GPU1占用；不使用用户名判断所有权。
        self.app_rows = f"{GPU0_UUID}, 6000, 2000\n{GPU1_UUID}, 5001, 20\n"
        result = GUARD.ensure_gpu1_available()
        self.assertEqual([row["pid"] for row in result["compute_processes"]], ["5001"])

    def test_query_errors_and_malformed_responses_fail_closed(self):
        for error in (FileNotFoundError("mock missing nvidia-smi"),
                      subprocess.CalledProcessError(1, "nvidia-smi"),
                      subprocess.TimeoutExpired("nvidia-smi", 10)):
            self.command.side_effect = error
            with self.assertRaises(GUARD.GPUUnavailable):
                GUARD.ensure_gpu1_available()
        self.command.side_effect = self.mock_command
        self.gpu_rows = "1, missing-columns\n"
        with self.assertRaises(GUARD.GPUUnavailable):
            GUARD.ensure_gpu1_available()
        self.gpu_rows = f"0, {GPU0_UUID}, 0, 0\n"
        with self.assertRaises(GUARD.GPUUnavailable):
            GUARD.ensure_gpu1_available()
        self.gpu_rows = f"1, {GPU1_UUID}, 0, 0\n"
        self.app_rows = f"{GPU1_UUID}, not-a-pid, 10\n"
        with self.assertRaises((GUARD.GPUUnavailable, ValueError)):
            GUARD.ensure_gpu1_available()

    def test_jsonl_appends_available_and_denied_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observations.jsonl"
            with patch.object(GUARD.time, "time", return_value=1234.5):
                GUARD.ensure_gpu1_available(evidence_path=path)
            first = path.read_bytes()
            self.app_rows = f"{GPU1_UUID}, 6000, 700\n"
            with patch.object(GUARD.time, "time", return_value=1235.5):
                with self.assertRaises(GUARD.GPUUnavailable):
                    GUARD.ensure_gpu1_available(evidence_path=path)
            self.assertTrue(path.read_bytes().startswith(first))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["status"] for row in rows], ["AVAILABLE", "OCCUPIED"])
            self.assertEqual([row["timestamp"] for row in rows], [1234.5, 1235.5])
            self.assertEqual(rows[1]["owner_pid"], 4242)
            self.assertEqual(rows[1]["foreign_processes"][0]["pid"], "6000")

    def test_ancestry_missing_cycle_and_depth_are_denied(self):
        self.assertTrue(GUARD.is_descendant(5001, 4242))
        self.assertFalse(GUARD.is_descendant(7000, 4242))
        self.parents.update({7001: 7002, 7002: 7001})
        self.assertFalse(GUARD.is_descendant(7001, 4242))
        with patch.object(GUARD, "_parent", side_effect=lambda pid: pid - 1):
            self.assertFalse(GUARD.is_descendant(9999, 4242))

    def test_proc_stat_parser_handles_spaces_and_parentheses(self):
        # _parent当前被setUp替换，单独载入一份实现测试文本解析。
        parser_module = load_guard()
        with patch.object(parser_module.Path, "read_text", return_value="5001 (name ) with spaces) S 5000 0 0"):
            self.assertEqual(parser_module._parent(5001), 5000)
        for value in ("broken", "5001 (name) S not-an-int"):
            with patch.object(parser_module.Path, "read_text", return_value=value):
                self.assertIsNone(parser_module._parent(5001))
        with patch.object(parser_module.Path, "read_text", side_effect=PermissionError("mock")):
            self.assertIsNone(parser_module._parent(5001))

    def test_explicit_selected_gpu_preserves_occupancy_and_uuid_checks(self):
        self.gpu_rows += f"2, GPU-selected-two, 0, 0\n"
        with patch.dict(os.environ, NANOVLLM_BENCH_GPU_INDEX="2",
                        CUDA_VISIBLE_DEVICES="GPU-selected-two"):
            self.assertEqual(GUARD.ensure_gpu1_available()["gpu"]["index"], "2")
            self.app_rows = "GPU-selected-two, 6000, 100\n"
            with self.assertRaises(GUARD.GPUUnavailable):
                GUARD.ensure_gpu1_available()
            self.app_rows = ""
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=GPU1_UUID):
                with self.assertRaises(GUARD.GPUUnavailable):
                    GUARD.ensure_gpu1_available()


class EntryAuthorizationTests(unittest.TestCase):
    """CPU入口不能借flag/缩写升级GPU；全部Popen/call或设备查询均被mock隔离。"""

    def test_cpu_cli_rejects_gpu_flags_before_subprocess(self):
        entry = load_benchmark("__main__")
        for option in ("--g", "--gp", "--gpu", "--gpu=true", "--gpu=false", "--unknown"):
            with self.subTest(option=option):
                with patch.object(entry.sys, "argv", ["benchmarks", "run", "metadata-cpu", option]):
                    with patch.object(entry.subprocess, "call") as launch, redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as caught:
                            entry.main()
                        self.assertNotEqual(caught.exception.code, 0)
                        launch.assert_not_called()
        for tail in ([], ["--help"]):
            with patch.object(entry.sys, "argv", ["benchmarks", "run", "metadata-cpu", *tail]):
                with patch.object(entry.subprocess, "call", return_value=0) as launch:
                    with self.assertRaises(SystemExit) as caught:
                        entry.main()
                    self.assertEqual(caught.exception.code, 0)
                    self.assertNotIn("--gpu", launch.call_args.args[0])

    def test_cpu_suite_rejects_gpu_flags_in_plan(self):
        suite = load_benchmark("runtime_v2_suite")
        mapping = suite.entries()
        for option in ("--g", "--gp", "--gpu", "--gpu=true", "--gpu=false", "--unknown"):
            plan = dict(stages=[dict(name="metadata", entry="metadata-cpu", args=[option])])
            with self.assertRaisesRegex(ValueError, "metadata-cpu only accepts"):
                suite.validate_plan(plan, mapping)
        for tail in ([], ["--help"]):
            suite.validate_plan(dict(stages=[dict(name="metadata", entry="metadata-cpu", args=tail)]), mapping)
        self.assertNotIn("metadata-gpu", suite.CPU_ENTRIES)
        with self.assertRaisesRegex(ValueError, "explicit --gpu"):
            suite.validate_plan(dict(stages=[dict(name="metadata", entry="metadata-gpu", args=[],
                                                  result_json="{stage_output}/result.json")]), mapping)

    def test_gpu_suite_does_not_launch_without_allow_gpu(self):
        suite = load_benchmark("runtime_v2_suite")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "plan.json"
            plan.write_text(json.dumps(dict(stages=[dict(name="metadata", entry="metadata-gpu",
                            args=["--gpu", "--output", "{stage_output}/metadata.json"],
                            result_json="{stage_output}/metadata.json")])), encoding="utf-8")
            args = SimpleNamespace(plan=plan, output=root / "session", resume=False,
                                   allow_gpu=False, keep_going=False)
            with patch.object(suite.subprocess, "Popen", side_effect=AssertionError("unauthorized worker launch")) as launch:
                with redirect_stdout(io.StringIO()):
                    code = suite.run(args)
            launch.assert_not_called()
            self.assertEqual(code, 2)
            state = json.loads((args.output / "suite.json").read_text(encoding="utf-8"))
            self.assertEqual(state["stages"][0]["status"], "NOT_RUN")
            self.assertEqual(state["stages"][0]["attempts"], [])

    def test_metadata_parser_disables_abbreviations_and_equal_boolean(self):
        metadata = load_benchmark("check_metadata_buffers")
        for option in ("--g", "--gp", "--gpu=true", "--gpu=false"):
            with patch.object(metadata.sys, "argv", ["check_metadata_buffers", option]):
                with patch.object(subprocess, "run", side_effect=AssertionError("guard must not run after invalid parse")) as query:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        metadata.main()
                    self.assertEqual(caught.exception.code, 2)
                    query.assert_not_called()

    def test_metadata_gpu_guard_precedes_torch_import(self):
        metadata = load_benchmark("check_metadata_buffers")
        imported_torch = []
        real_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name == "torch":
                imported_torch.append(name)
                raise AssertionError("torch imported before failed device guard")
            return real_import(name, *args, **kwargs)

        with patch.object(metadata.sys, "argv", ["check_metadata_buffers", "--gpu"]):
            with patch.object(subprocess, "run", side_effect=FileNotFoundError("mock query unavailable")):
                with patch.object(builtins, "__import__", side_effect=checked_import):
                    with self.assertRaisesRegex(RuntimeError, "occupancy query failed"):
                        metadata.main()
        self.assertEqual(imported_torch, [])

    def test_metadata_output_is_new_json_not_overwrite(self):
        metadata = load_benchmark("check_metadata_buffers")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "cpu.json"
            with patch.object(metadata.sys, "argv", ["check_metadata_buffers", "--output", str(output)]):
                with redirect_stdout(io.StringIO()):
                    metadata.main()
                before = output.read_bytes()
                result = json.loads(before)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["device"], "CPU fake asynchronous H2D")
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    metadata.main()
                self.assertEqual(output.read_bytes(), before)


def main():
    tests = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(group)
                               for group in (GuardTests, EntryAuthorizationTests)])
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    print(json.dumps(dict(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
                          device="CPU mocked only", failures=len(result.failures), errors=len(result.errors),
                          scope="no real GPU query; no reservation lock or race-free claim")))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
