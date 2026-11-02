"""纯CPU验证质量代理的覆盖规则和suite的安全恢复，不加载tokenizer/模型/GPU。"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name):
    source = (Path(__file__).resolve().parents[2] / ("tests/regression" if name.startswith("check_") else "benchmarks") / (name + ".py"))
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = load("runtime_v2_quality")
suite = load("runtime_v2_suite")


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def save(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def arguments(self, stage_definitions, **changes):
        plan = self.save("plan.json", dict(fixtures=[], stages=stage_definitions))
        return SimpleNamespace(plan=plan, output=self.directory / "session", resume=False,
                               allow_gpu=False, keep_going=False, **changes)

    def release_test_runner(self, args):
        """同一测试进程模拟上次suite已退出；真实resume仍必须查OS进程出生标记。"""
        path = args.output / "suite.json"
        state = json.loads(path.read_text())
        state["runner_identity"] = None
        suite.save_state(path, state)
        args.resume = True
        return state

    def successful_helper(self):
        path = self.directory / "cpu_success.py"
        path.write_text("import sys\nprint('CPU stage', sys.argv[1:])\n", encoding="utf-8")
        return path

    def test_window_and_query_coverage_exactly_once(self):
        for length in (2, 16, 64, 65, 128, 1025, 2507):
            targets = []
            for start, end in quality.window_ranges(length, 64):
                positions = []
                for q_start, q_end, prefill in quality.query_schedule(end - start - 1, 13, 8):
                    positions.extend(range(q_start, q_end))
                    if not prefill:
                        self.assertEqual(q_end - q_start, 1)
                self.assertEqual(positions, list(range(end - start - 1)))
                targets.extend(start + value for value in positions)
            self.assertEqual(targets, list(range(length - 1)))

    def test_corpus_rejects_whole_document_duplicates(self):
        text = "A distinct fictional ferry log describes a fog bank, a delayed departure, and a repaired compass. " * 2
        path = self.directory / "fixture.jsonl"
        path.write_text("\n".join(json.dumps(dict(id=identifier, text=text)) for identifier in ("a", "b")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate document"):
            quality.read_corpus(path)

    def test_prepare_insufficient_real_target_count_fails_before_output(self):
        path = self.directory / "fixture.jsonl"
        path.write_text(json.dumps(dict(id="a", text="甲" * 150)), encoding="utf-8")
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(range(150)))
        auto = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
        args = SimpleNamespace(output=self.directory / "quality", corpus=path, model="local-only")
        with patch.dict(sys.modules, transformers=SimpleNamespace(AutoTokenizer=auto)):
            with self.assertRaisesRegex(ValueError, "only 149 next-token targets"):
                quality.prepare(args)
        self.assertFalse(args.output.exists())

    def test_write_new_refuses_overwrite(self):
        path = self.directory / "result.json"
        quality.write_new(path, dict(status="FAIL"))
        with self.assertRaises(FileExistsError):
            quality.write_new(path, dict(status="PASS"))
        self.assertEqual(quality.load_json(path)["status"], "FAIL")

    def test_pid_identity_uses_live_process_not_lock_file(self):
        identity = suite.process_identity(os.getpid())
        self.assertIsNotNone(identity)
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            suite.ensure_terminated(identity)
        suite.ensure_terminated(dict(identity, birth="not-the-process-birth"))

    def test_gpu_stage_is_not_run_without_explicit_authorization(self):
        args = self.arguments([dict(name="quality", entry="quality", args=[], result_json="{stage_output}/comparison.json")])
        with redirect_stdout(io.StringIO()):
            code = suite.run(args)
        state = json.loads((args.output / "suite.json").read_text())
        self.assertEqual(code, 2)
        self.assertEqual((state["status"], state["stages"][0]["status"]), ("NOT_RUN", "NOT_RUN"))
        self.assertEqual(state["stages"][0]["attempts"], [])

    def test_stage_selection_rejects_empty_duplicate_and_unknown_names(self):
        plan = dict(stages=[dict(name="first"), dict(name="second")])
        self.assertEqual(suite.select_stages(plan, None), ["first", "second"])
        self.assertEqual(suite.select_stages(plan, " second , first "), ["first", "second"])
        for invalid in ("", " ", ",", "first,", ",first", "first,,second",
                        "first,first", "first, first", "missing", "FIRST"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                suite.select_stages(plan, invalid)

    def test_selection_preserves_full_plan_and_resume_runs_only_new_selection(self):
        helper = self.successful_helper()
        definitions = [dict(name=name, entry="adaptive", args=[name]) for name in ("first", "second", "third")]
        args = self.arguments(definitions, stages="second,first")
        original_plan = args.plan.read_bytes()
        with patch.object(suite, "entries", return_value={"adaptive": str(helper)}), redirect_stdout(io.StringIO()):
            self.assertEqual(suite.run(args), 0)
            state = self.release_test_runner(args)
            self.assertEqual(state["selected_stages"], ["first", "second"])
            self.assertEqual((state["selected_status"], state["status"]), ("PASS", "NOT_RUN"))
            self.assertEqual([row["status"] for row in state["stages"]], ["PASS", "PASS", "NOT_RUN"])
            self.assertEqual([len(row["attempts"]) for row in state["stages"]], [1, 1, 0])
            first_fingerprint = state["fingerprint"]
            self.assertEqual(json.loads((args.output / "plan.json").read_text())["stages"], definitions)
            args.stages = "third"
            self.assertEqual(suite.run(args), 0)
            state = self.release_test_runner(args)
            self.assertEqual((state["selected_status"], state["status"]), ("PASS", "PASS"))
            self.assertEqual(state["fingerprint"], first_fingerprint)
            self.assertEqual(args.plan.read_bytes(), original_plan)
            self.assertEqual([len(row["attempts"]) for row in state["stages"]], [1, 1, 1])
            # 已PASS阶段即使重新选中，也只核对旧证据，不启动第二个进程。
            args.stages = "first"
            with patch.object(suite.subprocess, "Popen", side_effect=AssertionError("PASS reran")):
                self.assertEqual(suite.run(args), 0)

    def test_resume_selection_still_checks_unselected_success_evidence(self):
        helper = self.successful_helper()
        args = self.arguments([dict(name=name, entry="adaptive", args=[name])
                               for name in ("first", "second")], stages="first")
        with patch.object(suite, "entries", return_value={"adaptive": str(helper)}), redirect_stdout(io.StringIO()):
            self.assertEqual(suite.run(args), 0)
            state = self.release_test_runner(args)
            Path(state["stages"][0]["attempts"][0]["log"]).write_text("tampered original log")
            args.stages = "second"
            with patch.object(suite.subprocess, "Popen", side_effect=AssertionError("spawned before hash check")):
                with self.assertRaisesRegex(RuntimeError, "evidence changed or disappeared"):
                    suite.run(args)

    def test_selected_gpu_still_requires_authorization_and_unselected_is_not_spawned(self):
        args = self.arguments([dict(name="cpu", entry="adaptive", args=[]),
            dict(name="gpu", entry="quality", args=[], result_json="{stage_output}/comparison.json")], stages="gpu")
        with patch.object(suite.subprocess, "Popen", side_effect=AssertionError("unauthorized GPU spawned")), redirect_stdout(io.StringIO()):
            self.assertEqual(suite.run(args), 2)
        state = json.loads((args.output / "suite.json").read_text())
        self.assertEqual((state["selected_status"], state["status"]), ("NOT_RUN", "NOT_RUN"))
        self.assertEqual([row["attempts"] for row in state["stages"]], [[], []])
        self.assertIn("not authorized", state["stages"][1]["reason"])

    def test_selected_exit_code_does_not_hide_unselected_failed_overall_status(self):
        helper = self.directory / "cpu_by_stage.py"
        helper.write_text("import sys\nprint(sys.argv[1])\nraise SystemExit(7 if sys.argv[1]=='bad' else 0)\n")
        args = self.arguments([dict(name=name, entry="adaptive", args=[name])
                               for name in ("bad", "good")], stages="bad")
        with patch.object(suite, "entries", return_value={"adaptive": str(helper)}), redirect_stdout(io.StringIO()):
            self.assertEqual(suite.run(args), 1)
            self.release_test_runner(args)
            args.stages = "good"
            self.assertEqual(suite.run(args), 0)
        state = json.loads((args.output / "suite.json").read_text())
        self.assertEqual((state["selected_status"], state["status"]), ("PASS", "FAIL"))
        self.assertEqual([row["status"] for row in state["stages"]], ["FAIL", "PASS"])
        self.assertEqual([len(row["attempts"]) for row in state["stages"]], [1, 1])

    def test_result_gate_preserves_numerical_review(self):
        path = self.save("review.json", dict(status="REVIEW_REQUIRED", quality_status="PASS", numerical_status="REVIEW_REQUIRED"))
        self.assertEqual(suite.result_gate(path)[0], "FAIL")
        path = self.save("notrun.json", dict(status="NOT_RUN"))
        self.assertEqual(suite.result_gate(path)[0], "NOT_RUN")
        path = self.save("pass.json", dict(status="PASS", quality_status="PASS", numerical_status="PASS_SAMPLED_LOGITS"))
        self.assertEqual(suite.result_gate(path)[0], "PASS")

    def test_result_gate_git_provenance_is_not_a_verdict(self):
        path = self.save("git.json", dict(status="PASS", provenance=dict(
            git_status="?? artifacts/FAIL_case/\n M tracked.py")))
        self.assertEqual(suite.result_gate(path), ("PASS", ["PASS"]))
        path = self.save("nested.json", dict(status="PASS", provenance=dict(
            git_status=""), checks=[dict(numerical_status="REVIEW_REQUIRED")]))
        self.assertEqual(suite.result_gate(path)[0], "FAIL")
        path = self.save("git_only.json", dict(provenance=dict(git_status="PASS")))
        self.assertEqual(suite.result_gate(path)[0], "FAIL")

    def test_external_config_content_is_part_of_resume_fingerprint(self):
        config = self.save("candidate.json", dict(mlp_backend="torch"))
        plan = dict(fixtures=[], stages=[dict(name="quality", entry="quality", args=["--candidate-config", str(config)], result_json="{stage_output}/comparison.json")])
        path = self.save("plan.json", plan)
        first = suite.fingerprint(path, plan)
        config.write_text('{"mlp_backend":"auto"}', encoding="utf-8")
        second = suite.fingerprint(path, plan)
        self.assertEqual(first["plan_sha256"], second["plan_sha256"])
        self.assertNotEqual(first["configs_sha256"], second["configs_sha256"])

    def test_resume_rejects_live_runner_and_changed_plan(self):
        args = self.arguments([dict(name="adaptive", entry="adaptive", args=[], enabled=False)])
        with redirect_stdout(io.StringIO()):
            suite.run(args)
        args.resume = True
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            suite.run(args)
        state_path = args.output / "suite.json"
        state = json.loads(state_path.read_text())
        state["runner_identity"] = None
        suite.save_state(state_path, state)
        args.plan.write_text('{"fixtures":[],"stages":[{"name":"adaptive","entry":"adaptive","args":[]}]}')
        with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
            suite.run(args)

    def test_failed_stage_resume_creates_new_attempt_and_preserves_old(self):
        flag = self.directory / "flag"
        helper = self.directory / "transient.py"
        helper.write_text("import pathlib,sys\np=pathlib.Path(sys.argv[1])\nexists=p.exists()\np.touch()\nraise SystemExit(0 if exists else 7)\n")
        args = self.arguments([dict(name="transient", entry="adaptive", args=[str(flag)])])
        with patch.object(suite, "entries", return_value={"adaptive": str(helper)}), redirect_stdout(io.StringIO()):
            self.assertEqual(suite.run(args), 1)
            state_path = args.output / "suite.json"
            state = json.loads(state_path.read_text())
            state["runner_identity"] = None
            suite.save_state(state_path, state)
            args.resume = True
            self.assertEqual(suite.run(args), 0)
        state = json.loads(state_path.read_text())
        attempts = state["stages"][0]["attempts"]
        self.assertEqual([row["exit_code"] for row in attempts], [7, 0])
        self.assertEqual([row["status"] for row in attempts], ["FAIL", "PASS"])
        self.assertNotEqual(attempts[0]["log"], attempts[1]["log"])

    def test_plan_refuses_shell_text_or_unstructured_gpu_gate(self):
        with self.assertRaises(ValueError):
            suite.validate_plan(dict(stages=[dict(name="x", entry="adaptive", args="echo bad")]), suite.entries())
        with self.assertRaisesRegex(ValueError, "result_json"):
            suite.validate_plan(dict(stages=[dict(name="x", entry="quality", args=[])]), suite.entries())

    def test_only_named_placeholders_are_rendered(self):
        value = '{"question":"what is {x}?","path":"{stage_output}"}'
        self.assertEqual(suite.render_argument(value, dict(stage_output="/new/output")),
                         '{"question":"what is {x}?","path":"/new/output"}')

    def test_equals_options_cannot_bypass_hashing(self):
        config = self.save("candidate.json", dict(mlp_backend="torch"))
        arguments = [f"--candidate-config={config}", "--output=/new/path"]
        self.assertEqual(list(suite.option_pairs(arguments)), [("--candidate-config", str(config)), ("--output", "/new/path")])
        plan = dict(fixtures=[], stages=[dict(name="quality", entry="quality", args=arguments,
                                             result_json="{stage_output}/comparison.json")])
        path = self.save("plan.json", plan)
        self.assertIn(str(config.resolve()), suite.fingerprint(path, plan)["configs_sha256"])

    def test_os_lock_prevents_two_writers_even_before_pid_check(self):
        with suite.exclusive_guard(self.directory):
            with self.assertRaisesRegex(RuntimeError, "OS output lock"):
                with suite.exclusive_guard(self.directory):
                    self.fail("second lock should not be acquired")

    def test_startup_gate_checks_child_identity_before_command(self):
        gate = self.save("gate.json", dict(child_pid=os.getpid() + 1))
        with self.assertRaisesRegex(RuntimeError, "does not belong"):
            suite.execute_gated(gate, [sys.executable, "-c", "raise SystemExit(0)"])

    def test_failed_partial_report_is_archived_without_rewriting(self):
        attempt = self.directory / "stage/attempt_001"
        attempt.mkdir(parents=True)
        log = attempt / "process.log"
        log.write_text("interrupted before completion", encoding="utf-8")
        raw = attempt / "report.json"
        raw.write_text('{"status":"RUNNING","evidence":"original"}', encoding="utf-8")
        before = raw.read_bytes()
        state = dict(status="FAIL", fingerprint={}, runner_identity=None,
            stages=[dict(name="stage", status="FAIL", attempts=[dict(status="FAIL", log=str(log))])])
        suite.save_state(self.directory / "suite.json", state)
        with redirect_stdout(io.StringIO()):
            suite.archive(self.directory)
        self.assertEqual(raw.read_bytes(), before)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.assertEqual(len(manifest["unfinished_reports"]), 1)
        self.assertEqual(manifest["suite_status"], "FAIL")
        with tarfile.open(self.directory / "raw-data.tar.gz") as packed:
            self.assertEqual(packed.extractfile("stage/attempt_001/report.json").read(), before)
        with self.assertRaises(FileExistsError):
            suite.archive(self.directory)

    def test_running_report_without_failed_owner_blocks_archive(self):
        state = dict(status="PASS", fingerprint={}, runner_identity=None, stages=[])
        suite.save_state(self.directory / "suite.json", state)
        self.save("unowned.json", dict(status="RUNNING"))
        with self.assertRaisesRegex(RuntimeError, "no terminal failed attempt"):
            suite.archive(self.directory)
        self.assertFalse((self.directory / "raw-data.tar.gz").exists())

    def test_archive_rechecks_success_evidence_instead_of_rehashing_mutation(self):
        for change in ("modified", "deleted"):
            with self.subTest(change=change):
                directory = self.directory / change
                attempt = directory / "quality/attempt_001"
                attempt.mkdir(parents=True)
                log = attempt / "process.log"
                report = attempt / "comparison.json"
                log.write_text("finished", encoding="utf-8")
                report.write_text('{"status":"PASS"}', encoding="utf-8")
                state = dict(status="PASS", fingerprint={}, runner_identity=None,
                    stages=[dict(name="quality", status="PASS", attempts=[dict(status="PASS", log=str(log),
                        log_sha256=suite.digest(log), result_json=str(report), result_sha256=suite.digest(report))])])
                suite.save_state(directory / "suite.json", state)
                if change == "modified":
                    report.write_text('{"status":"FAIL"}', encoding="utf-8")
                else:
                    report.unlink()
                with self.assertRaisesRegex(RuntimeError, "evidence changed or disappeared"):
                    suite.archive(directory)
                self.assertFalse((directory / "raw-data.tar.gz").exists())

    def test_popen_failure_cannot_turn_running_stage_into_pass(self):
        args = self.arguments([dict(name="adaptive", entry="adaptive", args=[])])
        with patch.object(suite.subprocess, "Popen", side_effect=OSError("spawn denied")):
            with self.assertRaises(OSError):
                suite.run(args)
        state = json.loads((args.output / "suite.json").read_text())
        self.assertEqual((state["status"], state["stages"][0]["status"]), ("FAIL", "FAIL"))

    def test_cleanup_escalates_group_even_after_leader_exit(self):
        signs, clock = [], [0.0]
        process = SimpleNamespace(pid=87654, _suite_identity=dict(pid=87654, birth="old"), poll=lambda: 0)
        def group_empty(group_id):
            if not signs or signs[-1] != 9:
                raise RuntimeError("a stubborn child remains")
        def monotonic():
            clock[0] += 40
            return clock[0]
        with patch.object(suite.os, "name", "posix"), patch.object(suite.os, "killpg", create=True, side_effect=lambda pid, sig: signs.append(sig)), \
                patch.object(suite, "process_identity", return_value=None), patch.object(suite, "ensure_group_empty", side_effect=group_empty), \
                patch.object(suite.time, "monotonic", side_effect=monotonic), patch.object(suite.time, "sleep"):
            suite.stop_owned(process)
        self.assertEqual(signs, [2, 15, 9])


if __name__ == "__main__":
    unittest.main(verbosity=2)
