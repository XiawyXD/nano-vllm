"""CPU fixture/measurement-report contracts; never a GPU acceptance substitute."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from benchmarks.cache_draft_experiment import (digest, load_manifest, sources, summarize,
                                              variants, workloads, write_new, prepare_trial_cache)
from tests.regression.check_draft_model import compare_logits


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


class HarnessTests(unittest.TestCase):
    def test_unique_control_cannot_reuse_warmup_prefixes(self):
        for dataset, expected_resets in (("synthetic_unique", 3), ("synthetic_recurring", 1)):
            case = dict(dataset=dataset, family="cache")
            with patch("benchmarks.cache_draft_experiment.reset_cache") as reset:
                states = [prepare_trial_cache(object(), case, repeat) for repeat in range(3)]
                self.assertEqual(reset.call_count, expected_resets)
                self.assertEqual(states[-1], "empty" if dataset == "synthetic_unique" else "warmed_by_previous_trial")

    def test_variant_controls_and_independent_inputs(self):
        mapping = variants("target", "draft")
        self.assertEqual(len(mapping), 18)
        for capacity in (12, 64):
            for mode in ("separate", "mixed"):
                baseline = mapping[f"cache_{mode}_off_{capacity}"]["engine"]
                for tier in ("sync", "async"):
                    candidate = mapping[f"cache_{mode}_{tier}_{capacity}"]["engine"]
                    changed = {key for key in candidate if candidate[key] != baseline[key]}
                    self.assertEqual(changed, {"host_kv_cache_bytes"} | (
                        {"host_kv_async_prefetch"} if tier == "async" else set()))
        all_cases = [workloads(Tokenizer(), split) for split in ("calibration", "validation")]
        fingerprints = [{digest(job["prompt_token_ids"]) for case in cases for job in case["jobs"]}
                        for cases in all_cases]
        self.assertFalse(fingerprints[0] & fingerprints[1])
        for cases in all_cases:
            self.assertEqual({c["concurrency"] for c in cases if c["family"] == "cache"}, {1, 4, 8})
            self.assertEqual({c["dataset"] for c in cases if c["family"] == "draft"},
                             {"chinese", "english", "code", "repeat"})

    def fixture(self, root):
        cases = [dict(id="case", family="cache", jobs=[dict(job=0), dict(job=1)])]
        mapping = {name: row for name, row in variants("target", "draft").items()
                   if name in ("cache_separate_off_12", "cache_separate_sync_12")}
        manifest = dict(rounds=2, warmup_trials=2, sha256="fixture-only", variants=mapping,
                        workloads=dict(validation=cases))
        for round_id in range(2):
            for name in mapping:
                folder = root / f"r{round_id}_{name}"
                folder.mkdir()
                files = []
                for index in range(3):
                    filename = f"trial{index}.json"
                    write_new(folder / filename, dict(status="PASS", manifest_sha256="fixture-only", case="case",
                        phase="steady" if index == 2 else "warmup", tokens_per_second=20 if "sync" in name else 10,
                        requests=[dict(job=i, output_token_ids=[i, 2]) for i in range(2)]))
                    files.append(filename)
                write_new(folder / "report.json", dict(status="PASS_EXECUTION_ONLY", trials=files,
                    manifest_sha256="fixture-only", split="validation", variant=name, round=round_id,
                    gpu=dict(gpu=dict(uuid="GPU-fixture"))))
        return manifest

    def test_paired_rounds_and_output_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = self.fixture(folder)
            report = summarize(manifest, folder, "validation")
            self.assertEqual(report["status"], "COMPLETE_MEASUREMENTS")
            self.assertEqual(report["comparisons"][0]["median_speed_ratio"], 2)
            trial = folder / "r1_cache_separate_sync_12/trial2.json"
            row = json.loads(trial.read_text())
            row["requests"][0]["output_token_ids"] = [9]
            trial.write_text(json.dumps(row))
            self.assertEqual(summarize(manifest, folder, "validation")["status"], "REVIEW_REQUIRED")

    def test_missing_warmup_or_profile_trial_cannot_pass(self):
        for kind in ("missing", "profile", "gpu"):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                manifest = self.fixture(folder)
                trial = folder / "r0_cache_separate_sync_12/trial0.json"
                if kind == "gpu":
                    report = folder / "r0_cache_separate_sync_12/report.json"
                    row = json.loads(report.read_text())
                    row["gpu"]["gpu"]["uuid"] = "GPU-other"
                    report.write_text(json.dumps(row))
                else:
                    row = json.loads(trial.read_text())
                    row["phase"] = "profile" if kind == "profile" else "steady"
                    trial.write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    summarize(manifest, folder, "validation")

    def test_manifest_tampering_fails_before_model_or_gpu_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = dict(source_sha256=sources(), model_files={})
            manifest["sha256"] = digest(manifest)
            manifest["source_sha256"]["nanovllm/config.py"] = "changed"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest or runtime source"):
                load_manifest(path)

    def test_existing_strict_logit_tolerance_is_not_relaxed(self):
        reference = torch.tensor([[0., 1., 2.], [1., 3., 2.]])
        self.assertTrue(compare_logits(reference, reference)["strict_allclose"])
        near = reference.clone()
        near[0, 0] += .02
        result = compare_logits(reference, near)
        self.assertFalse(result["strict_allclose"])
        self.assertEqual(result["argmax_equal"], 2)
        self.assertEqual((result["atol"], result["rtol"]), (.01, .01))
        near[0, 0] = float("nan")
        self.assertFalse(compare_logits(reference, near)["strict_allclose"])


if __name__ == "__main__":
    unittest.main()
