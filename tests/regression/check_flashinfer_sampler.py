"""不安装FlashInfer也可运行：适配器CPU合同测试，不代替真实CUDA内核验证。"""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.flashinfer_sampler import FlashInferSampler


def seq(**kwargs):
    x = dict(temperature=1., top_k=0, top_p=1., repetition_penalty=1.,
             seed=12, num_completion_tokens=0, token_ids=[1, 2])
    x.update(kwargs)
    return SimpleNamespace(**x)


class FakeOps:
    """只模拟过滤接口。故意把top-p边界并列全部保留，验证回退保护。"""
    @staticmethod
    def top_k_renorm_probs(probs, ks):
        rows = []
        for p, k in zip(probs, ks):
            q = p.clone()
            q[q < q.topk(int(k)).values[-1]] = 0
            rows.append(q/q.sum())
        return torch.stack(rows)

    @staticmethod
    def top_p_renorm_probs(probs, ps):
        rows = []
        for p, limit in zip(probs, ps):
            q = p.clone()
            sorted_p = q.sort(descending=True).values
            n = min(int((sorted_p.cumsum(0) <= limit).sum()), len(q)-1)
            q[q < sorted_p[n]] = 0
            rows.append(q/q.sum())
        return torch.stack(rows)


class Tests(unittest.TestCase):
    def adapter(self):
        adapter = FlashInferSampler.__new__(FlashInferSampler)
        adapter.ops, adapter.stats = FakeOps(), Counter()
        adapter.min_batch_size = 32
        return adapter

    def test_default_does_not_import_optional_backend(self):
        with patch("nanovllm.layers.flashinfer_sampler.FlashInferSampler", side_effect=AssertionError):
            self.assertEqual(Sampler().backend, "torch")

    def test_explicit_missing_dependency_errors(self):
        with patch("nanovllm.layers.flashinfer_sampler.version", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "requires"):
                Sampler("flashinfer")

    def test_unvalidated_version_errors(self):
        with patch("nanovllm.layers.flashinfer_sampler.version", return_value="999"):
            with self.assertRaisesRegex(RuntimeError, "validated"):
                Sampler("flashinfer")

    def test_cpu_uses_reference(self):
        with patch("nanovllm.layers.flashinfer_sampler.FlashInferSampler", return_value=self.adapter()):
            fi = Sampler("flashinfer")
        x = torch.tensor([[1., 2., 3.]])
        self.assertTrue(torch.equal(fi(x, [seq()]), Sampler()(x, [seq()])))

    def test_empty_and_shape(self):
        self.assertEqual(Sampler()(torch.empty(0, 5), []).numel(), 0)
        with self.assertRaises(ValueError):
            Sampler()(torch.ones(1, 5), [])

    def test_reference_input_immutable(self):
        x = torch.tensor([[1., 3., -2., 0.]])
        original = x.clone()
        Sampler()(x, [seq(top_k=3, top_p=.9, repetition_penalty=1.2)])
        self.assertTrue(torch.equal(x, original))

    def test_tied_top_p_is_ambiguous(self):
        _, bad = self.adapter().filtered_probs(torch.zeros(2, 8), [seq(top_p=.25), seq(top_p=1)])
        self.assertEqual(bad.tolist(), [True, False])

    def test_top_k_zero_and_above_vocab(self):
        x = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
        probs, bad = self.adapter().filtered_probs(x, [seq(top_k=0), seq(top_k=500)])
        torch.testing.assert_close(probs, x.softmax(-1))
        self.assertFalse(bad.any())

    def test_seed_reference_reorder(self):
        x = torch.randn(3, 30)
        params = [seq(seed=i) for i in range(3)]
        a = Sampler()(x, params)
        b = Sampler()(x[[2, 0, 1]], [params[i] for i in (2, 0, 1)])
        self.assertTrue(torch.equal(a[[2, 0, 1]], b))

    def test_invalid_min_batch(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                FlashInferSampler(value)

    def test_small_batch_does_not_enter_filter(self):
        adapter = self.adapter()
        x = torch.randn(3, 16)
        params = [seq(seed=None) for _ in range(3)]
        with patch("torch.cuda.is_current_stream_capturing", return_value=False), patch.object(adapter, "filtered_probs", side_effect=AssertionError):
            result = adapter(x, params, Sampler._reference)
        self.assertEqual(tuple(result.shape), (3,))
        self.assertEqual(adapter.stats["small_batch_reference_rows"], 3)

    def test_batch_filter_repairs_reference_support(self):
        from benchmarks.flashinfer_sampling import reference_probs
        x = torch.zeros(4, 20)
        x[:, :3] = torch.tensor([1., 2., 3.])
        params = [seq(top_k=10, top_p=p) for p in (.2, .5, .9, 1.)]
        actual = self.adapter().reference_filter_batch(x, params)
        for i, s in enumerate(params):
            torch.testing.assert_close(actual[i], reference_probs(x[i], s))

    def test_invalid_probabilities_fail_before_kernel(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.adapter().filtered_probs(torch.tensor([[float("nan"), 1.]]), [seq()])

    def test_config_rejects_unvalidated_combinations(self):
        from nanovllm.config import Config
        with patch("nanovllm.config.os.path.isdir", return_value=True):
            for kwargs in (dict(tensor_parallel_size=2), dict(batching_mode="mixed")):
                with self.assertRaisesRegex(ValueError, "TP1/separate"):
                    Config("unused", sampler_backend="flashinfer", **kwargs)

    def test_config_min_batch_before_model_loading(self):
        from nanovllm.config import Config
        with patch("nanovllm.config.os.path.isdir", return_value=True):
            with self.assertRaisesRegex(ValueError, "flashinfer_min_batch_size"):
                Config("unused", flashinfer_min_batch_size=0)


if __name__ == "__main__":
    unittest.main()
