"""CPU-only AWQ checkpoint layout and existing-loader integration contracts."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tempfile
import unittest

import torch
from safetensors.torch import save_file

from nanovllm.layers.awq_packed import (
    AWQPackedLinear, pack_awq_int4_cpu, unpack_awq_int4_cpu,
    reference_weight_cpu, validate_awq_config,
)
from nanovllm.utils.loader import load_model


def payload(k, n, seed=1):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randint(0, 16, (k, n), generator=generator, dtype=torch.int32)
    z = torch.randint(0, 16, (k // 128, n), generator=generator, dtype=torch.int32)
    s = (torch.rand((k // 128, n), generator=generator) * .125 + .005).half()
    return dict(qweight=pack_awq_int4_cpu(q), qzeros=pack_awq_int4_cpu(z), scales=s)


def fill(module, data, shard=None):
    for name, tensor in data.items():
        getattr(module, name).weight_loader(getattr(module, name), tensor, shard)


class AWQTests(unittest.TestCase):
    def test_known_nibble_word_not_just_roundtrip(self):
        q = torch.arange(8, dtype=torch.int32).reshape(1, 8)
        self.assertEqual(pack_awq_int4_cpu(q).item(), 0x75316420)
        packed = torch.tensor([[0x75316420]], dtype=torch.int32)
        self.assertTrue(torch.equal(unpack_awq_int4_cpu(packed), q))

    def test_signed_words_and_nibble_endpoints(self):
        source = torch.tensor([[0, 15, 1, 14, 2, 13, 3, 12], [15] * 8], dtype=torch.int32)
        encoded = pack_awq_int4_cpu(source)
        self.assertLess(encoded[0, 0].item(), 0)
        self.assertEqual(encoded[1, 0].item(), -1)
        self.assertTrue(torch.equal(source, unpack_awq_int4_cpu(encoded)))

    def test_random_pack_roundtrip_and_invalid_inputs(self):
        for rows, cols in ((1, 8), (128, 24), (256, 128)):
            values = torch.randint(0, 16, (rows, cols), dtype=torch.int32)
            self.assertTrue(torch.equal(values, unpack_awq_int4_cpu(pack_awq_int4_cpu(values))))
        for values in (torch.zeros(1, 7, dtype=torch.int32), torch.full((1, 8), -1),
                       torch.full((1, 8), 16), torch.zeros(1, 8)):
            with self.assertRaises(ValueError):
                pack_awq_int4_cpu(values)

    def test_direct_awq_zero_not_gptq_plus_one(self):
        q = torch.full((256, 8), 7, dtype=torch.int32)
        z = torch.tensor([[7] * 8, [8] * 8], dtype=torch.int32)
        s = torch.tensor([[.25] * 8, [.5] * 8], dtype=torch.float16)
        w = reference_weight_cpu(pack_awq_int4_cpu(q), pack_awq_int4_cpu(z), s)
        self.assertEqual(w[:128].abs().sum().item(), 0)
        self.assertTrue(torch.equal(w[128:], torch.full((128, 8), -.5, dtype=torch.float16)))

    def test_checkpoint_config_is_strict(self):
        base = dict(quant_method="awq", bits=4, group_size=128, zero_point=True, version="gemm")
        self.assertEqual(validate_awq_config(base), 128)
        for patch in (dict(quant_method="gptq"), dict(bits=8), dict(bits=True), dict(version="gemv"),
                      dict(group_size=64), dict(zero_point=False), dict(backend="marlin"),
                      dict(modules_to_not_convert=["q_proj"])):
            with self.assertRaises(ValueError):
                validate_awq_config(base | patch)

    def test_tp_dtype_bias_and_shape_guards(self):
        for overrides in (dict(tp_size=2), dict(tp_rank=1), dict(dtype=torch.bfloat16),
                          dict(bias=True), dict(group_size=64)):
            with self.assertRaises(ValueError):
                AWQPackedLinear(128, 16, device="cpu", **overrides)
        for k, n in ((0, 8), (127, 8), (128, 7)):
            with self.assertRaises(ValueError):
                AWQPackedLinear(k, n, device="cpu")

    def test_existing_loader_fuses_qkv_by_output_columns(self):
        model = torch.nn.Module()
        model.qkv_proj = AWQPackedLinear(256, (16, 8, 8), shard_ids=("q", "k", "v"), device="cpu")
        model.packed_modules_mapping = {f"{s}_proj": ("qkv_proj", s) for s in ("q", "k", "v")}
        shards = [payload(256, n, i) for i, n in enumerate((16, 8, 8))]
        tensors = {f"{s}_proj.{name}": value for s, data in zip(("q", "k", "v"), shards)
                   for name, value in data.items()}
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, directory + "/model.safetensors")
            load_model(model, directory)
        model.qkv_proj.assert_loaded()
        for name in ("qweight", "qzeros", "scales"):
            self.assertTrue(torch.equal(getattr(model.qkv_proj, name), torch.cat([p[name] for p in shards], dim=1)))
        expected = torch.cat([reference_weight_cpu(**p) for p in shards], dim=1)
        self.assertTrue(torch.equal(expected, reference_weight_cpu(model.qkv_proj.qweight,
            model.qkv_proj.qzeros, model.qkv_proj.scales)))

    def test_existing_loader_fuses_gate_up_and_loads_row_projection(self):
        model = torch.nn.Module()
        model.gate_up_proj = AWQPackedLinear(128, (24, 24), shard_ids=(0, 1), device="cpu")
        model.down_proj = AWQPackedLinear(256, 16, device="cpu")
        model.packed_modules_mapping = {"gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1)}
        shards = [payload(128, 24, 20 + i) for i in range(2)]
        down = payload(256, 16)
        tensors = {f"{s}.{name}": value for s, data in zip(("gate_proj", "up_proj"), shards)
                   for name, value in data.items()}
        tensors.update({f"down_proj.{name}": value for name, value in down.items()})
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, directory + "/model.safetensors")
            load_model(model, directory)
        model.gate_up_proj.assert_loaded()
        model.down_proj.assert_loaded()
        self.assertTrue(torch.equal(model.gate_up_proj.qweight, torch.cat([p["qweight"] for p in shards], 1)))
        self.assertTrue(torch.equal(model.down_proj.qweight, down["qweight"]))

    def test_missing_duplicate_and_wrong_shard_rejected(self):
        layer = AWQPackedLinear(128, 16, device="cpu")
        data = payload(128, 16)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            layer(torch.zeros(1, 128))
        layer.weight_loader(layer.qweight, data["qweight"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            layer.weight_loader(layer.qweight, data["qweight"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            layer.weight_loader(layer.qzeros, data["qzeros"], "q")

    def test_rejected_payload_does_not_mark_loaded(self):
        layer = AWQPackedLinear(128, 16, device="cpu")
        for values in (torch.zeros(1, 16), torch.zeros(2, 16, dtype=torch.float16),
                       torch.zeros(1, 16, dtype=torch.float16),
                       torch.full((1, 16), float("nan"), dtype=torch.float16)):
            with self.assertRaises(ValueError):
                layer.weight_loader(layer.scales, values)
            self.assertEqual(layer._loaded, set())

    def test_no_dense_storage_and_sealed_weights(self):
        layer = AWQPackedLinear(256, 128, device="cpu")
        data = payload(256, 128)
        fill(layer, data)
        layer.assert_loaded()
        self.assertFalse(hasattr(layer, "weight"))
        self.assertEqual(layer.stats()["dense_weight_bytes"], 0)
        storage = sum(p.numel() * p.element_size() for p in layer.parameters())
        self.assertEqual(storage, 256 * 128 // 2 + 2 * 128 // 2 + 2 * 128 * 2)
        with self.assertRaisesRegex(ValueError, "sealed"):
            fill(layer, data)

    def test_cpu_forward_never_silently_dequantizes(self):
        layer = AWQPackedLinear(128, 16, device="cpu")
        fill(layer, payload(128, 16))
        with self.assertRaisesRegex(ValueError, "CUDA FP16"):
            layer(torch.zeros(1, 128, dtype=torch.float16))


if __name__ == "__main__":
    unittest.main()
