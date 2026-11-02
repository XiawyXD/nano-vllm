import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import unittest
import torch
from nanovllm.layers.awq_packed import pack_awq_int4_cpu, unpack_awq_int4_cpu
from nanovllm.layers.awq_marlin_sidecar import metadata_cpu, validate_shape_dtype, UINT4_ID, derive_fp16_scales_cpu


class MarlinLayoutTests(unittest.TestCase):
    def test_dtype_mismatch_rejected_not_bit_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_shape_dtype(1024, 64, torch.bfloat16, torch.float16)

    def test_unsupported_shapes_rejected(self):
        for k, n in ((127, 64), (128, 72), (0, 64)):
            with self.assertRaises(ValueError):
                validate_shape_dtype(k, n, torch.float16, torch.float16)

    def test_native_dtypes_preserved_and_inputs_unchanged(self):
        for dtype in (torch.float16, torch.bfloat16):
            for k in (128, 256, 4096):
                n = 64
                zeros = (torch.arange(k // 128 * n).reshape(-1, n) % 16).int()
                packed = pack_awq_int4_cpu(zeros)
                scales = (torch.arange(k // 128 * n).reshape(-1, n) % 200 + 1).to(dtype)
                old_p, old_s = packed.clone(), scales.clone()
                zp, sp = metadata_cpu(packed, scales, k, n, dtype)
                perm = [i + 8 * j for i in range(8) for j in range(8)]
                self.assertTrue(torch.equal(unpack_awq_int4_cpu(zp), zeros.reshape(-1,64)[:,perm].reshape(-1,n)))
                s_perm = perm if k > 128 else [2*i+j for i in range(4) for j in (0,1,8,9,16,17,24,25)]
                self.assertTrue(torch.equal(sp, scales.reshape(-1,len(s_perm))[:,s_perm].reshape(-1,n)))
                self.assertEqual(sp.dtype, dtype)
                self.assertTrue(torch.equal(old_p, packed) and torch.equal(old_s, scales))

    def test_invalid_scales_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                metadata_cpu(torch.zeros(1, 8, dtype=torch.int32), torch.full((1,64),value,dtype=torch.float16),128,64)

    def test_scalar_id(self):
        self.assertEqual(UINT4_ID, 1125899906843648)

    def test_explicit_derived_cache_preserves_canonical(self):
        source = torch.tensor([[.125, .25, 1, 8]], dtype=torch.bfloat16)
        saved = source.clone()
        cache, report = derive_fp16_scales_cpu(source)
        self.assertEqual(cache.dtype, torch.float16)
        self.assertTrue(report["roundtrip_exact"])
        self.assertTrue(torch.equal(source, saved))
        for value in (1e30, 1e-30, 0, -1, float("nan")):
            with self.assertRaises(ValueError):
                derive_fp16_scales_cpu(torch.tensor([[value]], dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
