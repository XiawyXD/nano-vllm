"""Execute the actual tile/reduction kernels under Triton's CPU interpreter.

This verifies indexing/math, NOT GPU compilation, Tensor Core instructions,
latency, CUDA Graph behavior, or end-to-end model quality.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRITON_INTERPRET"] = "1"

import unittest
import torch

from tests.regression.check_awq_packed import payload
from nanovllm.layers.awq_packed import reference_weight_cpu
from nanovllm.layers.awq_triton_kernel import _awq_tile_gemm, _awq_reduce


class InterpreterTests(unittest.TestCase):
    def test_real_kernel_indexing_group_boundaries_tails_and_split(self):
        for m, k, n in ((1, 128, 8), (3, 256, 24), (17, 384, 72), (32, 256, 128)):
            data = payload(k, n, seed=42)
            x = (torch.randn(m, k, generator=torch.Generator().manual_seed(1)) * .1).half()
            expected = (x.float() @ reference_weight_cpu(**data).float()).half()
            for split in (1, 4):
                bm, bn, bk = 16, 64, 32
                partial = torch.empty((split, m, n), dtype=torch.float32)
                actual = torch.empty((m, n), dtype=torch.float16)
                _awq_tile_gemm[(((m + bm - 1) // bm) * ((n + bn - 1) // bn), split)](
                    x, data["qweight"], data["qzeros"], data["scales"], partial,
                    m, n, k, 128, split, bm, bn, bk)
                _awq_reduce[((m * n + 255) // 256,)](partial, actual, m * n, split, 256)
                torch.testing.assert_close(actual, expected, atol=.002, rtol=.002)


if __name__ == "__main__":
    unittest.main()
