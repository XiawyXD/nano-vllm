"""SM89 offline compilation: explicit compiler target, no CUDA context or launch."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("TRITON_INTERPRET", None)

import hashlib
import json
import torch
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from nanovllm.layers.awq_triton_kernel import _awq_tile_gemm, _awq_reduce


def main():
    if torch.cuda.is_initialized():
        raise RuntimeError("offline compiler must not initialize CUDA")
    target = GPUTarget("cuda", 89, 32)
    cases = []
    for m, k, n in ((1, 4096, 6144), (16, 4096, 24576), (32, 12288, 4096), (3, 384, 72)):
        split = 4 if m <= 16 else 1
        constants = dict(M=m, N=n, K=k, GROUP=128, SPLIT=split, BM=16 if m <= 16 else 32, BN=64, BK=32)
        source = ASTSource(_awq_tile_gemm, signature=dict(X="*fp16", W="*i32", Z="*i32", S="*fp16", P="*fp32"),
                           constexprs=constants)
        compiled = triton.compile(source, target=target, options=dict(num_warps=4))
        ptx = compiled.asm["ptx"]
        if ".target sm_89" not in ptx or "mma.sync.aligned" not in ptx:
            raise RuntimeError("SM89 Tensor Core instruction missing")
        reduction = ASTSource(_awq_reduce, signature=dict(P="*fp32", Y="*fp16"),
                              constexprs=dict(SIZE=m*n, SPLIT=split, BLOCK=256))
        triton.compile(reduction, target=target, options=dict(num_warps=4))
        cases.append(dict(m=m, k=k, n=n, split=split, sm89_mma=True,
                          ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(), cubin_bytes=len(compiled.asm["cubin"])))
    if torch.cuda.is_initialized():
        raise RuntimeError("offline compilation unexpectedly initialized CUDA")
    print(json.dumps(dict(status="PASS_OFFLINE_COMPILE_ONLY", cuda_initialized=False, cases=cases), indent=2))


if __name__ == "__main__":
    main()
