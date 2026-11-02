"""Optional vLLM 0.17.1 native AWQ-Marlin adapter; no vLLM package import.

GPU execution NOT YET VALIDATED. Native library is pinned to the verified
official wheel, not an arbitrary user-provided .so. CPU metadata permutations
are derived from vLLM v0.17.1 marlin_utils.py (Apache-2.0; see sidecar LICENSE).
Canonical checkpoint tensors are never mutated. No eager/Triton fallback.
"""
import hashlib
import copy
from pathlib import Path
import threading

import torch
from nanovllm.layers.awq_packed import pack_awq_int4_cpu, unpack_awq_int4_cpu

SO_SHA256 = "10ec98d19907ec8d31cc4425064c41da4a704789196020bd3c0f36de581d82b9"
WHEEL_SHA256 = "c52e892309532b4e51cb94d022c5e3c0087300cdb56e4645708601443299d871"
# vLLM ScalarType.uint(4,None): mantissa=4 at bit8, NanRepr.IEEE_754 at bit50.
UINT4_ID = (4 << 8) | (1 << 50)
_library_path = None
_load_lock = threading.Lock()


def load_verified_extension(library):
    """Explicit startup-only dlopen, with hash/ABI checks and op collision guard."""
    global _library_path
    path = Path(library).resolve(strict=True)
    with _load_lock:
        if _library_path is not None:
            if path != _library_path:
                raise RuntimeError("cannot replace a loaded Marlin extension")
            return
        if torch.__version__ != "2.10.0+cu128" or torch.version.cuda != "12.8" or not torch._C._GLIBCXX_USE_CXX11_ABI:
            raise RuntimeError("sidecar only ABI-probed with Torch2.10.0+cu128/CXX11 ABI")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != SO_SHA256:
            raise ValueError("unapproved native library checksum")
        if any(hasattr(torch.ops._C, op) for op in ("marlin_gemm", "awq_marlin_repack")):
            raise RuntimeError("existing _C Marlin registration has unknown provenance; use a fresh process")
        torch.ops.load_library(str(path))
        for op in ("marlin_gemm", "awq_marlin_repack"):
            if not hasattr(torch.ops._C, op):
                raise RuntimeError("verified extension did not register expected operation")
        _library_path = path


def validate_shape_dtype(k, n, scales_dtype, activation_dtype):
    if k <= 0 or k % 128 or n <= 0 or n % 64:
        raise ValueError("Marlin g128 requires positive K%128=0, N%64=0")
    if activation_dtype not in (torch.float16, torch.bfloat16) or scales_dtype != activation_dtype:
        raise ValueError("native Marlin requires scale dtype == activation dtype; no implicit conversion")


def metadata_cpu(qzeros, scales, k, n, activation_dtype=torch.float16):
    """Reference/initialization transform of O(K*N/128) metadata, not weights."""
    validate_shape_dtype(k, n, scales.dtype, activation_dtype)
    if (qzeros.device.type != "cpu" or scales.device.type != "cpu"
            or qzeros.dtype != torch.int32 or qzeros.shape != (k // 128, n // 8)
            or scales.shape != (k // 128, n)):
        raise ValueError("invalid CPU AWQ scale/zero metadata")
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("scales must be finite and positive")
    perm64 = [i + 8 * j for i in range(8) for j in range(8)]
    perm32 = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    permutation = perm64 if k > 128 else perm32
    permuted_scales = scales.reshape(-1, len(permutation))[:, permutation].reshape(-1, n).contiguous()
    zero = unpack_awq_int4_cpu(qzeros).reshape(-1, 64)[:, perm64].reshape(-1, n)
    permuted_zeros = pack_awq_int4_cpu(zero)
    return permuted_zeros, permuted_scales


def derive_fp16_scales_cpu(source):
    """Explicit lossless derived cache; never modifies canonical source scales."""
    if source.device.type != "cpu" or source.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("derived scale cache expects CPU FP16/BF16 source")
    result = source.to(torch.float16).clone()
    if (not bool(torch.isfinite(source).all()) or bool((source <= 0).any())
            or not bool(torch.isfinite(result).all()) or bool((result <= 0).any())
            or not torch.equal(source.float(), result.float())
            or not torch.equal(source, result.to(source.dtype))):
        raise ValueError("FP16 derived scale cache would not be positive finite exact")
    return result, dict(source_dtype=str(source.dtype), cache_dtype=str(result.dtype),
                        elements=source.numel(), value_exact=True, roundtrip_exact=True,
                        canonical_source_modified=False)


class MarlinAWQState:
    """Explicit prepared sidecar state; call prepare BEFORE CUDA graph capture.

    One state/workspace per concurrent execution stream. Same-stream graph
    replay is intended; cross-stream sharing is unsupported and rejected.
    User must acquire the GPU lease and run original GPU guard before prepare.
    """
    @classmethod
    @torch.no_grad()
    def prepare(cls, qweight, qzeros, scales, *, library, activation_dtype=torch.float16,
                allow_lossless_fp16_scale_cache=False):
        if qweight.ndim != 2:
            raise ValueError("qweight must be rank2")
        k, n = qweight.shape[0], qweight.shape[1] * 8
        if scales.dtype != activation_dtype and not (
                allow_lossless_fp16_scale_cache and activation_dtype == torch.float16
                and scales.dtype == torch.bfloat16):
            raise ValueError("scale dtype mismatch requires explicit lossless FP16 cache permission")
        validate_shape_dtype(k, n, activation_dtype, activation_dtype)
        if (qweight.device.type != "cuda" or qweight.dtype != torch.int32
                or any(t.device != qweight.device or not t.is_contiguous() for t in (qzeros, scales, qweight))
                or qzeros.dtype != torch.int32 or qzeros.shape != (k // 128, n // 8)
                or scales.shape != (k // 128, n)):
            raise ValueError("invalid contiguous CUDA AWQ payload")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Marlin prepare must run outside graph capture")
        if torch.cuda.get_device_capability(qweight.device) != (8, 9):
            raise RuntimeError("this sidecar prototype is scoped to SM89")
        load_verified_extension(library)
        # Startup-only CPU metadata handling avoids any runtime .item()/sync.
        source_scale_cpu = scales.cpu()
        scale_report = dict(source_dtype=str(scales.dtype), cache_dtype=str(scales.dtype),
                            value_exact=True, canonical_source_modified=False)
        if scales.dtype != activation_dtype:
            source_scale_cpu, scale_report = derive_fp16_scales_cpu(source_scale_cpu)
        zp_cpu, scale_cpu = metadata_cpu(qzeros.cpu(), source_scale_cpu, k, n, activation_dtype)
        state = cls()
        state.k, state.n, state.dtype, state.device = k, n, activation_dtype, qweight.device
        state.weight = torch.ops._C.awq_marlin_repack(qweight, k, n, 4, False)
        state.zeros = zp_cpu.to(qweight.device)
        state.scales = scale_cpu.to(qweight.device)
        sms = torch.cuda.get_device_properties(qweight.device).multi_processor_count
        state.workspace = torch.zeros(sms, device=qweight.device, dtype=torch.int32)
        state.empty = torch.empty(0, device=qweight.device, dtype=torch.int32)
        state.execution_stream = None
        state.python_dispatches = 0
        state.scale_cache_report = scale_report
        state.workspace_scope = "eager_single_stream"
        return state

    def for_graph(self, capture_stream):
        """Startup-only: fresh workspace PER GRAPH, shared immutable packed data.

        Explicit device synchronization ensures all preparation is complete.
        Call once per graph even if graphs use the same capture stream. Never
        share the returned workspace with eager or another concurrent graph.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("graph state must be prepared before capture")
        if capture_stream.device != self.device:
            raise ValueError("capture stream device mismatch")
        torch.cuda.synchronize(self.device)
        result = copy.copy(self)
        with torch.cuda.stream(capture_stream):
            result.workspace = torch.zeros_like(self.workspace)
        capture_stream.synchronize()
        result.execution_stream = capture_stream.cuda_stream
        result.python_dispatches = 0
        result.workspace_scope = "dedicated_graph_workspace"
        return result

    @torch.no_grad()
    def __call__(self, x):
        if (x.device != self.device or x.dtype != self.dtype or x.ndim < 2
                or x.shape[-1] != self.k or not x.is_contiguous()):
            raise ValueError("Marlin input must match prepared contiguous dtype/device/K")
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if self.execution_stream is None:
            self.execution_stream = stream
        elif self.execution_stream != stream:
            raise RuntimeError("Marlin workspace cannot be shared across streams")
        flat = x.reshape(-1, self.k)
        if not flat.shape[0]:
            return torch.empty((*x.shape[:-1], self.n), device=x.device, dtype=x.dtype)
        output = torch.ops._C.marlin_gemm(
            flat, None, self.weight, None, self.scales, None, None,
            self.zeros, self.empty, self.empty, self.workspace, UINT4_ID,
            flat.shape[0], self.n, self.k, True, False, True, False)
        self.python_dispatches += 1
        return output.reshape(*x.shape[:-1], self.n)

    def stats(self):
        tensors = (self.weight, self.zeros, self.scales, self.workspace, self.empty)
        return dict(backend="vllm_0_17_1_native_awq_marlin_sidecar", gpu_validation="PENDING",
                    persistent_bytes=sum(t.numel() * t.element_size() for t in tensors),
                    full_dequantized_weight_bytes=0, python_dispatches=self.python_dispatches,
                    counter_scope="Python calls only; graph replays do not reenter Python",
                    scale_cache=self.scale_cache_report, workspace_scope=self.workspace_scope,
                    note="caller-owned original AWQ payload adds to transient/resident memory until released")
