"""Strict TP1 AutoAWQ GEMM checkpoint adapter, without dense weight storage.

Format: qweight[K,N/8] int32, qzeros[K/G,N/8] int32, scales[K/G,N]
FP16/BF16 scale metadata; activations are FP16. Both integer tensors use AWQ's interleaved output-channel nibble order.
The loader may use the existing Parameter.weight_loader convention. Supported
shards are explicit, so q/k/v and gate/up cannot be silently concatenated in
the wrong order. GPTQ, GEMV, Marlin-repacked files and TP>1 are rejected.

Reference format: vllm-project/vllm v0.17.1 quantization/awq.py and
quantization/awq_triton.py (Apache-2.0). The CPU unpacker is validation-only;
forward never dequantizes a full weight matrix and never calls F.linear.
"""
from collections import Counter
from collections.abc import Mapping

import torch
from torch import nn

AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_UNPACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def validate_awq_config(config):
    if not isinstance(config, Mapping):
        raise ValueError("AWQ requires a quantization_config mapping")
    if (config.get("quant_method") != "awq" or type(config.get("bits")) is not int
            or config["bits"] != 4 or str(config.get("version", "")).lower() != "gemm"
            or config.get("zero_point") is not True
            or type(config.get("group_size")) is not int or config["group_size"] != 128):
        raise ValueError("supported checkpoint: AWQ GEMM, bits=4, group_size=128, zero_point=true")
    if config.get("backend", "autoawq") not in ("autoawq", "auto"):
        raise ValueError("unsupported AWQ checkpoint backend")
    if config.get("modules_to_not_convert") not in (None, []):
        raise ValueError("mixed skipped-module policies require explicit integration")
    return 128


def _cpu_integer_matrix(values):
    if values.device.type != "cpu" or values.ndim != 2:
        raise ValueError("packing references require a CPU matrix")
    if values.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError("packing references require integer values")


def pack_awq_int4_cpu(values):
    _cpu_integer_matrix(values)
    if values.shape[1] == 0 or values.shape[1] % 8:
        raise ValueError("AWQ output width must be a positive multiple of eight")
    if bool(((values < 0) | (values > 15)).any()):
        raise ValueError("AWQ unsigned nibbles must be in [0,15]")
    groups = values.to(torch.int64).reshape(values.shape[0], -1, 8)
    packed = torch.zeros(groups.shape[:2], dtype=torch.int64, device="cpu")
    for nibble, channel in enumerate(AWQ_PACK_ORDER):
        packed |= groups[:, :, channel] << (4 * nibble)
    return packed.to(torch.int32).contiguous()


def unpack_awq_int4_cpu(packed):
    _cpu_integer_matrix(packed)
    if packed.dtype != torch.int32:
        raise ValueError("AWQ packed storage must be int32")
    unsigned = packed.to(torch.int64) & 0xFFFFFFFF
    values = [(unsigned >> (4 * nibble)) & 15 for nibble in AWQ_UNPACK_ORDER]
    return torch.stack(values, dim=-1).reshape(packed.shape[0], -1).to(torch.int32)


def reference_weight_cpu(qweight, qzeros, scales, group_size=128):
    """Testing only: return dequantized [K,N] with kernel-equivalent FP16 rounding."""
    if scales.device.type != "cpu" or scales.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("reference scales must be CPU FP16/BF16")
    q = unpack_awq_int4_cpu(qweight)
    z = unpack_awq_int4_cpu(qzeros)
    k, n = q.shape
    if group_size != 128 or k % group_size or z.shape != (k // group_size, n) or scales.shape != z.shape:
        raise ValueError("invalid AWQ reference layout")
    groups = torch.arange(k, device="cpu") // group_size
    # AWQ stores the zero directly. GPTQ's historical zero-minus-one convention
    # must NOT be applied here.
    return ((q.float() - z[groups].float()) * scales[groups].float()).to(torch.float16)


class AWQPackedLinear(nn.Module):
    def __init__(self, input_size, output_sizes, *, shard_ids=None, group_size=128,
                 dtype=torch.float16, device=None, tp_size=1, tp_rank=0, bias=False):
        super().__init__()
        if tp_size != 1 or tp_rank != 0:
            raise ValueError("AWQ adapter currently supports TP1 only")
        if dtype != torch.float16 or bias:
            raise ValueError("AWQ adapter currently requires FP16 and bias=False")
        if type(input_size) is not int or input_size <= 0 or group_size != 128 or input_size % group_size:
            raise ValueError("AWQ input size must be a positive multiple of group_size=128")
        if type(output_sizes) is int:
            output_sizes = (output_sizes,)
        self.output_sizes = tuple(output_sizes)
        if not self.output_sizes or any(type(n) is not int or n <= 0 or n % 8 for n in self.output_sizes):
            raise ValueError("each AWQ output shard must be a positive multiple of eight")
        self.shard_ids = tuple(shard_ids) if shard_ids is not None else (None,)
        if len(self.shard_ids) != len(self.output_sizes) or len(set(self.shard_ids)) != len(self.shard_ids):
            raise ValueError("output_sizes requires unique explicit shard_ids")
        self.input_size, self.output_size = input_size, sum(self.output_sizes)
        self.group_size = group_size
        self.backend = "awq_packed_triton"
        self.qweight = nn.Parameter(torch.empty(input_size, self.output_size // 8,
                                    device=device, dtype=torch.int32), requires_grad=False)
        self.qzeros = nn.Parameter(torch.empty(input_size // group_size, self.output_size // 8,
                                   device=device, dtype=torch.int32), requires_grad=False)
        self.scales = nn.Parameter(torch.empty(input_size // group_size, self.output_size,
                                   device=device, dtype=dtype), requires_grad=False)
        self.register_parameter("bias", None)
        for parameter in (self.qweight, self.qzeros, self.scales):
            parameter.weight_loader = self.weight_loader
        self._loaded = set()
        self._ready = False
        self.counters = Counter()

    @torch.no_grad()
    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        if self._ready:
            raise ValueError("AWQ weights are sealed; no live KV/graph weight replacement")
        fields = {id(self.qweight): "qweight", id(self.qzeros): "qzeros", id(self.scales): "scales"}
        name = fields.get(id(param))
        if name is None or loaded_shard_id not in self.shard_ids:
            raise ValueError("unknown AWQ parameter or checkpoint shard")
        index = self.shard_ids.index(loaded_shard_id)
        key = (name, loaded_shard_id)
        if key in self._loaded:
            raise ValueError("duplicate AWQ checkpoint tensor")
        divisor = 1 if name == "scales" else 8
        offset = sum(self.output_sizes[:index]) // divisor
        count = self.output_sizes[index] // divisor
        expected = (param.shape[0], count)
        if loaded_weight.device.type != "cpu":
            raise ValueError("AWQ checkpoint validation expects CPU source tensors")
        # 官方 Qwen3-8B-AWQ 的 config 声明 FP16 激活，但 safetensors 的 scale
        # 是 BF16。保留原始 scale 数值，不偷偷做有损 BF16->FP16 转换。
        if name == "scales" and loaded_weight.dtype in (torch.float16, torch.bfloat16):
            if loaded_weight.dtype != param.dtype:
                if any(field == "scales" for field, _ in self._loaded):
                    raise ValueError("mixed scale dtypes within a fused AWQ module are unsupported")
                if tuple(loaded_weight.shape) != expected:
                    raise ValueError("invalid AWQ scale shape")
                if not bool(torch.isfinite(loaded_weight).all()) or bool((loaded_weight <= 0).any()):
                    raise ValueError("AWQ scales must be finite and positive")
                param.data = torch.empty_like(param, dtype=loaded_weight.dtype)
        if tuple(loaded_weight.shape) != expected or loaded_weight.dtype != param.dtype:
            raise ValueError(f"AWQ {key} expected {expected}/{param.dtype}, got {tuple(loaded_weight.shape)}/{loaded_weight.dtype}")
        if name == "scales" and (not bool(torch.isfinite(loaded_weight).all()) or bool((loaded_weight <= 0).any())):
            raise ValueError("AWQ scales must be finite and positive")
        param[:, offset:offset + count].copy_(loaded_weight)
        self._loaded.add(key)
        self._ready = len(self._loaded) == 3 * len(self.shard_ids)

    def assert_loaded(self):
        if not self._ready:
            missing = [(name, shard) for name in ("qweight", "qzeros", "scales")
                       for shard in self.shard_ids if (name, shard) not in self._loaded]
            raise RuntimeError(f"incomplete AWQ checkpoint: {missing}")

    def forward(self, x):
        self.assert_loaded()
        if self.backend == "awq_marlin":
            return self._active_marlin(x)
        # Lazy import keeps CPU loader/layout checks free of any GPU initialization.
        from nanovllm.layers.awq_triton_kernel import awq_packed_gemm
        result = awq_packed_gemm(x, self.qweight, self.qzeros, self.scales, self.group_size)
        self.counters["python_forward_dispatches"] += 1
        self.counters["python_input_rows"] += x.numel() // self.input_size
        return result

    def prepare_marlin(self, library):
        """启动期 repack；原始 checkpoint 不变，运行期仅保留一份 packed 权重。"""
        self.assert_loaded()
        if self.backend != "awq_packed_triton":
            raise RuntimeError("quant backend is sealed after preparation")
        from nanovllm.layers.awq_marlin_sidecar import MarlinAWQState
        state = MarlinAWQState.prepare(self.qweight, self.qzeros, self.scales,
            library=library, activation_dtype=torch.float16, allow_lossless_fp16_scale_cache=True)
        self._source_scale_dtype = str(self.scales.dtype)
        self._marlin_state = self._active_marlin = state
        self._marlin_graph_states = []
        for name in ("weight", "zeros", "scales", "workspace", "empty"):
            self.register_buffer("_marlin_"+name, getattr(state,name), persistent=False)
        # 重排后的格式仍是4bit；不是 K*N 个浮点权重。释放原packed副本，
        # empty Parameter 保留sealed loader接口。后续重新加载必须新建模型。
        for param in (self.qweight, self.qzeros, self.scales):
            param.data = torch.empty(0,device=param.device,dtype=param.dtype)
        self.backend = "awq_marlin"

    def _apply(self, fn, recurse=True):
        if self.backend == "awq_marlin":
            raise RuntimeError("prepared Marlin state is immutable; construct a new model for dtype/device changes")
        return super()._apply(fn,recurse=recurse)

    def stats(self):
        if self.backend == "awq_marlin":
            state = self._marlin_state
            return dict(backend=self.backend, ready=self._ready, input_size=self.input_size,
                output_size=self.output_size, group_size=self.group_size,
                packed_weight_bytes=state.weight.numel()*state.weight.element_size(),
                zero_bytes=state.zeros.numel()*state.zeros.element_size(),
                scale_bytes=state.scales.numel()*state.scales.element_size(),
                scale_dtype=self._source_scale_dtype, activation_dtype="torch.float16",
                scale_cache=state.scale_cache_report, dense_weight_bytes=0, canonical_packed_resident_bytes=0,
                workspace_bytes=sum(s.workspace.numel()*s.workspace.element_size()
                    for s in [state,*self._marlin_graph_states]), graph_workspaces=len(self._marlin_graph_states))
        return dict(self.counters, backend=self.backend, ready=self._ready,
                    counter_scope="Python dispatches only; CUDA graph replay does not re-enter Python",
                    input_size=self.input_size, output_size=self.output_size, group_size=self.group_size,
                    packed_weight_bytes=self.qweight.numel() * self.qweight.element_size(),
                    zero_bytes=self.qzeros.numel() * self.qzeros.element_size(),
                    scale_bytes=self.scales.numel() * self.scales.element_size(),
                    scale_dtype=str(self.scales.dtype), activation_dtype="torch.float16",
                    dense_weight_bytes=0)
