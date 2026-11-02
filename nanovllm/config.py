"""nano-vLLM 的全局运行配置。

这里的配置会同时被调度器、模型执行器和 KV Cache 管理器使用。
把它集中在一个对象中，便于在创建引擎时统一控制 batch 大小、显存预算、
张量并行以及 CUDA Graph 等行为。
"""

import os
import math
from dataclasses import dataclass

from transformers import AutoConfig
from nanovllm.runtime_profiles import resolve_runtime_profile, OPTIMIZED_PROFILE_STATUS


@dataclass(slots=True)
class Config:
    """推理引擎运行时需要的参数。

    `slots=True` 会用固定的字段布局保存对象，减少配置对象本身的开销。
    其中 `hf_config` 和 `eos` 会在初始化时根据模型目录补充。
    """

    model: str
    # 显式 packed AWQ GEMM W4A16；不会自动把量化 checkpoint 当普通权重加载。
    quantization: str | None = None
    draft_quantization: str | None = None
    quant_backend: str = "triton"
    draft_quant_backend: str = "triton"
    marlin_library: str | None = None
    model_dtype: str = "checkpoint"
    draft_dtype: str = "checkpoint"
    runtime_preset: str = "baseline"
    # None 区分“未设置”和显式覆盖，防止 preset 吞掉用户的消融配置。
    batching_mode: str | None = None
    speculative_policy: str | None = None
    metadata_backend: str | None = None
    enable_mixed_graph: bool | None = None
    mixed_attention_backend: str = "split"
    metadata_max_bytes: int = 64 * 2**20
    adaptive_max_batch_size: int = 16
    prefill_max_wait_steps: int = 8
    # 0 保持原混批；正数表示低于该 running Decode 数量时用分离轮次，
    # 防止少量 Decode 请求被大 chunk 的混合 Attention 拖慢 TPOT。
    mixed_min_active_decodes: int = 0
    # 仅在正数门槛命中时生效：separate 用纯 P/D 轮次；micro 优先 D，
    # 等待达到公平轮数时只注入一个 P token；legacy_fifo 实验按队首
    # prompt 的 chunk 顺序入队，未完成时暂不让后续短 prompt 提前 Decode。
    mixed_low_load_mode: str = "separate"
    # 一个 prefill batch 最多同时处理多少个 prompt token。
    max_num_batched_tokens: int = 16384
    # 单条请求在一次调度 step 中最多处理的 prompt token；None 表示不限制。
    prefill_chunk_size: int | None = None
    # 同时存在 waiting/running 时，连续执行多少个 prefill step 后强制执行 decode。
    # 0 表示保持原有的“只要有 prefill 就先做 prefill”策略。
    max_prefill_steps: int = 0
    # 显式实验：观测成本决定prefill预算，不保证每个请求都满足SLO。
    latency_aware_scheduling: bool = False
    # JSON只含离线校准参数；验证时不再更新。路径需属于当前模型校准结果。
    scheduler_cost_model_path: str | None = None
    scheduler_ttft_target_ms: float = 1000.
    scheduler_tpot_target_ms: float = 30.
    scheduler_min_prefill_tokens: int = 32
    # 显式桶只用于启动预捕获，不在请求路径临时capture。
    decode_graph_buckets: tuple[int, ...] | None = None
    # 0 表示不限制；正数表示仅在 decode batch 不超过该大小时尝试 draft。
    # 小模型高并发时普通 decode 已利用批处理，额外 verification 可能更慢。
    speculative_max_batch_size: int = 0
    # Optional native Qwen3 draft runner; v1 is Python-only, TP1, fixed greedy.
    draft_model: str | None = None
    # 原生草稿执行方式；eager 留作数值/性能对照，graph 在启动阶段录制。
    draft_execution: str = "eager"
    draft_graph_max_tokens: int = 8
    # target 验证也需消除逐算子 launch；与 draft 图分开开关以便消融。
    draft_verify_graph: bool = False
    # 独立实验：单个 packed metadata H2D，仍使用完全相同的验证图。
    draft_verify_metadata: bool = False
    # 独立实验，默认关闭：pinned draft 状态 + event 保护的异步 H2D。
    draft_async_metadata: bool = False
    # 独立实验，默认关闭：常见前缀块按真实 Q/K 长度在启动期捕获。
    draft_prefill_graph: bool = False
    # 单独消融冷前缀完整 chunk 后的 Q1 尾块；默认关闭，不做 query padding。
    draft_prefill_tail_graph: bool = False
    # TP1 greedy：候选不先回 CPU，验证后一次回传完整提交证据。
    draft_device_verification: bool = False
    # 启动期分配概率 proposal/图；请求还须显式选 rejection。首版仅 BF16 consistent target。
    enable_stochastic_speculation: bool = False
    # 一个 batch 最多包含多少个 Sequence（请求）。
    max_num_seqs: int = 512
    # 单条请求允许的最大上下文长度（prompt + completion）。
    max_model_len: int = 4096
    # 允许 KV Cache 使用的显存比例。
    gpu_memory_utilization: float = 0.9
    # Tensor Parallel 的 GPU 数量。
    tensor_parallel_size: int = 1
    # TP rendezvous 端口；None 时继承 NANOVLLM_MASTER_PORT，否则自动分配。
    tp_master_port: int | None = None
    # 一次 TP 命令/collective 的时限，以及加载权重/编译的启动等待时限。
    tp_timeout_seconds: float = 120.0
    tp_startup_timeout_seconds: float = 600.0
    tp_shutdown_timeout_seconds: float = 5.0
    # 单条 CPU 调度消息容量，不含 GPU KV 数据；超限请求会明确报错。
    tp_ipc_bytes: int = 2**20
    # 由 LLMEngine 每次启动生成，不应由用户手工复用。
    tp_shm_name: str | None = None
    # True 时不使用 CUDA Graph，便于调试但通常速度较慢。
    enforce_eager: bool = False
    # RMSNorm 后端：torch 使用现有 torch.compile，triton 使用自写融合算子。
    rmsnorm_backend: str = "torch"
    # 显式数值实验：固定 Triton GEMM tile；默认 Torch 不变。
    linear_backend: str = "torch"
    # 显式选择单请求数值一致路径；改变算术后端，不承诺与旧 Torch 逐位一致。
    numerical_backend: str = "default"
    stable_linear_packed: bool = False  # target-only physical layout, before Graph capture
    # Torch参考；Triton加速greedy；FlashInfer实验性随机采样需独立可选依赖。
    sampler_backend: str = "torch"
    # 仅FlashInfer使用：候选随机行少于该值时，省去metadata/同步开销，走Torch。
    flashinfer_min_batch_size: int = 32
    # MLP 实验后端：torch默认；triton人工门槛；auto为SM89已测形状白名单。
    mlp_backend: str = "torch"
    # 仅对人工triton模式生效；auto按固定离线形状表选择。
    mlp_min_tokens: int = 128
    # Decode Attention 后端：flash 使用 FlashAttention，triton 使用实验 kernel。
    decode_attention_backend: str = "flash"
    cascade_min_shared_tokens: int = 256
    # 调试/安全验收模式：每层同时计算参考，候选不逐元素相同则返回参考。
    # 不放宽容差，也不是加速模式；原始候选误差必须另行记录。
    cascade_verify_reference: bool = False
    # KV Cache 存储类型：model 使用模型 dtype；fp8 是实验性低精度 cache。
    kv_cache_dtype: str = "model"
    # 是否启用 prefix cache；FP8 现在可以通过 dequant kernel 读取 prefix。
    enable_prefix_caching: bool = True
    # 实验性 CPU 层：0 关闭。支持 TP1/model KV 的 separate/mixed 调度。
    host_kv_cache_bytes: int = 0
    host_kv_async_prefetch: bool = False
    host_kv_prefetch_max_inflight_blocks: int = 4
    host_kv_min_restore_tokens: int = 256
    host_kv_max_restore_blocks: int = 8
    # 两个独立实验：离线成本策略与有界批量传输；legacy 默认路径保留。
    host_kv_cost_model_path: str | None = None
    # 独立实验，默认关闭；首版保留 legacy async 传输并联合决定两级驻留。
    host_kv_joint_residency_path: str | None = None
    host_kv_admission_min_requests: int = 2
    host_kv_policy_history_entries: int = 4096
    host_kv_transfer_backend: str = "legacy"
    host_kv_transfer_batch_blocks: int = 4
    host_kv_transfer_slots: int = 2
    host_kv_transfer_max_bytes: int = 256 * 2**20
    host_kv_transfer_idle_fastpath: bool = False  # 独立实验：没有传输任务时跳过重复 CPU 工作。
    # mixed Graph 默认只允许重放；显式预热窗口内才开放捕获。
    allow_runtime_graph_capture: bool = False
    # Triton paged decode 的离线调优结果；CUDA Graph 运行时保持固定。
    decode_attention_block_n: int = 64
    decode_attention_num_warps: int | None = None
    # Hugging Face 解析出的模型结构配置；由 __post_init__ 填充。
    hf_config: AutoConfig | None = None
    # EOS token id；加载 tokenizer 后由 LLMEngine 填充。
    eos: int = -1
    # 一个 KV Cache block 保存多少个 token。必须是 256 的倍数。
    kvcache_block_size: int = 256
    # 根据显存预算计算出的 KV Cache block 数量。
    num_kvcache_blocks: int = -1
    # 可选真实物理页上限：在分配tensor前生效，便于受控容量实验或预留显存。
    # None沿用自动显存预算；不能用该上限突破显存预算计算出的安全容量。
    max_kvcache_blocks: int | None = None

    def __post_init__(self):
        """在加载权重和启动 worker 前拒绝无效配置，报出可读错误。"""
        resolved = resolve_runtime_profile(self.runtime_preset, {
            name: getattr(self, name) for name in (
                "batching_mode", "speculative_policy", "metadata_backend", "enable_mixed_graph")})
        for name, value in resolved.items():
            setattr(self, name, value)
        if not isinstance(self.enable_mixed_graph, bool):
            raise ValueError("enable_mixed_graph must be bool")
        if not isinstance(self.allow_runtime_graph_capture, bool):
            raise ValueError("allow_runtime_graph_capture must be bool")
        if type(self.latency_aware_scheduling) is not bool:
            raise ValueError("latency_aware_scheduling must be bool")
        if type(self.host_kv_async_prefetch) is not bool:
            raise ValueError("host_kv_async_prefetch must be bool")
        if self.host_kv_async_prefetch and not self.host_kv_cache_bytes:
            raise ValueError("async prefetch requires a positive host KV cache budget")
        if type(self.host_kv_prefetch_max_inflight_blocks) is not int or self.host_kv_prefetch_max_inflight_blocks < 1:
            raise ValueError("host_kv_prefetch_max_inflight_blocks must be positive")
        if self.latency_aware_scheduling and self.batching_mode != "separate":
            raise ValueError("latency policy currently supports separate scheduling only")
        if self.scheduler_cost_model_path is not None:
            import json, hashlib
            from pathlib import Path
            if not self.latency_aware_scheduling:
                raise ValueError("cost model requires explicit latency_aware_scheduling")
            model=json.loads(Path(self.scheduler_cost_model_path).read_text())
            if model.get("schema")!="nano-affine-v1":
                raise ValueError("unsupported scheduler cost model schema")
            if model.get("frozen") is not True:
                raise ValueError("scheduler cost model must be explicitly frozen")
            if model.get("model_config_sha256")!=hashlib.sha256(Path(self.model,"config.json").read_bytes()).hexdigest():
                raise ValueError("scheduler cost model belongs to a different model config")
            from nanovllm.engine.latency_policy import LatencyPolicy
            LatencyPolicy(cost_model=model)  # 在加载GPU权重前拒绝非有限/非法系数。
        for name in ("scheduler_ttft_target_ms", "scheduler_tpot_target_ms"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.scheduler_min_prefill_tokens) is not int or self.scheduler_min_prefill_tokens < 1:
            raise ValueError("scheduler_min_prefill_tokens must be positive")
        if type(self.mixed_min_active_decodes) is not int or self.mixed_min_active_decodes < 0:
            raise ValueError("mixed_min_active_decodes must be a non-negative integer")
        if self.mixed_low_load_mode not in ("separate", "micro", "legacy_fifo"):
            raise ValueError("mixed_low_load_mode must be separate, micro or legacy_fifo")
        if self.decode_graph_buckets is not None:
            buckets = self.decode_graph_buckets
            if (not isinstance(buckets, (tuple, list)) or not buckets
                    or any(type(b) is not int or b <= 0 or b > min(self.max_num_seqs, 512) for b in buckets)
                    or list(buckets) != sorted(set(buckets)) or buckets[-1] != min(self.max_num_seqs, 512)):
                raise ValueError("decode_graph_buckets must be sorted unique and cover max decode batch")
            self.decode_graph_buckets = tuple(buckets)
        if self.max_kvcache_blocks is not None and (
                type(self.max_kvcache_blocks) is not int or self.max_kvcache_blocks <= 0):
            raise ValueError("max_kvcache_blocks must be a positive integer or None")
        for name in ("host_kv_cache_bytes", "host_kv_min_restore_tokens", "host_kv_max_restore_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "host_kv_cache_bytes" else 1):
                raise ValueError(f"invalid {name}")
        if self.host_kv_cache_bytes and (
                self.tensor_parallel_size != 1 or self.kv_cache_dtype != "model"
                or not self.enable_prefix_caching):
            raise ValueError("host KV cache requires TP1, model KV and prefix caching")
        if self.host_kv_transfer_backend not in {"legacy", "batched"}:
            raise ValueError("host_kv_transfer_backend must be legacy or batched")
        if type(self.host_kv_transfer_idle_fastpath) is not bool:
            raise ValueError('host_kv_transfer_idle_fastpath must be bool')
        if self.host_kv_transfer_idle_fastpath and self.host_kv_transfer_backend!='batched':
            raise ValueError('idle cache fastpath requires the batched transfer backend')
        for name in ("host_kv_admission_min_requests", "host_kv_policy_history_entries",
                     "host_kv_transfer_batch_blocks", "host_kv_transfer_slots", "host_kv_transfer_max_bytes"):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 1 <= self.host_kv_transfer_batch_blocks <= 8 or not 2 <= self.host_kv_transfer_slots <= 8:
            raise ValueError("batched cache supports batch 1..8 and slots 2..8")
        if self.host_kv_transfer_backend == "batched" and (
                not self.host_kv_cache_bytes or not self.host_kv_async_prefetch):
            raise ValueError("batched cache requires a host budget and explicit async prefetch")
        if self.host_kv_cost_model_path is not None:
            if not self.host_kv_cache_bytes:
                raise ValueError("cache cost policy requires a host cache budget")
            from nanovllm.engine.cache_cost import CacheCostModel
            if not isinstance(self.host_kv_cost_model_path,str) or not os.path.isabs(self.host_kv_cost_model_path) or not os.path.isfile(self.host_kv_cost_model_path):
                raise ValueError("cache cost model requires an absolute existing calibration path")
            CacheCostModel.load(self.host_kv_cost_model_path).check_config(self)
        if self.host_kv_joint_residency_path is not None:
            if (not self.host_kv_cache_bytes or not self.host_kv_async_prefetch
                    or self.host_kv_transfer_backend != 'legacy' or self.host_kv_cost_model_path is not None):
                raise ValueError('joint residency requires legacy async host cache and no other cache cost policy')
            if (not isinstance(self.host_kv_joint_residency_path, str)
                    or not os.path.isabs(self.host_kv_joint_residency_path)
                    or not os.path.isfile(self.host_kv_joint_residency_path)):
                raise ValueError('joint residency requires an absolute existing calibration path')
            from nanovllm.engine.joint_cache import JointResidencyPolicy
            JointResidencyPolicy.load(self.host_kv_joint_residency_path).model.check_config(self)
        if (self.host_kv_cost_model_path is not None or self.host_kv_joint_residency_path is not None or self.host_kv_transfer_backend=='batched') and (
                self.quantization is not None or self.speculative_policy!='off'):
            raise ValueError('new host cache paths currently require dense non-speculative execution')
        if self.draft_model is not None:
            if not isinstance(self.draft_model, str) or not os.path.isdir(self.draft_model):
                raise ValueError("draft_model must be an existing model directory")
            if (self.tensor_parallel_size != 1 or self.kv_cache_dtype != "model"
                    or self.batching_mode != "mixed" or self.speculative_policy != "fixed"
                    or self.max_num_seqs != 1 or self.host_kv_cache_bytes):
                raise ValueError("draft_model requires TP1, model KV, mixed scheduling, fixed policy, "
                                 "max_num_seqs=1 and host KV disabled")
        if self.draft_execution not in {"eager", "graph"}:
            raise ValueError("draft_execution must be eager or graph")
        if type(self.draft_graph_max_tokens) is not int or not 1 <= self.draft_graph_max_tokens <= 16:
            raise ValueError("draft_graph_max_tokens must be an integer in [1, 16]")
        if type(self.draft_verify_graph) is not bool:
            raise ValueError("draft_verify_graph must be bool")
        if type(self.draft_verify_metadata) is not bool:
            raise ValueError("draft_verify_metadata must be bool")
        if type(self.draft_prefill_graph) is not bool:
            raise ValueError("draft_prefill_graph must be bool")
        if self.draft_prefill_graph and (self.draft_model is None or self.draft_execution != "graph" or self.enforce_eager):
            raise ValueError("draft_prefill_graph requires draft_model and active draft graphs")
        if type(self.draft_prefill_tail_graph) is not bool:
            raise ValueError("draft_prefill_tail_graph must be bool")
        if self.draft_prefill_tail_graph and (not self.draft_prefill_graph or self.enable_stochastic_speculation):
            raise ValueError("draft_prefill_tail_graph requires greedy draft_prefill_graph")
        if type(self.draft_device_verification) is not bool:
            raise ValueError("draft_device_verification must be bool")
        if self.draft_device_verification and (self.draft_model is None or self.draft_execution!="graph"
                or self.enforce_eager or not self.draft_verify_graph or not self.draft_verify_metadata or not self.draft_async_metadata
                or self.enable_stochastic_speculation or self.numerical_backend!="consistent"):
            raise ValueError("draft_device_verification requires greedy consistent native graphs with packed/pinned metadata")
        if type(self.draft_async_metadata) is not bool:
            raise ValueError("draft_async_metadata must be bool")
        if self.draft_async_metadata and (
                self.draft_model is None or self.draft_execution != "graph" or self.enforce_eager):
            raise ValueError("draft_async_metadata requires draft_model and active draft graphs")
        if self.draft_verify_metadata and (not self.draft_verify_graph or self.enforce_eager):
            raise ValueError("packed verification metadata requires active native verification graphs")
        if type(self.enable_stochastic_speculation) is not bool:
            raise ValueError("enable_stochastic_speculation must be bool")
        if self.enable_stochastic_speculation and (self.draft_model is None or self.numerical_backend != "consistent"):
            raise ValueError("stochastic speculation v1 requires native draft_model and consistent target backend")
        if (self.draft_execution != "eager" or self.draft_verify_graph) and self.draft_model is None:
            raise ValueError("draft graph options require draft_model")
        if not os.path.isdir(self.model):
            raise ValueError(f"model must be an existing directory: {self.model}")

        positive_fields = (
            "max_num_batched_tokens", "max_num_seqs", "max_model_len",
            "kvcache_block_size", "tensor_parallel_size", "decode_attention_block_n",
            "mlp_min_tokens",
            "metadata_max_bytes", "adaptive_max_batch_size", "prefill_max_wait_steps",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a multiple of 256")
        if self.tensor_parallel_size > 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if self.tp_master_port is None and "NANOVLLM_MASTER_PORT" in os.environ:
            try:
                self.tp_master_port = int(os.environ["NANOVLLM_MASTER_PORT"])
            except ValueError as error:
                raise ValueError("NANOVLLM_MASTER_PORT must be an integer") from error
        if self.tp_master_port is not None and (
                isinstance(self.tp_master_port, bool)
                or not isinstance(self.tp_master_port, int)
                or not 1 <= self.tp_master_port <= 65535):
            raise ValueError("tp_master_port must be None or an integer in [1, 65535]")
        for name in ("tp_timeout_seconds", "tp_startup_timeout_seconds", "tp_shutdown_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.tp_ipc_bytes, bool) or not isinstance(self.tp_ipc_bytes, int)
                or not 4096 <= self.tp_ipc_bytes <= 64 * 2**20):
            raise ValueError("tp_ipc_bytes must be an integer in [4096, 67108864]")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.prefill_chunk_size is not None:
            if (isinstance(self.prefill_chunk_size, bool)
                    or not isinstance(self.prefill_chunk_size, int)
                    or not 0 < self.prefill_chunk_size <= self.max_num_batched_tokens):
                raise ValueError("prefill_chunk_size must be an integer in [1, max_num_batched_tokens]")
        for name in ("max_prefill_steps", "speculative_max_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        choices = {
            "rmsnorm_backend": {"torch", "triton"},
            "linear_backend": {"torch", "triton_stable"},
            "numerical_backend": {"default", "consistent"},
            "sampler_backend": {"torch", "triton", "flashinfer"},
            "mlp_backend": {"torch", "triton", "auto"},
            "decode_attention_backend": {"flash", "triton", "flashinfer", "cascade"},
            "kv_cache_dtype": {"model", "fp8"},
            "batching_mode": {"separate", "mixed"},
            "speculative_policy": {"off", "fixed", "adaptive"},
            "metadata_backend": {"dynamic", "reuse"},
            "mixed_attention_backend": {"split", "varlen"},
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if type(self.flashinfer_min_batch_size) is not int or self.flashinfer_min_batch_size < 1:
            raise ValueError("flashinfer_min_batch_size must be a positive integer")
        if self.sampler_backend == "flashinfer" and (
                self.tensor_parallel_size != 1 or self.batching_mode != "separate"):
            raise ValueError("FlashInfer sampler currently validated only for TP1/separate")
        if self.decode_attention_backend in {"flashinfer", "cascade"} and (
                self.tensor_parallel_size != 1 or self.batching_mode != "separate"
                or self.kv_cache_dtype != "model"):
            raise ValueError("FlashInfer decode requires TP1/separate/model-dtype KV")
        if type(self.cascade_min_shared_tokens) is not int or self.cascade_min_shared_tokens<256:
            raise ValueError("cascade_min_shared_tokens must be an integer >=256")
        if type(self.cascade_verify_reference) is not bool:
            raise ValueError("cascade_verify_reference must be bool")
        if self.decode_attention_backend=="cascade" and not self.enforce_eager:
            raise ValueError("Cascade is eager-only; explicitly set enforce_eager=True")
        if self.speculative_policy == "adaptive" and self.batching_mode != "mixed":
            raise ValueError("speculative_policy='adaptive' requires batching_mode='mixed'")
        if self.enable_mixed_graph and self.batching_mode != "mixed":
            raise ValueError("enable_mixed_graph requires batching_mode='mixed'")
        if self.decode_attention_block_n & (self.decode_attention_block_n - 1):
            raise ValueError("decode_attention_block_n must be a power of two")
        if self.decode_attention_num_warps is not None and (
                isinstance(self.decode_attention_num_warps, bool)
                or self.decode_attention_num_warps not in {1, 2, 4, 8, 16}):
            raise ValueError("decode_attention_num_warps must be one of 1, 2, 4, 8, 16")
        if self.kv_cache_dtype == "fp8" and self.decode_attention_backend != "triton":
            raise ValueError("kv_cache_dtype='fp8' requires decode_attention_backend='triton'")

        self.hf_config = AutoConfig.from_pretrained(self.model)
        from nanovllm.utils.quant_config import configure_weight_format
        configure_weight_format(self.hf_config, self.quantization, self.model_dtype)
        for name, enabled in (("quant_backend", self.quantization), ("draft_quant_backend", self.draft_quantization)):
            if getattr(self, name) not in ("triton", "marlin"):
                raise ValueError(f"{name} must be triton or marlin")
            if getattr(self, name) == "marlin" and enabled != "awq":
                raise ValueError("Marlin backend requires explicit AWQ quantization")
        if "marlin" in (self.quant_backend, self.draft_quant_backend):
            if not self.marlin_library or not os.path.isabs(self.marlin_library) or not os.path.isfile(self.marlin_library):
                raise ValueError("Marlin requires an absolute verified sidecar library path")
        if self.draft_quantization is not None and self.draft_model is None:
            raise ValueError("draft_quantization requires draft_model")
        if self.quantization is not None or self.draft_quantization is not None:
            if self.tensor_parallel_size != 1 or self.kv_cache_dtype != "model" or self.max_num_seqs != 1:
                raise ValueError("AWQ v1 requires single-request TP1 and model-dtype KV")
            if self.enable_mixed_graph or self.allow_runtime_graph_capture:
                raise ValueError("AWQ v1 supports startup graphs only, not mixed/runtime capture")
        if self.quantization is not None and (self.linear_backend != "torch"
                or self.mlp_backend != "torch" or self.numerical_backend != "default"):
            raise ValueError("AWQ target cannot use BF16 stable linear/consistent or fused MLP")
        if self.numerical_backend == "consistent":
            if (self.tensor_parallel_size != 1 or self.max_num_seqs != 1
                    or self.kv_cache_dtype != "model" or self.hf_config.model_type != "qwen3"
                    or str(self.hf_config.dtype) != "torch.bfloat16"
                    or getattr(self.hf_config, "attention_bias", True)):
                raise ValueError("consistent numerical backend requires single-request TP1 BF16 bias-free Qwen3")
            if self.mlp_backend != "torch" or self.decode_attention_backend != "flash":
                raise ValueError("consistent numerical backend requires torch MLP and flash attention")
            # 这是 numerical_backend 显式选择的组成部分，effective 配置必须如实展示。
            self.linear_backend = "triton_stable"
            self.rmsnorm_backend = "triton"
        if type(self.stable_linear_packed) is not bool:
            raise ValueError("stable_linear_packed must be bool")
        if self.stable_linear_packed and self.numerical_backend != "consistent":
            raise ValueError("stable_linear_packed requires dense TP1 BF16 consistent target")
        if self.draft_model is not None:
            draft_config = AutoConfig.from_pretrained(self.draft_model)
            configure_weight_format(draft_config, self.draft_quantization, self.draft_dtype)
            if self.hf_config.model_type != "qwen3" or draft_config.model_type != "qwen3":
                raise ValueError("native draft runner currently supports dense Qwen3 only")
            if draft_config.vocab_size != self.hf_config.vocab_size:
                raise ValueError("draft and target vocabulary sizes differ")
            if min(self.max_model_len, self.hf_config.max_position_embeddings) > draft_config.max_position_embeddings:
                raise ValueError("draft model does not support the requested context length")
        if self.linear_backend == "triton_stable":
            if self.tensor_parallel_size != 1:
                raise ValueError("triton_stable linear is validated only for TP1")
            if str(self.hf_config.dtype) != "torch.bfloat16":
                raise ValueError("triton_stable linear requires BF16 model weights")
            if getattr(self.hf_config, "attention_bias", True):
                raise ValueError("triton_stable linear does not support QKV bias")
            if self.mlp_backend != "torch":
                raise ValueError("triton_stable linear cannot be combined with fused MLP")
        if self.tensor_parallel_size > 1:
            # 本项目使用等长分片，不实现 KV head 复制或非整除 vocab padding。
            for name in ("num_attention_heads", "num_key_value_heads", "hidden_size", "intermediate_size", "vocab_size"):
                value = getattr(self.hf_config, name, None)
                if not isinstance(value, int) or value <= 0 or value % self.tensor_parallel_size:
                    raise ValueError(f"{name}={value} must be divisible by tensor_parallel_size={self.tensor_parallel_size}")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    def runtime_summary(self):
        """仅输出运行时策略与有效回退，不把整个模型配置打印到日志。"""
        values = {name: getattr(self, name) for name in (
            "runtime_preset", "batching_mode", "speculative_policy", "metadata_backend", "draft_model",
            "draft_execution", "draft_graph_max_tokens", "draft_verify_graph",
            "draft_verify_metadata", "draft_async_metadata", "draft_prefill_graph", "draft_prefill_tail_graph", "draft_device_verification",
            "enable_stochastic_speculation",
            "quantization", "draft_quantization", "model_dtype", "draft_dtype",
            "quant_backend", "draft_quant_backend", "marlin_library",
            "enable_mixed_graph", "mixed_attention_backend", "metadata_max_bytes",
            "adaptive_max_batch_size", "prefill_max_wait_steps", "mlp_backend",
            "mixed_min_active_decodes",
            "mixed_low_load_mode",
            "tensor_parallel_size", "kv_cache_dtype", "enforce_eager", "sampler_backend",
            "flashinfer_min_batch_size",
            "rmsnorm_backend", "decode_attention_backend", "cascade_min_shared_tokens", "cascade_verify_reference", "max_num_seqs",
            "linear_backend", "numerical_backend", "stable_linear_packed",
            "max_num_batched_tokens", "max_model_len", "prefill_chunk_size",
            "max_prefill_steps", "speculative_max_batch_size", "enable_prefix_caching",
            "latency_aware_scheduling", "scheduler_cost_model_path", "scheduler_ttft_target_ms", "scheduler_tpot_target_ms",
            "scheduler_min_prefill_tokens", "decode_graph_buckets",
            "host_kv_cache_bytes", "host_kv_min_restore_tokens", "host_kv_max_restore_blocks",
            "host_kv_async_prefetch", "host_kv_prefetch_max_inflight_blocks",
            "host_kv_cost_model_path", "host_kv_joint_residency_path", "host_kv_admission_min_requests", "host_kv_policy_history_entries",
            "host_kv_transfer_backend", "host_kv_transfer_batch_blocks", "host_kv_transfer_slots", "host_kv_transfer_max_bytes", "host_kv_transfer_idle_fastpath",
            "allow_runtime_graph_capture",
            "kvcache_block_size", "num_kvcache_blocks", "max_kvcache_blocks", "gpu_memory_utilization")}
        values["optimized_profile_status"] = OPTIMIZED_PROFILE_STATUS
        values["decode_attention_status"] = (
            "EXPERIMENTAL_REVIEW_REQUIRED" if self.decode_attention_backend in {"flashinfer","cascade"} else "REFERENCE_OR_EXISTING")
        values["packed_supported"] = (self.tensor_parallel_size == 1 and self.kv_cache_dtype == "model"
                                      and self.decode_attention_backend not in {"flashinfer","cascade"})
        packed = self.batching_mode == "mixed" and values["packed_supported"]
        values["effective_batching_mode"] = "mixed" if packed else "separate"
        values["effective_speculative_policy"] = self.speculative_policy if packed or self.speculative_policy == "off" else "fixed"
        values["effective_mixed_graph"] = packed and self.enable_mixed_graph and not self.enforce_eager
        values["effective_draft_execution"] = self.draft_execution if self.draft_model and not self.enforce_eager else "eager"
        values["effective_draft_verify_graph"] = bool(self.draft_model and self.draft_verify_graph and not self.enforce_eager)
        values["effective_draft_async_metadata"] = bool(
            self.draft_model and self.draft_execution == "graph" and not self.enforce_eager
            and self.draft_async_metadata)
        return values
