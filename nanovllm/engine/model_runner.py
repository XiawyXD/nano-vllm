"""模型执行器：准备 batch、执行 Qwen3、维护 KV Cache 并采样。

Scheduler 决定“哪些请求本轮运行”；ModelRunner 决定“这些请求如何变成
GPU tensor 并通过模型”。rank 0 负责返回 token，其他 Tensor Parallel rank
通过进程间共享内存接收同样的调用。
"""

import os
from collections import Counter
from datetime import timedelta
from time import perf_counter
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.engine.verification_graph import VerificationGraphs
from nanovllm.engine.metadata_buffers import MetadataBuffers
from nanovllm.engine.packed_graph import PackedGraphs
from nanovllm.engine.packed_runner import run_plan as execute_plan
from nanovllm.engine.tp_runtime import encode_call, decode_call, wait_worker_status, common_kv_capacity


def decode_graph_batch_sizes(max_num_seqs: int) -> list[int]:
    """生成覆盖所有合法 decode batch 的分桶，最后一桶始终覆盖配置上限。

    例如 max_num_seqs=10 得到 [1, 2, 4, 8, 10]，不再让 batch 9/10
    找不到 graph。超过 512 的请求仍由 run_model 的 eager 路径处理。
    """
    maximum = min(max_num_seqs, 512)
    if maximum <= 0:
        raise ValueError("max_num_seqs must be positive")
    sizes = {size for size in (1, 2, 4, 8) if size <= maximum}
    sizes.update(range(16, maximum + 1, 16))
    sizes.add(maximum)
    return sorted(sizes)


class ModelRunner:
    """单个 Tensor Parallel rank 上的模型执行与 KV Cache 管理器。"""

    def __init__(self, config: Config, rank: int, event: Event | list[Event],
                 *, store=None, worker_connections=None):
        self.config = config
        hf_config = config.hf_config
        # block_size 决定分页 KV Cache 的逻辑粒度。
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.worker_connections = worker_connections or []
        self._closed = False
        self._failed = False
        self._owns_process_group = False
        self.draft_runner = None
        self.cache_transfer_workspace = None
        self.native_verification_graphs = None
        if config.draft_model is not None and config.draft_verify_graph and not config.enforce_eager:
            from nanovllm.engine.native_verification_graph import NativeVerificationGraphs
            self.native_verification_graphs = NativeVerificationGraphs(config.draft_graph_max_tokens,
                packed_metadata=getattr(config, "draft_verify_metadata", False))
        # 实验开关只影响多token验证；enforce_eager始终优先。
        # 当前仅验证单卡、模型dtype KV；TP/FP8自动保留原eager路径。
        self.verification_graphs = VerificationGraphs()
        self.packed_graphs = PackedGraphs(allow_capture=config.allow_runtime_graph_capture)
        self.metadata_buffers = MetadataBuffers(config.metadata_backend, config.metadata_max_bytes)
        self.last_plan_stats = {}
        self.graph_counters = Counter()
        self.last_model_path = "not_run"
        self.enable_verification_graph = os.environ.get("NANOVLLM_SPEC_GRAPH", "0") == "1"

        default_dtype = torch.get_default_dtype()
        try:
            self._initialize(config, rank, store)
        except BaseException:
            self._failed = True
            try:
                self.exit()
            except Exception:
                pass  # 保留最初的启动错误；父进程仍负责回收 worker。
            raise
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)
            reset_context()

    def _initialize(self, config, rank, store):
        """初始化一个 rank；TCPStore 在父进程提前保留端口，避免端口选择竞态。"""
        hf_config = config.hf_config
        if dist.is_initialized():
            raise RuntimeError("a process group already exists; use a separate process for each engine")
        torch.cuda.set_device(rank)
        if store is None:
            store = dist.TCPStore(
                "127.0.0.1", config.tp_master_port or 0, self.world_size,
                is_master=rank == 0,
                timeout=timedelta(seconds=config.tp_timeout_seconds),
                wait_for_workers=False,
            )
        dist.init_process_group(
            "nccl", store=store,
            world_size=self.world_size,
            rank=rank,
            timeout=timedelta(seconds=config.tp_timeout_seconds),
        )
        self._owns_process_group = True
        # 暂时切换到 checkpoint 声明的 dtype/device 来构造模型。
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 把运行时选择的 RMSNorm 后端传给模型层构造器。
        hf_config.nanovllm_rmsnorm_backend = config.rmsnorm_backend
        hf_config.nanovllm_linear_backend = config.linear_backend
        hf_config.nanovllm_numerical_backend = config.numerical_backend
        hf_config.nanovllm_mlp_backend = config.mlp_backend
        hf_config.nanovllm_mlp_min_tokens = config.mlp_min_tokens
        hf_config.nanovllm_decode_attention_backend = config.decode_attention_backend
        hf_config.nanovllm_decode_attention_block_n = config.decode_attention_block_n
        hf_config.nanovllm_decode_attention_num_warps = config.decode_attention_num_warps
        self.model = Qwen3ForCausalLM(hf_config)
        if getattr(config,"cascade_verify_reference",False):
            from nanovllm.layers.attention import Attention
            for module in self.model.modules():
                if isinstance(module,Attention):module.cascade_verify_reference=True
        # 从 safetensors 加载权重，并在各 TP 层切出当前 rank 的 shard。
        loading_started = perf_counter()
        load_model(self.model, config.model, quant_backend=config.quant_backend, marlin_library=config.marlin_library)
        torch.cuda.synchronize()
        self.weight_load_seconds = perf_counter() - loading_started
        self.stable_layout_stats = {"enabled": False}
        if config.stable_linear_packed:
            from nanovllm.layers.stable_layout import pack_stable_linears
            layout_started = perf_counter()
            self.stable_layout_stats = pack_stable_linears(self.model)
            torch.cuda.synchronize()
            self.stable_layout_stats["seconds"] = perf_counter() - layout_started
        if config.draft_model is not None:
            from nanovllm.engine.draft_runner import DraftRunner
            self.draft_runner = DraftRunner(config)
        self.sampler = Sampler(backend=config.sampler_backend,
                               flashinfer_min_batch_size=config.flashinfer_min_batch_size)
        # 先运行一次最大规模的 prefill，使后续 CUDA 内存/编译路径稳定。
        self.warmup_model()
        self.flashinfer_decode = None
        if config.decode_attention_backend == "cascade":
            from nanovllm.layers.cascade_attention import CascadeDecode
            self.flashinfer_decode = CascadeDecode(
                hf_config.num_attention_heads, hf_config.num_key_value_heads,
                hf_config.head_dim, self.block_size, config.max_model_len, hf_config.dtype,
                min_shared_tokens=config.cascade_min_shared_tokens)
        if config.decode_attention_backend == "flashinfer":
            from nanovllm.layers.flashinfer_attention import FlashInferDecode
            self.flashinfer_decode = FlashInferDecode(
                hf_config.num_attention_heads, hf_config.num_key_value_heads,
                hf_config.head_dim, self.block_size, config.max_model_len, hf_config.dtype,
                () if self.enforce_eager else (getattr(config, "decode_graph_buckets", None) or decode_graph_batch_sizes(min(config.max_num_seqs, 512))))
        if config.host_kv_transfer_backend == "batched":
            from nanovllm.engine.cache_transfer import CacheTransferWorkspace
            heads = hf_config.num_key_value_heads // self.world_size
            dim = getattr(hf_config,"head_dim",hf_config.hidden_size//hf_config.num_attention_heads)
            self.cache_transfer_workspace = CacheTransferWorkspace(
                (2,hf_config.num_hidden_layers,self.block_size,heads,dim),hf_config.dtype,
                batch_blocks=config.host_kv_transfer_batch_blocks,slots=config.host_kv_transfer_slots,
                max_bytes=config.host_kv_transfer_max_bytes,device=torch.device('cuda',rank))
        # 传输 staging 先计入常驻显存，再计算 KV 容量。
        self.allocate_kv_cache()
        if self.cache_transfer_workspace is not None:
            self.cache_transfer_workspace.warmup(self.kv_cache)
        if not self.enforce_eager:
            # 为常见 decode batch size 录制 CUDA Graph，减少 kernel launch 开销。
            self.capture_cudagraph()
        if self.native_verification_graphs is not None:
            self.native_verification_graphs.warmup(self.model, capacity=config.max_model_len,
                block_size=self.block_size, query_budget=config.max_num_batched_tokens)
        if self.world_size > 1:
            # 父进程创建且负责 unlink；各 rank 仅连接/关闭自己的 handle。
            self.shm = SharedMemory(name=config.tp_shm_name)

    def exit(self):
        """幂等本地清理，不用 barrier 等待可能已经崩溃的其他 rank。"""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if getattr(self, "native_verification_graphs", None) is not None:
            self.native_verification_graphs.close()
        self.verification_graphs.entries.clear()
        self.packed_graphs.entries.clear()
        self.metadata_buffers.frames.clear()
        if getattr(self, "draft_runner", None) is not None:
            self.draft_runner.close()
            self.draft_runner = None
        if hasattr(self, "shm"):
            self.shm.close()
            del self.shm
        self.__dict__.pop("graphs", None)
        self.__dict__.pop("graph_pool", None)
        workspace=getattr(self,"cache_transfer_workspace",None)
        if workspace is not None:
            workspace.fence()
            self.cache_transfer_workspace=None
        reset_context()
        if self._owns_process_group and dist.is_initialized():
            self._owns_process_group = False
            dist.destroy_process_group()

    def read_shm(self):
        """从共享内存读取一条 pickle 序列化的 `(method_name, args)`。"""
        assert self.world_size > 1 and self.rank > 0
        # 等待 rank 0 设置 event，表示共享内存中有新消息。
        if not self.event.wait(self.config.tp_timeout_seconds):
            raise TimeoutError("TP worker timed out waiting for IPC message")
        method_name, args = decode_call(self.shm.buf)
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """rank 0 把方法调用写入共享内存并唤醒所有 worker。"""
        assert self.world_size > 1 and self.rank == 0
        data = encode_call(method_name, args, len(self.shm.buf))
        self.shm.buf[:len(data)] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """统一调用入口：必要时先广播给其他 TP rank。"""
        if self._failed:
            raise RuntimeError("TP runtime failed; exit and recreate the engine")
        if self.world_size > 1 and self.rank == 0:
            # 序列化/容量错误发生在 dispatch 前，不污染 worker 状态。
            self.write_shm(method_name, *args)
        try:
            result = getattr(self, method_name)(*args)
            if self.world_size > 1 and self.rank == 0:
                # exit ACK 属于关闭流程，不能仍用普通调用的 120 秒时限。
                timeout = (self.config.tp_shutdown_timeout_seconds
                           if method_name == "exit" else self.config.tp_timeout_seconds)
                wait_worker_status(self.worker_connections, "done", timeout)
            return result
        except BaseException:
            self._failed = True
            failure_callback = getattr(self, "_failure_callback", None)
            if failure_callback is not None:
                try:
                    failure_callback()
                except Exception:
                    pass  # 不覆盖最初的 worker / collective 错误。
            raise

    def warmup_model(self):
        """用一组假 token 预热模型，触发编译并测量显存峰值。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # warmup Sequence 不需要真实 prompt，只要覆盖可能的最大形状。
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """按剩余显存分配分页 K/V cache，并把各层 Attention 指向对应 view。"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # peak/current 用来扣除 warmup 和模型参数曾经占用过的显存。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 每个 block 的字节数 = K/V(2) * 层数 * token 数 * KV heads * head_dim * cache dtype bytes。
        # FP8 cache 使用 1 byte/element；其 per-token scale 单独按 FP16 分配。
        cache_dtype = torch.float8_e4m3fn if config.kv_cache_dtype == "fp8" else hf_config.dtype
        cache_element_size = torch.empty((), dtype=cache_dtype).element_size()
        block_bytes = (
            2 * hf_config.num_hidden_layers * self.block_size
            * num_kv_heads * head_dim * cache_element_size
        )
        if config.kv_cache_dtype == "fp8":
            # K/V 各保存一个 FP16 scale，必须计入显存预算，避免 over-allocation。
            scale_element_size = torch.empty((), dtype=torch.float16).element_size()
            block_bytes += (
                2 * hf_config.num_hidden_layers * self.block_size * scale_element_size
            )
        workspace = max(peak - current, getattr(getattr(self, "draft_runner", None), "workspace_bytes", 0))
        if getattr(self, "native_verification_graphs", None) is not None:
            workspace += self.native_verification_graphs.reserve_bytes
        local_blocks = int(total * config.gpu_memory_utilization - used - workspace) // block_bytes
        if config.max_kvcache_blocks is not None:
            # 必须先限制真实分配，而不是分配大tensor后仅缩小scheduler页表。
            local_blocks = min(local_blocks, config.max_kvcache_blocks)
        # scheduler 在 rank0，但所有 rank 必须支持完全相同的物理 block id。
        # 先参与 MIN 再判断 0：不能让某一 rank 先抛错、其他 rank 卡在 collective。
        config.num_kvcache_blocks = common_kv_capacity(local_blocks, self.world_size)
        # 第一维 0/1 分别存 K 和 V，第二维按 transformer layer 切分。
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=cache_dtype,
        )
        if config.kv_cache_dtype == "fp8":
            # 每个物理 token 一个 K scale 和一个 V scale，便于 kernel 解量化。
            self.kv_cache_scales = torch.empty(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                dtype=torch.float16,
            )
        else:
            self.kv_cache_scales = None
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 每个 Attention 层拿到自己的 [num_blocks, block_size, H_kv, D] view。
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if self.kv_cache_scales is not None:
                    module.k_cache_scale = self.kv_cache_scales[0, layer_id]
                    module.v_cache_scale = self.kv_cache_scales[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把不同长度的 block_table padding 成二维 GPU tensor。"""
        max_len = max(len(seq.block_table) for seq in seqs)
        # -1 表示无效 block，便于 kernel 忽略 padding。
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        columns = (self.config.max_model_len + self.block_size - 1) // self.block_size
        block_tables = self._metadata_tensor("legacy_pages", block_tables, torch.int32,
                                             self.config.max_num_seqs * columns,
                                             shape=(len(seqs), max_len))
        return block_tables

    def _metadata_tensor(self, name, values, dtype, capacity, shape=None):
        """分离路径也共用同一有界元数据池；独立调用prepare_block_tables仍安全。"""
        pool = self.metadata_buffers
        own_round = not pool._active
        if own_round:
            pool.begin()
        try:
            return pool.tensor(name, values, dtype, shape=shape, max_elements=capacity)
        finally:
            if own_round:
                pool.end()

    def prepare_prefill(self, seqs: list[Sequence], return_all_logits: bool = False):
        """把变长 prompt/chunk 打包成 FlashAttention varlen 所需的输入。"""
        self.metadata_buffers.begin()
        try:
            return self._prepare_prefill(seqs, return_all_logits)
        finally:
            self.metadata_buffers.end()

    def _prepare_prefill(self, seqs: list[Sequence], return_all_logits: bool):
        """prefill的数值/Attention路径保持不变，只替换输入元数据的存储管理。"""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        prefill_sequence_ids = []
        prefill_positions = []
        block_tables = None
        for sequence_id, seq in enumerate(seqs):
            # start 是该请求已经在 cache 中的长度，当前只处理后面的 chunk。
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            # prefill 的 key 长度包含已有 prefix 和本轮 query。
            seqlen_k = end
            input_ids.extend(seq[start:end])
            if self.config.kv_cache_dtype == "fp8":
                # dequant kernel 输出每条请求从位置 0 到 end 的完整 K/V。
                prefill_sequence_ids.extend([sequence_id] * seqlen_k)
                prefill_positions.extend(range(seqlen_k))
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                # warmup Sequence 没有真实 block_table，不需要构造 slot 映射。
                continue
            # 将本轮 token 映射到它们在物理 KV Cache 中的 slot。
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            # K 比 Q 长说明有 prefix cache，需要把 block_table 传给 FlashAttention。
            block_tables = self.prepare_block_tables(seqs)
        # pinned CPU tensor + non_blocking 加速搬到 GPU。
        capacity = self.config.max_num_batched_tokens
        sequence_capacity = self.config.max_num_seqs + 1
        input_ids = self._metadata_tensor("legacy_ids", input_ids, torch.int64, capacity)
        positions = self._metadata_tensor("legacy_positions", positions, torch.int64, capacity)
        cu_seqlens_q = self._metadata_tensor("legacy_cu_q", cu_seqlens_q, torch.int32, sequence_capacity)
        cu_seqlens_k = self._metadata_tensor("legacy_cu_k", cu_seqlens_k, torch.int32, sequence_capacity)
        slot_mapping = self._metadata_tensor("legacy_slots", slot_mapping, torch.int32, capacity)
        if self.config.kv_cache_dtype == "fp8":
            # 这些两个小数组描述 flat K/V 输出中的“属于哪条请求、相对位置”。
            key_capacity = self.config.max_num_seqs * self.config.max_model_len
            prefill_sequence_ids = self._metadata_tensor("legacy_fp8_seq", prefill_sequence_ids, torch.int32, key_capacity)
            prefill_positions = self._metadata_tensor("legacy_fp8_positions", prefill_positions, torch.int32, key_capacity)
        else:
            prefill_sequence_ids = prefill_positions = None
        # Attention 从全局 Context 读取变长边界和 cache 映射。
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            prefill_sequence_ids,
            prefill_positions,
            return_all_logits=return_all_logits,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """准备 decode batch：每条请求只取最后一个 token 作为 query。"""
        self.metadata_buffers.begin()
        try:
            return self._prepare_decode(seqs)
        finally:
            self.metadata_buffers.end()

    def _prepare_decode(self, seqs: list[Sequence]):
        """原decode输入布局不变，所有动态字段每轮重新填充。"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # decode 的输入是上轮刚追加的 last_token。
            input_ids.append(seq.last_token)
            # 它在整条序列中的位置是 len(seq)-1。
            positions.append(len(seq) - 1)
            # Attention 需要知道可以读取多长的历史 KV。
            context_lens.append(len(seq))
            # 新 token 的 K/V 写入当前最后一个 block 的下一个 slot。
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        capacity = self.config.max_num_seqs
        input_ids = self._metadata_tensor("legacy_ids", input_ids, torch.int64, capacity)
        positions = self._metadata_tensor("legacy_positions", positions, torch.int64, capacity)
        slot_mapping = self._metadata_tensor("legacy_slots", slot_mapping, torch.int32, capacity)
        context_lens = self._metadata_tensor("legacy_lengths", context_lens, torch.int32, capacity)
        block_tables = self.prepare_block_tables(seqs)
        # decode 不需要 cu_seqlens，而需要 context_lens 和 block_tables。
        fi_metadata = None
        if getattr(self, "flashinfer_decode", None) is not None:
            # 从 CPU Sequence 构建，避免先送 GPU 再 .tolist() 的同步往返。
            fi_metadata = ([list(s.block_table) for s in seqs], [len(s) for s in seqs])
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    flashinfer_metadata=fi_metadata)
        return input_ids, positions

    def _plan_flashinfer_decode(self, graph_batch=None):
        """在真正选定 eager/Graph 后规划一次，各层只执行 run。"""
        manager = getattr(self, "flashinfer_decode", None)
        if manager is not None:
            context = get_context()
            if context.flashinfer_metadata is None:
                raise RuntimeError("missing FlashInfer decode metadata")
            context.flashinfer_wrapper = manager.plan(
                *context.flashinfer_metadata, self.config.num_kvcache_blocks, graph_batch)

    def prepare_sample(self, seqs: list[Sequence]):
        """收集每条请求的 temperature，并搬到 GPU。"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """执行一次 transformer；decode 小 batch 优先走 CUDA Graph。"""
        if not hasattr(self, "graph_counters"):
            self.graph_counters = Counter()
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if not is_prefill:
                self._plan_flashinfer_decode()
            # prefill 形状通常变化，直接 eager forward 更灵活。
            reason = "prefill" if is_prefill else "disabled" if self.enforce_eager else "batch_limit"
            self.graph_counters[f"decode_fallback_{reason}"] += 1
            self.last_model_path = "decode_eager" if not is_prefill else "prefill_eager"
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 分桶正常情况下覆盖全部 batch；缺图或静态页表容量不足时安全
            # 回退 eager，避免 next() / tensor copy 报错中断整个请求循环。
            graph_bs = next(
                (size for size in self.graph_bs if size >= bs and size in self.graphs),
                None,
            )
            graph_vars = self.graph_vars
            if (graph_bs is None
                    or context.block_tables.size(1) > graph_vars["block_tables"].size(1)):
                reason = "missing_bucket" if graph_bs is None else "page_table_capacity"
                self.graph_counters[f"decode_fallback_{reason}"] += 1
                self.last_model_path = "decode_eager"
                self._plan_flashinfer_decode()
                return self.model.compute_logits(self.model(input_ids, positions))
            graph = self.graphs[graph_bs]
            from nanovllm.layers.quant_graph import assert_graph_replay_stream
            assert_graph_replay_stream(self, input_ids.device)
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"].zero_()
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay 在固定 CUDA Graph 中执行 decode forward。
            self._plan_flashinfer_decode(graph_bs)
            graph.replay()
            self.graph_counters["decode_replays"] += 1
            self.graph_counters["decode_real_rows"] += bs
            self.graph_counters["decode_padding_rows"] += graph_bs-bs
            self.last_model_path = "decode_graph"
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """完成“准备输入 -> forward -> 采样 -> 清理 Context”的一次调用。"""
        try:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            # 得到每条 Sequence 最后位置对应的词表 logits。
            logits = self.run_model(input_ids, positions, is_prefill)
            # 只有 rank 0 采样并返回 token；其他 rank 只参与模型计算。
            return self.sampler(logits, seqs).tolist() if self.rank == 0 else None
        finally:
            # 输入准备、forward、采样中任一步失败，也不能留下旧 batch Context。
            reset_context()

    def run_plan(self, plan, sequences, stochastic_proposals=None):
        """显式 packed 执行入口；计划携带输入快照，不依赖 worker 的完整历史。"""
        return execute_plan(self, plan, sequences, stochastic_proposals)

    @torch.inference_mode()
    def run_device_verification(self, plan, sequences):
        from nanovllm.engine.device_draft import DeviceVerificationItem,DeviceVerificationResult
        from time import perf_counter
        pending=None;stream=torch.cuda.current_stream()
        started=perf_counter()
        try:
            if (not getattr(self.config,"draft_device_verification",False) or self.world_size!=1
                    or len(plan.items)!=1 or len(sequences)!=1
                    or not isinstance(plan.items[0],DeviceVerificationItem)):
                raise ValueError("device verification requires one configured TP1 request")
            item=plan.items[0];seq=sequences[0];q=item.query_len
            native=self.native_verification_graphs
            if (seq.seq_id!=item.request_id or seq.last_token!=item.last_token
                    or len(seq)-1!=item.start_pos or tuple(seq.block_table)!=item.block_table
                    or seq.temperature>1e-10 or seq.top_k or seq.top_p<1
                    or seq.repetition_penalty!=1 or getattr(seq,"speculative_sampling","greedy")!="greedy"
                    or native is None or not native.packed_metadata or q not in native.metadata_packets
                    or item.end_pos>self.config.max_model_len
                    or q>self.config.max_num_batched_tokens):
                raise ValueError("unsupported or stale device verification ticket")
            if any(type(p) is not int or not 0<=p<self.config.num_kvcache_blocks for p in item.block_table):
                raise ValueError("invalid device verification physical page")
            if len(item.block_table)*self.block_size<item.end_pos:
                raise ValueError("device verification has insufficient KV pages")
            pending=self.draft_runner.begin_device_proposal(item.request_id,seq.token_ids,item.count)
            # All CPU metadata is known without knowing candidate values. Staging
            # zeros are overwritten by a same-stream GPU copy before model replay.
            positions=list(range(item.start_pos,item.end_pos))
            slots=[item.block_table[p//self.block_size]*self.block_size+p%self.block_size for p in positions]
            if not native.prepare_cpu(q,[item.last_token]+[0]*item.count,positions,slots,item.block_table,item.end_pos):
                raise RuntimeError("device verification packet is unavailable")
            ids=native.entries[q][1]
            ids[1:].copy_(pending.ids)
            prepared_at=perf_counter()
            logits=native.replay_prepared(q)
            enqueued_at=perf_counter()
            predictions=self.sampler(logits,[seq]*q)
            host=torch.cat((pending.ids,predictions)).cpu().tolist()
            if len(host)!=item.count+q or any(type(t) is not int or not 0<=t<self.config.hf_config.vocab_size for t in host):
                raise RuntimeError("invalid device verification result layout")
            result=DeviceVerificationResult(item.request_id,tuple(host[:item.count]),tuple(host[item.count:]))
            self.draft_runner.finish_device_proposal(pending,result.draft)
            self.graph_counters["device_verification_replays"]+=1
            self.graph_counters["device_verification_result_transfers"]+=1
            self.last_plan_stats=dict(path="native_device_verification_graph",captured=False,metadata_grew=False,
                query_tokens=q,forward_calls=1,host_prepare_ms=(prepared_at-started)*1000,
                model_enqueue_ms=(enqueued_at-prepared_at)*1000,sample_and_wait_ms=(perf_counter()-enqueued_at)*1000,
                timing_scope="host_prepare includes async draft enqueue; wait includes draft and target completion")
            return result
        except BaseException:
            # Target may already be writing its reserved pages. Fence before the
            # Engine rolls back and returns them for another request.
            try:
                stream.synchronize()
            finally:
                if self.draft_runner is not None:self.draft_runner.abort_device_proposal()
            raise
        finally:
            reset_context()

    def draft_propose(self, request_id, tokens, limit):
        if self.draft_runner is None:
            raise ValueError("draft_model was not configured")
        return self.draft_runner.propose(request_id, tokens, limit)

    def draft_synchronize(self, request_id, tokens):
        if self.draft_runner is not None:
            self.draft_runner.synchronize(request_id, tokens)

    def draft_propose_probabilities(self, request_id, tokens, limit, temperature, seed):
        if self.draft_runner is None:
            raise ValueError("draft_model was not configured")
        return self.draft_runner.propose_probabilities(request_id, tokens, limit, temperature, seed)

    def draft_forget(self, request_id):
        if self.draft_runner is not None:
            self.draft_runner.forget(request_id)

    @torch.inference_mode()
    def run_speculative(self, seqs: list[Sequence]):
        """对一条临时扩展的 Sequence 返回每个 query 位置的 logits。"""
        try:
            input_ids, positions = self.prepare_prefill(seqs, return_all_logits=True)
            context = get_context()
            if getattr(self, "native_verification_graphs", None) is not None:
                logits = self.native_verification_graphs.run(input_ids, positions, context)
                if logits is not None:
                    return logits if self.rank == 0 else None
            hidden = None
            if (self.enable_verification_graph and not self.enforce_eager
                    and self.world_size == 1 and self.config.kv_cache_dtype == "model"
                    and context.block_tables is not None):
                hidden = self.verification_graphs.run(self.model, input_ids, positions, context)
            if hidden is None:
                logits = self.run_model(input_ids, positions, True)
            else:
                logits = self.model.compute_logits(hidden)
            return logits if self.rank == 0 else None
        finally:
            # 异常也清理上下文；CUDA执行异常不应被静默吞掉后继续使用cache。
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        """为一组典型 decode batch size 录制 CUDA Graph。"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 使用少量常见 batch size，实际 batch 会选择向上取整的 graph。
        self.graph_bs = list(getattr(config, "decode_graph_buckets", None) or decode_graph_batch_sizes(max_bs))
        self.graphs = {}
        self.graph_pool = None

        fi = getattr(self, "flashinfer_decode", None)
        if fi is not None:
            # capture 时没有真实请求。禁止 padding 写入 KV，同时给 dummy 读
            # 提供有限值，防止未初始化 cache 里的 NaN 污染捕获预热。
            slot_mapping.fill_(-1)
            context_lens.fill_(1)
            for module in self.model.modules():
                if hasattr(module, "k_cache") and module.k_cache.numel():
                    module.k_cache[0].zero_()
                    module.v_cache[0].zero_()

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            wrapper = None if fi is None else fi.plan([[0]], [1], config.num_kvcache_blocks, bs)
            # graph capture 前先把 Context 指向当前 bs 的静态 view。
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                        flashinfer_wrapper=wrapper)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            from nanovllm.layers.quant_graph import capture_model_graph
            with capture_model_graph(self.model, graph, pool=self.graph_pool):
                # capture 后，graph.replay() 会重复执行这一段模型 forward。
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
