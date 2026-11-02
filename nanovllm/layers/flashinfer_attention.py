"""FlashInfer 0.5.3 分页 decode 适配；prefill 仍由 FlashAttention 处理。

这里不是重写 Attention kernel，而是把调度器的页表安全地交给已有 kernel。
plan 描述“本轮读哪些页”，run 在每层读取该层的 K/V；所有层复用一次 plan。
"""
from collections import Counter
import importlib.metadata

import torch


def build_paged_metadata(block_tables, lengths, page_size, num_pages, batch_size=None):
    """CPU 页表 → CSR 三元组；不能把逻辑页编号误当成物理页编号。

    例如 tables=[[7,2],[5]], lengths=[257,10], page_size=256：
    indptr=[0,2,3]，indices=[7,2,5]，last_page_len=[1,10]。
    Graph 补齐行读物理页 0 的一个 token，但不写 cache、也不返回其输出。
    """
    if page_size <= 0 or num_pages <= 0 or len(block_tables) != len(lengths):
        raise ValueError("invalid page size/capacity or mismatched metadata rows")
    batch_size = len(lengths) if batch_size is None else batch_size
    if not lengths or batch_size < len(lengths):
        raise ValueError("empty decode or graph bucket smaller than actual batch")
    indptr, indices, last = [0], [], []
    for table, length in zip(block_tables, lengths):
        if length <= 0:
            raise ValueError("decode context length must be positive")
        pages = (length + page_size - 1) // page_size
        if len(table) < pages:
            raise ValueError("page table cannot cover the context")
        active = table[:pages]  # 不把预留但尚未写入的尾页交给 attention。
        if any(type(p) is not int or not 0 <= p < num_pages for p in active):
            raise ValueError("physical page outside allocated KV cache")
        indices.extend(active)
        indptr.append(len(indices))
        last.append((length - 1) % page_size + 1)
    for _ in range(batch_size - len(lengths)):
        indices.append(0)
        indptr.append(len(indices))
        last.append(1)
    return indptr, indices, last


class FlashInferDecode:
    """每个 ModelRunner 一份；仅支持单 CUDA stream 串行执行，不可并发复用。

    workspace 和各 Graph 桶的元数据在 KV 显存预算计算前分配。
    Graph 内只包含 run，plan 必须在 capture/replay 之前执行。
    """
    def __init__(self, num_heads, num_kv_heads, head_dim, page_size, max_model_len,
                 dtype, graph_batches=(), device="cuda"):
        try:
            version = importlib.metadata.version("flashinfer-python")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("FlashInfer decode requires scripts/with_flashinfer.sh (isolated optional environment)") from error
        if version != "0.5.3":
            raise RuntimeError(f"validated FlashInfer version is 0.5.3, found {version}")
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        if dtype != torch.bfloat16:
            raise ValueError("FlashInfer decode currently validated only for BF16")
        if head_dim != 128 or page_size != 256:
            raise ValueError("FlashInfer decode currently validated only for head_dim=128/page_size=256")
        self.device = torch.device(device)
        self.shape = (num_heads, num_kv_heads, head_dim, page_size)
        self.dtype = dtype
        self.max_pages = (max_model_len + page_size - 1) // page_size
        # 所有桶顺序重放，可以共享临时 workspace；不能跨 stream 并发写入。
        self.workspace = torch.zeros(128 * 2**20, dtype=torch.uint8, device=self.device)
        self.wrappers = {}
        self.stats = Counter()
        for batch in (None, *graph_batches):
            buffers = {}
            if batch is not None:
                buffers = dict(
                    paged_kv_indptr_buffer=torch.zeros(batch + 1, dtype=torch.int32, device=self.device),
                    paged_kv_indices_buffer=torch.zeros(batch * self.max_pages, dtype=torch.int32, device=self.device),
                    paged_kv_last_page_len_buffer=torch.ones(batch, dtype=torch.int32, device=self.device),
                )
            # 0.5.3 的 FA2 Tensor Core + no-split Graph 在实测中非法访问；
            # 对应上游 issue #4002。保留原库，改用专用 CUDA Core decode
            # planner（Graph 模式固定 launch 上界，metadata 标记有效任务）。
            # 这是内核实现选择，不意味着整个模型不使用 Tensor Core。
            self.wrappers[batch] = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace, kv_layout="NHD", use_cuda_graph=batch is not None,
                use_tensor_cores=False, backend="auto", **buffers)

    def plan(self, tables, lengths, num_pages, graph_batch=None):
        if max(lengths, default=0) > self.max_pages * self.shape[-1]:
            raise ValueError("context exceeds reserved metadata capacity")
        values = build_paged_metadata(tables, lengths, self.shape[-1], num_pages, graph_batch)
        # 显式 CPU，不能受 ModelRunner 初始化期间 default_device='cuda' 影响。
        tensors = [torch.tensor(v, dtype=torch.int32, device="cpu") for v in values]
        wrapper = self.wrappers[graph_batch]
        # 同步拷贝保护临时 CPU tensor 生命周期。pinned/event 原型未显示
        # 端到端收益，归档在 artifacts/flashinfer_attention_20260921 中。
        wrapper.plan(*tensors, *self.shape, q_data_type=self.dtype, kv_data_type=self.dtype,
                     sm_scale=self.shape[2] ** -0.5, non_blocking=False)
        self.stats["plans"] += 1
        self.stats["graph_plans" if graph_batch is not None else "eager_plans"] += 1
        self.stats["real_rows"] += len(lengths)
        self.stats["padded_rows"] += len(values[2]) - len(lengths)
        return wrapper
