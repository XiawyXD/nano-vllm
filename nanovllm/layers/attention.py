"""Attention 层和把 K/V 写入分页 KV Cache 的 Triton kernel。"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.attention_triton import (
    dequantize_paged_kvcache,
    paged_decode_attention,
)
from nanovllm.utils.context import get_context


FP8_E4M3_MAX = 448.0


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """一个 Triton program 把一个 token 的 K/V 写到对应物理 slot。"""
    # program_id(0) 对应输入中的第几个 token，而不是 batch 中第几个请求。
    idx = tl.program_id(0)
    # slot_mapping 把逻辑 token 映射到连续 KV Cache 的物理行。
    slot = tl.load(slot_mapping_ptr + idx)
    # -1 用于无效位置（例如 warmup/padding），不能写入 cache。
    if slot == -1:
        return
    # key/value 输入的布局是 [N, num_heads, head_dim] 展平后的行。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # cache 的一行保存该 token 的全部 head 和 head_dim。
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def store_kvcache_fp8_kernel(
    key_ptr, key_stride, value_ptr, value_stride, k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr, slot_mapping_ptr, D: tl.constexpr,
):
    """写入带 per-token scale 的 FP8 K/V cache。"""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, D)
    key = tl.load(key_ptr + idx * key_stride + offsets).to(tl.float32)
    value = tl.load(value_ptr + idx * value_stride + offsets).to(tl.float32)
    key_max = tl.maximum(tl.max(tl.abs(key), axis=0), 1e-8)
    value_max = tl.maximum(tl.max(tl.abs(value), axis=0), 1e-8)
    key_scale = key_max / 448.0
    value_scale = value_max / 448.0
    tl.store(k_scale_ptr + slot, key_scale)
    tl.store(v_scale_ptr + slot, value_scale)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key / key_scale)
    tl.store(v_cache_ptr + cache_offsets, value / value_scale)


def store_kvcache(
    key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor,
    v_cache: torch.Tensor, slot_mapping: torch.Tensor,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
):
    """检查张量布局并启动 K/V cache 写入 kernel。"""
    N, num_heads, head_dim = key.shape
    # 每个 token 的所有 head 总宽度；cache 的最后两维会被压平。
    D = num_heads * head_dim
    # kernel 使用连续的最后一维和已知 stride，先验证布局约束。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # N 个 program 分别处理 N 个 token。FP8 cache 需要额外保存 scale。
    if k_cache.dtype == torch.float8_e4m3fn:
        if k_scale_cache is None or v_scale_cache is None:
            raise ValueError("FP8 KV cache requires K/V scale tensors")
        assert k_scale_cache.ndim == 2 and v_scale_cache.ndim == 2
        assert k_scale_cache.shape == v_scale_cache.shape
        assert k_scale_cache.stride(1) == 1 and v_scale_cache.stride(1) == 1
        store_kvcache_fp8_kernel[(N,)](
            key, key.stride(0), value, value.stride(0), k_cache, v_cache,
            k_scale_cache, v_scale_cache, slot_mapping, D,
        )
    else:
        store_kvcache_kernel[(N,)](
            key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
        )


class Attention(nn.Module):
    """根据当前 Context 选择 prefill 或 decode 的 FlashAttention 路径。"""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        decode_backend="flash",
        decode_block_n=64,
        decode_num_warps=None,
    ):
        super().__init__()
        if decode_backend not in {"flash", "triton", "flashinfer", "cascade"}:
            raise ValueError(f"unknown decode attention backend: {decode_backend}")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.decode_backend = decode_backend
        self.decode_block_n = decode_block_n
        self.decode_num_warps = decode_num_warps
        self.numerical_backend = "default"
        # 初始化为空 tensor；ModelRunner.allocate_kv_cache 会替换成真正的 cache view。
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_cache_scale = self.v_cache_scale = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """执行 causal attention，并把当前 K/V 写入 KV Cache。"""
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            # 当前输入 token 的 K/V 先写入它们对应的物理 slot，供后续 decode 使用。
            store_kvcache(
                k, v, k_cache, v_cache, context.slot_mapping,
                self.k_cache_scale if self.k_cache_scale.numel() else None,
                self.v_cache_scale if self.v_cache_scale.numel() else None,
            )
        if self.numerical_backend == "consistent" and k_cache.numel():
            return self._consistent_attention(q, context)
        if context.attention_groups is not None:
            if self.decode_backend in {"flashinfer", "cascade"}:
                raise RuntimeError("FlashInfer decode does not yet support packed attention")
            # 所有 token 的 KV 已在上面写入一次。不同组只有 Attention 分路，
            # QKV、O projection 和 MLP 仍属于同一次 transformer forward。
            if k_cache.dtype == torch.float8_e4m3fn:
                raise RuntimeError("packed attention currently requires model-dtype KV")
            outputs = []
            for group in context.attention_groups:
                query = q[group.start:group.end]
                if group.kind in {"varlen", "varlen_current"}:
                    # 从逻辑位置 0 开始的完整 Prefill 没有历史 KV。
                    # 直接用本次 QKV 投影产生的连续 K/V 做 Attention，
                    # 与旧版普通 Prefill 的数值路径一致；上面的 store_kvcache
                    # 仍会把 K/V 写入分页缓存，供之后 Decode 使用。
                    if group.kind == "varlen_current":
                        group_k = k[group.start:group.end]
                        group_v = v[group.start:group.end]
                        block_table = None
                    else:
                        group_k, group_v = k_cache, v_cache
                        block_table = group.block_tables
                    output = flash_attn_varlen_func(
                        query, group_k, group_v,
                        cu_seqlens_q=group.cu_seqlens_q,
                        cu_seqlens_k=group.cu_seqlens_k,
                        max_seqlen_q=group.max_seqlen_q,
                        max_seqlen_k=group.max_seqlen_k,
                        softmax_scale=self.scale, causal=True,
                        block_table=block_table,
                    )
                elif group.kind == "decode":
                    if self.decode_backend == "triton":
                        output = paged_decode_attention(
                            query, k_cache, v_cache, group.context_lens,
                            group.block_tables, self.scale,
                            block_n=self.decode_block_n,
                            num_warps=self.decode_num_warps,
                        )
                    else:
                        output = flash_attn_with_kvcache(
                            query.unsqueeze(1), k_cache, v_cache,
                            cache_seqlens=group.context_lens,
                            block_table=group.block_tables,
                            softmax_scale=self.scale, causal=True,
                        ).squeeze(1)
                else:
                    raise ValueError(f"unknown packed attention kind: {group.kind}")
                outputs.append(output)
            return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        if context.is_prefill:
            block_table = context.block_tables
            if context.block_tables is not None:    # prefix cache 或 chunked prefill
                if k_cache.dtype == torch.float8_e4m3fn:
                    if context.prefill_sequence_ids is None or context.prefill_positions is None:
                        raise RuntimeError("FP8 prefill requires sequence/position maps")
                    # FlashAttention 不能直接读取 FP8 cache；先把完整 prefix
                    # 解量化成连续的模型 dtype，再走同一个 varlen reference。
                    k, v = dequantize_paged_kvcache(
                        k_cache,
                        v_cache,
                        self.k_cache_scale,
                        self.v_cache_scale,
                        context.block_tables,
                        context.prefill_sequence_ids,
                        context.prefill_positions,
                        k.dtype,
                    )
                    block_table = None
                else:
                    # 模型 dtype cache 可直接交给 FlashAttention 的 block_table。
                    k, v = k_cache, v_cache
            # varlen API 用 cu_seqlens 描述 batch 中不同长度的序列边界。
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=block_table,
            )
        else:    # decode
            # decode 每条请求只有一个 query；历史 K/V 从分页 cache 中读取。
            if self.decode_backend == "flashinfer" or (self.decode_backend=="cascade" and context.flashinfer_wrapper is not None):
                if context.flashinfer_wrapper is None:
                    raise RuntimeError("FlashInfer decode requires a fresh step-level plan")
                # Cache 已经写入当前 token；无需重复 RoPE，也无需复制整个 KV。
                o = context.flashinfer_wrapper.run(q, (k_cache, v_cache))
                if self.decode_backend=="cascade" and getattr(self,"cascade_verify_reference",False):
                    # 严格参考回退：候选逐元素完全相等时才采用，否则使用FA输出。
                    # 不使用较宽的allclose，也不把两次计算伪装成加速。GPU上比较，
                    # 不引入每层.item()的CPU同步；这只是显式可选的验收模式。
                    reference=flash_attn_with_kvcache(q.unsqueeze(1),k_cache,v_cache,
                        cache_seqlens=context.context_lens,block_table=context.block_tables,
                        softmax_scale=self.scale,causal=True).squeeze(1)
                    same=(o==reference).all(dim=-1,keepdim=True).all(dim=-2,keepdim=True)
                    o=torch.where(same,o,reference)
            elif self.decode_backend == "triton":
                # 实验 Triton 路径直接消费 [B, H, D]，不需要 FlashAttention 的
                # singleton sequence 维度。
                o = paged_decode_attention(
                    q, k_cache, v_cache, context.context_lens, context.block_tables, self.scale,
                    self.k_cache_scale if self.k_cache_scale.numel() else None,
                    self.v_cache_scale if self.v_cache_scale.numel() else None,
                    block_n=self.decode_block_n,
                    num_warps=self.decode_num_warps,
                )
            else:
                if k_cache.dtype == torch.float8_e4m3fn:
                    raise RuntimeError("FP8 KV cache requires decode_attention_backend='triton'")
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        # 返回形状通常为 [token_count, num_heads, head_dim]。
        return o

    def _consistent_attention(self, q, context):
        """TP1/single-request, one identical FA query per independent batch row.

        Lengths are computed on-device so fixed-capacity metadata can be replayed
        in a CUDA Graph. num_splits=1 prevents batch/length-dependent split-K
        reductions. Every query sees only its causal prefix, including when all
        verification K/V rows have already been stored. No host tensor reads.
        """
        if self.k_cache.dtype != torch.bfloat16:
            raise ValueError("consistent attention requires BF16 KV")

        def run(query, tables, end, lengths=None):
            if tables is None or tables.shape[0] != 1:
                raise ValueError("consistent attention requires one request/page table")
            count = query.shape[0]
            if lengths is None:
                lengths = end - count + torch.arange(1, count + 1, device=query.device, dtype=torch.int32)
            return flash_attn_with_kvcache(
                query.unsqueeze(1), self.k_cache, self.v_cache,
                cache_seqlens=lengths, block_table=tables.expand(count, -1),
                softmax_scale=self.scale, causal=True, num_splits=1,
            ).squeeze(1)

        if context.attention_groups is not None:
            if len(context.attention_groups) != 1:
                raise ValueError("consistent attention is scoped to a single request")
            group = context.attention_groups[0]
            query = q[group.start:group.end]
            if group.kind == "decode":
                return run(query, group.block_tables, None, group.context_lens)
            if group.kind in {"varlen", "varlen_current"}:
                if group.cu_seqlens_k.numel() != 2:
                    raise ValueError("consistent attention requires one varlen request")
                return run(query, group.block_tables, group.cu_seqlens_k[1])
            raise ValueError(f"unsupported consistent attention group: {group.kind}")
        if context.is_prefill:
            if context.cu_seqlens_k.numel() != 2:
                raise ValueError("consistent attention requires one varlen request")
            tables = context.block_tables
            if tables is None:
                # Legacy full prefill omits tables: its slots start at logical
                # position zero and provide the physical block at each boundary.
                block_size = self.k_cache.shape[1]
                tables = (context.slot_mapping[::block_size] // block_size).to(torch.int32).unsqueeze(0)
            return run(q, tables, context.cu_seqlens_k[1])
        return run(q, context.block_tables, None, context.context_lens)
