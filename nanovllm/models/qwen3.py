"""Qwen3 的最小推理实现。

模型结构保持 Hugging Face Qwen3 的权重命名和张量形状，但把执行路径
连接到 nano-vLLM 自己的 Tensor Parallel、Paged KV Cache 和 FlashAttention。
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.tensorcore_mlp import should_use_fused_swiglu, fused_swiglu
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 的 QKV 投影、RoPE、KV Cache Attention 和输出投影。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        decode_attention_backend: str = "flash",
        decode_attention_block_n: int = 64,
        decode_attention_num_warps: int | None = None,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rms_norm_backend: str = "torch",
        linear_backend: str = "torch",
        quantization: str | None = None,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        # 每个 rank 只计算全部 attention head 的一个分片。
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # q_size 是当前 rank 的 Q 宽度，kv_size 是当前 rank 的 K/V 宽度。
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # attention score 乘 1/sqrt(head_dim)，避免点积随维度变大。
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        if quantization == "awq":
            from nanovllm.layers.awq_packed import AWQPackedLinear
            self.qkv_proj = AWQPackedLinear(hidden_size, (self.q_size, self.kv_size, self.kv_size),
                                           shard_ids=("q", "k", "v"), tp_size=tp_size, bias=qkv_bias)
            self.o_proj = AWQPackedLinear(self.total_num_heads * self.head_dim,
                                         hidden_size, tp_size=tp_size)
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads,
                bias=qkv_bias, backend=linear_backend,
            )
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim, hidden_size,
                bias=False, backend=linear_backend,
            )
        if isinstance(rope_scaling, dict):
            # 某些 checkpoint 会把 rope_theta 放在 rope_scaling 配置内。
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            decode_backend=decode_attention_backend,
            decode_block_n=decode_attention_block_n,
            decode_num_warps=decode_attention_num_warps,
        )
        if not self.qkv_bias:
            # Qwen3 的无 bias 变体在 RoPE 前额外对 Q/K 做 RMSNorm。
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, backend=rms_norm_backend)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, backend=rms_norm_backend)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 一次矩阵乘同时得到拼接后的 Q、K、V。
        qkv = self.qkv_proj(hidden_states)
        # 按最后一维拆开，再整理为 [token_count, num_heads, head_dim]。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 对 Q/K 加入位置旋转；V 不做 RoPE。
        q, k = self.rotary_emb(positions, q, k)
        # Attention 根据 Context 自动选择 prefill 或 decode 路径。
        o = self.attn(q, k, v)
        # 合并 head 后通过输出投影回到 hidden_size。
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 的 SwiGLU 前馈网络。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mlp_backend: str = "torch",
        mlp_min_tokens: int = 128,
        linear_backend: str = "torch",
        quantization: str | None = None,
    ) -> None:
        super().__init__()
        # gate_up_proj 一次生成 gate 和 up 两支，输出宽度是 2*intermediate_size。
        self.quantization = quantization
        if quantization == "awq":
            from nanovllm.layers.awq_packed import AWQPackedLinear
            self.gate_up_proj = AWQPackedLinear(hidden_size, (intermediate_size, intermediate_size),
                                                shard_ids=(0, 1), tp_size=dist.get_world_size())
            self.down_proj = AWQPackedLinear(intermediate_size, hidden_size, tp_size=dist.get_world_size())
        else:
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size, [intermediate_size] * 2, bias=False, backend=linear_backend,
            )
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size, bias=False, backend=linear_backend,
            )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()
        # 后端在构造时固定；捕获 CUDA Graph 后不要原地修改这些属性。
        self.mlp_backend = mlp_backend
        self.mlp_min_tokens = mlp_min_tokens
        self.mlp_device_capability = (
            torch.cuda.get_device_capability(self.gate_up_proj.weight.device)
            if mlp_backend == "auto" and self.gate_up_proj.weight.is_cuda else None
        )

    def forward(self, x):
        if self.quantization == "awq":
            # packed 权重没有 dense .weight；只在 kernel 的小 tile 内解包。
            return self.down_proj(self.act_fn(self.gate_up_proj(x)))
        # 仅替换 gate/up + SwiGLU，不绕开 down_proj 的 TP AllReduce。
        # M 指实际 tensor 的 token 行数；Graph padding 后可能大于活跃请求数。
        # triton使用人工门槛；auto使用离线形状白名单；默认仍为Torch。
        if should_use_fused_swiglu(
            x, self.gate_up_proj.weight, self.mlp_backend,
            self.mlp_min_tokens, self.mlp_device_capability,
        ):
            x = fused_swiglu(x, self.gate_up_proj.weight)
        else:
            gate_up = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """一个 pre-norm decoder block：Attention + MLP + residual。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 自回归 decoder 的 self-attention，带 causal mask。
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rms_norm_backend=getattr(config, "nanovllm_rmsnorm_backend", "torch"),
            linear_backend=getattr(config, "nanovllm_linear_backend", "torch"),
            quantization=getattr(config, "nanovllm_quantization", None),
            decode_attention_backend=getattr(config, "nanovllm_decode_attention_backend", "flash"),
            decode_attention_block_n=getattr(config, "nanovllm_decode_attention_block_n", 64),
            decode_attention_num_warps=getattr(config, "nanovllm_decode_attention_num_warps", None),
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            mlp_backend=getattr(config, "nanovllm_mlp_backend", "torch"),
            mlp_min_tokens=getattr(config, "nanovllm_mlp_min_tokens", 128),
            linear_backend=getattr(config, "nanovllm_linear_backend", "torch"),
            quantization=getattr(config, "nanovllm_quantization", None),
        )
        # 两个子层各有一个 RMSNorm。
        rms_norm_backend = getattr(config, "nanovllm_rmsnorm_backend", "torch")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, backend=rms_norm_backend)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, backend=rms_norm_backend)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层没有传入旧 residual：保存 hidden_states，并归一化主分支。
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层融合 residual 相加与 RMSNorm，减少一次读写。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # self-attention 子层输出仍保持 hidden_size。
        hidden_states = self.self_attn(positions, hidden_states)
        # 第二个 residual + RMSNorm 后进入 MLP。
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """不包含 LM head 的 Qwen3 transformer 主体。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词表并行 embedding：每个 rank 保存部分词表。
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 堆叠 num_hidden_layers 个 decoder block。
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=getattr(config, "nanovllm_rmsnorm_backend", "torch"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # token id -> hidden states。
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            # 每层同时返回主分支和待融合 residual。
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一次把 residual 融入并做 RMSNorm。
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 transformer 主体加上词表输出头。"""
    packed_modules_mapping = {
        # safetensors 中的独立权重如何映射到合并后的 Parameter。
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # tied embeddings：输入 embedding 与输出 head 共享同一份权重存储。
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.numerical_backend = "default"
        backend = getattr(config, "nanovllm_numerical_backend", "default")
        if backend != "default":
            self.set_numerical_backend(backend)

    def set_numerical_backend(self, backend: str):
        """Explicit TP1 BF16 shape-consistent experiment, selected before capture.

        This is a different arithmetic backend from the old Torch reference:
        fixed-tile GEMM (including LM head), Triton RMSNorm, and identical
        single-query FlashAttention reductions for decode and verification.
        It does not promise equivalence to old Torch logits or support TP/batches.
        """
        if backend != "consistent":
            raise ValueError(f"unsupported explicit numerical backend: {backend}")
        if dist.get_world_size() != 1 or self.lm_head.weight.dtype != torch.bfloat16:
            raise ValueError("consistent numerical backend requires TP1 BF16")
        from nanovllm.layers.linear import LinearBase
        for module in self.modules():
            if isinstance(module, LinearBase):
                if module.bias is not None:
                    raise ValueError("consistent numerical backend requires bias-free Qwen3")
                module.backend = "triton_stable"
            elif isinstance(module, RMSNorm):
                module.backend = "triton"
            elif isinstance(module, Attention):
                module.numerical_backend = "consistent"
            elif isinstance(module, Qwen3MLP):
                module.mlp_backend = "torch"
        self.lm_head.backend = "triton_stable"
        self.numerical_backend = backend

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 只返回 transformer hidden states，LM head 由 compute_logits 单独调用。
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden -> vocab logits；prefill 时 ParallelLMHead 只取每条序列末位置。
        return self.lm_head(hidden_states)
