"""Stable BF16 linear with column-major physical weight storage.

Logical parameter shape stays [N,K]; startup changes stride to [1,N].
BM/BN/BK, dot order, FP32 accumulation and BF16 output remain 16/128/64.
Only dense target LinearBase modules are packed. Embedding/LM head remain
unchanged, preserving tied storage and keeping the startup transient bounded.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def packed_kernel(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    inner=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        ks=start*BK+inner
        x=tl.load(X+rows[:,None]*K+ks[None,:],(rows[:,None]<M)&(ks[None,:]<K),other=0.)
        w=tl.load(W+ks[:,None]*N+cols[None,:],(cols[None,:]<N)&(ks[:,None]<K),other=0.)
        acc=tl.dot(x,w,acc)
    tl.store(Y+rows[:,None]*N+cols[None,:],acc.to(tl.bfloat16),(rows[:,None]<M)&(cols[None,:]<N))

def packed_stable_bf16_linear(x, weight):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("packed stable linear requires matching 2D shapes")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("packed stable linear requires BF16")
    if not x.is_cuda or x.device != weight.device:
        raise ValueError("packed stable linear requires one CUDA device")
    n,k=weight.shape;m=x.shape[0]
    if n<1 or k<1 or weight.stride() != (1,n):
        raise ValueError("packed weight must have positive shape and stride (1,N)")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise ValueError("packed stable linear is inference-only")
    x=x.contiguous();out=torch.empty((m,n),dtype=x.dtype,device=x.device)
    if m:
        with torch.cuda.device(x.device):
            packed_kernel[(triton.cdiv(m,16),triton.cdiv(n,128))](
                x,weight,out,m,n,k,16,128,64,num_warps=4,num_stages=3)
    return out


@torch.inference_mode()
def pack_stable_linears(model):
    """Call once after loading, before warmup/KV budgeting/Graph capture.

    Preflight all modules before modifying parameters; an exception aborts engine
    initialization. No hidden hot-path packing, duplicate full model, or dtype
    change. State dict values keep their original logical matrix semantics.
    """
    from nanovllm.layers.linear import LinearBase
    modules=[m for m in model.modules() if isinstance(m,LinearBase)]
    if not modules:raise ValueError("no stable dense linear modules")
    for m in modules:
        if (m.backend != "triton_stable" or m.bias is not None or m.tp_size != 1
                or m.weight.dtype != torch.bfloat16 or not m.weight.is_cuda
                or getattr(m,"stable_packed_layout",False)):
            raise ValueError("packing requires fresh dense TP1 BF16 stable bias-free linears")
    total=0;largest=0
    for m in modules:
        size=m.weight.numel()*m.weight.element_size()
        m.weight.data=m.weight.detach().T.contiguous().T
        assert m.weight.stride()==(1,m.weight.shape[0])
        m.stable_packed_layout=True;total+=size;largest=max(largest,size)
    return dict(enabled=True,modules=len(modules),weight_bytes=total,
                largest_repacked_weight_bytes=largest,logical_layout="N,K",
                physical_stride="1,N",lm_head_packed=False)
