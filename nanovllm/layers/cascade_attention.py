"""两层Cascade：共同完整页作为一组，各请求独有后缀作为另一组。

普通prefix cache只避免重复prefill；Cascade进一步让一批不同Q共同读取
同一份历史K/V。两个局部softmax不能直接相加，FlashInfer用各自LSE合并。
当前仅eager/TP1/BF16；不把不支持的动态树结构假装成可重放Graph。
"""
from collections import Counter
import importlib.metadata
import torch
from nanovllm.layers.flashinfer_attention import build_paged_metadata


def shared_complete_pages(tables, lengths, page_size):
    """只认相同物理页，且每行至少留一个token在独有后缀。

    相同文本但不同物理页不算共享；可写尾页不能作为不可变共享页。
    """
    if len(tables)<2 or len(tables)!=len(lengths):
        return 0
    limit=min((n-1)//page_size for n in lengths)
    count=0
    for i in range(limit):
        if any(len(t)<=i or t[i]!=tables[0][i] for t in tables):
            break
        count+=1
    return count


class CascadeDecode:
    def __init__(self,num_heads,num_kv_heads,head_dim,page_size,max_model_len,dtype,
                 min_shared_tokens=256,device="cuda"):
        if importlib.metadata.version("flashinfer-python")!="0.5.3":
            raise RuntimeError("Cascade validated dependency is flashinfer-python==0.5.3")
        if dtype!=torch.bfloat16 or head_dim!=128 or page_size!=256:
            raise ValueError("Cascade currently requires BF16/head_dim128/page256")
        from flashinfer.cascade import MultiLevelCascadeAttentionWrapper
        self.shape=(num_heads,num_kv_heads,head_dim,page_size)
        self.dtype=dtype
        self.min_shared_tokens=min_shared_tokens
        self.max_model_len=max_model_len
        self.workspace=torch.zeros(128*2**20,dtype=torch.uint8,device=device)
        self.wrapper=MultiLevelCascadeAttentionWrapper(2,self.workspace,kv_layout="NHD")
        self.stats=Counter()

    def plan(self,tables,lengths,num_pages,graph_batch=None):
        if graph_batch is not None:
            raise ValueError("Cascade graph capture is not supported")
        if max(lengths,default=0)>self.max_model_len:
            raise ValueError("Cascade context exceeds capacity")
        page=self.shape[-1]
        build_paged_metadata(tables,lengths,page,num_pages)  # 先验证全部物理页范围。
        common=shared_complete_pages(tables,lengths,page)
        self.stats["plans"]+=1
        if common*page<self.min_shared_tokens:
            self.stats["fallback_no_shared_prefix"]+=1
            return None  # Attention明确回退原FlashAttention，不伪造Cascade命中。
        batch=len(lengths)
        suffix=build_paged_metadata([t[common:] for t in tables],
            [n-common*page for n in lengths],page,num_pages)
        def cpu(values):
            return torch.tensor(values,dtype=torch.int32,device="cpu")
        # level0: 一组B个query共享prefix；level1: B组各1个query读取独有后缀。
        # decode时所有历史token都可见，因此两个分段都不需要三角causal mask。
        self.wrapper.plan([cpu([0,batch]),cpu(list(range(batch+1)))],
            [cpu([0,common]),cpu(suffix[0])],
            [cpu(tables[0][:common]),cpu(suffix[1])],
            [cpu([page]),cpu(suffix[2])],*self.shape,causal=False,
            sm_scale=self.shape[2]**-.5,q_data_type=self.dtype,kv_data_type=self.dtype)
        self.stats["cascade_plans"]+=1
        self.stats["shared_prefix_tokens"]+=common*page
        self.stats["shared_rows"]+=batch
        return self.wrapper
