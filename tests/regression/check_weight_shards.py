"""只读真实8B checkpoint，在CPU验证Q/K/V、gate/up、row权重分片重建。

这证明loader切片索引正确，不等价于整个TP模型数值已通过。无需GPU/NCCL。
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors import safe_open
from nanovllm.layers.linear import QKVParallelLinear,MergedColumnParallelLinear,RowParallelLinear


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",type=Path,default=Path("/home/hrxu/huggingface/Qwen3-8B"))
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--all-layers",action="store_true")
    args=ap.parse_args();config=json.loads((args.model/"config.json").read_text())
    index=json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    def read(key):
        with safe_open(args.model/index[key],framework="pt",device="cpu") as f:return f.get_tensor(key)
    rows=[]
    for layer in (range(config["num_hidden_layers"]) if args.all_layers else (0,config["num_hidden_layers"]-1)):
        qkv=[read(f"model.layers.{layer}.self_attn.{k}_proj.weight") for k in ("q","k","v")]
        gu=[read(f"model.layers.{layer}.mlp.{k}_proj.weight") for k in ("gate","up")]
        down=read(f"model.layers.{layer}.mlp.down_proj.weight")
        collected={k:[] for k in ("q","k","v","gate","up","down")}
        for rank in (0,1):
            with patch("torch.distributed.get_world_size",return_value=2),patch("torch.distributed.get_rank",return_value=rank):
                q=QKVParallelLinear(config["hidden_size"],config["head_dim"],config["num_attention_heads"],config["num_key_value_heads"],bias=False).to(torch.bfloat16)
                for key,weight in zip(("q","k","v"),qkv):q.weight_loader(q.weight,weight,key)
                sizes=[w.shape[0]//2 for w in qkv]
                for key,w in zip(("q","k","v"),q.weight.data.split(sizes,dim=0)):collected[key].append(w.clone())
                m=MergedColumnParallelLinear(config["hidden_size"],[config["intermediate_size"]]*2,bias=False).to(torch.bfloat16)
                for i,weight in enumerate(gu):m.weight_loader(m.weight,weight,i)
                for key,w in zip(("gate","up"),m.weight.data.chunk(2,dim=0)):collected[key].append(w.clone())
                r=RowParallelLinear(config["intermediate_size"],config["hidden_size"],bias=False).to(torch.bfloat16)
                r.weight_loader(r.weight,down);collected["down"].append(r.weight.data.clone())
        for key,full in zip(("q","k","v","gate","up","down"),qkv+gu+[down]):
            rebuilt=torch.cat(collected[key],dim=1 if key=="down" else 0)
            equal=torch.equal(rebuilt,full);assert equal,(layer,key)
            rows.append(dict(layer=layer,component=key,shape=list(full.shape),bitwise_reconstruction=equal))
        print(json.dumps(dict(completed_layer=layer)),flush=True)
    with args.output.open("x") as f:json.dump(dict(status="PASS_WEIGHT_MAPPING_ONLY",rows=rows),f,indent=2)


if __name__=="__main__":main()
