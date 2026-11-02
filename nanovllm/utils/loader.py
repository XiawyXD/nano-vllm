"""从 safetensors 文件加载普通权重和 Tensor Parallel 分片权重。"""

import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """普通参数的加载方式：直接复制到已经分配好的参数张量。"""
    if param.shape != loaded_weight.shape:
        raise ValueError(f"weight shape mismatch: {tuple(param.shape)} != {tuple(loaded_weight.shape)}")
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, *, quant_backend="triton", marlin_library=None):
    """遍历模型目录中的 safetensors，并分发到对应的 Parameter。

    某些 Hugging Face 权重把 q/k/v 或 gate/up 分开保存，而 nano-vLLM
    为了减少 kernel 和模块数量把它们合并成一个 Parameter。此时通过
    `packed_modules_mapping` 把原始名称映射到合并参数，并传入 shard_id。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    from nanovllm.layers.awq_packed import AWQPackedLinear
    quant_modules = [m for m in model.modules() if isinstance(m, AWQPackedLinear)]
    loaded_storages = set()

    # 一个模型可能由多个 safetensors 分片组成。
    for file in glob(os.path.join(path, "*.safetensors")):
        # 在 CPU 上逐个读取，避免加载权重时额外占用 GPU 显存。
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 先检查它是否属于需要合并/切片的模块。
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 合并线性层自定义 loader，负责选取当前 rank 的分片。
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded_storages.add(param.untyped_storage().data_ptr())
                        break
                else:
                    # 普通参数直接按完整名称查找；若没有特殊 loader，就复制。
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded_storages.add(param.untyped_storage().data_ptr())
    if quant_modules:
        for module in quant_modules:
            module.assert_loaded()
        missing = [name for name, param in model.named_parameters()
                   if param.untyped_storage().data_ptr() not in loaded_storages]
        if missing:
            raise ValueError(f"incomplete AWQ checkpoint (including unquantized parameters): {missing}")
        if quant_backend == "marlin":
            for module in quant_modules:
                module.prepare_marlin(marlin_library)
        model.quantization_load_report = {"format": "awq_gemm_g128_zp", "packed_modules": len(quant_modules),
                                          "backend": quant_backend, "full_dequantized_weights": False}
