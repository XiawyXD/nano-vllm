"""在分配 GPU 权重之前验证 checkpoint 格式与激活 dtype。"""
import torch


def configure_weight_format(hf, quantization=None, dtype="checkpoint"):
    if quantization not in (None, "awq"):
        raise ValueError("supported quantization is None or explicit 'awq'")
    if dtype not in ("checkpoint", "float16", "bfloat16"):
        raise ValueError("dtype must be checkpoint, float16 or bfloat16")
    metadata = getattr(hf, "quantization_config", None)
    if quantization is None and metadata:
        raise ValueError("quantized checkpoint requires explicit quantization='awq'")
    if dtype != "checkpoint":
        hf.dtype = getattr(torch, dtype)
    if quantization == "awq":
        from nanovllm.layers.awq_packed import validate_awq_config
        validate_awq_config(metadata)
        if hf.model_type != "qwen3" or getattr(hf, "attention_bias", True):
            raise ValueError("AWQ v1 supports bias-free dense Qwen3 only")
        if hf.dtype != torch.float16:
            raise ValueError("AWQ packed kernel requires FP16 activation/scales; select float16 explicitly")
    hf.nanovllm_quantization = quantization
    return hf
