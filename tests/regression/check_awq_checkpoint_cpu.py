"""CPU-only real-checkpoint packed layout/scale audit; no remote model code."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from nanovllm.layers.awq_packed import AWQPackedLinear, validate_awq_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((args.model / "config.json").read_text())
    validate_awq_config(config["quantization_config"])
    if config["model_type"] != "qwen3" or config.get("auto_map"):
        raise ValueError("requires native dense Qwen3 without remote code")
    tensors, handles = {}, []
    try:
        for path in sorted(args.model.glob("*.safetensors")):
            handle = safe_open(path, framework="pt", device="cpu")
            handles.append(handle)
            for name in handle.keys():
                if name in tensors:
                    raise ValueError("duplicate checkpoint tensor")
                tensors[name] = handle
        prefixes = sorted(name.removesuffix(".qweight") for name in tensors if name.endswith(".qweight"))
        if len(prefixes) != config["num_hidden_layers"] * 7:
            raise ValueError("expected exactly seven quantized projections per layer")
        records = []
        h, inter = config["hidden_size"], config["intermediate_size"]
        q = config["num_attention_heads"] * config["head_dim"]
        kv = config["num_key_value_heads"] * config["head_dim"]
        shapes = dict(q_proj=(h, q), k_proj=(h, kv), v_proj=(h, kv), o_proj=(q, h),
                      gate_proj=(h, inter), up_proj=(h, inter), down_proj=(inter, h))
        for prefix in prefixes:
            k, n = shapes[prefix.rsplit(".", 1)[-1]]
            layer = AWQPackedLinear(k, n, device="cpu")
            source_dtype = None
            for field in ("qweight", "qzeros", "scales"):
                name = prefix + "." + field
                value = tensors[name].get_tensor(name)
                if field == "scales":
                    source_dtype = str(value.dtype)
                parameter = getattr(layer, field)
                parameter.weight_loader(parameter, value)
            layer.assert_loaded()
            records.append(dict(prefix=prefix, source_scale_dtype=source_dtype, **layer.stats()))
            del layer, parameter, value
        report = dict(status="PASS", model=str(args.model), tensor_count=len(tensors),
                      projection_count=len(records), cuda_initialized=torch.cuda.is_initialized(),
                      scale_casts=sum(r.get("checkpoint_scale_casts_bf16_to_fp16", 0) for r in records),
                      scale_cast_max_abs=max(r.get("checkpoint_scale_cast_max_abs", 0) for r in records),
                      records=records, scope="real CPU packed projection load, not model quality or GPU correctness")
        if report["cuda_initialized"]:
            raise RuntimeError("CPU audit unexpectedly initialized CUDA")
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k != "records"}))
    finally:
        handles.clear()


if __name__ == "__main__":
    main()
