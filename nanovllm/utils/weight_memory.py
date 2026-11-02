"""权重真实 storage 去重；packed backend buffer 也计入，不把 workspace 当权重。"""
def weight_storage_bytes(model):
    tensors = list(model.parameters())
    for module in model.modules():
        if getattr(module, "backend", None) == "awq_marlin":
            tensors.extend((module._marlin_state.weight,module._marlin_state.zeros,module._marlin_state.scales))
    return sum({t.untyped_storage().data_ptr():t.untyped_storage().nbytes()
                for t in tensors if t.numel()}.values())


def quantization_stats(model):
    from nanovllm.layers.awq_packed import AWQPackedLinear
    modules=[m.stats() for m in model.modules() if isinstance(m,AWQPackedLinear)]
    return dict(modules=len(modules),backends=sorted({m['backend'] for m in modules}),
        weight_storage_bytes=weight_storage_bytes(model),
        packed_weight_bytes=sum(m['packed_weight_bytes'] for m in modules),
        scale_bytes=sum(m['scale_bytes'] for m in modules),zero_bytes=sum(m['zero_bytes'] for m in modules),
        workspace_bytes=sum(m.get('workspace_bytes',0) for m in modules),
        full_dequantized_weight_bytes=sum(m['dense_weight_bytes'] for m in modules))
