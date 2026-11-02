"""启动期为每张 CUDA Graph 准备独立 Marlin workspace，权重只读共享。"""
from contextlib import contextmanager
import torch


@contextmanager
def model_graph_workspace(model, stream=None):
    layers = [m for m in getattr(model, "modules", lambda: ())() if getattr(m, "backend", None)=="awq_marlin"]
    if not layers:
        yield stream
        return
    if stream is None:
        stream = getattr(model, "_marlin_capture_stream", None)
        if stream is None:
            model._marlin_capture_stream = stream = torch.cuda.Stream(device=layers[0]._marlin_state.device)
    previous = []
    success = False
    try:
        for layer in layers:
            state = layer._marlin_state.for_graph(stream)
            name = f"_marlin_graph_workspace_{len(layer._marlin_graph_states)+1}"
            previous.append((layer,layer._active_marlin,state,name))
            layer._marlin_graph_states.append(state)
            layer.register_buffer(name,state.workspace,persistent=False)
            layer._active_marlin = state
        yield stream
        success = True
    finally:
        for layer, old, state, name in reversed(previous):
            layer._active_marlin = old
            if not success:
                layer._marlin_graph_states.remove(state)
                if hasattr(layer,name):
                    delattr(layer,name)
        # 成功 capture 的 workspace 是模型所有，在 engine/model 销毁时释放。
        # 本版仅支持有界 startup captures，不开放量化 runtime/mixed capture。


@contextmanager
def capture_model_graph(model, graph, pool=None):
    with model_graph_workspace(model) as stream:
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            yield


def assert_graph_replay_stream(owner, device):
    """在修改静态输入前绑定执行 stream；它不必等于 startup capture stream。"""
    if torch.device(device).type != "cuda":
        return  # CPU fake metadata tests have no CUDA stream.
    stream = torch.cuda.current_stream(device).cuda_stream
    previous = getattr(owner, "_replay_stream_id", None)
    if previous is not None and previous != stream:
        raise RuntimeError("graph static inputs/workspace cannot be shared across execution streams")
    owner._replay_stream_id = stream
