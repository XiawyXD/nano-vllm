"""One-token draft graph with GPU-resident autoregressive state.

Only the owning DraftRunner may use this graph, on its current CUDA stream.
Capture happens during startup, before target KV budgeting. A replay writes one
input into the draft's private KV, stores its prediction, and advances metadata.
The fixed max_seqlen_k is an upper bound; cu_k is the actual visible KV length.
"""
import torch

from nanovllm.utils.context import set_context, reset_context
from nanovllm.layers.quant_graph import model_graph_workspace, assert_graph_replay_stream


def _wait_metadata_copy(owner):
    """覆盖/释放 pinned host 状态前，只等待本 owner 上一次 H2D。

    copy 或 event.record 失败时没有有效 event，保留 host storage，改用
    原 copy stream 的 fence；失败不能把可能仍在读取的 host buffer 当作空闲。
    """
    if not owner.metadata_copy_pending:
        return
    if owner.metadata_event_recorded:
        if not owner.metadata_copy_event.query():
            owner.metadata_copy_event.synchronize()
            owner.metadata_host_waits += 1
    else:
        owner.metadata_copy_stream.synchronize()
        owner.metadata_host_waits += 1
    owner.metadata_copy_pending = False


def _copy_metadata_async(owner):
    # run 已在修改 host storage 之前检查同一执行 stream 与 query 边界。
    stream = torch.cuda.current_stream(owner.state.device)
    owner.metadata_copy_stream = stream
    owner.metadata_event_recorded = False
    owner.metadata_copy_pending = True
    owner.state.copy_(owner.host_state, non_blocking=True)
    owner.metadata_copy_event.record(stream)
    owner.metadata_event_recorded = True


class DraftTokenGraph:
    @torch.inference_mode()
    def __init__(self, model, block_table, capacity, max_tokens, device, *, async_metadata=False):
        if capacity < 1 or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("draft graph capacity and max_tokens must be positive")
        if type(async_metadata) is not bool:
            raise ValueError("async_metadata must be bool")
        if async_metadata and torch.device(device).type != "cuda":
            raise ValueError("async draft metadata requires CUDA pinned storage")
        self.model, self.block_table = model, block_table
        self.capacity, self.max_tokens = capacity, max_tokens
        # One host-to-device copy initializes token, position, and output index.
        self.state = torch.zeros(3, device=device, dtype=torch.int64)
        self.async_metadata = async_metadata
        self.host_state = (torch.zeros(3, device="cpu", dtype=torch.int64, pin_memory=True)
                           if async_metadata else torch.zeros(3, device="cpu", dtype=torch.int64))
        self.metadata_copy_event = torch.cuda.Event() if async_metadata else None
        self.metadata_copy_pending = self.metadata_event_recorded = False
        self.metadata_copy_stream = None
        self.metadata_host_waits = 0
        self.ids = self.state[:1]
        self.positions = self.state[1:2]
        self.index = self.state[2:3]
        self.slot = torch.zeros(1, device=device, dtype=torch.int32)
        self.cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        self.cu_k = torch.tensor([0, 1], device=device, dtype=torch.int32)
        self.outputs = torch.empty(max_tokens, device=device, dtype=torch.int64)
        self.graph = None
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with model_graph_workspace(model, stream):
                with torch.cuda.stream(stream):
                    # Compile Triton and initialize library state outside capture.
                    for _ in range(3):
                        self.state.zero_()
                        self._step()
                    self.state.zero_()
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self._step()
                torch.cuda.current_stream(device).wait_stream(stream)
                self.graph = graph
        finally:
            reset_context()

    def _step(self):
        self.slot.copy_(self.positions)
        torch.add(self.slot, 1, out=self.cu_k[1:])
        set_context(True, self.cu_q, self.cu_k, 1, self.capacity,
                    self.slot, block_tables=self.block_table)
        logits = self.model.compute_logits(self.model(self.ids, self.positions))
        self.ids.copy_(logits[-1].argmax().reshape(1))
        self.outputs.scatter_(0, self.index, self.ids)
        self.positions.add_(1)
        self.index.add_(1)

    @torch.inference_mode()
    def run(self, token, start, count):
        assert_graph_replay_stream(self, self.state.device)
        if count < 1 or count > self.max_tokens or start < 0 or start + count > self.capacity:
            raise ValueError("draft graph rollout exceeds output or KV capacity")
        if getattr(self, "async_metadata", False):
            _wait_metadata_copy(self)
        self.host_state[0] = 0 if isinstance(token, torch.Tensor) else token
        self.host_state[1] = start
        self.host_state[2] = 0
        if getattr(self, "async_metadata", False):
            _copy_metadata_async(self)
        else:
            # 默认参考路径仍为原来的 pageable、blocking H2D。
            self.state.copy_(self.host_state)
        if isinstance(token, torch.Tensor):
            self.ids.copy_(token.reshape(1))
        for _ in range(count):
            self.graph.replay()
        return self.outputs[:count]

    def close(self):
        if getattr(self, "async_metadata", False):
            _wait_metadata_copy(self)
            self.host_state = self.metadata_copy_event = None
        self.graph = None
        self.model = self.block_table = None


class DraftSuffixGraph:
    """Fixed-width varlen prefill for known committed suffixes (normally Q2).

    Unlike teacher forcing through Q1, this preserves the eager query shape.
    It writes every known suffix token and predicts only after the final one.
    """
    @torch.inference_mode()
    def __init__(self, model, block_table, capacity, width, device, *, async_metadata=False,
                 max_key_length=None, pool=None, needs_prediction=True):
        if type(width) is not int or not 1 <= width <= capacity:
            raise ValueError("draft suffix width must fit KV capacity")
        if type(async_metadata) is not bool:
            raise ValueError("async_metadata must be bool")
        if async_metadata and torch.device(device).type != "cuda":
            raise ValueError("async draft metadata requires CUDA pinned storage")
        self.model, self.block_table = model, block_table
        self.capacity, self.width = capacity, width
        self.max_key_length = capacity if max_key_length is None else max_key_length
        if type(self.max_key_length) is not int or not width <= self.max_key_length <= capacity:
            raise ValueError("draft graph K bound must fit query and capacity")
        if type(needs_prediction) is not bool:
            raise ValueError("needs_prediction must be bool")
        self.needs_prediction = needs_prediction
        self.state = torch.zeros(width + 1, device=device, dtype=torch.int64)
        self.async_metadata = async_metadata
        self.host_state = (torch.zeros(width + 1, device="cpu", dtype=torch.int64, pin_memory=True)
                           if async_metadata else torch.zeros(width + 1, device="cpu", dtype=torch.int64))
        self.metadata_copy_event = torch.cuda.Event() if async_metadata else None
        self.metadata_copy_pending = self.metadata_event_recorded = False
        self.metadata_copy_stream = None
        self.metadata_host_waits = 0
        self.ids, self.start = self.state[:width], self.state[width:]
        self.offsets = torch.arange(width, device=device, dtype=torch.int64)
        self.positions = torch.empty_like(self.offsets)
        self.slot = torch.empty(width, device=device, dtype=torch.int32)
        self.cu_q = torch.tensor([0, width], device=device, dtype=torch.int32)
        self.cu_k = torch.tensor([0, width], device=device, dtype=torch.int32)
        self.prediction = torch.empty(1, device=device, dtype=torch.int64)
        self.graph = None
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with model_graph_workspace(model, stream):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._step()
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, pool=pool):
                    self._step()
                torch.cuda.current_stream(device).wait_stream(stream)
                self.graph = graph
        finally:
            reset_context()

    def _step(self):
        torch.add(self.offsets, self.start, out=self.positions)
        self.slot.copy_(self.positions)
        torch.add(self.slot[-1:], 1, out=self.cu_k[1:])
        set_context(True, self.cu_q, self.cu_k, self.width,
                    getattr(self, "max_key_length", self.capacity),
                    self.slot, block_tables=self.block_table)
        hidden = self.model(self.ids, self.positions)
        if getattr(self, "needs_prediction", True):
            logits = self.model.compute_logits(hidden)
            self.prediction.copy_(logits[-1].argmax().reshape(1))

    @torch.inference_mode()
    def run(self, tokens, start):
        assert_graph_replay_stream(self, self.state.device)
        if (len(tokens) != self.width or start < 0
                or start + self.width > getattr(self, "max_key_length", self.capacity)):
            raise ValueError("draft suffix exceeds query width or KV capacity")
        if getattr(self, "async_metadata", False):
            _wait_metadata_copy(self)
        # CPU owned/pinned buffer: bulk fill avoids one Torch dispatch per token.
        # The existing event fence above still protects an outstanding async H2D.
        host = self.host_state.numpy()
        host[:self.width] = tokens
        host[self.width] = start
        if getattr(self, "async_metadata", False):
            _copy_metadata_async(self)
        else:
            self.state.copy_(self.host_state)
        self.graph.replay()
        return self.prediction if getattr(self, "needs_prediction", True) else None

    def close(self):
        if getattr(self, "async_metadata", False):
            _wait_metadata_copy(self)
            self.host_state = self.metadata_copy_event = None
        self.graph = None
        self.model = self.block_table = None
