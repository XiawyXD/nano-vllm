"""TP1 greedy 的显式 GPU proposal 协议；CPU 计划不伪造候选 token 值。"""
from dataclasses import dataclass
import torch
from nanovllm.engine.execution_plan import ExecutionRole

@dataclass(frozen=True, slots=True)
class DeviceVerificationItem:
    request_id: int
    last_token: int
    count: int
    start_pos: int
    block_table: tuple[int, ...]

    def __post_init__(self):
        if (type(self.request_id) is not int or type(self.block_table) is not tuple
                or type(self.count) is not int or self.count<1 or type(self.start_pos) is not int
                or self.start_pos<0 or type(self.last_token) is not int or self.last_token<0
                or not self.block_table or any(type(p) is not int or p<0 for p in self.block_table)):
            raise ValueError("invalid device verification metadata")
    @property
    def role(self): return ExecutionRole.VERIFY
    @property
    def query_len(self): return self.count+1
    @property
    def end_pos(self): return self.start_pos+self.query_len
    @property
    def kv_len(self): return self.end_pos
    @property
    def sample_eligible(self): return True
    @property
    def logit_positions(self): return tuple(range(self.query_len))
    @property
    def input_token_ids(self): raise RuntimeError("device proposal IDs are not CPU materialized")
    @property
    def draft(self): raise RuntimeError("device proposal IDs are not CPU materialized")

@dataclass(frozen=True, slots=True)
class PendingDeviceDraft:
    request_id: int
    history: tuple[int, ...]
    prefix: int
    count: int
    ids: torch.Tensor
    stream: object

    def __post_init__(self):
        if (type(self.request_id) is not int or type(self.history) is not tuple
                or not self.history or any(type(t) is not int or t<0 for t in self.history)
                or type(self.count) is not int or self.count<1
                or type(self.prefix) is not int or not 0<=self.prefix<len(self.history)
                or not isinstance(self.ids,torch.Tensor) or not self.ids.is_cuda
                or self.ids.dtype!=torch.int64 or self.ids.ndim!=1
                or self.ids.numel()!=self.count or not self.ids.is_contiguous()):
            raise ValueError("invalid pending GPU proposal")

@dataclass(frozen=True, slots=True)
class DeviceVerificationResult:
    request_id: int
    draft: tuple[int, ...]
    predictions: tuple[int, ...]
    def __post_init__(self):
        if (type(self.request_id) is not int or type(self.draft) is not tuple
                or type(self.predictions) is not tuple or not self.draft
                or len(self.predictions)!=len(self.draft)+1
                or any(type(t) is not int or t<0 for t in self.draft+self.predictions)):
            raise ValueError("invalid materialized device verification result")
