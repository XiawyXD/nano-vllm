"""不可变执行计划：CPU 调度与 GPU forward 之间的明确边界。

计划只携带本轮输入和页表快照，不携带可变 Sequence 或 Reservation；因此
worker 不需要完整生成历史，decode 的单 token 也不会被误当作完整 prompt。
"""
from dataclasses import dataclass
from enum import Enum


class ExecutionRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"


@dataclass(frozen=True, slots=True)
class ExecutionItem:
    """一条请求的本轮 query；logit_positions 是该 query 内需要输出的行。"""
    request_id: int
    role: ExecutionRole
    input_token_ids: tuple[int, ...]
    start_pos: int
    block_table: tuple[int, ...]
    draft: tuple[int, ...] = ()
    logit_positions: tuple[int, ...] = ()

    def __post_init__(self):
        if not self.input_token_ids or self.start_pos < 0:
            raise ValueError("execution item requires positive query length and non-negative start")
        if self.role is ExecutionRole.DECODE and len(self.input_token_ids) != 1:
            raise ValueError("decode requires exactly one query token")
        if self.role is ExecutionRole.VERIFY and len(self.input_token_ids) != len(self.draft) + 1:
            raise ValueError("verification query must be last_token + draft")
        if self.role is not ExecutionRole.VERIFY and self.draft:
            raise ValueError("draft is only valid for verification")
        if any(row < 0 or row >= self.query_len for row in self.logit_positions):
            raise ValueError("logit position is outside query")
        if tuple(sorted(set(self.logit_positions))) != self.logit_positions:
            raise ValueError("logit positions must be unique and ordered")

    @property
    def query_len(self) -> int:
        return len(self.input_token_ids)

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.query_len

    @property
    def kv_len(self) -> int:
        return self.end_pos

    @property
    def sample_eligible(self) -> bool:
        return bool(self.logit_positions)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """一次 transformer forward，P/V 在前、D 在后以便 Attention 分路。"""
    items: tuple[ExecutionItem, ...]
    step_id: int = 0

    def __post_init__(self):
        ids = [item.request_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("a request may occur only once in an execution plan")
        roles = {item.role for item in self.items}
        if ExecutionRole.PREFILL in roles and ExecutionRole.VERIFY in roles:
            raise ValueError("prefill and verification cannot share a plan")
        seen_decode = False
        for item in self.items:
            if item.role is ExecutionRole.DECODE:
                seen_decode = True
            elif seen_decode:
                raise ValueError("multi-query items must precede ordinary decode items")

    @property
    def kind(self) -> str:
        if self.has_prefill:
            return "PD_PACKED" if self.decode_tokens else "P_PREFILL"
        return "DV_PACKED" if self.has_verification else "D_NORMAL"

    @property
    def has_prefill(self) -> bool:
        return any(item.role is ExecutionRole.PREFILL for item in self.items)

    @property
    def has_verification(self) -> bool:
        return any(item.role is ExecutionRole.VERIFY for item in self.items)

    @property
    def num_tokens(self) -> int:
        return sum(item.query_len for item in self.items)

    @property
    def prefill_tokens(self) -> int:
        return sum(item.query_len for item in self.items if item.role is ExecutionRole.PREFILL)

    @property
    def decode_tokens(self) -> int:
        return sum(item.query_len for item in self.items if item.role is ExecutionRole.DECODE)

    @property
    def verification_tokens(self) -> int:
        return sum(item.query_len for item in self.items if item.role is ExecutionRole.VERIFY)

    @property
    def multi_query_tokens(self) -> int:
        return self.prefill_tokens + self.verification_tokens

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(item.request_id for item in self.items)

    @property
    def logits_indices(self) -> tuple[int, ...]:
        """需要执行 LM head 的 packed hidden 行；部分 prefill 不占任何一行。"""
        result, offset = [], 0
        for item in self.items:
            result.extend(offset + row for row in item.logit_positions)
            offset += item.query_len
        return tuple(result)

    @property
    def logits_layout(self) -> tuple[tuple[int, int, int], ...]:
        """(request_id, start, end) 标记紧凑 logits 中各请求对应的半开区间。"""
        result, offset = [], 0
        for item in self.items:
            count = len(item.logit_positions)
            result.append((item.request_id, offset, offset + count))
            offset += count
        return tuple(result)


@dataclass(slots=True)
class StepStats:
    """工作量和生成量分开计数，不能再用一个正负整数表示整个混合轮次。"""
    scheduled_prefill: int = 0
    query_decode: int = 0
    query_verify: int = 0
    generated_tokens: int = 0
    forward_calls: int = 0
    mixed: int = 0
    accepted_draft_tokens: int = 0

    @classmethod
    def from_plan(cls, plan: ExecutionPlan):
        return cls(scheduled_prefill=plan.prefill_tokens, query_decode=plan.decode_tokens,
                   query_verify=plan.verification_tokens,
                   mixed=int(plan.decode_tokens > 0 and (plan.has_prefill or plan.has_verification)))
