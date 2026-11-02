"""请求级增量输出事件。"""

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class GenerationEvent:
    """一次请求的一次增量输出。"""

    request_id: int
    token_id: int
    text: str
    token_index: int
    finished: bool
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        """返回适合 JSON/SSE 编码的普通字典。"""
        return asdict(self)
