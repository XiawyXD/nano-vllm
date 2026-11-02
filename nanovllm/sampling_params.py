"""控制每条请求如何从模型输出中采样 token。"""

from dataclasses import dataclass
import math


@dataclass(slots=True)
class SamplingParams:
    """一条请求对应的采样参数。

    支持 greedy、temperature、top-k、top-p 和 repetition penalty。
    参数放在请求对象上，因此同一 batch 中的请求可以使用不同策略。
    """

    # temperature=0 表示 greedy；大于 0 时才从概率分布采样。
    temperature: float = 1.0
    # 最多生成多少个 completion token，不包括 prompt token。
    max_tokens: int = 64
    # 是否忽略模型产生的 EOS token；True 时只由 max_tokens 结束。
    ignore_eos: bool = False
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    # 命中任意 stop 字符串后结束请求；字符串本身不会出现在 API 返回文本中。
    stop: str | list[str] | None = None
    speculative_ngram_size: int = 0
    speculative_max_draft_tokens: int = 4
    # None preserves the legacy n-gram opt-in; explicit off overrides it.
    speculative_method: str | None = None
    # 显式概率校正；默认仍是原来的 plain greedy 推测路径。
    speculative_sampling: str = "greedy"

    def __post_init__(self):
        """在入队前验证数值，避免无效输入让整个推理 worker 失败。"""
        for name in ("temperature", "top_p", "repetition_penalty"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.speculative_method not in (None, "off", "ngram", "draft_model"):
            raise ValueError("speculative_method must be off, ngram, draft_model or None")
        if self.speculative_sampling not in ("greedy", "rejection"):
            raise ValueError("speculative_sampling must be greedy or rejection")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        for name, minimum in (
            ("max_tokens", 1), ("top_k", 0), ("speculative_ngram_size", 0),
            ("speculative_max_draft_tokens", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.ignore_eos, bool):
            raise ValueError("ignore_eos must be a bool")
        if self.speculative_method == "ngram" and not self.speculative_ngram_size:
            raise ValueError("ngram method requires speculative_ngram_size > 0")
        if self.speculative_method == "draft_model" and self.speculative_ngram_size:
            raise ValueError("draft_model cannot also specify an n-gram proposer")
        if self.speculative_sampling == "rejection":
            # 普通 sampler 对 <=1e-10 有历史 greedy 语义，不能把它当 softmax p。
            if self.speculative_method != "draft_model" or self.temperature <= 1e-10:
                raise ValueError("rejection sampling requires draft_model and temperature > 1e-10")
            if self.top_k != 0 or self.top_p != 1.0 or self.repetition_penalty != 1.0:
                raise ValueError("rejection v1 supports temperature only; top-k/top-p/penalty are unsupported")
            if self.speculative_max_draft_tokens > 8:
                raise ValueError("rejection v1 supports at most 8 draft tokens")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ValueError("seed must be an int or None")
            # Sampler 每轮使用 seed + num_completion_tokens。保留 PyTorch
            # 支持的负 seed，但预先确保整个请求都不会溢出 manual_seed 范围。
            if not -(2**63) <= self.seed <= 2**64 - self.max_tokens:
                raise ValueError("seed plus generation position exceeds the supported 64-bit range")
        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            if not self.stop:
                raise ValueError("stop must contain non-empty strings")
            self.stop = [self.stop]
        else:
            if not isinstance(self.stop, (list, tuple)):
                raise ValueError("stop must be a string, a list of strings, or None")
            self.stop = list(self.stop)
            if not all(isinstance(item, str) and item for item in self.stop):
                raise ValueError("stop must contain non-empty strings")

    @property
    def use_speculative(self) -> bool:
        return self.resolved_speculative_method != "off"

    @property
    def resolved_speculative_method(self) -> str:
        return self.speculative_method or ("ngram" if self.speculative_ngram_size > 0 else "off")
