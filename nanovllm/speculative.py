"""轻量 n-gram draft proposer。

这个模块只负责从历史 token 中提出候选；目标模型的批量验证在
LLMEngine.generate_speculative 中完成。
"""


class NGramProposer:
    """从历史中寻找重复 n-gram，并连续提出后继 token。"""

    def __init__(self, ngram_size: int, max_draft_tokens: int = 4):
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        if max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")
        self.ngram_size = ngram_size
        self.max_draft_tokens = max_draft_tokens

    def propose(self, token_ids: list[int], max_tokens: int | None = None) -> list[int]:
        """根据最后一个 n-gram 提出候选。"""
        limit = min(max_tokens or self.max_draft_tokens, self.max_draft_tokens)
        if len(token_ids) <= self.ngram_size:
            return []
        transitions: dict[tuple[int, ...], list[int]] = {}
        for start in range(len(token_ids) - self.ngram_size):
            key = tuple(token_ids[start:start + self.ngram_size])
            transitions.setdefault(key, []).append(token_ids[start + self.ngram_size])

        history = list(token_ids)
        proposal = []
        for _ in range(limit):
            key = tuple(history[-self.ngram_size:])
            candidates = transitions.get(key)
            if not candidates:
                break
            next_token = candidates[-1]
            proposal.append(next_token)
            history.append(next_token)
        return proposal
