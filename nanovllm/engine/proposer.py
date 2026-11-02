"""Proposer contract and CPU-testable draft KV synchronization.

Only committed tokens enter the target Sequence. ``cached_tokens`` describes
inputs actually forwarded by the draft, including temporarily proposed inputs.
Synchronizing after target verification makes those inputs safe to reuse.
"""
from collections import Counter
from typing import Protocol


class Proposer(Protocol):
    def propose(self, token_ids: list[int], max_tokens: int) -> list[int]: ...
    def synchronize(self, token_ids: list[int]) -> None: ...
    def clear(self) -> None: ...


class DraftProposer:
    """One request, fixed-size physical KV, logical truncation after rejection.

    forward(tokens, start) writes these inputs to KV and returns the next greedy
    token. Physical stale tails can remain: subsequent calls overwrite them and
    expose only [0, end) to Attention.
    """

    def __init__(self, forward, capacity: int, chunk_size: int, rollout=None):
        if capacity < 1 or chunk_size < 1:
            raise ValueError("draft capacity and chunk_size must be positive")
        self.forward = forward
        self.capacity = capacity
        self.chunk_size = chunk_size
        # Optional device-resident rollout. It forwards the suffix and all but
        # the last proposed token, returning the entire candidate list at once.
        self.rollout = rollout
        self.cached_tokens = []
        self.counters = Counter()

    def clear(self):
        self.cached_tokens.clear()

    def synchronize(self, token_ids):
        common = 0
        for left, right in zip(self.cached_tokens, token_ids):
            if left != right:
                break
            common += 1
        # Keep at least the final committed token unforwarded, so the next
        # proposal can recover its logits without retaining a full vocab tensor.
        keep = min(common, max(0, len(token_ids) - 1))
        self.counters["trimmed_tokens"] += len(self.cached_tokens) - keep
        del self.cached_tokens[keep:]

    def _append(self, tokens):
        prediction = self.forward(tokens, len(self.cached_tokens))
        if type(prediction) is not int or prediction < 0:
            raise ValueError("draft forward must return a non-negative integer token")
        self.cached_tokens.extend(tokens)
        self.counters["forward_calls"] += 1
        self.counters["forwarded_tokens"] += len(tokens)
        return prediction

    def propose(self, token_ids, max_tokens):
        if max_tokens <= 0:
            return []
        if not token_ids:
            raise ValueError("draft requires a nonempty committed history")
        # The final proposal itself need not have a cached KV input yet.
        limit = min(max_tokens, self.capacity - len(token_ids) + 1)
        if limit <= 0:
            self.counters["capacity_fallbacks"] += 1
            self.clear()
            return []
        self.synchronize(token_ids)
        try:
            suffix = token_ids[len(self.cached_tokens):]
            if self.rollout is not None:
                draft = self.rollout(suffix, len(self.cached_tokens), limit)
                if (not isinstance(draft, list) or len(draft) != limit
                        or any(type(token) is not int or token < 0 for token in draft)):
                    raise ValueError("draft rollout must return the requested integer tokens")
                # Publish logical state only after the complete rollout succeeds.
                self.cached_tokens.extend(suffix)
                self.cached_tokens.extend(draft[:-1])
                self.counters["forward_calls"] += (len(suffix) + self.chunk_size - 1) // self.chunk_size + limit - 1
                self.counters["forwarded_tokens"] += len(suffix) + limit - 1
                self.counters["proposed_tokens"] += len(draft)
                self.counters["rollout_calls"] += 1
                return draft
            for start in range(0, len(suffix), self.chunk_size):
                prediction = self._append(suffix[start:start + self.chunk_size])
            draft = [prediction]
            for _ in range(limit - 1):
                draft.append(self._append([draft[-1]]))
            self.counters["proposed_tokens"] += len(draft)
            return draft
        except BaseException:
            # A partial failed write must never be reused as valid draft KV.
            self.clear()
            raise


def validate_tokenizers(target, draft):
    if target.get_vocab() != draft.get_vocab():
        raise ValueError("draft and target token-to-ID mappings differ")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "all_special_ids"):
        if getattr(target, name) != getattr(draft, name):
            raise ValueError(f"draft and target {name} differ")
