"""请求级增量 n-gram 索引，保留 reference 的“最近后继”语义。"""
from nanovllm.speculative import NGramProposer


class IncrementalNGramProposer(NGramProposer):
    """只接收同一请求的已提交、仅追加历史；不能索引未接受的 draft。

    首次建立 O(T*n) 索引，此后只扫描新提交 token 对应的窗口。
    字典只保留每个 n-gram 最近的后继，不保存所有历史后继列表。
    请求结束/取消时由引擎释放。若调用者要改写历史，必须 reset。
    """

    def __init__(self, ngram_size, max_draft_tokens=4):
        super().__init__(ngram_size, max_draft_tokens)
        self.reset()

    def reset(self):
        self.transitions = {}
        self.indexed_windows = 0

    def clear(self):
        self.reset()

    def synchronize(self, token_ids):
        # This proposer indexes committed append-only history lazily in propose.
        # A shortened history invalidates its index; rewritten history requires clear.
        if max(0, len(token_ids) - self.ngram_size) < self.indexed_windows:
            self.reset()

    def propose(self, token_ids, max_tokens=None):
        limit = self.max_draft_tokens if max_tokens is None else min(max_tokens, self.max_draft_tokens)
        if limit <= 0:
            return []
        n = self.ngram_size
        windows = max(0, len(token_ids) - n)
        # 防御缩短历史；同长度改写无法 O(1) 检测，须显式 reset。
        if windows < self.indexed_windows:
            self.reset()
        for start in range(self.indexed_windows, windows):
            self.transitions[tuple(token_ids[start:start+n])] = token_ids[start+n]
        self.indexed_windows = windows
        if not windows:
            return []
        # 推测时仅需最后 n 个 token，不再复制整个长上下文。
        tail = list(token_ids[-n:])
        proposal = []
        for _ in range(limit):
            next_token = self.transitions.get(tuple(tail))
            if next_token is None:
                break
            proposal.append(next_token)
            tail = tail[1:] + [next_token]
        return proposal
