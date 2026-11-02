"""单条请求在调度器中的状态记录。"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """请求生命周期的三个阶段。"""

    # 已进入 waiting 队列，但 prompt 还没有完整建立 KV Cache。
    WAITING = auto()
    # prompt 已完成 prefill，正在等待/执行 decode。
    RUNNING = auto()
    # 生成 EOS 或达到最大生成长度，资源已经可以释放。
    FINISHED = auto()
    # 用户主动取消请求；和 FINISHED 一样不会再进入调度队列。
    CANCELLED = auto()


class Sequence:
    """跟踪一个 prompt 以及它的一条自回归 completion。

    一个 Sequence 对应一个候选生成结果。`token_ids` 会随着 decode 不断增长，
    `block_table` 则把它的逻辑 token block 映射到 GPU KV Cache 的物理 block。
    """

    # BlockManager 会在创建引擎时同步设置这个值。
    block_size = 256
    # 全局递增 id，用来在批量请求结束后恢复输入顺序。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # 拷贝输入，避免调用方后续修改原始 list 影响引擎内部状态。
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        # 已经写入 KV Cache 的 token 数量。
        self.num_cached_tokens = 0
        # 本轮 scheduler 决定交给模型处理的 token 数量。
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        # 逻辑 block -> 物理 KV Cache block 的映射表。
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.stop = list(sampling_params.stop)
        self.finish_reason = None
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.repetition_penalty = sampling_params.repetition_penalty
        self.seed = sampling_params.seed
        self.speculative_ngram_size = sampling_params.speculative_ngram_size
        self.speculative_max_draft_tokens = sampling_params.speculative_max_draft_tokens
        self.speculative_method = sampling_params.speculative_method
        self.speculative_sampling = sampling_params.speculative_sampling

    @property
    def resolved_speculative_method(self):
        return self.speculative_method or ("ngram" if self.speculative_ngram_size > 0 else "off")

    @property
    def use_speculative(self):
        return self.resolved_speculative_method != "off"

    def __len__(self):
        # 让 len(seq) 等价于当前完整序列的 token 数。
        return self.num_tokens

    def __getitem__(self, key):
        # 允许 scheduler 用 seq[start:end] 取出 prompt 的一段。
        return self.token_ids[key]

    @property
    def is_finished(self):
        # 属性形式便于写成 `if seq.is_finished`。
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.CANCELLED)

    @property
    def num_completion_tokens(self):
        # completion 长度 = 当前总长度 - 原始 prompt 长度。
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        # 只返回不会改变的原始 prompt 部分。
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        # 返回模型后来生成的部分，不包含 prompt。
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 向上取整：最后一个 block 可以只装一部分 token。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个逻辑 block 实际已经使用了多少 token。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 取第 i 个逻辑 block 的 token，用于 prefix hash 和 cache 查找。
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # decode 每生成一个 token，就把它加入自回归历史。
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """为 Tensor Parallel 的 worker 进程生成更紧凑的 pickle 状态。

        prefill 需要完整 prompt；decode 时 worker 只需要上一个 token，
        因此 decode 状态只传 `last_token`，减少进程间通信量。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # 与 __getstate__ 对称地恢复 worker 侧的 Sequence。
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            # prefill 状态包含完整 token list。
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode 状态只保存最后一个 token；worker 不需要完整历史。
            self.token_ids = []
            self.last_token = last_state
