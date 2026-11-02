"""Paged KV Cache 的物理 block 管理和 prefix cache。"""

from collections import OrderedDict, deque
from dataclasses import dataclass

import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """只读 prefix 探测结果；reserve 前会再次验证，不能把探测当作已占用。"""
    block_ids: tuple[int, ...] = ()
    hashes: tuple[int, ...] = ()
    token_chunks: tuple[tuple[int, ...], ...] = ()
    cached_tokens: int = 0


@dataclass(slots=True)
class KVReservation:
    """一次尚未提交的页表扩展，仅由 CPU planner 持有，不发给 GPU worker。"""
    sequence: Sequence
    old_table: tuple[int, ...]
    old_cached_tokens: int
    old_scheduled_tokens: int
    start_pos: int
    reserved_end: int
    acquired_prefix_ids: tuple[int, ...] = ()
    allocated_ids: tuple[int, ...] = ()
    state: str = "active"


class Block:
    """一个物理 KV Cache block 的元数据。

    真正的 K/V 数值存放在 ModelRunner 分配的 GPU tensor 中；Block 只保存
    它的编号、引用计数、prefix hash 和对应 token ids，用来做调度与复用。
    """

    def __init__(self, block_id):
        self.block_id = block_id
        # 多条请求共享同一 prefix block 时，引用计数会大于 1。
        self.ref_count = 0
        # -1 表示这个 block 还没有已知的 prefix hash。
        self.hash = -1
        self.token_ids = []
        self.host_key = None

    def update(self, hash: int, token_ids: list[int]):
        # 一个完整 block 写入 KV Cache 后登记它的内容和 hash。
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # 从 free 列表重新取出 block 时，清除旧的 cache 身份。
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.host_key = None


class BlockManager:
    """管理逻辑 block 到物理 block 的映射。

    逻辑上每条 Sequence 都有连续的 block；物理上这些 block 可以分散在
    GPU 显存中，也可以通过 prefix hash 与其他请求共享。
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # prefix hash -> 物理 block id，用于查找可复用的 prefix。
        self.hash_to_block_id: dict[int, int] = dict()
        # OrderedDict 首端是最久未访问的 prefix hash。
        self.prefix_cache_lru: OrderedDict[int, int] = OrderedDict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.prefix_cache_lookups = 0
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_evictions = 0
        self._reservations: dict[int, KVReservation] = {}
        self.host_cache = None  # 由单卡 engine 注入；默认没有 CPU 层。
        self._host_load_leases = set()  # COPYING_IN页的额外引用，不属于任何请求。
        self.defer_host_stores = False
        self._pending_host_stores: dict[int, bytes] = {}

    def flush_host_stores(self):
        """Engine-only boundary: copy evicted bytes before any forward overwrites them.

        The CPU planner may reassign a page's metadata but never writes GPU KV.
        A deferred source can therefore be backed up after plan construction.
        H2D preparation must also flush these sources before reusing their pages.
        """
        if getattr(self.host_cache,"batched",False):
            self.host_cache.put_many([(key,bid) for bid,key in self._pending_host_stores.items()])
            self._pending_host_stores.clear()
            return
        for bid, key in list(self._pending_host_stores.items()):
            self.host_cache.put(key, bid)
            del self._pending_host_stores[bid]

    def observe_cache_request(self, seq):
        cache=self.host_cache
        policy=getattr(cache,"policy",None)
        if policy is None or getattr(seq,'_cache_observation_recorded',False):
            return
        keys,parent=[],None
        joint = getattr(policy, 'joint_residency', False)
        gpu_hashes, gpu_hash = [], -1
        for i in range(seq.num_blocks-1):
            tokens = seq.block(i)
            parent=cache.key(parent,tokens)
            keys.append(parent)
            if joint:
                gpu_hash = self.compute_hash(tokens, gpu_hash)
                gpu_hashes.append(gpu_hash)
        if joint:
            policy.observe(seq.seq_id, keys, len(keys)*self.block_size, gpu_hashes=gpu_hashes)
        else:
            policy.observe(seq.seq_id,keys,len(keys)*self.block_size)
        seq._cache_observation_recorded=True

    def discard_host_stores(self):
        if self.host_cache is not None:
            self.host_cache.counters["deferred_store_drops"] += len(self._pending_host_stores)
        self._pending_host_stores.clear()

    def reap_host_loads(self, blocking=False):
        cache=self.host_cache
        if cache is None or not getattr(cache,"asynchronous",False):
            return
        for bid in cache.reap(blocking=blocking):
            if bid not in self._host_load_leases:
                raise RuntimeError("completed host transfer has no GPU page lease")
            self._host_load_leases.remove(bid)
            self.blocks[bid].ref_count-=1
            if self.blocks[bid].ref_count==0:
                self._deallocate_block(bid)

    def probe_prefix(self, seq: Sequence) -> PrefixMatch:
        """寻找连续完整 prefix；至少留最后一个 token 重新 forward 得到 logits。"""
        if seq.block_table or not self.enable_prefix_caching:
            return PrefixMatch()
        ids, hashes, chunks = [], [], []
        previous = -1
        for index in range(max(0, (len(seq) - 1) // self.block_size)):
            tokens = seq.token_ids[index * self.block_size:(index + 1) * self.block_size]
            previous = self.compute_hash(tokens, previous)
            self.prefix_cache_lookups += 1
            block_id = self.hash_to_block_id.get(previous)
            if block_id is None or self.blocks[block_id].token_ids != tokens:
                self.prefix_cache_misses += 1
                break
            self.prefix_cache_hits += 1
            ids.append(block_id)
            hashes.append(previous)
            chunks.append(tuple(tokens))
        return PrefixMatch(tuple(ids), tuple(hashes), tuple(chunks), len(ids) * self.block_size)

    def _validate_prefix(self, match: PrefixMatch) -> None:
        """拒绝已被淘汰/替换的探测结果；只能共享真实已提交的完整页。"""
        if not (len(match.block_ids) == len(match.hashes) == len(match.token_chunks)):
            raise ValueError("invalid prefix match")
        for block_id, hash_value, tokens in zip(match.block_ids, match.hashes, match.token_chunks):
            block = self.blocks[block_id]
            if (self.hash_to_block_id.get(hash_value) != block_id or block.hash != hash_value
                    or tuple(block.token_ids) != tokens or len(tokens) != self.block_size):
                raise ValueError("prefix match became stale; probe again")

    def required_new_blocks(self, seq: Sequence, end_exclusive: int,
                            prefix_match: PrefixMatch | None = None) -> int:
        """返回本次要消耗的 free 页数量，包括尚未引用的命中 prefix 页。"""
        if end_exclusive < seq.num_cached_tokens or end_exclusive < 0:
            raise ValueError("reservation end precedes committed KV")
        required = (end_exclusive + self.block_size - 1) // self.block_size
        if seq.block_table:
            if prefix_match and prefix_match.block_ids:
                raise ValueError("cannot attach prefix to an existing page table")
            return max(0, required - len(seq.block_table))
        match = prefix_match or PrefixMatch()
        self._validate_prefix(match)
        if end_exclusive < match.cached_tokens:
            raise ValueError("reservation end precedes matched prefix")
        free_prefix = sum(self.blocks[index].ref_count == 0 for index in match.block_ids)
        return free_prefix + max(0, required - len(match.block_ids))

    def reserve_to(self, seq: Sequence, end_exclusive: int,
                   prefix_match: PrefixMatch | None = None) -> KVReservation:
        """事务式扩展到本轮 query 的半开结束位置，不预分配整个 prompt。

        不足时不改变请求/引用计数；全部 prefix 先固定引用，再分配新页，防止
        分配过程淘汰尚未占用的命中页。rollback 不恢复被淘汰的无引用缓存身份，
        因为 forward 可能已重写那些物理页；但请求和资源引用一定回到旧状态。
        """
        if seq.seq_id in self._reservations:
            raise RuntimeError("request already has an active KV reservation")
        match = prefix_match or PrefixMatch()
        needed = self.required_new_blocks(seq, end_exclusive, match)
        if needed > len(self.free_block_ids):
            raise MemoryError(f"KV reservation needs {needed} free blocks, has {len(self.free_block_ids)}")
        start = seq.num_cached_tokens if seq.block_table else match.cached_tokens
        if end_exclusive <= start:
            raise ValueError("reservation must contain at least one query token")
        if seq.block_table and start % self.block_size:
            tail = seq.block_table[start // self.block_size]
            if self.blocks[tail].ref_count != 1:
                raise RuntimeError("mutable KV tail must be exclusively owned")
        reservation = KVReservation(seq, tuple(seq.block_table), seq.num_cached_tokens,
                                    seq.num_scheduled_tokens, start, end_exclusive)
        acquired, allocated = [], []
        self._reservations[seq.seq_id] = reservation
        try:
            if not seq.block_table:
                for block_id in match.block_ids:
                    block = self.blocks[block_id]
                    if block.ref_count == 0:
                        self.free_block_ids.remove(block_id)
                        self.used_block_ids.add(block_id)
                    block.ref_count += 1
                    acquired.append(block_id)
                    seq.block_table.append(block_id)
                    self._touch_block(block_id)
                seq.num_cached_tokens = start
            required = (end_exclusive + self.block_size - 1) // self.block_size
            while len(seq.block_table) < required:
                block_id = self._allocate_block()
                allocated.append(block_id)
                seq.block_table.append(block_id)
            reservation.acquired_prefix_ids = tuple(acquired)
            reservation.allocated_ids = tuple(allocated)
            seq.num_scheduled_tokens = end_exclusive - start
            return reservation
        except BaseException:
            reservation.acquired_prefix_ids = tuple(acquired)
            reservation.allocated_ids = tuple(allocated)
            self.rollback(reservation)
            raise

    def commit(self, reservation: KVReservation, committed_end: int) -> None:
        """只登记已接受、已写入的完整页；未验证 draft 永远不能进入 prefix 索引。"""
        if reservation.state != "active":
            raise RuntimeError("KV reservation is no longer active")
        seq = reservation.sequence
        if not reservation.start_pos <= committed_end <= reservation.reserved_end:
            raise ValueError("committed end is outside reservation")
        if committed_end > len(seq.token_ids):
            raise ValueError("committed KV tokens are not in the real token history")
        seq.num_cached_tokens = reservation.start_pos
        seq.num_scheduled_tokens = committed_end - reservation.start_pos
        self.hash_blocks(seq)
        seq.num_cached_tokens = committed_end
        seq.num_scheduled_tokens = 0
        self.trim_to(seq, committed_end)
        reservation.state = "committed"
        self._reservations.pop(seq.seq_id, None)

    def extend_reservation(self, reservation: KVReservation, end_exclusive: int) -> None:
        """在所有 D 基础 query 已 pin 后，为其中一条增加 draft 临时槽位。"""
        if reservation.state != "active" or end_exclusive < reservation.reserved_end:
            raise ValueError("only active reservations may grow")
        seq = reservation.sequence
        needed = self.required_new_blocks(seq, end_exclusive)
        if needed > len(self.free_block_ids):
            raise MemoryError("not enough KV blocks for draft extension")
        old_length = len(seq.block_table)
        allocated = []
        try:
            required = (end_exclusive + self.block_size - 1) // self.block_size
            while len(seq.block_table) < required:
                block_id = self._allocate_block()
                allocated.append(block_id)
                seq.block_table.append(block_id)
        except BaseException:
            for block_id in reversed(seq.block_table[old_length:]):
                self.blocks[block_id].ref_count -= 1
                self._remove_prefix_index(block_id)
                self._deallocate_block(block_id)
            del seq.block_table[old_length:]
            raise
        reservation.allocated_ids += tuple(allocated)
        reservation.reserved_end = end_exclusive
        seq.num_scheduled_tokens = end_exclusive - reservation.start_pos

    def rollback(self, reservation: KVReservation) -> None:
        """撤销尚未提交的引用/页表扩展；已有 committed KV 前缀保持不变。"""
        if reservation.state != "active":
            return
        seq = reservation.sequence
        allocated = set(reservation.allocated_ids)
        for block_id in reversed(seq.block_table[len(reservation.old_table):]):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                if block_id in allocated:
                    self._remove_prefix_index(block_id)
                self._deallocate_block(block_id)
        seq.block_table[:] = reservation.old_table
        seq.num_cached_tokens = reservation.old_cached_tokens
        seq.num_scheduled_tokens = reservation.old_scheduled_tokens
        reservation.state = "rolled_back"
        self._reservations.pop(seq.seq_id, None)

    def trim_to(self, seq: Sequence, committed_end: int) -> None:
        """归还 committed frontier 之后的整页，页内旧字节由有效长度屏蔽。"""
        if committed_end < seq.num_cached_tokens:
            raise ValueError("cannot trim committed KV without explicit rollback")
        required = (committed_end + self.block_size - 1) // self.block_size
        while len(seq.block_table) > required:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._remove_prefix_index(block_id)
                self._deallocate_block(block_id)

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算带链式前缀的 block hash。

        当前 block 的 hash 同时依赖前一个 block 的 hash 和当前 token，
        因此只有前缀完全一致时，后续 block 才能安全共享。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 把前一个 hash 作为输入，形成链式 hash。
            h.update(prefix.to_bytes(8, "little"))
        # 使用 token id 的字节表示作为当前 block 内容。
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _remove_prefix_index(self, block_id: int) -> None:
        """删除一个 block 的 hash 和 LRU 索引，但不改变 block 引用计数。"""
        block = self.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            self.hash_to_block_id.pop(block.hash, None)
            self.prefix_cache_lru.pop(block.hash, None)

    def _touch_block(self, block_id: int) -> None:
        """把命中的 prefix block 移到 LRU 队尾。"""
        block = self.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            self.prefix_cache_lru.move_to_end(block.hash, last=True)

    def _evict_free_lru_block(self, offload=True) -> int | None:
        """选择最久未访问且当前未被引用的 prefix block。"""
        if not self.enable_prefix_caching:
            return None
        policy = getattr(self.host_cache, 'policy', None)
        if getattr(policy, 'joint_residency', False):
            selected = policy.select_gpu_victim((item for item in self.prefix_cache_lru.items()
                                                if item[1] not in self.used_block_ids))
            candidates = [selected] if selected is not None else []
        else:
            candidates = list(self.prefix_cache_lru.items())
        for prefix_hash, block_id in candidates:
            if block_id not in self.used_block_ids:
                if offload and self.host_cache is not None:
                    key = self.blocks[block_id].host_key
                    if self.defer_host_stores:
                        if key is not None:
                            self._pending_host_stores.setdefault(block_id, key)
                    else:
                        self.host_cache.put(key, block_id)
                self.prefix_cache_lru.pop(prefix_hash, None)
                if self.hash_to_block_id.get(prefix_hash) == block_id:
                    self.hash_to_block_id.pop(prefix_hash, None)
                self.free_block_ids.remove(block_id)
                self.prefix_cache_evictions += 1
                return block_id
        return None

    def _allocate_block(self, offload=True) -> int:
        """先使用没有有效缓存的空块；只有必要时才淘汰未引用的 LRU prefix。"""
        if not self.free_block_ids:
            raise RuntimeError("no free KV cache blocks")
        # 已有缓存且 ref_count=0 的 block 也在 free 队列中。直接选 LRU 会在
        # 仍有从未用过的空块时过早丢弃 prefix，所以先找没有有效 hash 索引的块。
        block_id = next(
            (candidate for candidate in self.free_block_ids
             if self.hash_to_block_id.get(self.blocks[candidate].hash) != candidate),
            None,
        )
        if block_id is not None:
            self.free_block_ids.remove(block_id)
        else:
            block_id = self._evict_free_lru_block(offload=offload)
            if block_id is None:
                block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if hasattr(self.host_cache,"evict_gpu_block"):
            self.host_cache.evict_gpu_block(block_id)
        # 如果这个物理 block 以前有 hash，先删除旧索引。
        self._remove_prefix_index(block_id)
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """把引用计数已经降为 0 的 block 放回 free 队列。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """检查 Sequence 能否分配，并返回可命中的完整 prefix block 数。

        最后一个可能不完整的 block 不参与 prefix cache 命中，因为它还可能
        在后续 token 到来后发生变化。
        """
        if not self.enable_prefix_caching:
            # 禁用 prefix cache 时，每个请求独占自己的 block。
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1

        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            self.prefix_cache_lookups += 1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # hash 冲突或 token 内容不一致，prefix 到这里结束。
                self.prefix_cache_misses += 1
                break
            self.prefix_cache_hits += 1
            self._touch_block(block_id)
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                # 已经被别的请求使用，当前请求可以共享它，不需要新 block。
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            # 没有足够物理 block，scheduler 需要等待或抢占其他请求。
            return -1
        return num_cached_blocks

    def restore_host_prefix(self, seq: Sequence, *, progressive: bool = False,
                            reserve_blocks: int = 1) -> int:
        """调度器显式回载，不能放进只读 probe_prefix。

        暂时 pin 已命中/已回载页，避免回载后一页时淘汰前一页。最终释放临时
        引用，由紧接着的 allocate 真正接管；失败也释放临时引用。
        仅作用于全新页表，且至少留最后一页正常计算 logits。

        progressive=True limits the reusable prefix to current free capacity,
        reserving query/decode headroom instead of preallocating the full prompt.
        The engine calls this before the CPU-only planner, and fences consumption
        of pending copies before forward. Submission is not proof of saved work.
        """
        cache = self.host_cache
        if cache is None or seq.block_table or not self.enable_prefix_caching:
            return 0
        if self._pending_host_stores:
            if getattr(cache,"batched",False):
                self.flush_host_stores()
            else:
                raise RuntimeError("flush deferred host stores before preparing H2D restores")
        if type(reserve_blocks) is not int or reserve_blocks < 1:
            raise ValueError("reserve_blocks must be a positive integer")
        candidates, parent, h = [], None, -1
        active_hits = 0
        prefix_cost = 0
        missing_so_far = 0
        for i in range(seq.num_blocks - 1):
            tokens = seq.block(i)
            h = self.compute_hash(tokens, h)
            parent = cache.key(parent, tokens)
            bid = self.hash_to_block_id.get(h)
            if bid is None or self.blocks[bid].token_ids != tokens:
                bid = None
                if not cache.contains(parent):
                    cache.counters["restore_stop_host_miss"] += 1
                    break
            if progressive:
                cost = int(bid is None or self.blocks[bid].ref_count == 0)
                if prefix_cost + cost + reserve_blocks > len(self.free_block_ids):
                    cache.counters["restore_stop_capacity"] += 1
                    break
                if bid is None and missing_so_far >= cache.max_restore_blocks:
                    break
                if (bid is None and getattr(cache, "asynchronous", False)
                        and missing_so_far >= cache.available_slots()):
                    cache.counters["restore_stop_inflight_limit"] += 1
                    break
                prefix_cost += cost
                missing_so_far += int(bid is None)
            active_hits += bid is not None and self.blocks[bid].ref_count > 0
            candidates.append((h, parent, tokens, bid))
        missing = sum(item[3] is None for item in candidates)
        if not missing:
            return 0
        # 与 legacy scheduler 全 prompt 预分配保持一致；不足时不做无益传输。
        if ((not progressive and seq.num_blocks - active_hits > len(self.free_block_ids))
                or missing * self.block_size < cache.min_restore_tokens):
            cache.counters["restore_skips"] += 1
            return 0
        policy=getattr(cache,"policy",None)
        if policy is not None:
            gpu_prefix=0
            for _,_,_,bid in candidates:
                if bid is None:
                    break
                gpu_prefix+=self.block_size
            if not policy.restore(len(seq),len(candidates)*self.block_size,gpu_prefix,missing,
                    len(getattr(cache,"inflight",{}))+len(getattr(cache,"pending_stores",{}))):
                cache.counters["cost_restore_skips"]+=1
                return 0
        pinned, loaded = [], 0
        batch=[]
        batched=getattr(cache,"batched",False)
        batch_capacity=cache.available_slots() if batched else 0
        joint = getattr(policy, 'joint_residency', False)
        restore_keys = [key for _, key, _, bid in candidates if bid is None] if joint else []
        if joint:
            policy.pin_restore_sources(restore_keys)
        try:
            # 先 pin 所有 GPU 命中（包括更后面的页），再分配回载页。
            for _, _, _, bid in candidates:
                if bid is not None:
                    block = self.blocks[bid]
                    if block.ref_count == 0:
                        self.free_block_ids.remove(bid)
                        self.used_block_ids.add(bid)
                    block.ref_count += 1
                    pinned.append(bid)
            for h, key, tokens, bid in candidates:
                if bid is None:
                    if loaded >= cache.max_restore_blocks:
                        break
                    if batched and loaded >= batch_capacity:
                        break
                    if getattr(cache,"asynchronous",False) and not cache.available_slots():
                        break
                    # 旧路径在回载时不备份 GPU 受害页，保护尚未提交的 CPU 源。
                    # 联合策略先 pin 本轮所有 CPU 源，再允许备份 GPU 受害页；
                    # 只有存在未受保护的 CPU 槽位时才能写回。复制/租约保持不变。
                    bid = self._allocate_block(offload=joint)
                    pinned.append(bid)
                    # mixed planner 会延迟备份，但这里即将用 H2D 覆盖同一物理页。
                    # 必须在提交 H2D 前备份仍属于旧 key 的字节；不能等下一次
                    # forward 前才 flush，否则会把新前缀的内容登记到旧 key 下。
                    # 本轮所有 CPU 源已 pin，备份不会淘汰尚未提交的回载源。
                    if self._pending_host_stores:
                        self.flush_host_stores()
                    if batched:
                        batch.append((h,key,tokens,bid))
                        loaded+=1
                        continue
                    cache.load(key, bid)
                    if getattr(cache,"asynchronous",False):
                        # 即使请求取消/还未allocate，目的页在event完成前仍是used。
                        self.blocks[bid].ref_count+=1
                        self._host_load_leases.add(bid)
                    block = self.blocks[bid]
                    block.update(h, tokens)
                    block.host_key = key
                    self.hash_to_block_id[h] = bid
                    self.prefix_cache_lru[h] = bid
                    loaded += 1
                    cache.counters["restore_submitted_tokens"] += self.block_size
                else:
                    self._touch_block(bid)
            if batch:
                cache.load_many([(key,bid) for _,key,_,bid in batch])
                for _,_,_,bid in batch:
                    self.blocks[bid].ref_count+=1
                    self._host_load_leases.add(bid)
                try:
                    for h,key,tokens,bid in batch:
                        block=self.blocks[bid]
                        block.update(h,tokens)
                        block.host_key=key
                        self.hash_to_block_id[h]=bid
                        self.prefix_cache_lru[h]=bid
                        cache.counters['restore_submitted_tokens']+=self.block_size
                except BaseException:
                    for h,_,_,bid in batch:
                        if self.hash_to_block_id.get(h)==bid:
                            self._remove_prefix_index(bid)
                    self.reap_host_loads(blocking=True)
                    raise
        finally:
            try:
                for bid in reversed(pinned):
                    self.blocks[bid].ref_count -= 1
                    if self.blocks[bid].ref_count == 0:
                        self._deallocate_block(bid)
            finally:
                if joint:
                    policy.unpin_restore_sources(restore_keys)
        return loaded

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为 Sequence 填充 block_table，并增加共享 block 的引用计数。"""
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # prefix block 正在被使用：共享物理存储，只增加 ref_count。
                block.ref_count += 1
            else:
                # prefix block 有缓存但当前处于 free 状态，重新占用它。
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            # 未命中的部分分配空 block，稍后由 Attention 写入 K/V。
            seq.block_table.append(self._allocate_block())
        # 已命中的完整 block 对应的 token 数量先记入 cache。
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放 Sequence 持有的 block，并处理 prefix 共享引用。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # 追加一个 token 后，只有跨入新 block（len % block_size == 1）时
        # 才需要额外物理 block；否则当前最后一个 block 仍有空间。
        return len(self.free_block_ids) >= max(0, seq.num_blocks - len(seq.block_table))

    def may_append(self, seq: Sequence):
        # decode 之前预留可能需要的下一个物理 block。
        # speculative 回滚可能保留了最后一个尚未缓存 token 的 block，
        # 因此根据实际缺口分配，不能仅用长度取模，否则会重复分配。
        self.ensure_blocks(seq)

    def ensure_blocks(self, seq: Sequence):
        """确保 Sequence 当前长度拥有足够物理 block。

        普通 decode 每次只跨越至多一个 block，所以 may_append 足够；
        speculative 一次暂时追加多个候选 token，需要使用这个批量版本。
        """
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self._allocate_block())

    def trim(self, seq: Sequence):
        """释放 Sequence 当前逻辑长度之外的临时 block。"""
        required = seq.num_blocks
        while len(seq.block_table) > required:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._remove_prefix_index(block_id)
                self._deallocate_block(block_id)

    def hash_blocks(self, seq: Sequence):
        """预计算并原子发布本轮完整 block 的 prefix 身份。

        hash/key 计算及 prefix 索引写入均可能失败。回滚只释放新分配页，
        不会撤销旧可写尾页上的 hash；因此发布前保存本轮涉及的元数据，
        发布失败后恢复 block、索引和 LRU 的旧状态。
        """
        if not self.enable_prefix_caching:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        # 只有完整跨过 block 边界时才有新的完整 block 可 hash。
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        parent = self.blocks[seq.block_table[start - 1]].host_key if start > 0 else None
        updates = []
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            host_key = self.host_cache.key(parent, token_ids) if self.host_cache is not None else None
            updates.append((block, h, token_ids, host_key))
            parent = host_key
        missing = object()
        block_before = [(block, block.hash, block.token_ids, block.host_key)
                        for block, _, _, _ in updates]
        index_before = {hash_value: self.hash_to_block_id.get(hash_value, missing)
                        for _, hash_value, _, _ in updates}
        lru_before = {hash_value: self.prefix_cache_lru.get(hash_value, missing)
                      for _, hash_value, _, _ in updates}
        try:
            for block, hash_value, token_ids, host_key in updates:
                block.update(hash_value, token_ids)
                if self.host_cache is not None:
                    block.host_key = host_key
                self.hash_to_block_id[hash_value] = block.block_id
                self.prefix_cache_lru[hash_value] = block.block_id
        except BaseException:
            for block, old_hash, old_tokens, old_host_key in block_before:
                block.hash = old_hash
                block.token_ids = old_tokens
                block.host_key = old_host_key
            for hash_value, old_block_id in index_before.items():
                if old_block_id is missing:
                    self.hash_to_block_id.pop(hash_value, None)
                else:
                    # 已存在的槽位无需新增 key；绕过仍在注入故障的包装器。
                    dict.__setitem__(self.hash_to_block_id, hash_value, old_block_id)
            for hash_value, old_block_id in lru_before.items():
                if old_block_id is missing:
                    self.prefix_cache_lru.pop(hash_value, None)
                else:
                    OrderedDict.__setitem__(self.prefix_cache_lru, hash_value, old_block_id)
            raise
        # 把可能分配内存的字典写入全部做完，再移动 LRU 顺序；已存在的
        # OrderedDict 键在值更新时不改变位置，失败路径可保留原次序。
        for _, hash_value, _, _ in updates:
            self.prefix_cache_lru.move_to_end(hash_value, last=True)


    def stats(self) -> dict:
        """返回 prefix cache 和物理 block 的可观测指标。"""
        return {
            "host_cache": self.host_cache.stats() if self.host_cache is not None else {},
            "free_blocks": len(self.free_block_ids),
            "used_blocks": len(self.used_block_ids),
            "host_load_leases": len(self._host_load_leases),
            "active_reservations": len(self._reservations),
            "pending_host_stores": len(self._pending_host_stores),
            "prefix_cache_entries": len(self.hash_to_block_id),
            "prefix_cache_lookups": self.prefix_cache_lookups,
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
            "prefix_cache_evictions": self.prefix_cache_evictions,
        }
