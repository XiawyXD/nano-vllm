"""有界 pinned slab + 固定 GPU staging，批量异步 D2H/H2D。

源 GPU 页在计算 stream 上先 gather，再允许同 stream 的下一次 forward 覆盖。
CPU 页仅在 D2H event 完成后发布；H2D 源、目的页由 slab pin 与 manager lease
分别保护。全部长期 buffer 均在 KV 容量预算前创建或计入明确的 CPU 预算。
"""
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import math
import torch

from nanovllm.engine.tiered_cache import HostPrefixCache


def contiguous_segments(slots):
    """返回 (lane_start,lane_end,host_start,host_end)，不假设空槽永远连续。"""
    if not slots:
        return []
    result, begin = [], 0
    for i in range(1,len(slots)+1):
        if i == len(slots) or slots[i] != slots[i-1]+1:
            result.append((begin,i,slots[begin],slots[i-1]+1))
            begin = i
    return result


@dataclass
class TransferSlot:
    stage: torch.Tensor
    ids_cpu: torch.Tensor
    ids_gpu: torch.Tensor
    packed: object
    started: object
    done: object
    job: object = None


class CacheTransferWorkspace:
    """ModelRunner 在 allocate_kv_cache 之前创建；cache 退出后才释放。"""
    def __init__(self, page_shape, dtype, *, batch_blocks, slots, max_bytes, device):
        if batch_blocks < 1 or slots < 2:
            raise ValueError("batched cache requires positive batch and at least two slots")
        self.page_shape = tuple(page_shape)
        self.page_bytes = math.prod(page_shape)*torch.empty((),dtype=dtype,device='cpu').element_size()
        self.resident_bytes = slots*batch_blocks*(self.page_bytes+8)
        if self.resident_bytes > max_bytes:
            raise ValueError("batched cache staging exceeds its byte budget")
        self.batch_blocks = batch_blocks
        self.owner = torch.cuda.current_stream(device)
        self.read_stream = torch.cuda.Stream(device=device)
        self.write_stream = torch.cuda.Stream(device=device)
        self.slots = [TransferSlot(
            torch.empty((batch_blocks,*page_shape),dtype=dtype,device=device),
            torch.empty(batch_blocks,dtype=torch.int64,device='cpu',pin_memory=True),
            torch.empty(batch_blocks,dtype=torch.int64,device=device),
            torch.cuda.Event(), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(slots)]
        self.host_index_bytes = slots*batch_blocks*8

    def assert_owner(self):
        if torch.cuda.current_stream(self.owner.device).cuda_stream != self.owner.cuda_stream:
            raise RuntimeError("batched cache is bound to its original execution stream")

    def warmup(self, kv):
        from nanovllm.layers.kv_transfer import move_pages
        slot = self.slots[0]
        slot.ids_cpu.zero_()
        slot.ids_gpu.copy_(slot.ids_cpu,non_blocking=True)
        move_pages(kv,slot.stage,slot.ids_gpu,1,gather=True)
        move_pages(kv,slot.stage,slot.ids_gpu,1,gather=False)
        self.owner.synchronize()  # startup-only compilation/layout warmup

    def fence(self):
        self.owner.synchronize()
        self.read_stream.synchronize()
        self.write_stream.synchronize()


class BatchedHostPrefixCache(HostPrefixCache):
    asynchronous = True
    batched = True

    def __init__(self, kv, max_bytes, namespace, *, workspace, policy=None,
                 min_restore_tokens=256,max_restore_blocks=8,max_inflight_blocks=4,idle_fastpath=False):
        super().__init__(kv,max_bytes,namespace,min_restore_tokens,max_restore_blocks)
        if not kv.is_cuda or not kv.is_contiguous():
            raise ValueError("batched host cache requires contiguous CUDA KV")
        if workspace.page_shape != self.page_shape or workspace.page_bytes != self.page_bytes:
            raise ValueError("cache workspace layout mismatch")
        self.workspace = workspace
        self.policy = policy
        self.max_inflight_blocks = min(max_inflight_blocks,len(workspace.slots)*workspace.batch_blocks)
        # Index staging is charged to the same pinned memory cap.
        self.capacity = (max_bytes-workspace.host_index_bytes)//self.page_bytes
        if self.capacity < 1:
            raise ValueError("host budget cannot hold one KV page and index buffers")
        self.slab = torch.empty((self.capacity,*self.page_shape),dtype=kv.dtype,device='cpu',pin_memory=True)
        self.free_slots = deque(range(self.capacity))
        self.pages = OrderedDict()  # key -> slab slot; only completed D2H is visible
        self.pending_stores = {}
        self.inflight = {}  # destination -> (key,host_slot,done-event)
        self.source_pins = Counter()
        self.completed_loads = deque()
        self.unconsumed = {}
        self.failed = False
        self.idle_fastpath = idle_fastpath

    @property
    def needs_reap(self):
        # 完成队列也必须保留：GPU 已复制完，不代表 manager 已释放目的页 lease。
        return bool(self.pending_stores or self.inflight or self.completed_loads)

    def _poll(self, blocking=False):
        for slot in self.workspace.slots:
            job = slot.job
            if job is None:
                continue
            if blocking:
                slot.done.synchronize()
                self.counters['drain_host_waits'] += 1
            if not slot.done.query():
                continue
            self.counters[job['direction']+'_device_ms'] += slot.started.elapsed_time(slot.done)
            if job['direction'] == 'd2h':
                for key,host_slot in zip(job['keys'],job['hosts']):
                    self.pending_stores.pop(key)
                    self.pages[key] = host_slot
                    self.pages.move_to_end(key)
                    self.counters['stores'] += 1
            else:
                for bid,host_slot in zip(job['bids'],job['hosts']):
                    self.source_pins[host_slot] -= 1
                    if not self.source_pins[host_slot]:
                        del self.source_pins[host_slot]
                    self.inflight.pop(bid)
                    self.completed_loads.append(bid)
                    self.counters['loads'] += 1
            self.counters[job['direction']+'_completed_bytes'] += len(job['keys'])*self.page_bytes
            slot.job = None

    def _check(self):
        self.workspace.assert_owner()
        if self.failed:
            raise RuntimeError('batched cache failed; recreate the engine')

    def _validate_pairs(self, pairs):
        if len({bid for _,bid in pairs}) != len(pairs):
            raise ValueError('duplicate GPU destination/source in one cache batch')
        if any(type(bid) is not int or not 0 <= bid < self.kv.shape[2] for _,bid in pairs):
            raise ValueError('cache batch GPU page out of range')

    def _host_slots(self, count):
        while len(self.free_slots) < count:
            victim = next((key for key,index in self.pages.items() if not self.source_pins[index]),None)
            if victim is None:
                return None
            self.free_slots.append(self.pages.pop(victim))
            self.counters['evictions'] += 1
        return [self.free_slots.popleft() for _ in range(count)]

    def _slot(self, direction):
        self._poll()
        # Optional backup must leave a ring slot for request-critical H2D.
        if direction == 'd2h' and sum(s.job is not None and s.job['direction']=='d2h'
                for s in self.workspace.slots) >= len(self.workspace.slots)-1:
            return None
        return next((s for s in self.workspace.slots if s.job is None),None)

    def _fail_after_fence(self, old_loads):
        self.failed = True
        self.workspace.fence()
        for slot in self.workspace.slots:
            job=slot.job
            if job is None:
                continue
            if job['direction']=='d2h':
                for key,host_slot in zip(job['keys'],job['hosts']):
                    self.pending_stores.pop(key,None)
                    self.free_slots.append(host_slot)
            slot.job=None
        for bid,(_,host_slot,_) in list(self.inflight.items()):
            self.source_pins[host_slot]-=1
            if not self.source_pins[host_slot]:
                del self.source_pins[host_slot]
            if bid in old_loads:
                self.completed_loads.append(bid)
            del self.inflight[bid]
            self.unconsumed.pop(bid,None)

    def put(self, key, block_id):
        self.put_many([(key,block_id)])

    def put_many(self, pairs):
        self._check()
        pairs=list(pairs)
        self._validate_pairs(pairs)
        self._poll()
        if any(bid in self.inflight for _,bid in pairs):
            raise RuntimeError('cannot back up an unfinished H2D destination')
        candidates=[]
        for key,bid in pairs:
            if key is None or key in self.pages or key in self.pending_stores:
                continue
            if self.policy is not None and not self.policy.admit(key):
                continue
            candidates.append((key,bid))
        from nanovllm.layers.kv_transfer import move_pages
        width=self.workspace.batch_blocks
        for begin in range(0,len(candidates),width):
            group=candidates[begin:begin+width]
            slot=self._slot('d2h')
            if slot is not None and getattr(self.policy,'capacity_aware',False):
                # 先规划再修改 slab：pending 写回和 H2D 源页都不在候选内。
                accepted,victims=self.policy.plan_admission([key for key,_ in group],
                    (key for key,host in self.pages.items() if not self.source_pins[host]),len(self.free_slots))
                self.counters['capacity_store_skips']+=len(group)-len(accepted)
                selected=set(accepted)
                group=[pair for pair in group if pair[0] in selected]
                if not group:continue
                for victim in victims:
                    self.free_slots.append(self.pages.pop(victim))
                    self.counters['evictions']+=1
            hosts=self._host_slots(len(group)) if slot is not None else None
            if slot is None or hosts is None:
                self.counters['bounded_store_skips']+=len(candidates)-begin
                break
            keys,bids=map(list,zip(*group))
            slot.job=dict(direction='d2h',keys=keys,bids=bids,hosts=hosts)
            for key,host in zip(keys,hosts):
                self.pending_stores[key]=host
            try:
                for lane,bid in enumerate(bids):
                    slot.ids_cpu[lane]=bid
                slot.ids_gpu[:len(bids)].copy_(slot.ids_cpu[:len(bids)],non_blocking=True)
                # Same-stream order protects source pages without blocking CPU.
                move_pages(self.kv,slot.stage,slot.ids_gpu,len(bids),gather=True)
                slot.packed.record(self.workspace.owner)
                with torch.cuda.stream(self.workspace.write_stream):
                    self.workspace.write_stream.wait_event(slot.packed)
                    slot.started.record()
                    for a,b,c,d in contiguous_segments(hosts):
                        self.slab[c:d].copy_(slot.stage[a:b],non_blocking=True)
                        self.counters['d2h_dma_calls']+=1
                    slot.done.record()
            except BaseException:
                self._fail_after_fence(set(self.inflight))
                raise
            self.counters['d2h_batches']+=1
            self.counters['pack_launches']+=1
            self.counters['d2h_bytes']+=len(group)*self.page_bytes

    def contains(self, key):
        self._poll()
        return key in self.pages

    def available_slots(self):
        self._poll()
        free=sum(s.job is None for s in self.workspace.slots)*self.workspace.batch_blocks
        return min(free,self.max_inflight_blocks-len(self.inflight))

    def load(self, key, block_id):
        self.load_many([(key,block_id)])

    def load_many(self, pairs):
        self._check()
        pairs=list(pairs)
        self._validate_pairs(pairs)
        self._poll()
        if len(pairs)>self.available_slots() or any(bid in self.inflight for _,bid in pairs):
            raise RuntimeError('batched restore exceeds reserved transfer capacity')
        if any(key not in self.pages for key,_ in pairs):
            raise KeyError('CPU cache page is not READY')
        from nanovllm.layers.kv_transfer import move_pages
        old_loads=set(self.inflight)
        width=self.workspace.batch_blocks
        try:
            for begin in range(0,len(pairs),width):
                group=pairs[begin:begin+width]
                slot=self._slot('h2d')
                if slot is None:
                    raise RuntimeError('reserved cache transfer slot disappeared')
                keys,bids=map(list,zip(*group))
                hosts=[self.pages[key] for key in keys]
                slot.job=dict(direction='h2d',keys=keys,bids=bids,hosts=hosts)
                for key,bid,host in zip(keys,bids,hosts):
                    self.source_pins[host]+=1
                    self.inflight[bid]=(key,host,slot.done)
                    self.unconsumed[bid]=key
                    self.pages.move_to_end(key)
                for lane,bid in enumerate(bids):
                    slot.ids_cpu[lane]=bid
                with torch.cuda.stream(self.workspace.read_stream):
                    self.workspace.read_stream.wait_stream(self.workspace.owner)
                    slot.started.record()
                    slot.ids_gpu[:len(bids)].copy_(slot.ids_cpu[:len(bids)],non_blocking=True)
                    for a,b,c,d in contiguous_segments(hosts):
                        slot.stage[a:b].copy_(self.slab[c:d],non_blocking=True)
                        self.counters['h2d_dma_calls']+=1
                    move_pages(self.kv,slot.stage,slot.ids_gpu,len(bids),gather=False)
                    slot.done.record()
                self.counters['h2d_batches']+=1
                self.counters['scatter_launches']+=1
                self.counters['h2d_bytes']+=len(group)*self.page_bytes
                self.counters['h2d_submissions']+=len(group)
        except BaseException:
            self._fail_after_fence(old_loads)
            raise

    def wait_for_blocks(self, block_ids):
        self._check()
        for bid in set(block_ids):
            ticket=self.inflight.get(bid)
            if ticket is not None:
                self.workspace.owner.wait_event(ticket[2])
                self.counters['consumer_wait_events']+=1

    def mark_consumed(self, block_ids):
        for bid in set(block_ids):
            if bid in self.unconsumed:
                del self.unconsumed[bid]
                self.counters['consumed_prefetch_bytes']+=self.page_bytes

    def evict_gpu_block(self, bid):
        if bid in self.unconsumed:
            del self.unconsumed[bid]
            self.counters['unused_prefetch_bytes']+=self.page_bytes

    def reap(self, blocking=False):
        self._poll(blocking)
        completed=list(self.completed_loads)
        self.completed_loads.clear()
        return completed

    def clear(self):
        self.workspace.assert_owner()
        self._poll(blocking=True)
        self.pages.clear()
        self.pending_stores.clear()
        self.free_slots=deque(range(self.capacity))
        self.counters['unused_prefetch_bytes']+=len(self.unconsumed)*self.page_bytes
        self.unconsumed.clear()
        if self.policy is not None:
            self.policy.history.clear()

    def stats(self):
        self._poll()
        return dict(self.counters,entries=len(self.pages),resident_bytes=len(self.pages)*self.page_bytes,
                    pinned_pool_bytes=self.capacity*self.page_bytes+self.workspace.host_index_bytes,
                    max_bytes=self.max_bytes,page_bytes=self.page_bytes,capacity_blocks=self.capacity,
                    staging_bytes=self.workspace.resident_bytes,
                    staging_slots=len(self.workspace.slots),batch_blocks=self.workspace.batch_blocks,
                    pending_blocks=len(self.inflight),pending_store_blocks=len(self.pending_stores),
                    pending_bytes=(len(self.inflight)+len(self.pending_stores))*self.page_bytes,
                    transfer_mode='bounded_batched_async_d2h_h2d',
                    policy=self.policy.stats() if self.policy is not None else {})
