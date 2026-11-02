"""实验性 GPU–CPU 两层前缀缓存（进程私有，默认关闭）。

第一版刻意采用有完成栅栏的复制：non_blocking 只允许驱动异步提交，
不代表实现了计算/传输重叠。只有 event 完成，CPU 页或 GPU 页才可发布。
这让释放、取消、页复用的生命周期容易验证；性能收益必须另行测量。
"""
from collections import Counter, OrderedDict
import hashlib
import struct
from time import perf_counter

import torch


class HostPrefixCache:
    asynchronous = False
    def __init__(self, kv, max_bytes, namespace, min_restore_tokens=256, max_restore_blocks=8, *, policy=None):
        # kv 的维度为 [K/V, layer, physical_block, token, kv_head, head_dim]。
        if kv.ndim != 6 or max_bytes <= 0 or max_restore_blocks <= 0:
            raise ValueError("invalid host cache shape or capacity")
        self.kv = kv
        self.max_bytes = max_bytes
        self.namespace = namespace
        self.min_restore_tokens = min_restore_tokens
        self.max_restore_blocks = max_restore_blocks
        self.page_shape = tuple(kv.shape[:2]) + tuple(kv.shape[3:])
        self.page_bytes = kv[:, :, 0].numel() * kv.element_size()
        self.capacity = max_bytes // self.page_bytes
        self.pages = OrderedDict()
        self.counters = Counter()
        self.policy = policy

    def key(self, parent, tokens):
        # SHA256 链包含全部历史前缀，而不只比较当前页。
        # namespace 每个引擎唯一，防止跨引擎/模型配置复用；不提供引擎内租户隔离。
        digest = hashlib.sha256(self.namespace.encode())
        digest.update(parent or bytes(32))
        digest.update(struct.pack("<" + "q" * len(tokens), *tokens))
        return digest.digest()

    def _copy(self, destination, source, direction):
        start = perf_counter()
        destination.copy_(source, non_blocking=self.kv.is_cuda)
        if self.kv.is_cuda:
            # 在当前计算 stream 排队，保证先前 KV 写入/读取完成后再复制。
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.kv.device))
            event.synchronize()
        self.counters[direction + "_bytes"] += self.page_bytes
        self.counters[direction + "_wall_ms"] += (perf_counter() - start) * 1000

    def put(self, key, block_id):
        if key is None or not self.capacity:
            return
        if key in self.pages:
            self.pages.move_to_end(key)
            return
        if self.policy is not None and not self.policy.admit(key):
            return
        # 先移除旧页，再创建新页；不让存活 CPU tensor 超过容量上限。
        if getattr(self.policy,'capacity_aware',False):
            accepted,victims=self.policy.plan_admission([key],self.pages,self.capacity-len(self.pages))
            if not accepted:
                self.counters['capacity_store_skips']+=1
                return
            for victim in victims:
                del self.pages[victim]
                self.counters['evictions']+=1
        else:
            while len(self.pages) >= self.capacity:
                self.pages.popitem(last=False)
                self.counters["evictions"] += 1
        page = torch.empty(self.page_shape, dtype=self.kv.dtype, device="cpu",
                           pin_memory=self.kv.is_cuda)
        self._copy(page, self.kv[:, :, block_id], "d2h")
        # 拷贝失败时不会发布不完整的 CPU 数据；CUDA 错误向上抛出。
        self.pages[key] = page
        self.counters["stores"] += 1

    def contains(self, key):
        return key in self.pages

    def load(self, key, block_id):
        page = self.pages[key]
        self._copy(self.kv[:, :, block_id], page, "h2d")
        self.pages.move_to_end(key)
        self.counters["loads"] += 1

    def clear(self):
        self.pages.clear()
        if self.policy is not None:
            self.policy.history.clear()

    def stats(self):
        return dict(self.counters, entries=len(self.pages),
                    resident_bytes=len(self.pages) * self.page_bytes,
                    max_bytes=self.max_bytes, page_bytes=self.page_bytes,
                    capacity_blocks=self.capacity,
                    transfer_mode="event_fenced_no_overlap",
                    policy=self.policy.stats() if self.policy is not None else {})


class AsyncHostPrefixCache(HostPrefixCache):
    """只异步预取H2D，D2H仍同步；独立stream不代表收益已经成立。

    inflight同时持有CPU源引用和完成event。BlockManager另持GPU页租约，
    因而CPU源不会被LRU释放、GPU目的页也不会在COPYING_IN时复用。
    """
    asynchronous = True

    def __init__(self, *args, max_inflight_blocks=4, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.kv.is_cuda or max_inflight_blocks < 1:
            raise ValueError("async host cache requires CUDA and positive inflight capacity")
        self.max_inflight_blocks = max_inflight_blocks
        self.transfer_stream = torch.cuda.Stream(device=self.kv.device)
        self.kv.record_stream(self.transfer_stream)
        self.inflight = {}

    def available_slots(self):
        return self.max_inflight_blocks-len(self.inflight)

    def put(self, key, block_id):
        if key is None or not self.capacity:
            return
        if key in self.pages:
            self.pages.move_to_end(key)
            return
        if self.policy is not None and not self.policy.admit(key):
            return
        busy_keys = {ticket[0] for ticket in self.inflight.values()}
        # 联合策略会在回载时备份 GPU 受害页；后续尚未提交的源页也必须保护。
        busy_keys.update(getattr(self.policy, 'restore_pins', ()))
        if getattr(self.policy,'capacity_aware',False):
            accepted,victims=self.policy.plan_admission([key],
                (k for k in self.pages if k not in busy_keys),self.capacity-len(self.pages))
            if not accepted:
                self.counters['capacity_store_skips']+=1
                return
            for victim in victims:
                del self.pages[victim]
                self.counters['evictions']+=1
        while len(self.pages) >= self.capacity:
            victim = next((k for k in self.pages if k not in busy_keys), None)
            if victim is None:
                # 容量全被传输源占用：跳过备份，后续重算；不能突破pinned容量。
                self.counters["busy_store_skips"] += 1
                return
            del self.pages[victim]
            self.counters["evictions"] += 1
        page = torch.empty(self.page_shape,dtype=self.kv.dtype,device="cpu",pin_memory=True)
        self._copy(page,self.kv[:,:,block_id],"d2h")
        self.pages[key]=page
        self.counters["stores"]+=1

    def load(self, key, block_id):
        if not self.available_slots() or block_id in self.inflight:
            raise RuntimeError("async restore capacity exceeded or duplicate destination")
        source = self.pages[key]
        start = torch.cuda.Event(enable_timing=True)
        done = torch.cuda.Event(enable_timing=True)
        # 等待之前对GPU页的计算，再开始回载。后续不相关计算可在原stream继续。
        self.transfer_stream.wait_stream(torch.cuda.current_stream(self.kv.device))
        try:
            with torch.cuda.stream(self.transfer_stream):
                start.record()
                self.kv[:,:,block_id].copy_(source,non_blocking=True)
                done.record()
        except BaseException:
            # 未发布的页也可能已排队写入；失败传播前必须停止该页被提前复用。
            self.transfer_stream.synchronize()
            raise
        self.inflight[block_id]=(key,source,start,done)
        self.pages.move_to_end(key)
        self.counters["h2d_submissions"]+=1
        self.counters["h2d_bytes"]+=self.page_bytes

    def wait_for_blocks(self, block_ids):
        for bid in set(block_ids):
            ticket=self.inflight.get(bid)
            if ticket is not None:
                # 只给真正消费该页的计算stream加依赖；不在CPU等待全部GPU工作。
                torch.cuda.current_stream(self.kv.device).wait_event(ticket[3])
                self.counters["consumer_wait_events"]+=1

    def reap(self, blocking=False):
        ready=[]
        for bid,(_,source,start,done) in list(self.inflight.items()):
            if blocking:
                done.synchronize()
            if done.query():
                self.counters["h2d_device_ms"]+=start.elapsed_time(done)
                self.counters["loads"]+=1
                del self.inflight[bid]
                ready.append(bid)
        return ready

    def clear(self):
        # 引擎先通过BlockManager回收租约，这里再防御性排空源生命周期。
        self.reap(blocking=True)
        super().clear()

    def stats(self):
        return dict(super().stats(),pending_blocks=len(self.inflight),
                    pending_bytes=len(self.inflight)*self.page_bytes,
                    transfer_mode="async_h2d_sync_d2h",max_inflight_blocks=self.max_inflight_blocks)
