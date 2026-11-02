"""离线校准、运行时只读的前缀缓存成本决策；默认路径不创建此对象。"""
from bisect import bisect_left
from collections import Counter, OrderedDict, deque
import hashlib
import json
import math
from pathlib import Path


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _lerp(points, x):
    """有序非负表；区间内插值，区间外按端点斜率保守线性延伸。"""
    if x <= 0:
        return 0.0
    xs = [row[0] for row in points]
    i = bisect_left(xs, x)
    if i == 0:
        return points[0][1] * x / points[0][0] if xs[0] else points[0][1]
    if i == len(points):
        return points[-1][1] * x / max(1, points[-1][0])
    a, b = points[i - 1], points[i]
    return a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])


class CacheCostModel:
    """保存真实 (Q, past-K) wall time 与每种传输后端的批量耗时表。"""
    def __init__(self, document):
        if document.get("schema") != "nano-cache-cost-v1":
            raise ValueError("unknown cache cost calibration schema")
        self.document = document
        self.identity = document["identity"]
        self.chunk_size = self.identity["chunk_size"]
        self.block_size = self.identity["block_size"]
        self.cold_reuse_probability = document.get('cold_reuse_probability',0.0)
        replacement = document.get('replacement', {})
        if not isinstance(replacement, dict) or replacement.get('kind', 'none') not in ('none','capacity_value'):
            raise ValueError('invalid cache replacement policy')
        self.replacement_kind = replacement.get('kind', 'none')
        self.reuse_window_requests = replacement.get('reuse_window_requests', 64)
        self.victim_scan_limit = replacement.get('victim_scan_limit', 64)
        if (type(self.reuse_window_requests) is not int or not 1 <= self.reuse_window_requests <= 64
                or type(self.victim_scan_limit) is not int or not 1 <= self.victim_scan_limit <= 64):
            raise ValueError('cache replacement bounds must be integers in 1..64')
        if (type(self.cold_reuse_probability) not in (int,float)
                or not math.isfinite(self.cold_reuse_probability) or not 0 <= self.cold_reuse_probability <= 1):
            raise ValueError('invalid calibrated cold reuse probability')
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("invalid calibrated chunk size")
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError("invalid calibrated block size")
        self.query = {}
        for row in document["query_ms"]:
            q, past, ms = row["q"], row["past"], row["ms"]
            if type(q) is not int or q < 1 or type(past) is not int or past < 0 or not _positive(ms):
                raise ValueError("invalid query calibration row")
            if (q, past) in self.query:
                raise ValueError("duplicate query calibration row")
            self.query[q, past] = float(ms)
        self.qs = sorted({q for q, _ in self.query})
        self.pasts = sorted({p for _, p in self.query})
        if not self.qs or not self.pasts or any((q,p) not in self.query for q in self.qs for p in self.pasts):
            raise ValueError("query calibration must be a nonempty rectangular grid")
        self.transfers = {}
        for direction in ("h2d", "d2h"):
            rows = document["transfer_ms"][direction]
            if (not rows or any(type(n) is not int or n < 1 or not _positive(ms) for n,ms in rows)
                    or [n for n,_ in rows] != sorted({n for n,_ in rows})):
                raise ValueError("invalid transfer calibration table")
            self.transfers[direction] = [(n, float(ms)) for n,ms in rows]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def check_config(self, config):
        identity = self.identity
        expected = dict(model=str(Path(config.model).resolve()),
                        config_sha256=hashlib.sha256((Path(config.model)/"config.json").read_bytes()).hexdigest(),
                        chunk_size=config.prefill_chunk_size or config.max_num_batched_tokens,
                        block_size=config.kvcache_block_size, linear_backend=config.linear_backend,
                        numerical_backend=config.numerical_backend, rmsnorm_backend=config.rmsnorm_backend,
                        model_dtype=config.model_dtype,
                        transport=("batched" if config.host_kv_transfer_backend == "batched" else "legacy"))
        if config.host_kv_transfer_backend=='batched':
            expected.update(batch_blocks=config.host_kv_transfer_batch_blocks,slots=config.host_kv_transfer_slots)
        if any(identity.get(k) != v for k,v in expected.items()):
            raise ValueError("cache cost calibration does not match model/backend/chunk/layout")
        import os
        if os.environ.get("CUDA_VISIBLE_DEVICES") and identity.get("gpu_uuid") != os.environ["CUDA_VISIBLE_DEVICES"]:
            raise ValueError("cache cost calibration belongs to another physical GPU")

    def query_ms(self, q, past):
        by_q = []
        for width in self.qs:
            rows = [(p, self.query[width,p]) for p in self.pasts]
            # past=0 is meaningful, unlike a zero-byte transfer.
            if past <= rows[0][0]:
                cost = rows[0][1]
            else:
                cost = _lerp(rows, past)
            by_q.append((width, cost))
        return _lerp(by_q, q)

    def prefill_ms(self, total, cached=0):
        value = 0.0
        while cached < total:
            q = min(self.chunk_size, total-cached)
            value += self.query_ms(q, cached)
            cached += q
        return value

    def transfer_ms(self, direction, blocks):
        return _lerp(self.transfers[direction], blocks)


class CacheCostPolicy:
    """仅用实际不同请求的历史热度，元数据有界；不在线修改校准参数。"""
    def __init__(self, model, *, min_observations=2, history_capacity=4096):
        if type(min_observations) is not int or min_observations < 1:
            raise ValueError("invalid cache admission observations")
        if type(history_capacity) is not int or history_capacity < 1:
            raise ValueError("invalid cache history capacity")
        self.model = model
        self.min_observations = min_observations
        self.history_capacity = history_capacity
        self.history = OrderedDict()
        self.counters = Counter()
        self.capacity_aware = model.replacement_kind == 'capacity_value'
        self.epoch = 0
        self.last_observation = None
        self.value_cache = OrderedDict()

    def observe(self, request_id, keys, horizon_tokens):
        if self.capacity_aware and self.last_observation != request_id:
            self.epoch += 1
            self.last_observation = request_id
        for key in keys:
            old = self.history.get(key)
            if old is None:
                old = dict(count=0, request=None, horizon=0)
            if old["request"] != request_id:
                old["count"] += 1
                old["request"] = request_id
                self.counters["distinct_request_observations"] += 1
                if self.capacity_aware:
                    seen = old.setdefault('seen', deque(maxlen=self.model.reuse_window_requests))
                    seen.append(self.epoch)
            old["horizon"] = max(old["horizon"], horizon_tokens)
            self.history[key] = old
            self.history.move_to_end(key)
            while len(self.history) > self.history_capacity:
                self.history.popitem(last=False)
                self.counters["history_evictions"] += 1

    def _page_saving(self, tokens):
        """校准表只读；相同前缀长度的每页重算节省可有界复用。"""
        if tokens not in self.value_cache:
            blocks = max(1,tokens//self.model.block_size)
            self.value_cache[tokens] = (self.model.prefill_ms(tokens+1)
                - self.model.prefill_ms(tokens+1,tokens))/blocks
            self.counters['value_computations'] += 1
            if len(self.value_cache)>128:self.value_cache.popitem(last=False)
        self.value_cache.move_to_end(tokens)
        return self.value_cache[tokens]

    def retention_value(self, key):
        """已有页的 D2H 是沉没成本，保留价值不能再次扣除它。"""
        row=self.history.get(key)
        if row is None:return 0.0
        seen=row.get('seen',())
        if self.capacity_aware:
            while seen and seen[0] <= self.epoch-self.model.reuse_window_requests:seen.popleft()
            count=len(seen)
            if not count:return 0.0
        else:count=row['count']
        expected=self.model.cold_reuse_probability if count<self.min_observations else max(1,count-1)
        return max(0.0,expected*(self._page_saving(row['horizon'])-self.model.transfer_ms('h2d',1)))

    def plan_admission(self, keys, residents, free_slots):
        """只做 CPU 规划，不改缓存。调用方先排除复制中的源页和未完成写回。

        空间有富余时保留首次复用机会；空间不足时，新页净收益需高于
        被替换页的保留价值。只扫描最旧的固定窗口，避免与 CPU 容量线性增长。
        """
        from itertools import islice
        keys=list(keys)
        if free_slots>=len(keys):
            self.counters['capacity_checks']+=len(keys)
            self.counters['capacity_free_accepts']+=len(keys)
            return keys,[]
        eligible=list(islice(iter(residents),self.model.victim_scan_limit))
        ranked=sorted((self.retention_value(k),i,k) for i,k in enumerate(eligible))
        accepted=[];victims=[];index=0
        for key in keys:
            self.counters['capacity_checks']+=1
            if free_slots>0:
                accepted.append(key);free_slots-=1
                self.counters['capacity_free_accepts']+=1
                continue
            incoming=self.retention_value(key)-self.model.transfer_ms('d2h',1)
            if index<len(ranked) and incoming>ranked[index][0]:
                accepted.append(key);victims.append(ranked[index][2]);index+=1
                self.counters['capacity_replacements']+=1
            else:self.counters['capacity_rejects']+=1
        self.counters['victim_keys_scanned']+=len(eligible)
        return accepted,victims

    def admit(self, key):
        self.counters["admission_checks"] += 1
        row = self.history.get(key)
        if row is None:
            self.counters["admission_cold_skips"] += 1
            return False
        tokens = row["horizon"]
        blocks = max(1, tokens // self.model.block_size)
        # Amortize the full reusable prefix benefit over its dependency chain.
        saved = self._page_saving(tokens) if self.capacity_aware else (self.model.prefill_ms(tokens+1)-self.model.prefill_ms(tokens+1,tokens))/blocks
        cold=row['count'] < self.min_observations
        expected_reuses=self.model.cold_reuse_probability if cold else max(1,row['count']-1)
        score = ((self.retention_value(key) if self.capacity_aware else expected_reuses * (saved-self.model.transfer_ms("h2d",1)))
                 - self.model.transfer_ms("d2h",1))
        accepted = score > 0
        self.counters["admission_accepts" if accepted else "admission_cost_skips"] += 1
        if accepted and cold:self.counters['admission_prior_accepts']+=1
        return accepted

    def restore(self, total_tokens, prefix_tokens, gpu_prefix_tokens, missing_blocks, queued_blocks=0):
        saved = (self.model.prefill_ms(total_tokens,gpu_prefix_tokens)
                 - self.model.prefill_ms(total_tokens,prefix_tokens))
        # Conservative: charge outstanding traffic, do not assume hidden overlap.
        cost = self.model.transfer_ms("h2d",missing_blocks+queued_blocks)
        accepted = saved > cost
        self.counters["restore_checks"] += 1
        self.counters["restore_accepts" if accepted else "restore_cost_skips"] += 1
        self.counters["predicted_saved_ms_sum"] += max(0.0, saved)
        self.counters["predicted_reload_ms_sum"] += cost
        return accepted

    def stats(self):
        return dict(self.counters, history_entries=len(self.history), history_capacity=self.history_capacity,
                    min_observations=self.min_observations,cold_reuse_probability=self.model.cold_reuse_probability,
                    replacement_kind=self.model.replacement_kind,
                    reuse_window_requests=self.model.reuse_window_requests if self.capacity_aware else None,
                    victim_scan_limit=self.model.victim_scan_limit if self.capacity_aware else None,
                    calibration_identity=self.model.identity)
