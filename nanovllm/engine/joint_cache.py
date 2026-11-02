"""GPU/CPU 联合前缀驻留实验：近期请求、前缀依赖与副本感知。

这里只决定元数据的准入/替换，不提交 CUDA 工作。复制仍由原异步缓存
负责；CPU 源 pin、GPU 目的页 lease 和完成 event 都不能被策略跳过。
"""
from collections import Counter, OrderedDict
from itertools import islice
import json
from pathlib import Path

from nanovllm.engine.cache_cost import CacheCostModel, CacheCostPolicy


class JointResidencyPolicy(CacheCostPolicy):
    joint_residency = True
    capacity_aware = True

    def __init__(self, document, manager=None):
        if document.get('schema') != 'nano-joint-residency-v1':
            raise ValueError('unknown joint residency calibration schema')
        self.document = document
        self.model = CacheCostModel(document['cost_model'])
        self.scan_limit = document['victim_scan_limit']
        self.history_capacity = document.get('history_capacity', 4096)
        if (type(self.scan_limit) is not int or not 8 <= self.scan_limit <= 64
                or type(self.history_capacity) is not int or not 1 <= self.history_capacity <= 4096):
            raise ValueError('invalid bounded joint residency metadata')
        self.manager = manager
        self.history = OrderedDict()
        self.counters = Counter()
        self.epoch = 0
        self.last_observation = None
        self.restore_pins = Counter()

    @classmethod
    def load(cls, path, manager=None):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')), manager)

    def observe(self, request_id, keys, horizon_tokens, *, gpu_hashes=()):
        """每条实际执行请求只计一次；不使用后续请求或实验 split 标签。"""
        if self.last_observation == request_id:
            return
        keys = list(keys)
        if len(keys) != len(gpu_hashes):
            raise ValueError('joint residency needs matching CPU/GPU prefix identities')
        self.epoch += 1
        self.last_observation = request_id
        parent = None
        for depth, (key, gpu_hash) in enumerate(zip(keys, gpu_hashes), 1):
            self.history[key] = dict(last=self.epoch, parent=parent, depth=depth,
                                     gpu_hash=gpu_hash, horizon=horizon_tokens)
            self.history.move_to_end(key)
            self.counters['distinct_request_observations'] += 1
            parent = key
        while len(self.history) > self.history_capacity:
            self.history.popitem(last=False)
            self.counters['history_evictions'] += 1

    def _on_gpu(self, key):
        row = self.history.get(key)
        if row is None or self.manager is None:
            return False
        bid = self.manager.hash_to_block_id.get(row['gpu_hash'])
        return bid is not None and self.manager.blocks[bid].host_key == key

    def select_gpu_victim(self, eligible):
        """只比较无引用页；同一请求前缀优先保留靠前的页面。"""
        choices = list(islice(eligible, self.scan_limit))
        if not choices:
            return None
        cache = self.manager.host_cache
        def rank(item):
            _, bid = item
            key = self.manager.blocks[bid].host_key
            row = self.history.get(key, {})
            return (row.get('last', -1), int(not cache.contains(key)),
                    -row.get('depth', 0))
        selected = min(choices, key=rank)
        self.counters['gpu_victim_checks'] += 1
        self.counters['gpu_victim_keys_scanned'] += len(choices)
        if selected != choices[0]:
            self.counters['gpu_victim_changed'] += 1
        return selected

    def admit(self, key):
        """不再用旧热点累计频次阻止第一次复用的前缀进入 CPU。"""
        self.counters['admission_checks'] += 1
        accepted = key in self.history
        self.counters['admission_accepts' if accepted else 'admission_unknown_skips'] += 1
        return accepted

    def plan_admission(self, keys, residents, free_slots):
        """只规划、不修改页表。调用方先排除在途源页及本轮恢复源 pin。"""
        keys = list(keys)
        eligible = list(islice(residents, self.scan_limit)) if free_slots < len(keys) else []
        cache = self.manager.host_cache if self.manager is not None else None
        def rank(key):
            row = self.history.get(key, {})
            parent = row.get('parent')
            duplicate = self._on_gpu(key)
            orphan = (parent is not None and not self._on_gpu(parent)
                      and cache is not None and not cache.contains(parent))
            # GPU 已有完整副本的 CPU 页先让位；其次清理失去祖先的页。
            return (int(not duplicate), int(not orphan), row.get('last', -1),
                    -row.get('depth', 0))
        ranked = sorted(eligible, key=rank)
        accepted, victims = [], []
        for key in keys:
            self.counters['capacity_checks'] += 1
            if free_slots:
                accepted.append(key)
                free_slots -= 1
            elif ranked:
                victim = ranked.pop(0)
                accepted.append(key)
                victims.append(victim)
                self.counters['cpu_gpu_duplicate_victims'] += int(self._on_gpu(victim))
                self.counters['capacity_replacements'] += 1
            else:
                self.counters['capacity_pinned_skips'] += 1
        self.counters['victim_keys_scanned'] += len(eligible)
        return accepted, victims

    def pin_restore_sources(self, keys):
        self.restore_pins.update(keys)
        self.counters['restore_source_pins'] += len(keys)

    def unpin_restore_sources(self, keys):
        for key in keys:
            if self.restore_pins[key] <= 0:
                raise RuntimeError('joint restore source pin underflow')
            self.restore_pins[key] -= 1
            if not self.restore_pins[key]:
                del self.restore_pins[key]

    def stats(self):
        return dict(self.counters, kind='joint_recent_prefix', history_entries=len(self.history),
                    history_capacity=self.history_capacity, victim_scan_limit=self.scan_limit,
                    active_restore_pins=sum(self.restore_pins.values()),
                    calibration_identity=self.model.identity)
