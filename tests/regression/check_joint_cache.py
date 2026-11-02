"""CPU regressions for joint residency; no CUDA, model, or server is required.

Run after overlaying the staged runtime on the source tree:
    python -B -m tests.regression.check_joint_cache
An optional --output PATH saves the JSON summary also printed to stdout.
The async fixture uses real CPU tensor copies, AsyncHostPrefixCache.put/reap,
and BlockManager; only pinned allocation and H2D submission/events are faked.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.cache_cost import CacheCostModel, CacheCostPolicy
from nanovllm.engine.joint_cache import JointResidencyPolicy
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.tiered_cache import AsyncHostPrefixCache, HostPrefixCache


def calibration(**overrides):
    """Small, valid old-format cost model nested in the joint document."""
    document = dict(
        schema='nano-joint-residency-v1', victim_scan_limit=8,
        history_capacity=32,
        cost_model=dict(
            schema='nano-cache-cost-v1', identity=dict(chunk_size=2, block_size=2),
            query_ms=[dict(q=q, past=p, ms=10 + q + .1 * p)
                      for q in (1, 2) for p in (0, 2, 4, 8)],
            transfer_ms=dict(h2d=[[1, .1], [2, .2], [4, .4]],
                             d2h=[[1, .1], [2, .2], [4, .4]])))
    document.update(overrides)
    return document


class CPUEvent:
    def __init__(self):
        self.complete = False

    def synchronize(self):
        self.complete = True

    def query(self):
        return self.complete

    def elapsed_time(self, other):
        return 0.0


class JointCacheTests(unittest.TestCase):
    def setUp(self):
        size_patch = patch.object(Sequence, 'block_size', 2)
        size_patch.start()
        self.addCleanup(size_patch.stop)
        for name in ('_lazy_init', 'init', 'set_device', 'Stream', 'Event',
                     'current_stream', 'synchronize', 'device_count'):
            guard = patch.object(torch.cuda, name,
                                 side_effect=AssertionError('CPU test attempted CUDA: ' + name))
            guard.start()
            self.addCleanup(guard.stop)
        # Async.put requests pinned CPU memory; exercise its actual replacement
        # and copy code with pageable CPU memory, without involving a CUDA driver.
        real_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            self.assertEqual(torch.device(kwargs.get('device', 'cpu')).type, 'cpu')
            kwargs['pin_memory'] = False
            return real_empty(*args, **kwargs)

        allocator = patch('nanovllm.engine.tiered_cache.torch.empty', side_effect=cpu_empty)
        allocator.start()
        self.addCleanup(allocator.stop)

    def state(self, blocks=3, capacity=4, asynchronous=False, document=None):
        kv = torch.zeros(2, 2, blocks, 2, 1, 2, dtype=torch.float32, device='cpu')
        manager = BlockManager(blocks, 2)
        policy = JointResidencyPolicy(document or calibration(), manager)
        page_bytes = kv[:, :, 0].numel() * kv.element_size()
        cache = (AsyncHostPrefixCache.__new__(AsyncHostPrefixCache)
                 if asynchronous else HostPrefixCache.__new__(HostPrefixCache))
        HostPrefixCache.__init__(cache, kv, capacity * page_bytes, 'joint-cpu-test',
                                 min_restore_tokens=2, max_restore_blocks=8, policy=policy)
        manager.host_cache = cache
        if asynchronous:
            cache.inflight = {}
            cache.max_inflight_blocks = 8
            cache.submissions = []
            cache.before_load = None

            def cpu_load(key, bid):
                if cache.before_load is not None:
                    cache.before_load(key, bid)
                if not cache.available_slots() or bid in cache.inflight:
                    raise RuntimeError('invalid fake H2D destination')
                source = cache.pages[key]
                cache._copy(kv[:, :, bid], source, 'h2d')
                cache.inflight[bid] = (key, source, CPUEvent(), CPUEvent())
                cache.pages.move_to_end(key)
                cache.submissions.append((key, bid))
                cache.counters['h2d_submissions'] += 1

            cache.load = cpu_load
        return SimpleNamespace(kv=kv, manager=manager, cache=cache, policy=policy)

    def observe(self, state, tokens):
        seq = Sequence(list(tokens))
        state.manager.observe_cache_request(seq)
        return seq

    def identities(self, state, seq):
        keys, hashes, parent, previous = [], [], None, -1
        for i in range(seq.num_blocks - 1):
            parent = state.cache.key(parent, seq.block(i))
            previous = state.manager.compute_hash(seq.block(i), previous)
            keys.append(parent)
            hashes.append(previous)
        return keys, hashes

    def seed_gpu(self, state, tokens, values=None, keep=False):
        """Commit complete prefix pages via the real reservation lifecycle."""
        self.assertEqual(len(tokens) % 2, 0)
        seq = self.observe(state, list(tokens) + [99991])
        reservation = state.manager.reserve_to(seq, len(tokens))
        ids = tuple(seq.block_table)
        values = list(values) if values is not None else [100 + i for i in range(len(ids))]
        self.assertEqual(len(values), len(ids))
        for bid, value in zip(ids, values):
            state.kv[:, :, bid].fill_(value)
        state.manager.commit(reservation, len(tokens))
        keys, hashes = self.identities(state, seq)
        if not keep:
            state.manager.deallocate(seq)
        return SimpleNamespace(seq=seq, ids=ids, keys=keys, hashes=hashes, values=values)

    def host_page(self, state, key, value):
        # Initial CPU snapshots; subsequent admissions use the real put method.
        state.cache.pages[key] = torch.full(state.cache.page_shape, float(value),
                                             dtype=state.kv.dtype, device='cpu')

    def assert_page(self, page, value):
        self.assertTrue(torch.equal(page, torch.full_like(page, float(value))))

    def assert_ownership(self, state, drained=False):
        manager = state.manager
        free, used = list(manager.free_block_ids), manager.used_block_ids
        self.assertEqual(len(free), len(set(free)))
        self.assertFalse(set(free) & used)
        self.assertEqual(set(free) | used, set(range(len(manager.blocks))))
        for block in manager.blocks:
            self.assertGreaterEqual(block.ref_count, 0)
            self.assertEqual(block.block_id in used, block.ref_count > 0)
        self.assertEqual(dict(manager.prefix_cache_lru), manager.hash_to_block_id)
        for hash_value, bid in manager.hash_to_block_id.items():
            self.assertEqual(manager.blocks[bid].hash, hash_value)
            self.assertIsNotNone(manager.blocks[bid].host_key)
        self.assertLessEqual(len(state.cache.pages), state.cache.capacity)
        self.assertTrue(manager._host_load_leases <= used)
        if drained:
            self.assertFalse(used)
            self.assertFalse(manager._host_load_leases)
            self.assertFalse(manager._reservations)
            self.assertFalse(getattr(state.cache, 'inflight', {}))
            self.assertFalse(state.policy.restore_pins)

    def restore_fixture(self, document=None):
        state = self.state(asynchronous=True, document=document)
        target = self.observe(state, [1, 2, 3, 4, 9])
        keys, hashes = self.identities(state, target)
        for key, value in zip(keys, (21, 22)):
            self.host_page(state, key, value)
        spares = []
        for tokens in ([30, 31, 9], [40, 41, 9]):
            seq = self.observe(state, tokens)
            key = self.identities(state, seq)[0][0]
            self.host_page(state, key, -1)
            spares.append(key)
        victims = self.seed_gpu(state, [10, 11, 12, 13, 14, 15], [101, 102, 103])
        # Without the all-candidate pins, the older CPU source tail is the first
        # victim of the first D2H backup, before that tail's H2D can be submitted.
        self.assertEqual(state.policy.plan_admission([victims.keys[-1]],
                         state.cache.pages, 0)[1], [keys[-1]])
        return state, target, keys, hashes, victims, spares

    def test_gpu_victim_uses_request_recency_despite_old_frequency_and_lru(self):
        state = self.state(blocks=2)
        old = self.seed_gpu(state, [10, 11], [17])
        recent = self.seed_gpu(state, [20, 21], [27])
        for _ in range(40):
            self.observe(state, old.seq.token_ids)
        self.observe(state, recent.seq.token_ids)
        state.manager._touch_block(old.ids[0])
        self.assertEqual(next(iter(state.manager.prefix_cache_lru)), recent.hashes[0])
        reservation = state.manager.reserve_to(Sequence([70, 71, 9]), 2)
        self.assertEqual(reservation.allocated_ids, old.ids)
        self.assertIn(recent.hashes[0], state.manager.hash_to_block_id)
        self.assertNotIn(old.hashes[0], state.manager.hash_to_block_id)
        self.assert_page(state.cache.pages[old.keys[0]], 17)
        state.manager.rollback(reservation)
        self.assert_ownership(state, drained=True)

    def test_gpu_chain_tail_is_replaced_before_its_leading_prefix(self):
        state = self.state()
        chain = self.seed_gpu(state, [1, 2, 3, 4, 5, 6], [11, 12, 13])
        reservation = state.manager.reserve_to(Sequence([70, 71, 9]), 2)
        self.assertEqual(reservation.allocated_ids, (chain.ids[-1],))
        match = state.manager.probe_prefix(chain.seq)
        self.assertEqual(match.block_ids, chain.ids[:2])
        self.assertEqual(match.cached_tokens, 4)
        for bid, value in zip(chain.ids[:2], chain.values[:2]):
            self.assert_page(state.kv[:, :, bid], value)
        state.manager.rollback(reservation)
        self.assert_ownership(state, drained=True)

    def test_gpu_shared_referenced_pages_are_never_victims(self):
        state = self.state(blocks=4)
        active = self.seed_gpu(state, [1, 2, 3, 4], [11, 12], keep=True)
        other = self.seed_gpu(state, [10, 11, 12, 13])
        shared = Sequence(active.seq.token_ids)
        reservation = state.manager.reserve_to(shared, len(shared),
                                                state.manager.probe_prefix(shared))
        self.assertEqual([state.manager.blocks[b].ref_count for b in active.ids], [2, 2])
        extra = state.manager.reserve_to(Sequence([70, 71, 9]), 2)
        self.assertIn(extra.allocated_ids[0], other.ids)
        for bid, hash_value, value in zip(active.ids, active.hashes, active.values):
            self.assertEqual(state.manager.hash_to_block_id[hash_value], bid)
            self.assertEqual(state.manager.blocks[bid].ref_count, 2)
            self.assert_page(state.kv[:, :, bid], value)
        self.assert_ownership(state)
        state.manager.rollback(extra)
        state.manager.rollback(reservation)
        state.manager.deallocate(active.seq)
        self.assert_ownership(state, drained=True)

    def test_cpu_duplicate_yields_to_sole_leading_prefix_even_when_newer(self):
        state = self.state(capacity=2)
        keeper = self.identities(state, self.observe(state, [1, 2, 9]))[0][0]
        duplicate = self.seed_gpu(state, [10, 11], [27])
        self.host_page(state, keeper, 17)
        self.host_page(state, duplicate.keys[0], 27)
        incoming = self.identities(state, self.observe(state, [70, 71, 9]))[0][0]
        state.kv[:, :, 2].fill_(37)
        state.cache.put(incoming, 2)
        self.assertEqual(set(state.cache.pages), {keeper, incoming})
        self.assert_page(state.cache.pages[keeper], 17)
        self.assert_page(state.cache.pages[incoming], 37)
        self.assertEqual(state.manager.hash_to_block_id[duplicate.hashes[0]], duplicate.ids[0])
        self.assert_page(state.kv[:, :, duplicate.ids[0]], 27)
        self.assert_ownership(state, drained=True)

    def test_cpu_orphan_yields_to_older_sole_leading_prefix(self):
        state = self.state(capacity=2)
        keeper = self.identities(state, self.observe(state, [1, 2, 9]))[0][0]
        missing_parent, orphan = self.identities(state,
            self.observe(state, [10, 11, 12, 13, 9]))[0]
        self.host_page(state, keeper, 17)
        self.host_page(state, orphan, 27)
        self.assertFalse(state.cache.contains(missing_parent))
        incoming = self.identities(state, self.observe(state, [70, 71, 9]))[0][0]
        state.kv[:, :, 0].fill_(37)
        state.cache.put(incoming, 0)
        self.assertEqual(set(state.cache.pages), {keeper, incoming})
        self.assert_page(state.cache.pages[keeper], 17)
        self.assert_ownership(state, drained=True)

    def test_first_use_is_admitted_even_after_many_old_hot_requests(self):
        state = self.state(capacity=1)
        for _ in range(40):
            hot = self.identities(state, self.observe(state, [1, 2, 9]))[0][0]
        state.kv[:, :, 0].fill_(17)
        state.cache.put(hot, 0)
        first_use = self.identities(state, self.observe(state, [70, 71, 9]))[0][0]
        state.kv[:, :, 1].fill_(37)
        state.cache.put(first_use, 1)
        self.assertEqual(list(state.cache.pages), [first_use])
        state.kv[:, :, 1].fill_(-99)
        self.assert_page(state.cache.pages[first_use], 37)
        with patch.object(state.cache, '_copy', wraps=state.cache._copy) as transfer:
            state.cache.put(b'unobserved-prefix', 2)
            transfer.assert_not_called()
        self.assertEqual(list(state.cache.pages), [first_use])

    def test_all_cpu_sources_survive_victim_backups_before_h2d_submission(self):
        state, target, keys, _, victims, spares = self.restore_fixture()
        cache, manager = state.cache, state.manager
        sources = {key: cache.pages[key] for key in keys}
        puts = []
        real_put = cache.put

        def assert_sources():
            self.assertEqual(state.policy.restore_pins, Counter(keys))
            for key, value in zip(keys, (21, 22)):
                self.assertIs(cache.pages[key], sources[key])
                self.assert_page(cache.pages[key], value)

        def checked_put(key, bid):
            assert_sources()
            puts.append((key, len(cache.submissions)))
            real_put(key, bid)
            assert_sources()

        cache.before_load = lambda key, bid: assert_sources()
        with patch.object(cache, 'put', side_effect=checked_put):
            self.assertEqual(manager.restore_host_prefix(target), 2)
        self.assertEqual(puts, [(victims.keys[2], 0), (victims.keys[1], 1)])
        self.assertEqual([key for key, _ in cache.submissions], keys)
        self.assertFalse(state.policy.restore_pins)
        for key, value in zip(victims.keys[1:], victims.values[1:]):
            self.assert_page(cache.pages[key], value)
        self.assertTrue(all(key not in cache.pages for key in spares))
        self.assertEqual(cache.counters['d2h_bytes'], 2 * cache.page_bytes)
        for (_, bid), value in zip(cache.submissions, (21, 22)):
            self.assert_page(state.kv[:, :, bid], value)
            self.assertEqual(manager.blocks[bid].ref_count, 1)
        self.assertEqual(manager._host_load_leases, {bid for _, bid in cache.submissions})
        self.assert_ownership(state)
        # Attach and cancel a real reservation while copies are incomplete. The
        # transfer leases must retain those pages after request references go.
        cache.before_load = None
        reservation = manager.reserve_to(target, len(target), manager.probe_prefix(target))
        manager.rollback(reservation)
        manager.reap_host_loads()
        self.assertEqual(manager.used_block_ids, manager._host_load_leases)
        self.assertEqual(len(manager.used_block_ids), 2)
        manager.reap_host_loads(blocking=True)
        self.assert_ownership(state, drained=True)

    def test_deferred_restore_backs_up_old_logical_bytes_before_each_h2d(self):
        state, target, keys, hashes, victims, _ = self.restore_fixture()
        cache, manager = state.cache, state.manager
        manager.defer_host_stores = True  # PlannedScheduler also sets this for legacy transfers.
        old_pages = {bid: (key, value) for bid, key, value in
                     zip(victims.ids, victims.keys, victims.values)}
        sources = {key: cache.pages[key] for key in keys}
        copies = []
        real_put = cache.put

        def assert_sources():
            self.assertEqual(state.policy.restore_pins, Counter(keys))
            for key, value in zip(keys, (21, 22)):
                self.assertIs(cache.pages[key], sources[key])
                self.assert_page(cache.pages[key], value)

        def checked_put(key, bid):
            assert_sources()
            old_key, old_value = old_pages[bid]
            self.assertEqual(key, old_key)
            self.assertEqual(manager._pending_host_stores.get(bid), old_key)
            self.assertEqual(manager.blocks[bid].ref_count, 1)
            self.assertNotIn(bid, manager._host_load_leases)
            # Snapshot identity comes from before allocation, not the bytes left
            # after H2D. A byte-perfect round trip of a mislabeled page must fail.
            self.assert_page(state.kv[:, :, bid], old_value)
            real_put(key, bid)
            self.assert_page(cache.pages[old_key], old_value)
            copies.append(('d2h', old_key, bid))
            assert_sources()

        def before_load(key, bid):
            assert_sources()
            old_key, old_value = old_pages[bid]
            self.assertFalse(manager._pending_host_stores,
                             'victim backup must finish before H2D overwrites its GPU page')
            self.assertEqual(copies[-1], ('d2h', old_key, bid))
            self.assert_page(cache.pages[old_key], old_value)
            self.assert_page(state.kv[:, :, bid], old_value)
            copies.append(('h2d', key, bid))

        cache.before_load = before_load
        with patch.object(cache, 'put', side_effect=checked_put):
            self.assertEqual(manager.restore_host_prefix(target, progressive=True), 2)
        self.assertEqual(copies, [
            ('d2h', victims.keys[2], victims.ids[2]), ('h2d', keys[0], victims.ids[2]),
            ('d2h', victims.keys[1], victims.ids[1]), ('h2d', keys[1], victims.ids[1])])
        self.assertEqual(cache.counters['d2h_bytes'], 2 * cache.page_bytes)
        self.assertFalse(manager._pending_host_stores)
        self.assertFalse(state.policy.restore_pins)
        destinations = {bid for _, bid in cache.submissions}
        self.assertEqual(manager._host_load_leases, destinations)
        self.assertEqual(set(cache.inflight), destinations)
        for (key, bid), hash_value, value in zip(cache.submissions, hashes, (21, 22)):
            self.assertEqual(manager.blocks[bid].host_key, key)
            self.assertEqual(manager.hash_to_block_id[hash_value], bid)
            self.assertEqual(manager.blocks[bid].ref_count, 1)
            self.assert_page(state.kv[:, :, bid], value)
        manager.reap_host_loads()  # Fake events are incomplete until the blocking drain.
        self.assertEqual(manager.used_block_ids, destinations)
        self.assert_ownership(state)
        manager.reap_host_loads(blocking=True)
        self.assert_ownership(state, drained=True)

    def test_deferred_backup_failure_precedes_h2d_and_engine_discard_drains(self):
        for successful_loads in (0, 1):
            with self.subTest(successful_loads=successful_loads):
                state, target, keys, hashes, victims, _ = self.restore_fixture()
                cache, manager = state.cache, state.manager
                manager.defer_host_stores = True
                old_pages = {bid: (key, value) for bid, key, value in
                             zip(victims.ids, victims.keys, victims.values)}
                sources = {key: cache.pages[key] for key in keys}
                failed_bid = victims.ids[2 - successful_loads]
                failed_key, failed_value = old_pages[failed_bid]
                puts, failed_copies = [], []
                real_put = cache.put

                def fail_backup(destination, source, direction):
                    self.assertEqual(direction, 'd2h')
                    self.assert_page(source, failed_value)
                    failed_copies.append(failed_bid)
                    raise RuntimeError('injected deferred backup failure before H2D')

                def checked_put(key, bid):
                    self.assertEqual(state.policy.restore_pins, Counter(keys))
                    self.assertEqual(key, old_pages[bid][0])
                    self.assertEqual(manager._pending_host_stores.get(bid), key)
                    self.assert_page(state.kv[:, :, bid], old_pages[bid][1])
                    self.assertEqual(manager.blocks[bid].ref_count, 1)
                    puts.append((key, bid, len(cache.submissions)))
                    if bid == failed_bid:
                        with patch.object(cache, '_copy', side_effect=fail_backup):
                            real_put(key, bid)
                    else:
                        real_put(key, bid)

                with patch.object(cache, 'put', side_effect=checked_put):
                    with self.assertRaisesRegex(RuntimeError, 'deferred backup failure before H2D'):
                        manager.restore_host_prefix(target, progressive=True)
                self.assertEqual(failed_copies, [failed_bid])
                self.assertEqual(len(puts), successful_loads + 1)
                self.assertEqual(puts[-1], (failed_key, failed_bid, successful_loads))
                self.assertEqual(len(cache.submissions), successful_loads)
                self.assertNotIn(failed_bid, cache.inflight)
                self.assertNotIn(failed_bid, manager.used_block_ids)
                self.assertEqual(manager.blocks[failed_bid].ref_count, 0)
                self.assertEqual(manager.blocks[failed_bid].hash, -1)
                self.assertNotIn(hashes[successful_loads], manager.hash_to_block_id)
                self.assertNotIn(failed_key, cache.pages)
                self.assertFalse(state.policy.restore_pins)
                self.assertFalse(target.block_table)
                self.assertEqual(target.num_cached_tokens, 0)
                self.assertEqual(manager.probe_prefix(target).cached_tokens, 2 * successful_loads)
                self.assertFalse(manager._reservations)  # Restore failed before planning began.
                for key, value in zip(keys, (21, 22)):
                    self.assertIs(cache.pages[key], sources[key])
                    self.assert_page(cache.pages[key], value)
                # A failed put is still pending. LLMEngine._step_planned discards
                # it on error; retrying that promise after page reuse is unsafe.
                self.assertEqual(manager._pending_host_stores, {failed_bid: failed_key})
                manager.discard_host_stores()
                self.assertFalse(manager._pending_host_stores)
                self.assertEqual(cache.counters['deferred_store_drops'], 1)
                destinations = {bid for _, bid in cache.submissions}
                self.assertEqual(manager._host_load_leases, destinations)
                self.assertEqual(set(cache.inflight), destinations)
                manager.reap_host_loads()
                self.assertEqual(manager.used_block_ids, destinations)
                self.assert_ownership(state)
                manager.reap_host_loads(blocking=True)
                self.assert_ownership(state, drained=True)

    def test_copy_failures_unpin_every_source_and_release_gpu_references(self):
        for direction, successful_loads in (('d2h', 0), ('h2d', 0), ('h2d', 1)):
            with self.subTest(direction=direction, successful_loads=successful_loads):
                state, target, keys, hashes, _, _ = self.restore_fixture()
                cache, manager = state.cache, state.manager
                real_copy = cache._copy
                old_index = dict(manager.hash_to_block_id)

                def failing_copy(destination, source, kind):
                    if kind == direction and len(cache.submissions) == successful_loads:
                        raise RuntimeError('injected CPU copy failure')
                    real_copy(destination, source, kind)

                with patch.object(cache, '_copy', side_effect=failing_copy):
                    with self.assertRaisesRegex(RuntimeError, 'injected CPU copy failure'):
                        manager.restore_host_prefix(target)
                self.assertFalse(state.policy.restore_pins)
                self.assertFalse(target.block_table)
                self.assertEqual(target.num_cached_tokens, 0)
                self.assertEqual(len(manager._host_load_leases), successful_loads)
                self.assertEqual(manager.probe_prefix(target).cached_tokens, 2 * successful_loads)
                self.assertNotIn(hashes[successful_loads], manager.hash_to_block_id)
                if direction == 'd2h':
                    self.assertEqual(manager.hash_to_block_id, old_index)
                for key, value in zip(keys, (21, 22)):
                    self.assert_page(cache.pages[key], value)
                self.assert_ownership(state)
                manager.reap_host_loads(blocking=True)
                self.assert_ownership(state, drained=True)

    def test_async_put_excludes_inflight_sources_after_request_pins_are_gone(self):
        state, target, keys, _, _, _ = self.restore_fixture()
        self.assertEqual(state.manager.restore_host_prefix(target), 2)
        self.assertFalse(state.policy.restore_pins)
        # All capacity is now held by the two in-flight sources. No D2H copy
        # may overwrite them, even for an otherwise admissible new prefix.
        for key in list(state.cache.pages):
            if key not in keys:
                del state.cache.pages[key]
        state.cache.capacity = 2
        incoming = self.identities(state, self.observe(state, [70, 71, 9]))[0][0]
        with patch.object(state.cache, '_copy', wraps=state.cache._copy) as transfer:
            state.cache.put(incoming, 0)
            transfer.assert_not_called()
        self.assertEqual(set(state.cache.pages), set(keys))
        state.manager.reap_host_loads(blocking=True)
        self.assert_ownership(state, drained=True)

    def test_restore_pins_later_gpu_hit_before_allocating_missing_root(self):
        state = self.state(asynchronous=True)
        chain = self.seed_gpu(state, [1, 2, 3, 4], [21, 22])
        state.cache.put(chain.keys[0], chain.ids[0])
        state.manager._remove_prefix_index(chain.ids[0])
        target = self.observe(state, chain.seq.token_ids)
        # Fill both the unused page and the root's now-unindexed page, so the
        # oldest eligible GPU identity would be the later hit unless pinned.
        self.seed_gpu(state, [30, 31], [31])
        self.seed_gpu(state, [40, 41], [41])
        self.assertEqual(state.policy.select_gpu_victim(state.manager.prefix_cache_lru.items()),
                         (chain.hashes[1], chain.ids[1]))
        self.assertEqual(state.manager.restore_host_prefix(target), 1)
        self.assertEqual(state.manager.hash_to_block_id[chain.hashes[1]], chain.ids[1])
        self.assert_page(state.kv[:, :, chain.ids[1]], 22)
        self.assertEqual(state.manager.probe_prefix(target).cached_tokens, 4)
        state.manager.reap_host_loads(blocking=True)
        self.assert_ownership(state, drained=True)

    def test_victim_iterators_are_bounded_and_free_space_needs_no_scan(self):
        state = self.state(blocks=9, capacity=9)
        chain = self.seed_gpu(state, list(range(18)))

        def bounded(items, seen):
            for item in items[:8]:
                seen.append(item)
                yield item
            raise AssertionError('victim iterator read beyond scan limit')

        gpu_seen, cpu_seen = [], []
        items = list(zip(chain.hashes, chain.ids))
        self.assertEqual(state.policy.select_gpu_victim(bounded(items, gpu_seen)), items[7])
        for key in chain.keys:
            self.host_page(state, key, 1)
        accepted, victims = state.policy.plan_admission([b'new'],
            bounded(chain.keys, cpu_seen), 0)
        self.assertEqual((accepted, victims), ([b'new'], [chain.keys[7]]))
        self.assertEqual((len(gpu_seen), len(cpu_seen)), (8, 8))
        self.assertEqual(state.policy.counters['gpu_victim_keys_scanned'], 8)
        self.assertEqual(state.policy.counters['victim_keys_scanned'], 8)
        self.assertEqual(state.policy.plan_admission([b'new'], bounded([], []), 1),
                         ([b'new'], []))

    def test_request_observation_is_once_per_request_and_history_stays_bounded(self):
        state = self.state(blocks=2, document=calibration(history_capacity=4))
        old = self.seed_gpu(state, [1, 2, 3, 4])
        before = copy.deepcopy(state.policy.history)
        for _ in range(20):
            state.manager.observe_cache_request(old.seq)
        self.assertEqual(state.policy.history, before)
        self.assertEqual(state.policy.epoch, 1)
        for i in range(100):
            current = self.observe(state, [100 + 2 * i, 101 + 2 * i, 9])
            self.assertLessEqual(len(state.policy.history), 4)
        self.assertEqual(len(state.policy.history), 4)
        self.assertFalse(set(old.keys) & state.policy.history.keys())
        self.assertFalse(state.policy.admit(old.keys[0]))
        self.assertTrue(state.policy.admit(self.identities(state, current)[0][0]))
        # History expiry must not strand a valid, unreferenced GPU page.
        reservation = state.manager.reserve_to(Sequence([800, 801, 9]), 2)
        self.assertIn(reservation.allocated_ids[0], old.ids)
        state.manager.rollback(reservation)
        self.assert_ownership(state, drained=True)

    def test_joint_restore_keeps_original_cost_model_and_backlog_decisions(self):
        document = calibration()
        joint = JointResidencyPolicy(document)
        legacy = CacheCostPolicy(CacheCostModel(copy.deepcopy(document['cost_model'])))
        for arguments, expected in (((9, 8, 0, 4, 0), True),
                                    ((9, 8, 0, 4, 10000), False),
                                    ((9, 8, 8, 1, 0), False)):
            with self.subTest(arguments=arguments):
                decision = joint.restore(*arguments)
                self.assertEqual(decision, expected)
                self.assertEqual(decision, legacy.restore(*arguments))
        slow = calibration()
        slow['cost_model']['transfer_ms']['h2d'] = [[1, 1000], [4, 4000]]
        state, target, _, _, _, _ = self.restore_fixture(slow)
        index, free = dict(state.manager.hash_to_block_id), list(state.manager.free_block_ids)
        with patch.object(state.cache, 'put', wraps=state.cache.put) as store, \
                patch.object(state.cache, 'load', wraps=state.cache.load) as load:
            self.assertEqual(state.manager.restore_host_prefix(target), 0)
            store.assert_not_called()
            load.assert_not_called()
        self.assertEqual(state.manager.hash_to_block_id, index)
        self.assertEqual(list(state.manager.free_block_ids), free)
        self.assertEqual(state.cache.counters['cost_restore_skips'], 1)
        self.assert_ownership(state, drained=True)

    def test_invalid_metadata_bounds_and_mismatched_identities_are_rejected(self):
        for options in (dict(victim_scan_limit=7), dict(victim_scan_limit=65),
                        dict(victim_scan_limit=True), dict(history_capacity=0),
                        dict(history_capacity=4097), dict(history_capacity=1.5),
                        dict(schema='unknown')):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    JointResidencyPolicy(calibration(**options))
        policy = JointResidencyPolicy(calibration())
        with self.assertRaises(ValueError):
            policy.observe(1, [b'a', b'b'], 4, gpu_hashes=[123])
        self.assertFalse(policy.history)
        self.assertEqual(policy.epoch, 0)

    def config_fixture(self, folder):
        root = Path(folder)
        (root / 'config.json').write_text('{}', encoding='utf-8')
        document = calibration()
        document['cost_model']['identity'].update(
            model=str(root.resolve()), config_sha256=hashlib.sha256(b'{}').hexdigest(),
            block_size=256, linear_backend='torch', numerical_backend='default',
            rmsnorm_backend='torch', model_dtype='checkpoint', transport='legacy')
        path, old_path = root / 'joint.json', root / 'old-cost.json'
        path.write_text(json.dumps(document), encoding='utf-8')
        old_path.write_text(json.dumps(document['cost_model']), encoding='utf-8')
        options = dict(host_kv_joint_residency_path=str(path), host_kv_cache_bytes=4096,
                       host_kv_async_prefetch=True, speculative_policy='off',
                       prefill_chunk_size=2, max_num_batched_tokens=2)
        return options, old_path

    def test_valid_joint_config_loads_without_cuda_and_validates_cost_identity(self):
        from nanovllm.config import Config
        hf = SimpleNamespace(max_position_embeddings=4096, dtype=torch.bfloat16,
                             model_type='qwen3', attention_bias=False)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict('os.environ', CUDA_VISIBLE_DEVICES=''), \
                patch('nanovllm.config.AutoConfig.from_pretrained', return_value=hf) as auto:
            options, _ = self.config_fixture(folder)
            config = Config(folder, **options)
            auto.assert_called_once_with(folder)
            self.assertEqual(config.runtime_summary()['host_kv_joint_residency_path'],
                             options['host_kv_joint_residency_path'])
            auto.reset_mock()
            with self.assertRaisesRegex(ValueError, 'does not match'):
                Config(folder, **dict(options, prefill_chunk_size=1))
            auto.assert_not_called()

    def test_joint_mixed_config_is_accepted_without_cuda(self):
        from nanovllm.config import Config
        hf = SimpleNamespace(max_position_embeddings=4096, dtype=torch.bfloat16,
                             model_type='qwen3', attention_bias=False)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict('os.environ', CUDA_VISIBLE_DEVICES=''), \
                patch('nanovllm.config.AutoConfig.from_pretrained', return_value=hf) as auto:
            options, _ = self.config_fixture(folder)
            config = Config(folder, **dict(options, batching_mode='mixed'))
            auto.assert_called_once_with(folder)
            summary = config.runtime_summary()
            self.assertEqual(summary['batching_mode'], 'mixed')
            self.assertEqual(summary['host_kv_joint_residency_path'],
                             options['host_kv_joint_residency_path'])
            self.assertTrue(summary['packed_supported'])
            self.assertTrue(summary['enable_prefix_caching'])
            self.assertTrue(summary['host_kv_async_prefetch'])
            self.assertEqual(summary['host_kv_transfer_backend'], 'legacy')

    def test_invalid_joint_flag_combinations_fail_before_model_or_cuda_initialization(self):
        from nanovllm.config import Config
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict('os.environ', CUDA_VISIBLE_DEVICES=''), \
                patch('nanovllm.config.AutoConfig.from_pretrained') as auto:
            options, old_path = self.config_fixture(folder)
            cases = (
                (dict(host_kv_cache_bytes=0), 'host KV cache budget'),
                (dict(host_kv_async_prefetch=False), 'joint residency requires'),
                (dict(host_kv_transfer_backend='batched'), 'joint residency requires'),
                (dict(host_kv_cost_model_path=str(old_path)), 'joint residency requires'),
                (dict(tensor_parallel_size=2), 'host KV cache requires'),
                (dict(enable_prefix_caching=False), 'host KV cache requires'),
                (dict(kv_cache_dtype='fp8'), 'host KV cache requires'),
                (dict(quantization='awq'), 'dense non-speculative'),
                (dict(speculative_policy='fixed'), 'dense non-speculative'),
                (dict(speculative_policy='adaptive', batching_mode='mixed'), 'dense non-speculative'),
                (dict(host_kv_joint_residency_path='relative.json'), 'absolute existing'),
                (dict(host_kv_joint_residency_path=str(Path(folder) / 'missing.json')), 'absolute existing'),
                (dict(host_kv_joint_residency_path=123), 'absolute existing'))
            for changed, message in cases:
                with self.subTest(changed=changed):
                    with self.assertRaisesRegex(ValueError, message):
                        Config(folder, **dict(options, **changed))
                    auto.assert_not_called()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--output', type=Path, help='Optional JSON summary path')
    options, unittest_args = parser.parse_known_args()
    program = unittest.main(module=__name__, argv=[sys.argv[0], *unittest_args], exit=False)
    result = program.result
    passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    report = dict(status='PASS' if passed else 'FAIL', device='cpu',
                  tests_run=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), skipped=len(result.skipped))
    if options.output is not None:
        options.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
