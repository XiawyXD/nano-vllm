"""Real asynchronous bytes, joint victim backup, cancellation and exception cleanup."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deferred', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    report = dict(status='RUNNING', guard=ensure_gpu1_available(), checks=[])
    import torch
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.joint_cache import JointResidencyPolicy
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.tiered_cache import AsyncHostPrefixCache
    old = Sequence.block_size
    Sequence.block_size = 256
    cache = None
    try:
        kv = torch.empty((2, 2, 12, 256, 1, 8), dtype=torch.bfloat16, device='cuda')
        manager = BlockManager(12, 256)
        manager.defer_host_stores = a.deferred
        policy = JointResidencyPolicy.load(a.policy, manager)
        page_bytes = kv[:, :, 0].numel()*kv.element_size()
        cache = AsyncHostPrefixCache(kv, 8*page_bytes, 'joint-gpu', policy=policy,
                                     min_restore_tokens=256, max_restore_blocks=8,
                                     max_inflight_blocks=4)
        manager.host_cache = cache
        expected_cpu_values = {}

        def seed(base, length):
            seq = Sequence([base+i % 37 for i in range(length)])
            manager.allocate(seq, 0)
            manager.flush_host_stores()  # same boundary as the real engine before forward
            for i, bid in enumerate(seq.block_table):
                kv[:, :, bid].fill_(base+i)
            seq.num_scheduled_tokens = length
            manager.observe_cache_request(seq)
            manager.hash_blocks(seq)
            parent = None
            for i in range((length-1)//256):
                parent = cache.key(parent, seq.block(i))
                expected_cpu_values[parent] = base+i
            manager.deallocate(seq)
            return seq

        first = seed(100, 1025)
        pressure = [manager._allocate_block() for _ in range(12)]
        manager.flush_host_stores()
        for bid in pressure:
            kv[:, :, bid].fill_(-9)
            manager.blocks[bid].ref_count = 0
            manager._deallocate_block(bid)
        assert len(cache.pages) == 4
        seed(200, 1025)
        seed(300, 1025)
        seed(400, 513)
        before = cache.counters['stores']
        with torch.cuda.stream(cache.transfer_stream):
            torch.cuda._sleep(100_000_000)
        again = Sequence(first.token_ids)
        assert manager.restore_host_prefix(again) == 4
        if a.deferred:
            # Deterministically exercise the ordering observed in the real
            # counterexample: H2D finishes before the later engine flush.
            cache.wait_for_blocks(manager._host_load_leases)
            torch.cuda.current_stream().synchronize()
        # Simulate the later engine boundary too: the old code publishes NEW
        # H2D bytes under the OLD victim key here. Test identity, not just that
        # a copy agrees with the already overwritten physical source.
        manager.flush_host_stores()
        for key, page in cache.pages.items():
            assert torch.equal(page, torch.full_like(page, expected_cpu_values[key])), 'CPU prefix key has bytes from another logical page'
        assert cache.counters['stores'] > before, 'GPU victims were not backed up during restore'
        assert not policy.restore_pins
        protected = set(manager._host_load_leases)
        assert len(protected) == 4
        manager.allocate(again, manager.can_allocate(again))
        manager.flush_host_stores()
        restored = list(again.block_table[:4])
        manager.deallocate(again)  # cancel before the consumer forward
        other = [manager._allocate_block() for _ in range(len(manager.free_block_ids))]
        assert not protected.intersection(other)
        for bid in other:
            manager.blocks[bid].ref_count = 0
            manager._deallocate_block(bid)
        manager.reap_host_loads(blocking=True)
        for i, bid in enumerate(restored):
            assert torch.equal(kv[:, :, bid], torch.full_like(kv[:, :, bid], 100+i))
        assert not manager.used_block_ids and not manager._host_load_leases
        assert all(block.ref_count == 0 for block in manager.blocks)
        report['checks'].append(dict(case='real_restore_backups_and_cancel_lease', status='PASS',
                                     victim_backup_pages=cache.counters['stores']-before))

        # Keep the first submitted H2D alive when the next submission fails.
        manager.reap_host_loads(blocking=True)
        manager.flush_host_stores()
        cache.clear()
        for bid in list(manager.free_block_ids):
            manager._remove_prefix_index(bid)
        pressure = [manager._allocate_block() for _ in range(12)]
        manager.flush_host_stores()
        for bid in pressure:
            manager.blocks[bid].ref_count = 0
            manager._deallocate_block(bid)
        # A fresh self-contained prefix avoids depending on pages evicted above.
        source = seed(500, 513)
        pressure = [manager._allocate_block() for _ in range(12)]
        manager.flush_host_stores()
        for bid in pressure:
            manager.blocks[bid].ref_count = 0
            manager._deallocate_block(bid)
        ordinary_load = cache.load
        submitted = 0
        def fail_second(key, bid):
            nonlocal submitted
            submitted += 1
            if submitted == 2:
                raise RuntimeError('injected second H2D failure')
            return ordinary_load(key, bid)
        with patch.object(cache, 'load', side_effect=fail_second):
            try:
                manager.restore_host_prefix(Sequence(source.token_ids))
            except RuntimeError as error:
                assert 'injected' in str(error)
            else:
                raise AssertionError('injected failure was not propagated')
        assert submitted == 2 and not policy.restore_pins
        manager.reap_host_loads(blocking=True)
        assert not manager.used_block_ids and not manager._host_load_leases
        assert all(block.ref_count == 0 for block in manager.blocks)
        assert cache.stats()['resident_bytes'] <= cache.max_bytes
        report['checks'].append(dict(case='partial_submission_failure_cleanup', status='PASS'))
        report['status'] = 'PASS'
    except BaseException:
        import traceback
        report.update(status='FAIL', traceback=traceback.format_exc())
        raise
    finally:
        if cache is not None:
            cache.clear()
        Sequence.block_size = old
        with a.output.open('x') as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
