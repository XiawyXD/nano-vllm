"""纯 CPU 验收：事务 KV、混合计划、公平调度和验证提交，不加载模型。"""
from collections import OrderedDict
import json
import pickle
import random
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.execution_plan import ExecutionRole, ExecutionPlan, ExecutionItem
from nanovllm.engine.planned_scheduler import PlannedScheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def expect(error, action):
    try:
        action()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def config(*, blocks=32, block_size=4, budget=8, seqs=4, chunk=2, wait_steps=8,
           min_decodes=0, max_prefill_steps=0, low_load_mode="separate"):
    Sequence.block_size = block_size
    return SimpleNamespace(max_num_seqs=seqs, max_num_batched_tokens=budget,
                           prefill_chunk_size=chunk, max_prefill_steps=max_prefill_steps, eos=-1,
                           kvcache_block_size=block_size, num_kvcache_blocks=blocks,
                           enable_prefix_caching=True, max_model_len=2048,
                           prefill_max_wait_steps=wait_steps,
                           mixed_min_active_decodes=min_decodes,
                           mixed_low_load_mode=low_load_mode)


def sequence(tokens, maximum=4, *, speculative=False):
    return Sequence(tokens, SamplingParams(temperature=0, max_tokens=maximum,
                    speculative_ngram_size=2 if speculative else 0,
                    speculative_max_draft_tokens=100 if speculative else 4))


def invariant(manager, seqs):
    """从所有请求的真实页表重新计算引用计数，不能只看 free+used 总数。"""
    counts = [0] * len(manager.blocks)
    for seq in seqs:
        assert len(seq.block_table) == len(set(seq.block_table))
        assert 0 <= seq.num_cached_tokens <= len(seq.block_table) * manager.block_size
        for block in seq.block_table:
            counts[block] += 1
    assert counts == [block.ref_count for block in manager.blocks]
    free = list(manager.free_block_ids)
    assert len(free) == len(set(free))
    assert set(free).isdisjoint(manager.used_block_ids)
    assert set(free) | manager.used_block_ids == set(range(len(manager.blocks)))
    assert manager.used_block_ids == {index for index, count in enumerate(counts) if count}
    for hash_value, block_id in manager.hash_to_block_id.items():
        block = manager.blocks[block_id]
        assert block.hash == hash_value and len(block.token_ids) == manager.block_size


def snapshot(seq):
    return (tuple(seq.token_ids), tuple(seq.block_table), seq.num_cached_tokens, seq.num_scheduled_tokens)


def check_transactions():
    config(block_size=4)
    manager = BlockManager(6, 4)
    seq = sequence(list(range(20)))
    reservation = manager.reserve_to(seq, 4)
    assert len(seq.block_table) == 1 and len(seq) == 20 and seq.num_cached_tokens == 0
    manager.commit(reservation, 4)
    assert len(manager.hash_to_block_id) == 1
    before = snapshot(seq)
    reservation = manager.reserve_to(seq, 8)
    assert len(seq.block_table) == 2
    manager.rollback(reservation)
    manager.rollback(reservation)
    assert snapshot(seq) == before
    expect(MemoryError, lambda: manager.reserve_to(seq, 28))
    assert snapshot(seq) == before
    invariant(manager, [seq])

    # 共享已提交完整页，变动尾页仍独占；rollback 恢复 prefix 引用。
    shared = sequence(list(range(9)))
    match = manager.probe_prefix(shared)
    assert match.cached_tokens == 4
    reservation = manager.reserve_to(shared, 5, match)
    assert manager.blocks[seq.block_table[0]].ref_count == 2
    assert manager.blocks[shared.block_table[-1]].ref_count == 1
    manager.rollback(reservation)
    assert not shared.block_table and manager.blocks[seq.block_table[0]].ref_count == 1
    invariant(manager, [seq, shared])

    reservation = manager.reserve_to(seq, 5)
    manager.commit(reservation, 5)
    tail = manager.blocks[seq.block_table[-1]]
    tail.ref_count = 2
    expect(RuntimeError, lambda: manager.reserve_to(seq, 6))
    tail.ref_count = 1
    invariant(manager, [seq])


def check_prefix_fix_and_rollback():
    config(block_size=2)
    manager = BlockManager(3, 2)
    seed = sequence([1, 2, 3, 4])
    manager.commit(manager.reserve_to(seed, 4), 4)
    manager.deallocate(seed)
    target = sequence([1, 2, 3, 4, 5])
    match = manager.probe_prefix(target)
    assert match.cached_tokens == 4
    reservation = manager.reserve_to(target, 5, match)
    # 必须先占住两页缓存，再分配唯一空页，不能淘汰刚命中的 prefix。
    assert tuple(target.block_table[:2]) == match.block_ids
    assert len(manager.hash_to_block_id) == 2
    manager.rollback(reservation)
    invariant(manager, [seed, target])
    assert len(manager.hash_to_block_id) == 2


def check_failed_hash_commit_preserves_prefix():
    """跨多个完整页的 commit 中途失败，不得索引先前的可写尾页。"""
    config(block_size=4)
    manager = BlockManager(8, 4)
    seq = sequence(list(range(16)))
    manager.commit(manager.reserve_to(seq, 3), 3)
    old_block_id = seq.block_table[0]
    old_hash = manager.blocks[old_block_id].hash
    old_tokens = manager.blocks[old_block_id].token_ids
    old_index = dict(manager.hash_to_block_id)
    old_lru = tuple(manager.prefix_cache_lru.items())

    reservation = manager.reserve_to(seq, 12)
    original_hash = manager.compute_hash
    calls = 0

    def fail_second_hash(token_ids, prefix=-1):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-block hash failure")
        return original_hash(token_ids, prefix)

    manager.compute_hash = fail_second_hash
    try:
        expect(RuntimeError, lambda: manager.commit(reservation, 12))
    finally:
        manager.compute_hash = original_hash
        manager.rollback(reservation)

    assert calls == 2
    assert seq.num_cached_tokens == 3 and seq.num_scheduled_tokens == 0
    assert seq.block_table == [old_block_id]
    assert manager.blocks[old_block_id].hash == old_hash
    assert manager.blocks[old_block_id].token_ids == old_tokens
    assert manager.hash_to_block_id == old_index
    assert tuple(manager.prefix_cache_lru.items()) == old_lru
    invariant(manager, [seq])


def check_failed_prefix_publication_preserves_state():
    """索引/LRU 写入中途失败后，旧尾页和被覆盖的 active prefix 都要恢复。"""
    class FailPersistentDict(dict):
        def __init__(self, values, fail_at):
            super().__init__(values)
            self.fail_at = fail_at
            self.calls = 0

        def __setitem__(self, key, value):
            self.calls += 1
            if self.calls >= self.fail_at:
                raise MemoryError("injected prefix-index write failure")
            return super().__setitem__(key, value)

    class FailPersistentOrderedDict(OrderedDict):
        def __init__(self, values, fail_at):
            self.armed = False
            super().__init__(values)
            self.fail_at = fail_at
            self.calls = 0
            self.armed = True

        def __setitem__(self, key, value):
            if not self.armed:
                return super().__setitem__(key, value)
            self.calls += 1
            if self.calls >= self.fail_at:
                raise MemoryError("injected prefix-LRU write failure")
            return super().__setitem__(key, value)

    def state(manager, seq):
        return (
            tuple(seq.block_table), seq.num_cached_tokens, seq.num_scheduled_tokens,
            tuple((block.ref_count, block.hash, tuple(block.token_ids), block.host_key)
                  for block in manager.blocks),
            dict(manager.hash_to_block_id), tuple(manager.prefix_cache_lru.items()),
            tuple(sorted(manager.free_block_ids)), tuple(sorted(manager.used_block_ids)),
            tuple(sorted(manager._reservations)),
        )

    for target in ("index", "lru"):
        for fail_at in (1, 2):
            for existing_prefix in (False, True):
                for host_cache in (False, True):
                    config(block_size=4)
                    manager = BlockManager(8, 4)
                    if host_cache:
                        manager.host_cache = SimpleNamespace(
                            key=lambda parent, tokens: (parent, tuple(tokens)))
                    seeded = None
                    if existing_prefix:
                        seeded = sequence(list(range(16)))
                        manager.commit(manager.reserve_to(seeded, 8), 8)
                    seq = sequence(list(range(16)))
                    manager.commit(manager.reserve_to(seq, 3), 3)
                    before = state(manager, seq)
                    reservation = manager.reserve_to(seq, 8)
                    failing = (FailPersistentDict(manager.hash_to_block_id, fail_at)
                               if target == "index" else
                               FailPersistentOrderedDict(manager.prefix_cache_lru, fail_at))
                    if target == "index":
                        manager.hash_to_block_id = failing
                    else:
                        manager.prefix_cache_lru = failing
                    try:
                        expect(MemoryError, lambda: manager.commit(reservation, 8))
                    finally:
                        manager.rollback(reservation)
                    assert failing.calls >= fail_at
                    assert state(manager, seq) == before
                    invariant(manager, [seq] + ([seeded] if seeded is not None else []))
                    # 故障解除后，同一请求仍能正常推进并供后续 prefix 命中。
                    failing.fail_at = 1 << 60
                    manager.commit(manager.reserve_to(seq, 8), 8)
                    probe = sequence(list(range(16)))
                    assert manager.probe_prefix(probe).cached_tokens == 8


def check_mixed_and_progressive():
    planner = PlannedScheduler(config(budget=4, chunk=2))
    first, second = sequence([1, 2, 3], 3), sequence([11, 12, 13, 14, 15], 2)
    planner.add(first)
    plan = planner.schedule_plan()
    assert plan.kind == "P_PREFILL" and not plan.logits_indices
    assert len(first.block_table) == 1 and first.num_cached_tokens == 0
    planner.commit_item(plan.items[0], [])
    plan = planner.schedule_plan()
    planner.commit_item(plan.items[0], [9])
    planner.add(second)
    plan = planner.schedule_plan()
    assert plan.kind == "PD_PACKED"
    assert [item.role for item in plan.items] == [ExecutionRole.PREFILL, ExecutionRole.DECODE]
    assert plan.logits_indices == (2,)
    assert pickle.loads(pickle.dumps(plan)) == plan
    for item in plan.items:
        planner.commit_item(item, [] if not item.sample_eligible else [10])
    assert second.num_cached_tokens == 2 and first.num_cached_tokens == 4
    invariant(planner.block_manager, [first, second])
    for _ in range(10):
        if planner.is_finished():
            break
        plan = planner.schedule_plan()
        for item in plan.items:
            planner.commit_item(item, [20] if item.sample_eligible else [])
        invariant(planner.block_manager, [first, second])
    assert planner.is_finished() and not planner.block_manager.used_block_ids


def check_low_load_mixing_gate():
    """低并发分离轮次并保持旧 prefill 配额；达到门槛才 P+D。"""
    planner = PlannedScheduler(config(blocks=64, budget=8, chunk=None,
                                      min_decodes=2, max_prefill_steps=1))
    first = sequence([1, 2, 3], 4)
    second = sequence([11, 12, 13], 4)
    third = sequence([21, 22, 23], 4)
    planner.add(first)
    plan = planner.schedule_plan()
    assert plan.kind == "P_PREFILL"
    planner.commit_item(plan.items[0], [9])
    assert planner.prefill_steps_since_decode == 1
    planner.add(second)

    # 只有一个 running Decode；先按配额做 D，再给等待请求一轮纯 P。
    plan = planner.schedule_plan()
    assert plan.kind == "D_NORMAL"
    planner.commit_item(plan.items[0], [10])
    assert planner.prefill_steps_since_decode == 0
    plan = planner.schedule_plan()
    assert plan.kind == "P_PREFILL"
    planner.commit_item(plan.items[0], [11])
    assert planner.prefill_steps_since_decode == 1

    # 两个 running Decode 达到门槛；再次到来的 P 可以与 D 同轮。
    planner.add(third)
    plan = planner.schedule_plan()
    assert plan.kind == "PD_PACKED"
    assert sum(item.role is ExecutionRole.DECODE for item in plan.items) == 2
    assert sum(item.role is ExecutionRole.PREFILL for item in plan.items) == 1
    for item in plan.items:
        planner.commit_item(item, [12] if item.sample_eligible else [])
    assert planner.prefill_steps_since_decode == 0
    invariant(planner.block_manager, [first, second, third])
    for seq in (first, second, third):
        planner.cancel(seq)
    assert not planner.block_manager.used_block_ids


    # 旧配置 0 意味着 Prefill 无连续轮次上限；门控启用后不得让 Decode 饿死。
    planner = PlannedScheduler(config(blocks=32, budget=8, chunk=None,
                                      min_decodes=2, max_prefill_steps=0))
    first = sequence([1, 2, 3], 4)
    second = sequence([11, 12, 13], 4)
    planner.add(first)
    plan = planner.schedule_plan()
    planner.commit_item(plan.items[0], [9])
    planner.add(second)
    plan = planner.schedule_plan()
    assert plan.kind == "D_NORMAL"
    planner.commit_item(plan.items[0], [10])
    for seq in (first, second):
        planner.cancel(seq)
    invariant(planner.block_manager, [first, second])
    assert not planner.block_manager.used_block_ids

    # 连续新到达的 prompt 也不能把同一个 Decode 永久压在等待队列后面。
    planner = PlannedScheduler(config(blocks=64, budget=8, chunk=None,
                                      min_decodes=2, max_prefill_steps=0))
    old = sequence([1, 2], 20)
    planner.add(old)
    plan = planner.schedule_plan()
    planner.commit_item(plan.items[0], [9])
    for index in range(4):
        newcomer = sequence([100 + index, 200 + index], 4)
        planner.add(newcomer)
        plan = planner.schedule_plan()
        assert plan.kind == "D_NORMAL" and plan.request_ids == (old.seq_id,)
        planner.commit_item(plan.items[0], [10 + index])
        plan = planner.schedule_plan()
        assert plan.kind == "P_PREFILL" and plan.request_ids == (newcomer.seq_id,)
        planner.commit_item(plan.items[0], [20 + index])
        planner.cancel(newcomer)
        invariant(planner.block_manager, [old, newcomer])
    planner.cancel(old)
    assert not planner.block_manager.used_block_ids

    # 校准候选门槛 9：低于门槛可分离，但达到门槛必须恢复真正 P+D。
    planner = PlannedScheduler(config(blocks=256, budget=64, seqs=16, chunk=None,
                                      min_decodes=9, max_prefill_steps=1))
    running = [sequence([1000 + index, 2000 + index, 3000 + index], 4)
               for index in range(9)]
    for seq in running:
        planner.add(seq)
    plan = planner.schedule_plan()
    assert plan.kind == "P_PREFILL" and len(plan.items) == 9
    for item in plan.items:
        planner.commit_item(item, [9])
    waiting = sequence([7, 8, 9], 4)
    planner.add(waiting)
    plan = planner.schedule_plan()
    assert plan.kind == "PD_PACKED"
    assert sum(item.role is ExecutionRole.DECODE for item in plan.items) == 9
    assert sum(item.role is ExecutionRole.PREFILL for item in plan.items) == 1
    for item in plan.items:
        planner.commit_item(item, [11] if item.sample_eligible else [])
    for seq in running + [waiting]:
        planner.cancel(seq)
    assert not planner.block_manager.used_block_ids


def check_low_load_micro_prefill():
    """八轮公平规则可缩小测试为两轮；到期只插入 1 个 P token。"""
    planner = PlannedScheduler(config(blocks=64, budget=8, chunk=None,
                                      min_decodes=2, wait_steps=2,
                                      low_load_mode="micro"))
    running = sequence([1, 2, 3], 10)
    waiting = sequence([11, 12, 13], 4)
    planner.add(running)
    plan = planner.schedule_plan()
    planner.commit_item(plan.items[0], [9])
    planner.add(waiting)
    for output in (20, 21):
        plan = planner.schedule_plan()
        assert plan.kind == "D_NORMAL" and plan.request_ids == (running.seq_id,)
        planner.commit_item(plan.items[0], [output])
    plan = planner.schedule_plan()
    assert plan.kind == "PD_PACKED" and plan.prefill_tokens == 1
    assert {item.request_id for item in plan.items} == {running.seq_id, waiting.seq_id}
    for item in plan.items:
        planner.commit_item(item, [22] if item.sample_eligible else [])
    assert waiting.num_cached_tokens == 1 and len(waiting.token_ids) == 3
    invariant(planner.block_manager, [running, waiting])
    planner.cancel(running)
    planner.cancel(waiting)
    assert not planner.block_manager.used_block_ids


def check_low_load_legacy_fifo():
    """低负载实验：长队首 prompt 完成前不让后续短 prompt 先 Decode。"""
    planner = PlannedScheduler(config(blocks=512, budget=512, seqs=4, chunk=128,
                                      min_decodes=9, max_prefill_steps=1,
                                      low_load_mode="legacy_fifo"))
    long_request = sequence(list(range(768)), 4)
    short_requests = [sequence([1000 + index] * 96, 4) for index in range(3)]
    for seq in (long_request, *short_requests):
        planner.add(seq)
    for _ in range(5):
        plan = planner.schedule_plan()
        assert plan.kind == "P_PREFILL" and len(plan.items) == 1
        assert plan.items[0].request_id == long_request.seq_id
        assert plan.items[0].query_len == 128 and not plan.items[0].sample_eligible
        planner.commit_item(plan.items[0], [])
        assert len(planner.waiting) == 4 and not planner.running
    plan = planner.schedule_plan()
    assert plan.kind == "P_PREFILL" and plan.prefill_tokens == 416
    assert len(plan.items) == 4 and all(item.sample_eligible for item in plan.items)
    for item in plan.items:
        planner.commit_item(item, [42])
    assert not planner.waiting and len(planner.running) == 4
    plan = planner.schedule_plan()
    assert plan.kind == "D_NORMAL" and len(plan.items) == 4
    planner.rollback_plan()
    for seq in (long_request, *short_requests):
        planner.cancel(seq)
    invariant(planner.block_manager, [long_request, *short_requests])
    assert not planner.block_manager.used_block_ids


def check_verify_and_failure():
    planner = PlannedScheduler(config(block_size=2, blocks=32, budget=8, chunk=None))
    spec, normal = sequence([1, 2, 3], 10, speculative=True), sequence([10, 11], 5)
    planner.add(spec)
    plan = planner.schedule_plan()
    planner.commit_item(plan.items[0], [4])
    planner.add(normal)
    proposed = []

    def provider(seq, limit):
        proposed.append(seq.seq_id)
        return [6, 7, 8][:limit]

    plan = planner.schedule_plan(provider)
    assert plan.kind == "PD_PACKED" and not proposed
    for item in plan.items:
        planner.commit_item(item, [12] if item.request_id == normal.seq_id else [5])
    before = snapshot(spec)
    plan = planner.schedule_plan(provider)
    assert plan.kind == "DV_PACKED" and spec.token_ids == list(before[0])
    assert plan.verification_tokens == 4 and plan.decode_tokens == 1
    verify = plan.items[0]
    assert verify.role is ExecutionRole.VERIFY
    planner.commit_item(verify, [6, 99])
    assert spec.token_ids == [1, 2, 3, 4, 5, 6, 99]
    assert spec.num_cached_tokens == 6 and len(spec.block_table) == 3
    planner.commit_item(plan.items[1], [13])
    assert not any(7 in block.token_ids or 8 in block.token_ids for block in planner.block_manager.blocks if block.hash != -1)
    invariant(planner.block_manager, [spec, normal])
    before = [snapshot(seq) for seq in (spec, normal)]
    plan = planner.schedule_plan(provider)
    planner.rollback_plan()
    assert [snapshot(seq) for seq in (spec, normal)] == before
    invariant(planner.block_manager, [spec, normal])

    # provider 或 stop 判断失败时，未提交 draft/输出不能污染真实历史。
    def broken(*args):
        raise RuntimeError("injected CPU failure")

    expect(RuntimeError, lambda: planner.schedule_plan(broken))
    assert [snapshot(seq) for seq in (spec, normal)] == before
    plan = planner.schedule_plan(allow_verification=False)
    item = plan.items[0]
    history = list(planner.get_sequence(item.request_id).token_ids)
    expect(RuntimeError, lambda: planner.commit_item(item, [31], stop_checker=broken))
    assert planner.get_sequence(item.request_id).token_ids == history
    planner.rollback_plan()
    invariant(planner.block_manager, [spec, normal])


def check_fairness_pressure_and_cancel():
    planner = PlannedScheduler(config(blocks=32, budget=1, seqs=1, chunk=1, wait_steps=2))
    running, waiting = sequence([1], 20), sequence([21, 22], 2)
    planner.add(running)
    planner.commit_item(planner.schedule_plan().items[0], [2])
    planner.add(waiting)
    seen = None
    for step in range(10):
        plan = planner.schedule_plan()
        item = plan.items[0]
        if item.request_id == waiting.seq_id:
            seen = step
        planner.commit_item(item, [42] if item.sample_eligible else [])
        if seen is not None:
            break
    assert seen is not None and seen <= 2
    planner.cancel(running)
    planner.cancel(waiting)
    assert not planner.block_manager.used_block_ids

    planner = PlannedScheduler(config(blocks=3, block_size=2, budget=2, chunk=2))
    oldest, younger = sequence([1, 2, 3], 3), sequence([11, 12, 13], 3)
    planner.add(oldest)
    planner.add(younger)
    completion_order = []
    for _ in range(30):
        if planner.is_finished():
            break
        plan = planner.schedule_plan()
        assert plan.items
        # 本轮 pin 的请求在其他请求的 reserve/preemption 后仍持有相同页表。
        for item in plan.items:
            assert tuple(planner.get_sequence(item.request_id).block_table) == item.block_table
        for item in plan.items:
            seq = planner.get_sequence(item.request_id)
            planner.commit_item(item, [99] if item.sample_eligible else [])
            if seq.is_finished:
                completion_order.append(seq.seq_id)
        invariant(planner.block_manager, [oldest, younger])
    assert planner.is_finished() and completion_order[0] == oldest.seq_id
    assert planner.total_preemptions >= 1
    expect(ValueError, lambda: planner.add(sequence(list(range(8)), 2)))


def check_verify_bound_and_stop():
    planner = PlannedScheduler(config(blocks=512, block_size=2, budget=256, chunk=None, seqs=2))
    seqs = [sequence([base, base + 1], 200, speculative=True) for base in (1000, 1100)]
    for seq in seqs:
        planner.add(seq)
    for item in planner.schedule_plan().items:
        planner.commit_item(item, [5])
    plan = planner.schedule_plan(lambda seq, limit: [6] * limit)
    assert plan.verification_tokens == 128 and plan.kind == "DV_PACKED"
    for item in plan.items:
        outputs = list(item.draft) + [9]
        actual = planner.commit_item(item, outputs, stop_checker=lambda seq: seq.num_completion_tokens == 3)
        assert actual == [6, 6]
    assert planner.is_finished() and not planner.block_manager.used_block_ids
    assert not planner._plan_sequences and not planner._reservations
    invariant(planner.block_manager, seqs)

    planner = PlannedScheduler(config(blocks=512, block_size=2, budget=256, chunk=None, seqs=129))
    large = [sequence([10000 + index], 3, speculative=True) for index in range(129)]
    for seq in large:
        planner.add(seq)
    for item in planner.schedule_plan().items:
        planner.commit_item(item, [5])
    proposed = []
    plan = planner.schedule_plan(lambda seq, limit: proposed.append(seq.seq_id) or [6] * limit)
    assert plan.num_tokens == 129 and not plan.has_verification and not proposed
    planner.rollback_plan()
    for seq in large:
        planner.cancel(seq)
    invariant(planner.block_manager, large)


def check_randomized_pressure():
    """固定随机种子做状态机压力：小物理池、随机回滚/接受率，每轮重算引用。"""
    for seed in range(200):
        # 五组复用相同 40 个请求序列：关闭、门槛 2/9、micro、legacy FIFO。
        mode_index = seed // 40
        gate = (0, 2, 9, 9, 9)[mode_index]
        low_load_mode = ("separate", "separate", "separate", "micro", "legacy_fifo")[mode_index]
        rng = random.Random(seed % 40)
        block_size = rng.choice([2, 4])
        blocks = rng.randint(3, 8)
        planner = PlannedScheduler(config(blocks=blocks, block_size=block_size,
                                           budget=rng.randint(1, 12), seqs=rng.randint(1, 5),
                                           chunk=rng.randint(1, 5), min_decodes=gate,
                                           max_prefill_steps=1 if gate else 0,
                                           low_load_mode=low_load_mode))
        planner.prefill_max_wait_steps = 3
        original_preempt = planner._preempt_candidate

        def checked_preempt(protected):
            before = len(planner.last_preempted_ids)
            result = original_preempt(protected)
            assert not set(planner.last_preempted_ids[before:]).intersection(protected)
            return result

        planner._preempt_candidate = checked_preempt
        seqs = []
        for index in range(8):
            maximum = rng.randint(1, min(4, blocks * block_size - 1))
            length = rng.randint(1, min(7, blocks * block_size - maximum + 1))
            seq = sequence([100 * (index + 1) + i for i in range(length)], maximum,
                           speculative=bool(index % 2))
            planner.add(seq)
            seqs.append(seq)
        for step in range(600):
            if planner.is_finished():
                break
            plan = planner.schedule_plan(lambda seq, limit: [51] * min(limit, 3))
            assert plan.items and len(plan.request_ids) == len(set(plan.request_ids))
            for item in plan.items:
                assert tuple(planner.get_sequence(item.request_id).block_table) == item.block_table
            invariant(planner.block_manager, seqs)
            if step % 13 == 0:
                planner.rollback_plan()
                invariant(planner.block_manager, seqs)
                continue
            for item in plan.items:
                if not item.sample_eligible:
                    outputs = []
                elif item.role is ExecutionRole.VERIFY:
                    accepted = rng.randint(0, len(item.draft))
                    outputs = list(item.draft[:accepted]) + [52]
                else:
                    outputs = [52]
                planner.commit_item(item, outputs)
            invariant(planner.block_manager, seqs)
        assert planner.is_finished(), (seed, [(seq.seq_id, len(seq), seq.num_cached_tokens) for seq in seqs])
        assert not planner.block_manager.used_block_ids
        assert not planner._plan_sequences and not planner._active_items and not planner._reservations
        assert not planner.block_manager._reservations


def main():
    checks = [check_transactions, check_prefix_fix_and_rollback,
              check_failed_hash_commit_preserves_prefix,
              check_failed_prefix_publication_preserves_state, check_mixed_and_progressive,
              check_low_load_mixing_gate, check_low_load_micro_prefill,
              check_low_load_legacy_fifo,
              check_verify_and_failure, check_fairness_pressure_and_cancel, check_verify_bound_and_stop,
              check_randomized_pressure]
    for check in checks:
        check()
    print(json.dumps({"status": "PASS", "device": "CPU", "checks": [check.__name__ for check in checks]}, indent=2))


if __name__ == "__main__":
    main()
