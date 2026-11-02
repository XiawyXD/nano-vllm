"""单请求 native draft 的启动期验证图；不在真实请求里 capture。

每个 Q 都是真实 query 数，不 padding token。页表宽度与 K 上界固定，
每次 replay 更新位置、slot、真实 K 长度和物理页表；跨页和拒绝后均不能复用旧元数据。
Transformer 和 LM head 一并录制，输出必须在下次 replay 前消费。
"""
from collections import Counter
from dataclasses import fields

import torch

from nanovllm.utils.context import Context, get_context, set_context
from nanovllm.layers.quant_graph import assert_graph_replay_stream


class NativeVerificationGraphs:
    def __init__(self, max_tokens=8, reserve_bytes=256 * 2**20, *, packed_metadata=False):
        self.max_tokens = max_tokens
        self.reserve_bytes = reserve_bytes
        self.entries = {}
        self.pool = None
        self.counters = Counter()
        self.resident_bytes = 0
        self.packed_metadata = packed_metadata
        self.metadata_packets = {}

    @staticmethod
    def make_packet(query, columns, device):
        """混合整型字段共享 byte storage；避免 int64->int32 转换 kernel。"""
        size = 20 * query + 8 + 4 * columns
        host = torch.zeros(size, dtype=torch.uint8, device="cpu", pin_memory=str(device).startswith("cuda"))
        packet = torch.zeros(size, dtype=torch.uint8, device=device)
        def views(buffer):
            return (buffer[:8*query].view(torch.int64),
                    buffer[8*query:16*query].view(torch.int64),
                    buffer[16*query:20*query].view(torch.int32),
                    buffer[20*query:20*query+8].view(torch.int32),
                    buffer[20*query+8:].view(torch.int32).view(1,columns))
        return dict(host=host, packet=packet, host_views=views(host), device_views=views(packet),
                    event=None, stream=None)

    def prepare_cpu(self, query, ids, positions, slots, pages, end):
        """调用方已完成 plan 边界/页号验证；此处只接受完整单请求 VERIFY。"""
        record = self.metadata_packets.get(query)
        if record is None:
            return False
        assert_graph_replay_stream(self, record["packet"].device)
        columns = record["host_views"][-1].size(1)
        if len(pages)>columns:
            raise ValueError("page table exceeds native graph capacity")
        stream = torch.cuda.current_stream()
        if record["stream"] not in (None, stream.cuda_stream):
            raise RuntimeError("native metadata packet cannot be shared across streams")
        record["stream"] = stream.cuda_stream
        event = record["event"]
        if event is not None and not event.query():
            event.synchronize()
            self.counters["metadata_host_waits"] += 1
        cpu_ids, cpu_pos, cpu_slots, cpu_k, cpu_pages = record["host_views"]
        cpu_ids.numpy()[:] = ids
        cpu_pos.numpy()[:] = positions
        cpu_slots.numpy()[:] = slots
        cpu_k.numpy()[:] = [0,end]
        cpu_pages.zero_()
        cpu_pages.numpy()[0,:len(pages)] = pages
        record["packet"].copy_(record["host"], non_blocking=True)
        if event is None:
            record["event"] = event = torch.cuda.Event()
        event.record(stream)
        self.counters["metadata_packet_h2d_copies"] += 1
        self.counters["metadata_packet_h2d_bytes"] += record["host"].numel()
        return True

    def replay_prepared(self, query):
        if query not in self.metadata_packets:
            raise ValueError("query has no prepared native metadata packet")
        assert_graph_replay_stream(self, self.metadata_packets[query]["packet"].device)
        self.entries[query][0].replay()
        self.counters["replays"] += 1
        self.counters["packed_metadata_replays"] += 1
        return self.entries[query][-1]

    @staticmethod
    def _context(context):
        set_context(**{field.name: getattr(context, field.name) for field in fields(Context)})

    @torch.inference_mode()
    def warmup(self, model, *, capacity, block_size, query_budget):
        previous = get_context()
        device = next(model.parameters()).device
        page_columns = (capacity + block_size - 1) // block_size
        # The startup dummy uses only page zero. No live request exists yet.
        maximum = min(self.max_tokens + 1, query_budget, capacity, block_size)
        before = torch.cuda.memory_allocated(device)
        try:
            for query in range(maximum, 1, -1):
                ids = torch.zeros(query, dtype=torch.int64, device=device)
                positions = torch.arange(query, dtype=torch.int64, device=device)
                context = Context(is_prefill=True,
                    cu_seqlens_q=torch.tensor([0, query], dtype=torch.int32, device=device),
                    cu_seqlens_k=torch.tensor([0, query], dtype=torch.int32, device=device),
                    max_seqlen_q=query, max_seqlen_k=capacity,
                    slot_mapping=positions.to(torch.int32),
                    block_tables=torch.zeros((1, page_columns), dtype=torch.int32, device=device),
                    return_all_logits=True)
                if self.packed_metadata:
                    record = self.make_packet(query, page_columns, device)
                    ids, positions, context.slot_mapping, context.cu_seqlens_k, context.block_tables = record["device_views"]
                    positions.copy_(torch.arange(query, device=device))
                    context.slot_mapping.copy_(positions)
                    context.cu_seqlens_k[1] = query
                    self.metadata_packets[query] = record
                self._context(context)
                for _ in range(2):
                    model.compute_logits(model(ids, positions))
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                from nanovllm.layers.quant_graph import capture_model_graph
                with capture_model_graph(model, graph, pool=self.pool):
                    logits = model.compute_logits(model(ids, positions))
                if self.pool is None:
                    self.pool = graph.pool()
                self.entries[query] = (graph, ids, positions, context, logits)
                self.counters["startup_captures"] += 1
            torch.cuda.synchronize(device)
            self.resident_bytes = max(0, torch.cuda.memory_allocated(device) - before)
            if self.resident_bytes > self.reserve_bytes:
                raise RuntimeError("native verification graph exceeded reserved memory; use eager verification")
        finally:
            self._context(previous)

    @torch.inference_mode()
    def run(self, input_ids, positions, context):
        assert_graph_replay_stream(self, input_ids.device)
        query = input_ids.numel()
        entry = self.entries.get(query)
        groups = context.attention_groups
        if groups is None:
            metadata = context
            valid = (context.is_prefill and context.return_all_logits
                     and context.cu_seqlens_q is not None and context.cu_seqlens_q.numel() == 2
                     and context.block_tables is not None and context.block_tables.size(0) == 1)
        else:
            valid = (len(groups) == 1 and groups[0].kind == "varlen"
                     and groups[0].start == 0 and groups[0].end == query
                     and groups[0].block_tables.size(0) == 1)
            metadata = groups[0] if valid else None
        if entry is None or not valid:
            self.counters["fallbacks"] += 1
            return None
        graph, ids, pos, static, logits = entry
        columns = metadata.block_tables.size(1)
        if columns > static.block_tables.size(1):
            self.counters["fallbacks"] += 1
            return None
        ids.copy_(input_ids)
        pos.copy_(positions)
        static.slot_mapping.copy_(context.slot_mapping)
        static.cu_seqlens_k.copy_(metadata.cu_seqlens_k)
        static.block_tables.zero_()
        static.block_tables[:, :columns].copy_(metadata.block_tables)
        graph.replay()
        self.counters["replays"] += 1
        return logits

    def stats(self):
        return dict(self.counters, entries=len(self.entries), resident_bytes=self.resident_bytes,
                    reserved_bytes=self.reserve_bytes, max_draft_tokens=self.max_tokens,
                    packed_metadata=self.packed_metadata)

    def close(self):
        self.entries.clear()
        # 正常返回已有 D2H 等待；异常退出仍须等待 pinned H2D 完成才释放 host storage。
        for record in self.metadata_packets.values():
            if record["event"] is not None:
                record["event"].synchronize()
        self.metadata_packets.clear()
        self.pool = None
