"""Opt-in probability-producing wrapper for an existing native DraftRunner.

Construct BEFORE target KV budgeting, after owner.model/KV/greedy startup setup.
No changes to DraftRunner are required in this module. The integration owns the
explicit Config gate, attaches this wrapper, and calls close() before owner.close().
Q1/Q2 graphs capture logits, NEVER greedy argmax; sampling happens outside capture
on the same device/current stream. CPU tests can exercise use_graph=False with a
fake owner. CUDA graph operation still requires separate real-GPU verification.
Only one active request and one stream may use an owner at a time.
"""
from collections import Counter
from dataclasses import dataclass

import torch

from nanovllm.engine.speculative_sampling import (
    make_generators, sample_draft, speculative_correction, validate_sampling_options,
)
from nanovllm.utils.context import set_context, reset_context
from nanovllm.layers.quant_graph import model_graph_workspace, assert_graph_replay_stream


@dataclass
class StochasticProposal:
    token_ids: list[int]
    probabilities: torch.Tensor  # [K,V], owned FP64 q rows actually used for sampling
    ids: torch.Tensor            # [K], owned int64 device tokens, not a graph scratch view


class _LogitsGraph:
    """Startup capture of one real Q width, model + last-row LM head only."""
    @torch.inference_mode()
    def __init__(self, owner, width):
        self.model = owner.model
        self.block_table = owner.block_table
        self.width, self.capacity = width, owner.capacity
        device = owner.device
        self.ids = torch.zeros(width, dtype=torch.int64, device=device)
        self.start = torch.zeros((), dtype=torch.int64, device=device)
        self.offsets = torch.arange(width, dtype=torch.int64, device=device)
        self.cu_q = torch.tensor([0,width], dtype=torch.int32, device=device)
        self.cu_k = torch.tensor([0,width], dtype=torch.int32, device=device)
        self.graph = None
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with model_graph_workspace(self.model, stream):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._forward()
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.logits = self._forward()
                torch.cuda.current_stream(device).wait_stream(stream)
                self.graph = graph
        finally:
            reset_context()

    def _forward(self):
        positions = self.start + self.offsets
        slots = positions.to(torch.int32)
        self.cu_k[1:].copy_(slots[-1:] + 1)
        set_context(True, self.cu_q, self.cu_k, self.width, self.capacity,
                    slots, block_tables=self.block_table)
        return self.model.compute_logits(self.model(self.ids,positions))[-1]

    @torch.inference_mode()
    def run(self, ids, start):
        assert_graph_replay_stream(self, self.ids.device)
        if ids.shape != (self.width,) or ids.dtype != torch.int64 or ids.device != self.ids.device:
            raise ValueError('logits graph requires exact-width device int64 IDs')
        if type(start) is not int or start < 0 or start+self.width > self.capacity:
            raise ValueError('logits graph exceeds draft KV capacity')
        self.ids.copy_(ids)
        self.start.fill_(start)
        self.graph.replay()
        return self.logits

    def close(self):
        self.graph = None
        self.model = self.block_table = None
        self.__dict__.pop('logits',None)


class StochasticDraft:
    """Shared logical cache, private owner KV, request-local independent RNGs.

    propose(history,limit,temperature,seed,request_id) returns one candidate list
    D2H per round and retains q/ids on-device. Feed proposal.ids/q plus real
    target p to speculative_correction(..., generators=self.generators).
    Do not reseed acceptance/target streams per round. clear() and any failed
    rollout discard RNG and logical cache together; physical stale KV is masked.
    """
    @torch.inference_mode()
    def __init__(self, owner, *, use_graph=True, max_tokens=8):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8:
            raise ValueError('stochastic draft supports K in [1,8]')
        if owner.proposer.cached_tokens or owner.request_id is not None:
            raise ValueError('construct stochastic wrapper only before requests/target KV budget')
        self.owner, self.max_tokens = owner, max_tokens
        self.device = torch.device(owner.device)
        self.vocab_size = owner.model.lm_head.weight.shape[0]
        if owner.capacity < 1 or owner.chunk_size < 1:
            raise ValueError('owner requires positive capacity and chunk size')
        self.graphs = {}
        self.generators = None
        self.request_id = self.seed = self.temperature = None
        self.counters = Counter()
        self.resident_bytes = self.reserved_bytes = self.workspace_bytes = 0
        self.closed = False
        try:
            if use_graph:
                if self.device.type != 'cuda':
                    raise ValueError('logits CUDA graphs require a CUDA owner')
                before = torch.cuda.memory_allocated(self.device)
                reserved_before = torch.cuda.memory_reserved(self.device)
                for width in (1,2):
                    if width <= min(owner.chunk_size,owner.capacity):
                        self.graphs[width] = _LogitsGraph(owner,width)
                torch.cuda.synchronize(self.device)
                self.resident_bytes = max(0,torch.cuda.memory_allocated(self.device)-before)
                self.reserved_bytes = max(0,torch.cuda.memory_reserved(self.device)-reserved_before)
            if self.device.type == 'cuda':
                self._warmup_sampling_workspace()
        except BaseException:
            self.close()
            raise
        finally:
            reset_context()

    @torch.inference_mode()
    def _warmup_sampling_workspace(self):
        # Include retained q, p, FP64 normalization/CDF/correction intermediates
        # before target budgeting. This is allocation warmup, not model evidence.
        rng = make_generators(0x53504543,device=self.device)
        logits = torch.zeros(self.vocab_size,device=self.device,dtype=torch.float32)
        samples = [sample_draft(logits,1.,generators=rng) for _ in range(self.max_tokens)]
        q = torch.stack([s.probabilities for s in samples])
        ids = torch.stack([s.ids for s in samples])
        p = torch.cat((q,q[-1:]),0)
        result = speculative_correction(ids,q,p,generators=rng)
        torch.cuda.synchronize(self.device)
        del result, p, q, ids, samples, logits
        memory = torch.cuda.memory_stats(self.device)
        self.workspace_bytes = max(0,memory['allocated_bytes.all.peak']-memory['allocated_bytes.all.current'])
        self.owner.workspace_bytes = max(getattr(self.owner,'workspace_bytes',0),self.workspace_bytes)

    @torch.inference_mode()
    def _logits(self, ids, start):
        width = ids.numel()
        if width < 1 or start < 0 or start+width > self.owner.capacity:
            raise ValueError('draft logits forward exceeds capacity')
        graph = self.graphs.get(width)
        try:
            if graph is not None:
                result = graph.run(ids,start)
                self.counters['logits_graph_replays'] += 1
            else:
                positions = torch.arange(start,start+width,device=self.device,dtype=torch.int64)
                cu_q = torch.tensor([0,width],device=self.device,dtype=torch.int32)
                cu_k = torch.tensor([0,start+width],device=self.device,dtype=torch.int32)
                set_context(True,cu_q,cu_k,width,start+width,positions.to(torch.int32),
                            block_tables=self.owner.block_table)
                result = self.owner.model.compute_logits(self.owner.model(ids,positions))[-1]
                self.counters['eager_logits_forwards'] += 1
            self.counters['forward_calls'] += 1
            self.counters['forwarded_tokens'] += width
            return result
        finally:
            reset_context()

    @torch.inference_mode()
    def propose(self, history, limit, temperature, seed, request_id):
        if self.closed:
            raise RuntimeError('stochastic draft wrapper is closed')
        try:
            validate_sampling_options(temperature)
            if request_id is None:
                raise ValueError('request_id must identify the active request')
            if type(limit) is not int or not 1 <= limit <= self.max_tokens:
                raise ValueError('requested stochastic draft length exceeds configured K')
            if (not isinstance(history,(list,tuple)) or not history
                    or any(type(t) is not int or not 0 <= t < self.vocab_size for t in history)):
                raise ValueError('history must be nonempty valid committed token IDs')
            if self.request_id != request_id or self.owner.request_id != request_id:
                self.clear()
                self.generators = make_generators(seed,device=self.device)
                self.request_id = self.owner.request_id = request_id
                self.seed, self.temperature = seed, temperature
            elif self.seed != seed or self.temperature != temperature:
                raise ValueError('seed/temperature cannot change within one active request')
            count = min(limit,self.owner.capacity-len(history)+1)
            if count <= 0:
                self.clear()
                return StochasticProposal([],torch.empty((0,self.vocab_size),device=self.device,dtype=torch.float64),
                                          torch.empty(0,device=self.device,dtype=torch.int64))
            proposer = self.owner.proposer
            proposer.synchronize(history)
            start = len(proposer.cached_tokens)
            suffix = list(history[start:])
            # Synchronize guarantees at least one committed token is unforwarded.
            if not suffix:
                raise RuntimeError('owner synchronization failed to retain the last input')
            for offset in range(0,len(suffix),self.owner.chunk_size):
                chunk = torch.tensor(suffix[offset:offset+self.owner.chunk_size],device=self.device,dtype=torch.int64)
                logits = self._logits(chunk,start+offset)
            samples = []
            for index in range(count):
                sample = sample_draft(logits,temperature,generators=self.generators)
                samples.append(sample)
                if index+1 < count:
                    # Invalid probabilities produce -1; keep a failing GPU rollout
                    # memory-safe until its single status/token transfer below.
                    logits = self._logits(sample.ids.clamp_min(0).reshape(1),len(history)+index)
            ids = torch.stack([s.ids for s in samples])
            probabilities = torch.stack([s.probabilities for s in samples])
            token_ids = ids.cpu().tolist()  # Sole proposal D2H; no vocabulary copy.
            self.counters['candidate_host_transfers'] += 1
            if any(not 0 <= t < self.vocab_size for t in token_ids):
                raise ValueError('invalid draft probabilities; no logical rollout state published')
            # Publish only after the complete rollout and the single transfer succeed.
            proposer.cached_tokens.extend(suffix)
            proposer.cached_tokens.extend(token_ids[:-1])
            proposer.counters['proposed_tokens'] += count
            proposer.counters['rollout_calls'] += 1
            proposer.counters['forward_calls'] += (len(suffix)+self.owner.chunk_size-1)//self.owner.chunk_size+count-1
            proposer.counters['forwarded_tokens'] += len(suffix)+count-1
            self.counters['proposed_tokens'] += count
            return StochasticProposal(token_ids,probabilities,ids)
        except BaseException:
            self.clear()
            raise

    def synchronize(self, history, request_id=None):
        if self.closed:
            raise RuntimeError('stochastic draft wrapper is closed')
        try:
            if request_id is None or request_id == self.owner.request_id:
                self.owner.proposer.synchronize(history)
        except BaseException:
            self.clear()
            raise

    def clear(self):
        self.owner.proposer.clear()
        self.owner.request_id = None
        self.generators = None
        self.request_id = self.seed = self.temperature = None

    def stats(self):
        return dict(self.counters,execution='logits_graph' if self.graphs else 'eager_logits',
                    graph_widths=sorted(self.graphs),resident_bytes=self.resident_bytes,
                    reserved_bytes=self.reserved_bytes,workspace_bytes=self.workspace_bytes,
                    cached_tokens=len(self.owner.proposer.cached_tokens),rng_active=self.generators is not None,
                    gpu_validation='NOT_RUN',probability_dtype='float64',correction_dtype='float64')

    def close(self):
        if self.closed:
            return
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        self.clear()
        self.closed = True
        reset_context()
