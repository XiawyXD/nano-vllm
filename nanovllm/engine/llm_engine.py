"""nano-vLLM 的高层推理引擎。

本文件把 tokenizer、scheduler 和 ModelRunner 串起来：用户只需要调用
`LLM.generate`，引擎会自动完成请求入队、prefill/decode 迭代和结果收集。
"""

import atexit
import os
import uuid
from datetime import timedelta
from dataclasses import fields
from dataclasses import asdict
from collections import Counter, OrderedDict
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.incremental_ngram import IncrementalNGramProposer
from nanovllm.streaming import GenerationEvent
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.planned_scheduler import PlannedScheduler, KVProgressBlocked
from nanovllm.engine.execution_plan import ExecutionRole, StepStats
from nanovllm.engine.adaptive_speculation import AdaptiveSpeculationController
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.tp_runtime import tp_worker_entry, wait_worker_status, stop_processes


class LLMEngine:
    """面向用户的批量生成入口。"""

    def __init__(self, model, **kwargs):
        # 只把 Config 声明过的 kwargs 交给配置对象，避免无关参数泄漏。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        # Sequence.block_size 是类级别配置，供所有 Sequence 计算 block 数。
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self._worker_connections = []
        self._tp_shm = None
        self._tp_store = None
        self.model_runner = None
        self._exited = False
        # 先加载纯 CPU tokenizer；失败时还没有 worker 或共享内存要清理。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if config.draft_model is not None:
            from nanovllm.engine.proposer import validate_tokenizers
            validate_tokenizers(self.tokenizer, AutoTokenizer.from_pretrained(config.draft_model, use_fast=True))
        # spawn 进程上下文适合 CUDA：每个 Tensor Parallel rank 都有独立进程。
        ctx = mp.get_context("spawn")
        try:
            if config.tensor_parallel_size > 1:
                if config.tensor_parallel_size > torch.cuda.device_count():
                    raise ValueError("tensor_parallel_size exceeds visible CUDA devices")
                # TCPStore 自己绑定 port=0 并保持 socket，无“查空端口再释放”竞态。
                self._tp_store = dist.TCPStore(
                    "127.0.0.1", config.tp_master_port or 0, config.tensor_parallel_size,
                    is_master=True, timeout=timedelta(seconds=config.tp_timeout_seconds),
                    wait_for_workers=False,
                )
                config.tp_master_port = self._tp_store.port
                config.tp_shm_name = f"nanovllm_{os.getpid()}_{uuid.uuid4().hex}"
                self._tp_shm = SharedMemory(name=config.tp_shm_name, create=True, size=config.tp_ipc_bytes)
                for i in range(1, config.tensor_parallel_size):
                    event = ctx.Event()
                    receiver, sender = ctx.Pipe(duplex=False)
                    process = ctx.Process(target=tp_worker_entry,
                                          args=(config, i, event, sender, os.getpid()))
                    try:
                        process.start()
                    except BaseException:
                        receiver.close()
                        sender.close()
                        raise
                    sender.close()
                    self.ps.append(process)
                    self.events.append(event)
                    self._worker_connections.append((i, receiver))
                self.model_runner = ModelRunner(config, 0, self.events, store=self._tp_store,
                                                worker_connections=self._worker_connections)
                wait_worker_status(self._worker_connections, "ready", config.tp_startup_timeout_seconds)
                # 已 dispatch 的 TP 调用失败后，不保留孤儿 worker 等待下一条消息。
                self.model_runner._failure_callback = lambda: self._cleanup_tp_resources(force=True)
            else:
                self.model_runner = ModelRunner(config, 0, self.events)
        except BaseException:
            try:
                self._cleanup_tp_resources(force=True)
            except Exception:
                pass  # 保留原始启动异常，不让二次清理错误掩盖根因。
            raise
        # scheduler 需要 EOS id 来判断请求何时完成。
        config.eos = self.tokenizer.eos_token_id
        self._packed_enabled = (config.batching_mode == "mixed"
                                and config.tensor_parallel_size == 1
                                and config.kv_cache_dtype == "model")
        self.scheduler = PlannedScheduler(config) if self._packed_enabled else Scheduler(config)
        if config.host_kv_cache_bytes:
            from nanovllm.engine.tiered_cache import HostPrefixCache, AsyncHostPrefixCache
            cache_type = AsyncHostPrefixCache if config.host_kv_async_prefetch else HostPrefixCache
            cache_options = dict(max_inflight_blocks=config.host_kv_prefetch_max_inflight_blocks) if config.host_kv_async_prefetch else {}
            if config.host_kv_cost_model_path is not None:
                from nanovllm.engine.cache_cost import CacheCostModel, CacheCostPolicy
                cache_options['policy']=CacheCostPolicy(CacheCostModel.load(config.host_kv_cost_model_path),
                    min_observations=config.host_kv_admission_min_requests,
                    history_capacity=config.host_kv_policy_history_entries)
            if config.host_kv_joint_residency_path is not None:
                from nanovllm.engine.joint_cache import JointResidencyPolicy
                cache_options['policy'] = JointResidencyPolicy.load(
                    config.host_kv_joint_residency_path, self.scheduler.block_manager)
            if config.host_kv_transfer_backend == 'batched':
                from nanovllm.engine.cache_transfer import BatchedHostPrefixCache
                cache_type=BatchedHostPrefixCache
                cache_options['workspace']=self.model_runner.cache_transfer_workspace
                cache_options['idle_fastpath']=config.host_kv_transfer_idle_fastpath
            try:
                self.scheduler.block_manager.host_cache = cache_type(
                    self.model_runner.kv_cache, config.host_kv_cache_bytes,
                    namespace=uuid.uuid4().hex,
                    min_restore_tokens=config.host_kv_min_restore_tokens,
                    max_restore_blocks=config.host_kv_max_restore_blocks, **cache_options)
            except BaseException:
                self._cleanup_tp_resources(force=True)
                raise
            if config.host_kv_transfer_backend == 'batched':
                self.scheduler.block_manager.defer_host_stores=True
        # request_id -> Sequence，供 HTTP server 查询状态、取消请求和读取结果。
        # Sequence 完成后仍暂存到这里，直到上层消费者调用 forget_request。
        self._requests: dict[int, Sequence] = {}
        self._last_step_events: list[GenerationEvent] = []
        self._stream_texts: dict[int, str] = {}
        self._last_speculative_stats: dict[str, int | float] = {}
        self._init_runtime_state()
        # 无论用户是否显式调用 exit，程序结束时都尝试释放进程和 CUDA 资源。
        atexit.register(self.exit)

    def _init_runtime_state(self):
        """创建有界观测/控制状态；请求完成、取消、抢占时清理请求级记录。"""
        self.adaptive_controller = AdaptiveSpeculationController()
        self._runtime_counts = Counter()
        self._event_times = {}
        self._seen_plan_shapes = OrderedDict()
        self._last_step_stats = StepStats()
        if (getattr(self.config, "batching_mode", "separate") == "mixed"
                and not getattr(self, "_packed_enabled", False)):
            reason = "tp" if getattr(self.config, "tensor_parallel_size", 1) > 1 else "fp8"
            self._runtime_counts[f"configuration_fallback_{reason}"] += 1

    def _forget_runtime_request(self, request_id):
        """只清理一个请求，不清除跨请求复用的有界成本模型。"""
        if hasattr(self, "adaptive_controller"):
            self.adaptive_controller.forget(request_id)
            self._event_times.pop(request_id, None)
        runner = getattr(self, "model_runner", None)
        if getattr(getattr(self, "config", None), "draft_model", None) is not None and runner is not None:
            runner.draft_forget(request_id)

    def exit(self):
        """停止 worker 并销毁 Tensor Parallel 的通信组。

        `LLMEngine` 同时注册了 atexit 清理回调，因此显式调用 `exit()`
        后进程退出时还会再次进入这里；把清理做成幂等，避免二次释放报错。
        """
        if getattr(self, "_exited", False):
            return
        self._exited = True
        cache = getattr(getattr(getattr(self, "scheduler", None), "block_manager", None), "host_cache", None)
        if cache is not None:
            if getattr(cache,"asynchronous",False):
                self.scheduler.block_manager.reap_host_loads(blocking=True)
            cache.clear()
            self.scheduler.block_manager.host_cache = None
            del cache  # 释放 HostPrefixCache 持有的 GPU KV tensor 引用。
        self._cleanup_tp_resources(force=False)

    def set_graph_capture_enabled(self, enabled: bool):
        """只在独占引擎的预热阶段调用；不能与 HTTP worker 并发操作。

        预热结束后设为 False，已有形状仍 replay，陌生形状直接 eager。
        此接口只控制 mixed packed Graph，不影响启动时普通 decode Graph。
        """
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be bool")
        self.model_runner.packed_graphs.allow_capture = enabled

    def _cleanup_tp_resources(self, *, force: bool) -> None:
        """正常退出发送 exit；失败时只清理本引擎进程/IPC，不等待全组 barrier。

        NCCL/CUDA 致命错误后不能复用同一引擎；此处是有界子进程回收，
        不是 GPU 上下文恢复机制。底层驱动自身失去响应仍可能需要重启服务进程。
        """
        runner = getattr(self, "model_runner", None)
        self.model_runner = None
        processes = getattr(self, "ps", [])
        timeout = getattr(self.config, "tp_shutdown_timeout_seconds", 5.0)
        try:
            if runner is not None:
                runner._failure_callback = None
                if force or getattr(runner, "_failed", False):
                    stop_processes(processes, 0)
                    runner.exit()
                else:
                    try:
                        runner.call("exit")
                    except Exception:
                        stop_processes(processes, 0)
                        runner.exit()
        finally:
            stop_processes(processes, timeout)
            for _, connection in getattr(self, "_worker_connections", []):
                connection.close()
            self._worker_connections = []
            shm = getattr(self, "_tp_shm", None)
            if shm is not None:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                self._tp_shm = None
            self._tp_store = None
            self.ps = []
            self.events = []

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        """把一条字符串或 token-id prompt 转成 Sequence 并放入 waiting。"""
        if sampling_params.use_speculative:
            self._validate_speculative_params(sampling_params)
        if sampling_params.resolved_speculative_method == "draft_model":
            if not getattr(self.config, "draft_model", None):
                raise ValueError("draft_model method requires Config.draft_model")
            if (getattr(self.config,"draft_device_verification",False)
                    and sampling_params.speculative_max_draft_tokens>getattr(self.config,"draft_graph_max_tokens",8)):
                raise ValueError("device verification request K exceeds captured graph limit")
        if (getattr(sampling_params, "speculative_sampling", "greedy") == "rejection"
                and not getattr(self.config, "enable_stochastic_speculation", False)):
            raise ValueError("rejection requests require Config.enable_stochastic_speculation=True")
        if getattr(self.config, "draft_model", None) is not None and not self.scheduler.is_finished():
            raise ValueError("draft_model v1 allows only one unfinished request; submit sequentially")
        if isinstance(prompt, str):
            # 字符串只在入口处 tokenize，内部调度全部使用整数 token id。
            prompt = self.tokenizer.encode(prompt)
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("prompt must be a non-empty string or list of token ids")
        # 非法输入在 CPU 入口拒绝，不能让 embedding 越界变成 CUDA 致命错误。
        hf_config = getattr(self.config, "hf_config", None)
        vocab_size = getattr(hf_config, "vocab_size", None)
        for token in prompt:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("prompt token ids must be non-negative integers")
            if vocab_size is not None and token >= vocab_size:
                raise ValueError(f"prompt token id {token} exceeds model vocabulary size {vocab_size}")
        if len(prompt) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("prompt + max_tokens exceeds max_model_len")
        # Admission 看的是整张卡的物理上限，而非当前剩余空块：
        # 当前拥塞可以排队；单条请求独占所有 block 仍装不下则必须立即拒绝。
        # 最后一个输出 token 不会再次 forward，因此不需要它的 KV slot。
        manager = self.scheduler.block_manager
        required_tokens = len(prompt) + sampling_params.max_tokens - 1
        required_blocks = (required_tokens + manager.block_size - 1) // manager.block_size
        if required_blocks > len(manager.blocks):
            raise ValueError(
                f"request needs up to {required_blocks} KV blocks, but physical capacity "
                f"is {len(manager.blocks)}; reduce prompt/max_tokens or increase KV capacity")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        self._requests[seq.seq_id] = seq
        return seq

    def get_request(self, request_id: int) -> Sequence | None:
        """按 request id 返回内部 Sequence；HTTP 层只读这个快照对象。"""
        return self._requests.get(request_id)

    def forget_request(self, request_id: int) -> None:
        """删除已经被上层消费的请求元数据，不影响已释放的 KV Cache。"""
        self._requests.pop(request_id, None)
        self._forget_runtime_request(request_id)

    def cancel_request(self, request_id: int) -> bool:
        """取消一条尚未完成的请求并释放其 waiting/running 资源。"""
        seq = self._requests.get(request_id)
        if seq is None or seq.is_finished:
            return False
        self.scheduler.cancel(seq)
        seq._ngram_proposer = None
        self._stream_texts.pop(request_id, None)
        self._forget_runtime_request(request_id)
        return True

    def _decode_completion(self, seq: Sequence) -> str:
        """完整解码已生成 token；关闭空格清理，避免后续 token 改写已发文本。"""
        try:
            return self.tokenizer.decode(seq.completion_token_ids,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        except TypeError:
            # CPU 状态机测试中的最小 tokenizer 仅接受 ids 参数。
            return self.tokenizer.decode(seq.completion_token_ids)

    def completion_text(self, seq: Sequence, *, streaming: bool = False) -> str:
        """统一流式和非流式的 stop 语义，返回当前可安全展示的文本。

        stop 可能横跨多个 token。未结束时保留最长的“stop 前缀后缀”，
        例如 stop='23'、当前文本='12'，只能发送 '1'，不能先发 '2' 再撤回。
        UTF-8 字节 token 尚不完整时，tokenizer 常解码出尾部 U+FFFD；同样
        等后续 token 补全再发送。正常结束时释放未匹配 stop 的保留尾部。
        """
        text = self._decode_completion(seq)
        positions = [index for stop in seq.stop if (index := text.find(stop)) >= 0]
        if positions:
            return text[:min(positions)]
        if streaming and not seq.is_finished:
            text = text.rstrip("\ufffd")
            hold = 0
            for stop in seq.stop:
                for size in range(1, min(len(stop) - 1, len(text)) + 1):
                    if text.endswith(stop[:size]):
                        hold = max(hold, size)
            if hold:
                text = text[:-hold]
        return text

    def _apply_stop_sequences(self, seqs) -> None:
        """在 scheduler 追加 token 后检查 stop；同轮触及长度上限也保留 stop 原因。"""
        for seq in seqs:
            if not seq.stop or seq.num_completion_tokens == 0:
                continue
            text = self._decode_completion(seq)
            if any(stop in text for stop in seq.stop):
                seq.finish_reason = "stop"
                if seq.is_finished:
                    continue
                seq.status = SequenceStatus.FINISHED
                self.scheduler.total_finished_requests += 1
                self.scheduler.block_manager.deallocate(seq)
                if seq in self.scheduler.running:
                    self.scheduler.running.remove(seq)

    def _stream_delta(self, seq: Sequence) -> str:
        full_text = self.completion_text(seq, streaming=True)
        previous = self._stream_texts.get(seq.seq_id, "")
        if not full_text.startswith(previous):
            # SSE 不能撤回已发送内容。不可静默重新发送 full_text 导致重复输出。
            raise RuntimeError("tokenizer revised already emitted text; unstable incremental decode")
        delta = full_text[len(previous):]
        self._stream_texts[seq.seq_id] = full_text
        return delta

    def _make_step_events(self, seqs, previous_lengths):
        events = []
        for seq in seqs:
            current_length = seq.num_completion_tokens
            if current_length <= previous_lengths.get(seq.seq_id, current_length):
                continue
            token_id = seq.completion_token_ids[-1]
            reason = None
            if seq.is_finished:
                reason = seq.finish_reason or (
                    "stop" if (not seq.ignore_eos and token_id == self.scheduler.eos) else "length"
                )
            events.append(GenerationEvent(seq.seq_id, token_id, self._stream_delta(seq), current_length - 1, seq.is_finished, reason))
            if seq.is_finished:
                # 请求结果可以继续保留，但不应继续占用草稿索引内存。
                seq._ngram_proposer = None
                self._stream_texts.pop(seq.seq_id, None)
        return events

    def step(self):
        """返回 (本轮完成结果, StepStats)，生成数与各类 query 数分别计量。"""
        if not hasattr(self, "_runtime_counts"):
            self._init_runtime_state()
        outputs, stats = self._step_planned() if getattr(self, "_packed_enabled", False) else self._step_legacy()
        self._last_step_stats = stats
        self._runtime_counts["steps"] += 1
        for key, value in asdict(stats).items():
            self._runtime_counts[key] += value
        return outputs, stats

    def _step_legacy(self):
        """保留默认分离调度路径，仅把带符号统计迁移到结构化统计。"""
        started = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        manager = self.scheduler.block_manager
        cache = getattr(manager,"host_cache",None)
        fast_cache = getattr(cache,'idle_fastpath',False)
        if fast_cache:cache._check()  # 空路径也检查 failed 状态和原始 stream 所有权。
        if getattr(cache,'batched',False) and (not fast_cache or manager._pending_host_stores):
            manager.flush_host_stores()
        if cache is not None and getattr(cache,"asynchronous",False):
            # 只预取未入本轮batch的未来请求，给当前计算留出重叠机会。
            if not fast_cache or cache.pages or cache.pending_stores:
                selected={s.seq_id for s in seqs}
                for candidate in list(self.scheduler.waiting)[:4]:
                    if candidate.seq_id not in selected and not candidate.block_table and (
                            len(manager.free_block_ids) >= candidate.num_blocks+2):
                        if manager.restore_host_prefix(candidate):
                            cache.counters["lookahead_requests"]+=1
                            break
            if not fast_cache or cache.inflight:
                cache.wait_for_blocks([bid for s in seqs for bid in s.block_table])
        previous_lengths = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        stats = StepStats(scheduled_prefill=sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else 0,
                          query_decode=0 if is_prefill else len(seqs), forward_calls=1)
        # ModelRunner 内部会准备输入、forward、采样并返回本轮 token ids。
        speculative_limit = getattr(self.config, "speculative_max_batch_size", 0)
        has_speculative = any(seq.use_speculative for seq in seqs)
        allow_speculative = ((speculative_limit == 0 or len(seqs) <= speculative_limit)
                             and getattr(self.config, "speculative_policy", "fixed") != "off")
        if not is_prefill and has_speculative and allow_speculative:
            stats = self._decode_mixed(seqs)
        else:
            if not is_prefill and has_speculative and not allow_speculative:
                # plain-greedy 请求退回同一个 target 的普通 decode，保持采样语义。
                # 此处跳过 proposer/额外 forward；通过计数器让降级可观测。
                spec_stats = self._last_speculative_stats
                spec_stats["batch_gate_skips"] = spec_stats.get("batch_gate_skips", 0) + 1
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            if getattr(cache,'policy',None) is not None:
                for seq in seqs:
                    if not fast_cache or not getattr(seq,'_cache_observation_recorded',False):
                        manager.observe_cache_request(seq)
            if hasattr(cache,'mark_consumed') and (not fast_cache or cache.unconsumed):
                cache.mark_consumed(bid for seq in seqs for bid in seq.block_table)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # stop 是字符串级规则，需要 tokenizer，因此放在 engine 而不是 scheduler。
        self._apply_stop_sequences(seqs)
        self._last_step_events = self._make_step_events(seqs, previous_lengths)
        # 只返回本轮刚刚完成的 Sequence，未完成的请求留在 scheduler 队列。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        stats.generated_tokens = sum(seq.num_completion_tokens - previous_lengths[seq.seq_id] for seq in seqs)
        if cache is not None and getattr(cache,"asynchronous",False) and (not fast_cache or cache.needs_reap):
            manager.reap_host_loads()
        policy = getattr(self.scheduler, "latency_policy", None)
        if policy is not None:
            ended = perf_counter()
            policy.observe(seqs, {s.seq_id for s in seqs if s.num_completion_tokens > previous_lengths[s.seq_id]},
                           is_prefill, stats.scheduled_prefill, (ended-started)*1000, ended)
        for seq in seqs:
            if seq.is_finished:
                self._forget_runtime_request(seq.seq_id)
        return outputs, stats

    def _adaptive_bucket(self, seqs):
        """普通/推测轮共用桶；不能把 draft 长度放进 key 导致成本不可比较。"""
        context_bucket = (max((len(seq) for seq in seqs), default=0) + 127) // 128
        signature = tuple(getattr(self.config, name, None) for name in (
            "mixed_attention_backend", "decode_attention_backend", "metadata_backend",
            "rmsnorm_backend", "mlp_backend", "enable_mixed_graph", "enforce_eager"))
        return len(seqs), context_bucket, signature

    def _step_planned(self):
        """Engine owns transfers; the planner still performs CPU metadata work only."""
        manager = self.scheduler.block_manager
        cache = getattr(manager, "host_cache", None)
        fast_cache = getattr(cache,'idle_fastpath',False)
        # proposal 发生在 schedule_plan 前；规划失败时 seqs 尚未建立，也必须
        # 清掉两侧事务及私有随机状态，不能只依赖执行阶段的异常清理。
        request_ids = {seq.seq_id for seq in (*self.scheduler.running, *self.scheduler.waiting)}
        try:
            if cache is not None:
                if fast_cache:cache._check()
                if not fast_cache or manager._pending_host_stores:manager.flush_host_stores()
                if not fast_cache or cache.needs_reap:manager.reap_host_loads()
                # Old lookahead copies must not prevent all scheduling progress.
                if manager._host_load_leases and not manager.free_block_ids:
                    manager.reap_host_loads(blocking=True)
                decode_blocks = sum(max(0, (len(seq) + manager.block_size - 1)
                                        // manager.block_size - len(seq.block_table))
                                    for seq in self.scheduler.running)
                candidates=list(self.scheduler.waiting)[:4] if not fast_cache or cache.pages or cache.pending_stores else ()
                for candidate in candidates:
                    if not candidate.block_table and manager.restore_host_prefix(
                            candidate, progressive=True, reserve_blocks=1 + decode_blocks):
                        cache.counters["mixed_restore_requests"] += 1
                        break
            try:
                return self._step_planned_impl()
            except KVProgressBlocked:
                if not manager._host_load_leases:
                    raise
                # Free pages can still be insufficient for an entire GPU prefix
                # match. Finish bounded outstanding transfers and retry once;
                # do not launch another speculative lookahead on the retry.
                for request_id in self.scheduler.last_preempted_ids:
                    self._forget_runtime_request(request_id)
                cache.counters["mixed_progress_waits"] += 1
                manager.reap_host_loads(blocking=True)
                return self._step_planned_impl()
        except BaseException:
            # Failed/rolled-back plans need not keep a cache backup promise.
            # Never publish a deferred source after a failed forward may write it.
            try:
                self.scheduler.rollback_plan()
            finally:
                for request_id in request_ids:
                    self._forget_runtime_request(request_id)
                manager.discard_host_stores()
            raise
        finally:
            if cache is not None and (not fast_cache or cache.needs_reap or manager._host_load_leases):
                manager.reap_host_loads(blocking=self.scheduler.is_finished())

    def _step_planned_impl(self):
        """预留 -> 单次 packed forward -> 验证/逐 token 提交 -> 成本反馈。

        草稿只保存在 ExecutionItem，不追加进 Sequence。只有本轮没有 P 时
        才启用 proposer；整个批次真实耗时只记录一次，不分摊给单个请求。
        """
        started = perf_counter()
        policy = getattr(self.config, "speculative_policy", "fixed")
        adaptive = policy == "adaptive"
        controller = self.adaptive_controller
        decision_lengths, decisions = {}, {}
        bucket, round_id = None, None
        preemptions_cleared = False
        native_drafts = {}
        native_probabilities = {}
        device_requests = {}
        # GPU proposal work belongs to the engine, before CPU plan construction.
        # v1 has one unfinished request, so no P round or multi-request admission
        # can invalidate this proposal. Planner may still shorten it to its budget.
        if policy != "off" and not self.scheduler.waiting:
            manager = self.scheduler.block_manager
            for seq in self.scheduler.running:
                if seq.resolved_speculative_method != "draft_model":
                    continue
                available = (len(seq.block_table) + len(manager.free_block_ids)) * manager.block_size - len(seq)
                limit = min(seq.speculative_max_draft_tokens,
                            seq.max_tokens - seq.num_completion_tokens - 1,
                            self.config.max_model_len - len(seq),
                            self.config.max_num_batched_tokens - 1, 127, available)
                if limit > 0:
                    if getattr(self.config,"draft_device_verification",False):
                        device_requests[seq.seq_id]=limit
                        continue
                    if getattr(seq, "speculative_sampling", "greedy") == "rejection":
                        proposal = self.model_runner.draft_propose_probabilities(
                            seq.seq_id, seq.token_ids, limit, seq.temperature, seq.seed)
                        native_probabilities[seq.seq_id] = proposal
                        native_drafts[seq.seq_id] = proposal.token_ids
                    else:
                        native_drafts[seq.seq_id] = self.model_runner.draft_propose(seq.seq_id, seq.token_ids, limit)

        def draft_provider(seq, limit):
            nonlocal bucket, round_id, preemptions_cleared
            selected = list(self.scheduler._plan_sequences.values())
            if not preemptions_cleared:
                for request_id in self.scheduler.last_preempted_ids:
                    self._forget_runtime_request(request_id)
                preemptions_cleared = True
            if bucket is None:
                bucket = self._adaptive_bucket(selected)
                if adaptive:
                    round_id = controller.begin_round()
            static_limit = getattr(self.config, "speculative_max_batch_size", 0)
            if static_limit and len(selected) > static_limit:
                self._runtime_counts["gate_batch_limit"] += 1
                return ()
            if adaptive:
                if len(selected) > self.config.adaptive_max_batch_size:
                    self._runtime_counts["gate_adaptive_batch_limit"] += 1
                    return ()
                decision = controller.decide(seq.seq_id, bucket, min(limit, 8))
                decisions[seq.seq_id] = decision
                limit = min(limit, decision.draft_tokens)
                # 首先获得8个整个batch的普通D样本；每16个D轮再整体校准，
                # 不能只把一条请求关掉后将D+V轮冒充普通D成本。
                if controller.needs_normal_samples(bucket) or round_id % 16 == 0:
                    self._runtime_counts["gate_normal_calibration"] += 1
                    limit = 0
                decision_lengths[seq.seq_id] = decision.draft_tokens if limit else 0
            if limit <= 0:
                return ()
            if seq.resolved_speculative_method == "draft_model":
                draft = native_drafts.get(seq.seq_id, ())[:limit]
                self._runtime_counts["proposer_empty"] += int(not draft)
                return draft
            proposer = getattr(seq, "_ngram_proposer", None)
            if proposer is None:
                proposer = IncrementalNGramProposer(seq.speculative_ngram_size, seq.speculative_max_draft_tokens)
                seq._ngram_proposer = proposer
            draft = proposer.propose(seq.token_ids, limit)
            self._runtime_counts["proposer_empty"] += int(not draft)
            return draft

        plan = self.scheduler.schedule_plan(draft_provider if policy != "off" and not device_requests else None,
                                            allow_verification=policy != "off" and not device_requests)
        if device_requests and len(plan.items)==1 and plan.items[0].role is ExecutionRole.DECODE:
            count=device_requests.get(plan.items[0].request_id,0)
            if count:plan=self.scheduler.upgrade_device_verification(plan,count)
        if not preemptions_cleared:
            for request_id in self.scheduler.last_preempted_ids:
                self._forget_runtime_request(request_id)
        seqs = [self.scheduler.get_sequence(item.request_id) for item in plan.items]
        previous = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        stats = self.scheduler.last_step_stats
        if not plan.items:
            self._last_step_events = []
            return [], StepStats()
        if adaptive and not plan.has_prefill and round_id is None:
            round_id = controller.begin_round()
            bucket = self._adaptive_bucket(seqs)
        if plan.has_prefill:
            self._runtime_counts["gate_prefill_round"] += 1
        self._runtime_counts[f"batch_{plan.kind}"] += 1
        reserved = sum(len(item.block_table) * self.scheduler.block_size for item in plan.items)
        self._runtime_counts["reserved_tokens_peak"] = max(self._runtime_counts["reserved_tokens_peak"], reserved)
        accepted = {}
        try:
            manager = self.scheduler.block_manager
            cache = getattr(manager, "host_cache", None)
            fast_cache = getattr(cache,'idle_fastpath',False)
            if cache is not None and (not fast_cache or manager._pending_host_stores):
                manager.flush_host_stores()
            if cache is not None and getattr(cache, "asynchronous", False) and (not fast_cache or cache.inflight):
                cache.wait_for_blocks(bid for item in plan.items for bid in item.block_table)
            from nanovllm.engine.device_draft import DeviceVerificationItem
            if len(plan.items)==1 and isinstance(plan.items[0],DeviceVerificationItem):
                result=self.model_runner.call("run_device_verification",plan,seqs)
                plan=self.scheduler.materialize_device_verification(plan,result)
                predictions={result.request_id:list(result.predictions)}
            else:
                predictions = (self.model_runner.call("run_plan", plan, seqs, native_probabilities)
                               if native_probabilities else self.model_runner.call("run_plan", plan, seqs))
            if hasattr(cache,'mark_consumed') and (not fast_cache or cache.unconsumed):
                cache.mark_consumed(bid for item in plan.items for bid in item.block_table)
            if getattr(cache,'policy',None) is not None:
                for seq in seqs:
                    if not fast_cache or not getattr(seq,'_cache_observation_recorded',False):
                        manager.observe_cache_request(seq)
            stats.forward_calls = 1
            # 先验证整个返回布局，再提交任何请求，避免接口错误造成部分提交。
            if set(predictions) != set(plan.request_ids):
                raise RuntimeError("runner returned wrong request IDs")
            from nanovllm.engine.packed_runner import StochasticVerification
            seq_map = {seq.seq_id: seq for seq in seqs}
            for item in plan.items:
                prediction = predictions[item.request_id]
                rejection = (item.role is ExecutionRole.VERIFY
                    and getattr(seq_map[item.request_id], "speculative_sampling", "greedy") == "rejection")
                if rejection:
                    if (not isinstance(prediction, StochasticVerification)
                            or type(prediction.accepted) is not int
                            or not 0 <= prediction.accepted <= len(item.draft)
                            or len(prediction.tokens) != prediction.accepted + 1
                            or any(type(token) is not int or not 0 <= token < self.config.hf_config.vocab_size
                                   for token in prediction.tokens)
                            or prediction.tokens[:-1] != item.draft[:prediction.accepted]):
                        raise RuntimeError("runner returned invalid probabilistic correction")
                elif isinstance(prediction, StochasticVerification) or len(prediction) != len(item.logit_positions):
                    raise RuntimeError("runner returned wrong logits row count")
            for item, seq in zip(plan.items, seqs):
                rows = predictions[item.request_id]
                count = 0
                if isinstance(rows, StochasticVerification):
                    count, tokens = rows.accepted, list(rows.tokens)
                elif item.role is ExecutionRole.VERIFY:
                    while count < len(item.draft) and rows[count] == item.draft[count]:
                        count += 1
                    tokens = list(item.draft[:count]) + [rows[count]]
                else:
                    tokens = rows
                generated = self.scheduler.commit_item(
                    item, tokens, stop_checker=lambda current: bool(current.stop) and any(
                        stop in self._decode_completion(current) for stop in current.stop))
                if seq.resolved_speculative_method == "draft_model":
                    self.model_runner.draft_synchronize(seq.seq_id, seq.token_ids)
                accepted[item.request_id] = min(count, len(generated))
            self._last_step_events = self._make_step_events(seqs, previous)
        except BaseException:
            self.scheduler.rollback_plan()
            for seq in seqs:
                self._forget_runtime_request(seq.seq_id)
            self._runtime_counts["step_errors"] += 1
            raise
        finished_at = perf_counter()
        spec = self._last_speculative_stats
        proposed = sum(len(item.draft) for item in plan.items)
        for key, amount in {"draft_rounds": int(plan.has_verification),
                            "target_verification_calls": int(plan.has_verification),
                            "fallback_decode_calls": int(not plan.has_prefill and not plan.has_verification),
                            "proposed_tokens": proposed, "accepted_tokens": sum(accepted.values())}.items():
            spec[key] = spec.get(key, 0) + amount
        spec["acceptance_rate"] = spec.get("accepted_tokens", 0) / max(1, spec.get("proposed_tokens", 0))
        info = getattr(self.model_runner, "last_plan_stats", {})
        for timing in ("host_prepare_ms", "model_enqueue_ms", "sample_and_wait_ms"):
            self._runtime_counts[timing] += info.get(timing, 0.0)
        if info.get("captured"):
            self._runtime_counts["capture_forward_calls"] += 3  # 两次warmup和一次capture。
        shape = (plan.kind, tuple((item.role, item.query_len) for item in plan.items),
                 tuple(len(item.block_table) for item in plan.items))
        first_shape = shape not in self._seen_plan_shapes
        self._seen_plan_shapes[shape] = None
        self._seen_plan_shapes.move_to_end(shape)
        while len(self._seen_plan_shapes) > 256:
            self._seen_plan_shapes.popitem(last=False)
        excluded = ("prefill" if plan.has_prefill else "capture" if info.get("captured") else
                    "grow" if info.get("metadata_grew") else "jit" if first_shape else None)
        if adaptive and not plan.has_prefill:
            requested = tuple(decision_lengths.get(item.request_id, 0) if item.draft else 0 for item in plan.items)
            controller.observe_batch(bucket, (finished_at - started) * 1000, stats.generated_tokens,
                                     len(seqs), "spec" if plan.has_verification else "normal", requested,
                                     excluded_reason=excluded)
            for item, seq in zip(plan.items, seqs):
                if item.request_id in decisions:
                    # EOS/stop截断的尾部没有被观察，不能按拒绝样本训练接受率。
                    censored = seq.is_finished and seq.num_completion_tokens - previous[seq.seq_id] <= accepted[item.request_id]
                    controller.observe_request(item.request_id, len(item.draft), accepted[item.request_id],
                                               excluded_reason=excluded or ("censored" if censored else None))
        # 一次verification事件可能含多个token，只记这一个真实事件时间。
        for event in self._last_step_events:
            last = self._event_times.get(event.request_id)
            pure_decode = not plan.has_prefill
            if adaptive and pure_decode and last is not None and last[1] and last[2] == bucket:
                controller.observe_interval(bucket, (finished_at - last[0]) * 1000,
                                            normal=not plan.has_verification, excluded_reason=excluded)
            self._event_times[event.request_id] = (finished_at, pure_decode, bucket)
        for seq in seqs:
            if seq.is_finished:
                self._forget_runtime_request(seq.seq_id)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, stats

    def _decode_mixed(self, seqs):
        """同一调度轮里的普通 decode 和多 token 验证分别执行。

        普通请求保留原来的 decode/CUDA Graph 路径；有 draft 的请求使用
        varlen forward。共享队列和 token budget 不代表必须共享一次 kernel。
        draft 仅是临时输入，验证结束后逐 token 提交，才能正确处理 EOS/stop。
        """
        manager = self.scheduler.block_manager
        stats = self._last_speculative_stats
        accepted_before = stats.get("accepted_tokens", 0)
        for key in ("draft_rounds", "proposed_tokens", "accepted_tokens",
                    "target_verification_calls", "fallback_decode_calls"):
            stats.setdefault(key, 0)
        # 每条已调度请求先保留一个基础 decode token；剩余预算才分配给 draft。
        extra_budget = self.scheduler.max_num_batched_tokens - len(seqs)
        verify, ordinary = [], []
        for seq in seqs:
            remaining = seq.max_tokens - seq.num_completion_tokens
            limit = min(seq.speculative_max_draft_tokens, remaining - 1,
                        extra_budget, self.config.max_model_len - len(seq))
            draft = []
            if seq.use_speculative and limit > 0:
                # 仅在追加临时 draft 之前更新索引；拒绝的 token 不会污染历史。
                proposer = getattr(seq, "_ngram_proposer", None)
                if proposer is None:
                    proposer = IncrementalNGramProposer(
                        seq.speculative_ngram_size, seq.speculative_max_draft_tokens)
                    seq._ngram_proposer = proposer
                draft = proposer.propose(seq.token_ids, limit)
            # 临时候选不能挤占其他已安排请求的 KV。内存不足时缩短 draft；
            # 缩短到零就正常 decode，不因为优化路径而造成分配失败。
            while draft:
                required = (len(seq) + len(draft) + seq.block_size - 1) // seq.block_size
                if required - len(seq.block_table) <= len(manager.free_block_ids):
                    break
                draft.pop()
            if not draft:
                ordinary.append(seq)
                continue
            old_tokens, old_cached = list(seq.token_ids), seq.num_cached_tokens
            for token in draft:
                seq.append_token(token)
            seq.num_scheduled_tokens = len(draft) + 1
            seq.is_prefill = True
            manager.ensure_blocks(seq)
            verify.append((seq, draft, old_tokens, old_cached))
            extra_budget -= len(draft)
        if ordinary:
            try:
                ids = self.model_runner.call("run", ordinary, False)
                self.scheduler.postprocess(ordinary, ids, False)
                stats["fallback_decode_calls"] += 1
            except Exception:
                self._restore_drafts(verify, trim=True)
                raise
        if verify:
            stats["draft_rounds"] += 1
            stats["target_verification_calls"] += 1
            stats["proposed_tokens"] += sum(len(item[1]) for item in verify)
            predictions = None
            try:
                logits = self.model_runner.call("run_speculative", [item[0] for item in verify])
                # 一次传回全部 argmax，避免每个 token 单独触发 GPU/CPU 同步。
                predictions = logits.argmax(dim=-1).tolist()
            finally:
                # 即使 forward 报错，也不能让未验证的 draft 留在逻辑序列中。
                self._restore_drafts(verify, trim=predictions is None)
            offset = 0
            for seq, draft, tokens, cached in verify:
                rows = predictions[offset:offset + len(draft) + 1]
                offset += len(draft) + 1
                accepted = 0
                while accepted < len(draft) and rows[accepted] == draft[accepted]:
                    accepted += 1
                generated = draft[:accepted] + [rows[accepted]]
                for index, token in enumerate(generated):
                    # postprocess 每次结算一个已计算输入位置，再追加一个输出；
                    # 最后追加的 token 尚未缓存，始终保持 cached == len - 1。
                    seq.num_scheduled_tokens = 1
                    self.scheduler.postprocess([seq], [token], False)
                    self._apply_stop_sequences([seq])
                    if index < accepted:
                        stats["accepted_tokens"] += 1
                    if seq.is_finished:
                        break
                if not seq.is_finished:
                    manager.trim(seq)
        stats["acceptance_rate"] = stats["accepted_tokens"] / max(1, stats["proposed_tokens"])
        return StepStats(query_decode=len(ordinary), query_verify=sum(len(item[1]) + 1 for item in verify),
                         forward_calls=int(bool(ordinary)) + int(bool(verify)),
                         accepted_draft_tokens=stats["accepted_tokens"] - accepted_before)

    def _restore_drafts(self, items, *, trim):
        """恢复真实 token 历史；forward 失败时一并归还临时 block。"""
        for seq, draft, tokens, cached in items:
            seq.token_ids = tokens
            seq.num_tokens = len(tokens)
            seq.last_token = tokens[-1]
            seq.num_cached_tokens = cached
            seq.num_scheduled_tokens = 0
            seq.is_prefill = False
            if trim:
                self.scheduler.block_manager.trim(seq)

    def is_finished(self):
        # waiting 和 running 都为空时，所有请求都已经完成。
        return self.scheduler.is_finished()

    def step_events(self) -> list[GenerationEvent]:
        """返回最近一次 step 产生的增量事件。"""
        return list(self._last_step_events)

    def generate_stream(self, prompt: str | list[int], sampling_params: SamplingParams):
        """增量生成；speculative 的一个事件可能包含多个已接受 token。

        event.token_index 是本次增量最后一个 token 的下标，不能把事件数量
        当作生成 token 数。取消在 step 之间生效。
        """
        seq = self.add_request(prompt, sampling_params)
        try:
            # yield 把控制权交回调用方；调用方可能在此时 cancel_request。
            # 已取消请求没有后续 finished event，不能继续对空队列无限 step。
            while not seq.is_finished:
                self.step()
                for event in self._last_step_events:
                    if event.request_id == seq.seq_id:
                        yield event
                        if event.finished:
                            return
        finally:
            # 调用方关闭 generator 或 forward 报错时，也释放剩余请求资源。
            if not seq.is_finished:
                self.cancel_request(seq.seq_id)
            self.forget_request(seq.seq_id)

    stream = generate_stream

    def speculative_stats(self) -> dict[str, int | float]:
        return dict(self._last_speculative_stats)

    def runtime_metrics(self) -> dict:
        """HTTP使用有界聚合，绝不按request ID建立永久指标标签。"""
        if not hasattr(self, "_runtime_counts"):
            self._init_runtime_state()
        runner = self.model_runner
        graph = getattr(runner, "packed_graphs", None)
        fi_attention = getattr(runner, "flashinfer_decode", None)
        pool = getattr(runner, "metadata_buffers", None)
        effective = self.config.runtime_summary() if hasattr(self.config, "runtime_summary") else {}
        effective["packed_enabled"] = getattr(self, "_packed_enabled", False)
        if not effective["packed_enabled"]:
            effective["effective_batching_mode"] = "separate"
            effective["effective_speculative_policy"] = ("off" if getattr(self.config, "speculative_policy", "fixed") == "off" else "fixed")
        else:
            effective["effective_batching_mode"] = "mixed"
            effective["effective_speculative_policy"] = self.config.speculative_policy
        from nanovllm.utils.weight_memory import quantization_stats
        return dict(runtime=dict(self._runtime_counts), last_step=asdict(self._last_step_stats),
                    target_weight_load_seconds=getattr(runner, "weight_load_seconds", None),
                    target_quantization=(quantization_stats(runner.model) if getattr(runner,"model",None) is not None else {}),
                    speculative=self.speculative_stats(), adaptive=self.adaptive_controller.stats(include_buckets=False),
                    decode_graph=dict(getattr(runner, "graph_counters", {})),
                    flashinfer_attention=dict(fi_attention.stats) if fi_attention is not None else {},
                    packed_graph=graph.stats() if graph is not None else {},
                    native_verification_graph=(runner.native_verification_graphs.stats()
                        if getattr(runner, "native_verification_graphs", None) is not None else {}),
                    host_cache=(self.scheduler.block_manager.host_cache.stats()
                                if getattr(self.scheduler.block_manager, "host_cache", None) is not None else {}),
                    draft_model=(runner.draft_runner.stats()
                                 if getattr(runner, "draft_runner", None) is not None else {}),
                    metadata=pool.stats() if pool is not None else {}, effective=effective)

    @staticmethod
    def _validate_speculative_params(sampling_params: SamplingParams) -> None:
        """检查当前 reference speculative 路径要求的采样模式。"""
        if not sampling_params.use_speculative:
            raise ValueError("speculative decoding requires an enabled proposer")
        if getattr(sampling_params, "speculative_sampling", "greedy") == "rejection":
            if (sampling_params.resolved_speculative_method != "draft_model"
                    or sampling_params.temperature <= 1e-10 or sampling_params.top_k != 0
                    or sampling_params.top_p != 1.0 or sampling_params.repetition_penalty != 1.0
                    or sampling_params.speculative_max_draft_tokens > 8):
                raise ValueError("rejection v1 requires native draft, temperature-only sampling and K<=8")
            return
        if sampling_params.temperature > 1e-10:
            raise ValueError("speculative decoding requires temperature=0")
        if sampling_params.top_k or sampling_params.top_p < 1.0 or sampling_params.repetition_penalty != 1.0:
            raise ValueError("speculative decoding requires plain greedy sampling")

    def generate_speculative_batch(self, prompts, sampling_params):
        """兼容旧入口；所有请求现在通过同一个 step/scheduler 推进。"""
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        for item in params:
            self._validate_speculative_params(item)
        return self.generate(prompts, params, use_tqdm=False)

    def generate_speculative(self, prompt, sampling_params):
        """单请求也复用统一调度，避免两套 cache 回滚逻辑产生差异。"""
        return self.generate_speculative_batch([prompt], sampling_params)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """批量生成文本，并按输入顺序返回结果字典列表。"""
        # 进度条的总数是请求数，而不是 token 数。
        if not isinstance(sampling_params, list):
            # 一个 SamplingParams 可以广播给所有 prompts。
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("prompts and sampling_params must have the same length")
        if getattr(self.config, "draft_model", None) is not None and len(prompts) != 1:
            raise ValueError("draft_model v1 generate accepts exactly one prompt")
        for item in sampling_params:
            if item.use_speculative:
                self._validate_speculative_params(item)
        self._last_speculative_stats = {}
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        for prompt, sp in zip(prompts, sampling_params):
            # 先把全部请求放进 waiting，再由 scheduler 做连续 batching。
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            # 每一轮可能是一个 prefill batch 或一个 decode batch。
            output, stats = self.step()
            elapsed = perf_counter() - t
            if stats.scheduled_prefill:
                prefill_throughput = stats.scheduled_prefill / elapsed
            if stats.generated_tokens:
                decode_throughput = stats.generated_tokens / elapsed
            # 在进度条后显示最近测得的两种吞吐率。
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                # 记录已经结束的请求，并让进度条前进一个请求。
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # seq_id 是全局递增的，排序后恢复 prompts 的输入顺序。
        results = []
        for seq_id in sorted(outputs):
            seq = self._requests[seq_id]
            results.append({"text": self.completion_text(seq), "token_ids": outputs[seq_id]})
            # generate 已把结果交给调用方，不再永久保留整条请求历史。
            self.forget_request(seq_id)
        return results
