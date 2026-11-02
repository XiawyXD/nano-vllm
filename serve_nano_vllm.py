"""启动 nano-vLLM HTTP/SSE 服务的命令行入口。"""

import argparse
import os
import json

from nanovllm import LLM
from nanovllm.server import ServiceLimits, serve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--runtime-preset", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--batching-mode", choices=("separate", "mixed"), default=None)
    parser.add_argument("--speculative-policy", choices=("off", "fixed", "adaptive"), default=None)
    parser.add_argument("--metadata-backend", choices=("dynamic", "reuse"), default=None)
    parser.add_argument("--mixed-attention-backend", choices=("split", "varlen"), default="split")
    parser.add_argument("--mixed-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adaptive-max-batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--tp-master-port", type=int, default=None,
                        help="NCCL rendezvous port; omitted uses env override or a reserved automatic port")
    parser.add_argument("--tp-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--tp-startup-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--tp-ipc-bytes", type=int, default=2**20)
    # 调度策略是负载相关的：显式暴露参数，便于复现 benchmark 的配置。
    # 不把某一个小样本的最优配置静默设为所有用户的默认值。
    parser.add_argument("--prefill-chunk-size", type=int, default=None)
    parser.add_argument("--max-prefill-steps", type=int, default=0)
    parser.add_argument("--latency-aware-scheduling", action="store_true",
                        help="experimental SLO heuristic; measured regression on initial workload; default off")
    parser.add_argument("--scheduler-ttft-target-ms", type=float, default=1000.)
    parser.add_argument("--scheduler-cost-model-path", default=None,
                        help="frozen affine prefill cost fitted on independent calibration inputs")
    parser.add_argument("--scheduler-tpot-target-ms", type=float, default=30.)
    parser.add_argument("--scheduler-min-prefill-tokens", type=int, default=32)
    parser.add_argument("--decode-graph-buckets", type=int, nargs="+", default=None,
                        help="sorted unique batch buckets; final bucket must cover min(max-num-seqs,512)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--host-kv-cache-bytes", type=int, default=0)
    parser.add_argument("--host-kv-async-prefetch", action="store_true",
                        help="experimental prefetch: legacy has synchronous D2H; batched enables bounded async D2H/H2D")
    parser.add_argument("--host-kv-prefetch-max-inflight-blocks", type=int, default=4)
    parser.add_argument("--max-kv-cache-blocks", type=int, default=None)
    parser.add_argument("--host-kv-min-restore-tokens", type=int, default=256)
    parser.add_argument("--host-kv-max-restore-blocks", type=int, default=8)
    parser.add_argument("--host-kv-cost-model-path", default=None)
    parser.add_argument("--host-kv-joint-residency-path", default=None)
    parser.add_argument("--host-kv-admission-min-requests", type=int, default=2)
    parser.add_argument("--host-kv-policy-history-entries", type=int, default=4096)
    parser.add_argument("--host-kv-transfer-backend", choices=("legacy","batched"), default="legacy")
    parser.add_argument("--host-kv-transfer-batch-blocks", type=int, default=4)
    parser.add_argument("--host-kv-transfer-slots", type=int, default=2)
    parser.add_argument("--host-kv-transfer-max-bytes", type=int, default=256*2**20)
    parser.add_argument("--host-kv-transfer-idle-fastpath", action="store_true")
    parser.add_argument("--rmsnorm-backend", choices=("torch","triton"), default="torch")
    parser.add_argument("--allow-runtime-graph-capture", action="store_true",
                        help="experimental: allow request-time mixed Graph capture; default replay-only")
    parser.add_argument("--speculative-max-batch-size", type=int, default=0)
    parser.add_argument("--sampler-backend", choices=("torch", "triton", "flashinfer"), default="torch")
    parser.add_argument("--decode-attention-backend", choices=("flash", "triton", "flashinfer", "cascade"), default="flash")
    parser.add_argument("--cascade-min-shared-tokens", type=int, default=256)
    parser.add_argument("--cascade-verify-reference", action="store_true",
                        help="debug: run both Cascade and FA, exact mismatch falls back to FA; not an acceleration")
    parser.add_argument("--flashinfer-min-batch-size", type=int, default=32)
    parser.add_argument("--mlp-backend", choices=("torch", "triton", "auto"), default="torch")
    parser.add_argument("--mlp-min-tokens", type=int, default=128)
    parser.add_argument("--max-inflight-requests", type=int, default=128)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-stream-events", type=int, default=128)
    parser.add_argument("--finished-request-ttl-seconds", type=float, default=300.0)
    parser.add_argument("--max-retained-requests", type=int, default=256)
    args = parser.parse_args()
    limits = ServiceLimits(
        max_inflight_requests=args.max_inflight_requests,
        request_timeout_seconds=args.request_timeout_seconds,
        max_stream_events=args.max_stream_events,
        finished_request_ttl_seconds=args.finished_request_ttl_seconds,
        max_retained_requests=args.max_retained_requests,
    )
    engine = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        runtime_preset=args.runtime_preset,
        batching_mode=args.batching_mode,
        speculative_policy=args.speculative_policy,
        metadata_backend=args.metadata_backend,
        mixed_attention_backend=args.mixed_attention_backend,
        enable_mixed_graph=args.mixed_graph,
        adaptive_max_batch_size=args.adaptive_max_batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        tp_master_port=args.tp_master_port,
        tp_timeout_seconds=args.tp_timeout_seconds,
        tp_startup_timeout_seconds=args.tp_startup_timeout_seconds,
        tp_ipc_bytes=args.tp_ipc_bytes,
        prefill_chunk_size=args.prefill_chunk_size,
        max_prefill_steps=args.max_prefill_steps,
        latency_aware_scheduling=args.latency_aware_scheduling,
        scheduler_cost_model_path=args.scheduler_cost_model_path,
        scheduler_ttft_target_ms=args.scheduler_ttft_target_ms,
        scheduler_tpot_target_ms=args.scheduler_tpot_target_ms,
        scheduler_min_prefill_tokens=args.scheduler_min_prefill_tokens,
        decode_graph_buckets=args.decode_graph_buckets,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=not args.disable_prefix_caching,
        host_kv_cache_bytes=args.host_kv_cache_bytes,
        host_kv_async_prefetch=args.host_kv_async_prefetch,
        host_kv_prefetch_max_inflight_blocks=args.host_kv_prefetch_max_inflight_blocks,
        max_kvcache_blocks=args.max_kv_cache_blocks,
        host_kv_min_restore_tokens=args.host_kv_min_restore_tokens,
        host_kv_max_restore_blocks=args.host_kv_max_restore_blocks,
        host_kv_cost_model_path=args.host_kv_cost_model_path,
        host_kv_joint_residency_path=args.host_kv_joint_residency_path,
        host_kv_admission_min_requests=args.host_kv_admission_min_requests,
        host_kv_policy_history_entries=args.host_kv_policy_history_entries,
        host_kv_transfer_backend=args.host_kv_transfer_backend,
        host_kv_transfer_batch_blocks=args.host_kv_transfer_batch_blocks,
        host_kv_transfer_slots=args.host_kv_transfer_slots,
        host_kv_transfer_max_bytes=args.host_kv_transfer_max_bytes,
        host_kv_transfer_idle_fastpath=args.host_kv_transfer_idle_fastpath,
        rmsnorm_backend=args.rmsnorm_backend,
        allow_runtime_graph_capture=args.allow_runtime_graph_capture,
        speculative_max_batch_size=args.speculative_max_batch_size,
        sampler_backend=args.sampler_backend,
        decode_attention_backend=args.decode_attention_backend,
        cascade_min_shared_tokens=args.cascade_min_shared_tokens,
        cascade_verify_reference=args.cascade_verify_reference,
        flashinfer_min_batch_size=args.flashinfer_min_batch_size,
        mlp_backend=args.mlp_backend,
        mlp_min_tokens=args.mlp_min_tokens,
    )
    print(json.dumps({"effective_runtime": engine.config.runtime_summary()}, ensure_ascii=False), flush=True)
    serve(engine, args.host, args.port, model_name=args.model, limits=limits)


if __name__ == "__main__":
    main()
