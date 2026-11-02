"""统一实验入口：只分派脚本，不改变原有基准的参数或统计口径。"""
import argparse
from pathlib import Path
import subprocess
import sys

ENTRIES = {'cache-draft-cpu': ('module:tests.regression.check_cache_draft', 'CPU：混合回载、页租约与独立draft状态机'),
 'draft-model': ('module:tests.regression.check_draft_model', 'GPU：原生draft、greedy与固定历史严格logits'),
 'cache-draft-experiment': ('module:benchmarks.cache_draft_experiment', '冻结输入、五轮独立进程缓存/draft对照'),
 'cache-draft-acceptance': ('module:benchmarks.validate_cache_draft', 'CPU回归；可选真实GPU、性能矩阵与30分钟混合缓存soak'),
 'flashinfer-attention': ('module:benchmarks.flashinfer_attention',
                          'CPU/GPU check、micro、model：分页decode与Graph重放'),
 'flashinfer-sampling': ('module:benchmarks.flashinfer_sampling',
                         'GPU：check/micro/model，需独立FlashInfer环境和UUID守卫'),
 'flashinfer-cpu': ('module:tests.regression.check_flashinfer_sampler', 'CPU：参考语义、回退与可选依赖隔离'),
 'tiered-tests': ('module:tests.regression.check_tiered_cache', 'CPU默认；--gpu需UUID守卫：分层KV生命周期'),
 'tiered-model': ('module:tests.regression.check_tiered_model', 'GPU：清空物理KV后的CPU回载生成一致性'),
 'tiered-performance': ('module:benchmarks.tiered_graph_experiment', '缓存/Graph/调度/SLO统一消融'),
 'cache-runtime': ('module:benchmarks.cache_runtime_experiment', '成本感知/有界批量传输：校准、冻结、五轮独立进程对照'),
 'cache-policy-v2': ('module:benchmarks.cache_policy_experiment', '容量价值：独立TRAIN/DEV/final、原路径与五轮对照'),
 'cache-idle': ('module:benchmarks.cache_idle_experiment', '空传输路径：同源码开关、旧异步基线、自然容量五轮'),
 'cache-joint-mixed': ('benchmarks/joint_mixed_experiment.py', 'joint+mixed身份时序修复：固定历史、四基线与五轮对照'),
 'cache-joint': ('benchmarks/joint_cache_experiment.py', '联合驻留：独立TRAIN/DEV、冻结后旧异步/GPU-only五轮对照'),
 'cache-joint-check': ('module:tests.regression.check_joint_cache_gpu', 'GPU：联合驻留真实备份、源pin、取消与部分提交失败'),
 'cache-transfer-check': ('module:tests.regression.check_cache_transfer_gpu', 'GPU：异步页传输位级一致、取消/复用与异常fence'),
 'cache-runtime-check': ('module:tests.regression.check_cache_runtime_gpu', 'GPU：真实模型固定历史logits、有效KV、页边界与生命周期'),
 'runtime-v2-suite': ('module:benchmarks.runtime_v2_suite', '六阶段统一入口：哈希匹配恢复和不可覆盖证据'),
 'execution-plan': ('module:tests.regression.check_execution_plan', 'CPU：渐进KV事务与公平调度'),
 'planned-engine': ('module:tests.regression.check_planned_engine', 'CPU：真实planner+引擎混合与adaptive联动'),
 'packed-metadata-cpu': ('module:tests.regression.check_packed_metadata_cpu', 'CPU：混批行映射/Graph元数据与异常清理'),
 'metadata-cpu': ('module:tests.regression.check_metadata_buffers', 'CPU：两帧H2D事件生命周期；仅无参数或--help'),
 'metadata-gpu': ('module:tests.regression.check_metadata_buffers', 'GPU：显式--gpu；真实地址/值检查，必须通过UUID守卫'),
 'adaptive': ('module:tests.regression.check_adaptive_speculation', 'CPU：有界成本模型与推测升降档'),
 'runtime-v2-harness': ('module:tests.regression.check_runtime_v2_harness', 'CPU：证据恢复、锁和进程回收'),
 'gpu-guard': ('module:tests.regression.check_gpu_guard', 'CPU mocks：GPU1边界与进程所有权守卫'),
 'evaluation-harness': ('module:tests.regression.check_runtime_v2_evaluation',
                        'CPU：性能/profiler/soak工具自测，不加载模型'),
 'packed-runtime': ('module:tests.regression.check_packed_runtime', 'GPU：真实P+D/D+V logits/KV/Graph对照'),
 'runtime-v2-engine': ('module:tests.regression.check_runtime_v2_engine', 'GPU：真实引擎混批/回滚/终止与独立共同历史数值轨'),
 'quality': ('module:benchmarks.runtime_v2_quality', '独立固定历史22k+token质量代理'),
 'runtime-v2-performance': ('module:benchmarks.runtime_v2_performance', 'GPU：七组合五轮交错消融'),
 'runtime-v2-profile': ('module:benchmarks.runtime_v2_profile', 'GPU：无hooks四组合计时与独立CPU/CUDA诊断'),
 'runtime-v2-soak': ('module:benchmarks.runtime_v2_soak', 'GPU：真实HTTP边界、30分钟监控与清理重启'),
 'tensorcore-check': ('module:tests.regression.check_tensorcore', '单卡：BF16 GEMM/SwiGLU 数值、输入契约和Graph'),
 'tensorcore-benchmarks.bench_offline': ('module:benchmarks.bench_tensorcore', '单卡：真实形状下五轮交错算子基准'),
 'tensorcore-model': ('module:tests.regression.check_tensorcore_model', '单卡：实际Qwen权重与端到端后端冒烟'),
 'tensorcore-compare': ('module:benchmarks.compare_tensorcore_models', 'CPU：跨后端固定历史logits与token比较'),
 'mlp-dispatch': ('module:tests.regression.check_mlp_dispatch', 'CPU：auto形状白名单和Torch回退'),
 'tp-runtime': ('module:tests.regression.check_tp_runtime', 'CPU：TP控制协议、容量协商和资源回收'),
 'soak-session': ('module:benchmarks.run_release_soak_session', '启动并回收自身服务：smoke + 30分钟soak'),
 'archive': ('module:benchmarks.archive_release', '归档已完成实验并逐文件校验SHA256'),
 'release-matrix': ('module:benchmarks.release_matrix', '投递版：同配置五轮交错性能矩阵'),
 'release-soak': ('module:benchmarks.release_soak', '投递版：HTTP并发与30分钟资源回收'),
 'engine-hardening': ('module:tests.regression.check_engine_hardening', 'CPU：stop、UTF8、KV admission、输入校验'),
 'runtime-hardening': ('module:tests.regression.check_runtime_hardening', 'CPU：Graph/LRU/Context/采样输入不变性'),
 'service-hardening': ('module:tests.regression.check_service_hardening', 'CPU：服务容量、deadline、断连、异常传播'),
 'tokenizer-stream': ('module:tests.regression.check_tokenizer_stream', '真实tokenizer：中文和emoji增量完整性'),
 'gpu-edges': ('module:tests.regression.check_release_gpu_edges', 'GPU：非整齐Graph桶与可选FP8组合'),
 'environment': ('module:benchmarks.capture_environment', '记录版本、GPU、代码哈希及模型配置'),
 'verify-matrix': ('module:benchmarks.verification_matrix', '动态长短请求：验证graph命中/回退与延迟'),
 'verify-graph': ('module:tests.regression.check_verification_graph', '验证CUDA Graph：eager对照、稳态和冷启动'),
 'matrix': ('module:benchmarks.micro.bench_serving_matrix', '系统性能：混合请求、TTFT/TPOT、调度策略'),
 'unified': ('module:benchmarks.micro.bench_unified_scheduler', '真实 GPU：统一调度、取消、SSE 回归'),
 'unit': ('module:tests.regression.test_unified_scheduler', 'CPU：调度状态机确定性回归'),
 'api': ('module:tests.regression.check_openai_api', 'CPU：HTTP/OpenAI 接口回归'),
 'sampling': ('module:tests.regression.check_sampling_gpu', '采样正确性'),
 'sampler': ('module:benchmarks.micro.bench_fused_sampler', '采样算子性能'),
 'attention': ('module:benchmarks.micro.bench_decode_attention', 'decode attention 算子性能'),
 'profile': ('module:benchmarks.profile_runtime', 'CPU/CUDA 时间线，独立无 profiler 计时'),
 'ngram': ('module:tests.regression.check_ngram', '增量 proposer 等价性及 CPU 微基准')}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "list-all", "run"])
    parser.add_argument("name", nargs="?", choices=ENTRIES)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action in {"list", "list-all"}:
        primary = ("environment", "runtime-v2-suite", "runtime-v2-performance", "runtime-v2-profile", "quality", "runtime-v2-soak", "tiered-performance", "cache-draft-experiment")
        selected = ENTRIES if args.action == "list-all" else {name: ENTRIES[name] for name in primary}
        for name, (path, description) in selected.items():
            print(f"{name:12} {path:40} {description}")
        if args.action == "list":
            print("CPU checks: python -m tests list; all retained commands: python -m benchmarks list-all")
        return
    if not args.name:
        parser.error("run 需要实验名；先执行 list 查看")
    path = ENTRIES[args.name][0]
    tail = args.args[1:] if args.args[:1] == ["--"] else args.args
    if args.name == "metadata-cpu" and tail not in ([], ["--help"]):
        parser.error("metadata-cpu only accepts no args or --help; use metadata-gpu for GPU checks")
    if args.name == "metadata-gpu" and "--gpu" not in tail and tail != ["--help"]:
        parser.error("metadata-gpu requires the explicit --gpu flag")
    command = [sys.executable]
    command += ["-m", path.split(":", 1)[1]] if path.startswith("module:") else [path]
    raise SystemExit(subprocess.call(command + tail, cwd=Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    main()
