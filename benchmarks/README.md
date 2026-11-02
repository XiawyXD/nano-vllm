# 性能与评测工具

`python -m benchmarks list` 只看 8 个主线入口；`python -m benchmarks list-all` 查看全部仍有效的兼容命令。CPU 测试优先使用 `python -m tests`，不需要在这里逐个找 check 文件。

- runtime_v2_gpu_guard.py：物理 UUID、进程所有权与空闲检查。
- runtime_v2_performance.py / release_matrix.py：交错性能对照。
- runtime_v2_profile.py / profile_runtime.py：独立 profiler。
- runtime_v2_quality.py / evaluate_w4_quality.py：固定历史质量评测。
- runtime_v2_soak.py / release_soak.py：HTTP 生命周期与负载。
- runtime_v2_suite.py：显式计划、证据、恢复与 CPU/GPU 权限边界。
- cache_draft_experiment.py / tiered_graph_experiment.py：缓存与 draft 的独立实验工具。
- cache_runtime_experiment.py：成本感知/有界批量传输，calibration、DEV、冻结、六臂五轮及自然容量负对照；入口 `python -m benchmarks run cache-runtime -- --help`，详见[本轮记录](../docs/results/CACHE_RUNTIME_OPTIMIZATION.md)。
- cache_policy_experiment.py / cache_idle_experiment.py：容量机会成本与同源码空路径对照；原阈值、失败分组和明确停止点见[后续记录](../docs/results/CACHE_POLICY_FOLLOWUP.md)。
- micro/：次要算子基准；新结果应单独命名，不能覆盖旧证据。

历史 five_stage/followup/diagnose/replay 一次性工具在外部归档；冻结最终复现仍按原 candidate 执行。当前支持范围见 [最终验收](../docs/FULL_OPT_20261025/README.md)。


`python -m benchmarks run cache-joint -- --help`：新 TRAIN/DEV、冻结、GPU-only/旧异步/joint 五轮交错；容量压力与正常显存分组。结果与命令见 `docs/results/CACHE_JOINT_RESIDENCY.md`。


`python -m benchmarks run cache-joint-mixed -- --help`：四条固定路径（GPU-only separate、旧异步 separate、旧异步 mixed、修复的 joint mixed），全新独立数据、冻结及两组五轮；基线 decode Graph 保持开启。报告见 `docs/results/JOINT_MIXED_FIX.md`。
