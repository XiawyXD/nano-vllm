# 回归测试

推荐入口：`python -B -m tests run core`；完整 CPU 检查：`python -B -m tests run extended`。清单：`python -m tests list`。不依赖 pytest；每项保留原来的断言，子进程隔离各类 mock。

|类别|代表入口|
|---|---|
|生命周期/调度|regression/check_service_hardening.py、check_engine_hardening.py、check_execution_plan.py|
|KV/draft|regression/check_cache_draft.py、check_tiered_cache.py、check_draft_model.py|
|成本/批量传输|regression/check_cache_cost.py（extended 已包含）、check_cache_transfer_gpu.py（显式 --gpu）、check_cache_runtime_gpu.py（显式后端、可选跨进程参考）|
|空传输快路径|regression/check_cache_idle.py（CPU：真实调度/逻辑KV/完成队列）；GPU模型检查可加 --idle-fastpath；传输GPU检查同时覆盖新容量策略源页保护|
|概率接受拒绝|regression/check_speculative_sampling.py、check_stochastic_engine.py|
|Graph|regression/check_quant_graph_cpu.py、check_draft_graph.py、check_native_verification_graph.py|
|量化|regression/check_awq_packed.py、check_marlin_layout_cpu.py、check_marlin_gpu.py、check_marlin_draft_lifecycle.py|
|测量工具自身|regression/check_gpu_guard.py、check_runtime_v2_harness.py、check_runtime_v2_evaluation.py|

GPU 或依赖真实 checkpoint 的检查不在默认 CPU runner 中。先读对应模块 `--help`、检查空闲 GPU、设置完整 UUID/index，再按原支持范围运行。例如：

```bash
python -m tests.regression.check_marlin_draft_lifecycle --help
python -m tests.regression.check_stochastic_engine --help
```

旧直接模块名 `benchmarks.check_service_hardening` → `tests.regression.check_service_hardening`。常用统一入口 `python -m benchmarks run service-hardening` 保留。所有文件迁移见 [path_map.json](../archive/path_map.json)。

早期 tests/test_scheduler_policy.py 的旧 fixture 与对应阶段材料保留在 archive，当前调度回归使用以上状态机和执行计划测试。CPU 通过不替代真实 GPU/跨进程/性能验收。

需显式输入的 CPU 检查：check_weight_shards 需要 --model/--output；check_marlin_sidecar_cpu 需要 --bundle/--directory/--output。它们不是无依赖自测，不放入 extended；可用 --help 查看完整契约。首次扩展调用的两个参数错误保留在外部 cpu-after-001。

GPU候选协议专项：`python -m tests.regression.check_device_draft`（CPU）；`python -m tests.regression.check_device_draft_gpu --variant legacy --output NEW_JSON`；新路径改用`--variant device --compare LEGACY_JSON`（明确空闲GPU环境）。GPU模块逐轮比较logits和两侧有效KV，覆盖边界、停止、取消及异常fence；详细命令见[执行优化记录](../docs/results/DRAFT_OPTIMIZATION.md)。

GPU布局回归：`python -B -m tests.regression.check_device_draft_gpu --variant device --packed-linear --compare <reference.json> --output <new.json>`。先按同模块legacy模式生成参考，必须选空闲GPU并设置完整UUID及物理index。


`python -m tests run cache-joint`：16 项 CPU 联合驻留状态/源页保护/异常/范围检查。真实 GPU 搬运用 `tests.regression.check_joint_cache_gpu --policy ABSOLUTE_JSON --output NEW_JSON`；固定历史模型用 `check_cache_runtime_gpu --backend legacy --joint-policy ABSOLUTE_JSON`。joint 只支持 separate；mixed 有保留的输出反例。


joint+mixed 修复后 `cache-joint` 为 19 项 CPU 检查，新增旧 key 字节身份、备份失败先于 H2D 及 mixed Config 正向检查。真实 GPU `check_joint_cache_gpu --deferred --policy ABS_JSON --output NEW_JSON` 会强制覆盖后才到晚期 flush 的顺序，旧源码身份断言失败，修复版通过；非 deferred 路径也保留。详见 `docs/results/JOINT_MIXED_FIX.md`。
