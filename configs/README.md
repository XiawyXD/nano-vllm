# 五套独立配置

这些 JSON 从最终冻结交付原样复制，含 config、status 和 decision。它们是配置证据，不会因文件存在自动被启用。只有 ordinary_service 是当前已验收通用服务起点；量化 target 的推荐范围限定在已通过的重复文本类别。

缓存、FP16 draft、量化 target、量化 draft 分开验收，不能任意叠加。所有数值和失败结论见 [最终报告](../docs/FULL_OPT_20261025/README.md)。严格启动用 scripts/serve_validated.py；不要把整份 JSON 直接作为 LLM kwargs，需区分外层 metadata 与内层 config。

新增默认关闭实验：[GPU候选直连验证](experiments/fp16_draft_device_verification.json)。它只通过了相对既有draft的限定增量验收，普通服务流式门槛仍失败；不是第六套已验收服务配置。详见[执行优化记录](../docs/results/DRAFT_OPTIMIZATION.md)。

新增实验配置：`experiments/consistent_packed_linear.json`（普通consistent）和`experiments/fp16_draft_packed_linear.json`（原FP16 draft搭配packed target）。新开关默认关闭，量化draft支持范围未扩大。

2026-10-28 缓存实验配置：`experiments/cache_cost.json`、`cache_batched.json`、`cache_combined.json`。它们分别启用成本策略、批量异步传输和二者组合，均不替换上面的已验收配置；校准绑定本轮物理 GPU/模型/后端，结果与负对照见[完整记录](../docs/results/CACHE_RUNTIME_OPTIMIZATION.md)。

后续默认关闭配置：`experiments/cache_capacity_value.json`（冷热混合的限定相对收益，工作集切换失败）与 `cache_idle_fastpath.json`（CPU空路径已消除，整体未达收益门槛）。各自记录测试源码目录，不能把不同冻结版本的收益相乘；详见[后续实验](../docs/results/CACHE_POLICY_FOLLOWUP.md)。


`experiments/cache_joint_residency.json` 是显式关闭的 separate 联合驻留实验。包含正式源码/配置/证据绑定，不能把压力组加速解释成正常显存的通用收益，也不能启用 mixed。


`experiments/cache_joint_mixed.json` 绑定修复后的真实 mixed 路径及本轮冻结结果。旧 mixed 状态错误已修复；跨 query 形状数值误差仍单独记录。该配置默认关闭，不将限定收益升级为通用服务推荐。
