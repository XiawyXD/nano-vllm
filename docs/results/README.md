# 实验报告索引

报告按功能组织；源码、模型、精度、输入、基线与原始数据位置在各报告中记录。

|主题|报告|主要内容|
|---|---|---|
|Draft / Graph / BF16 布局|[DRAFT_OPTIMIZATION](DRAFT_OPTIMIZATION.md)|prefill/tail Graph、GPU 候选直连、固定数值路径权重布局、流式间隔与启动/显存成本|
|最新联合缓存|[JOINT_MIXED_FIX](JOINT_MIXED_FIX.md)|mixed 页身份错误的根因和修复、共同历史 KV/logits、两组四路径五轮对照|
|联合驻留历史|[CACHE_JOINT_RESIDENCY](CACHE_JOINT_RESIDENCY.md)|联合策略的原设计与旧 mixed 反例；后续修复见上一行|
|缓存成本与传输|[CACHE_RUNTIME_OPTIMIZATION](CACHE_RUNTIME_OPTIMIZATION.md)|离线成本模型、有界 staging、批量异步与容量负对照|
|缓存策略后续|[CACHE_POLICY_FOLLOWUP](CACHE_POLICY_FOLLOWUP.md)|容量价值、空路径检查及有限停止规则|
|2026-10-25 冻结交付|[配置决定](../FULL_OPT_20261025/README.md)|普通服务、分层缓存、FP16 draft、量化 target、量化 draft 的独立结论|

重点阅读顺序：最新联合缓存 → Draft/Graph → 冻结配置决定。学习主线见[代码阅读目录](../learning/README.md)。

原始实验数据保留在服务器外部归档，本仓库保存可读报告和复用工具。报告里的服务器绝对路径用于定位已有证据；从新克隆复现时，按对应工具生成新的输出目录，不覆盖旧结果。
