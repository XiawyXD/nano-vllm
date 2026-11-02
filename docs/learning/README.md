# nano-vLLM 源码学习入口

按AIInfraGuide的nano-vLLM栏目的3.1–3.6组织基础课程，再补充本分支的服务、计划调度、draft、缓存、量化和评测实现。全部对照与补充集中在[一本讲义](AIINFRAGUIDE_COMPANION.md)，不拆成大量重复说明文件。

要动手运行、部署和打断点，从[动手运行与调试指南](02-running-and-debugging.md)开始；包含各 case 的命令、预期、报告位置和支持边界。

|顺序|主题|阅读入口|
|---|---|---|
|3.1|项目全景与四层架构|[先读本节](AIINFRAGUIDE_COMPANION.md#s31)|
|3.2|配置、入口与generate全流程|[课程](AIINFRAGUIDE_COMPANION.md#s32) → [逐行请求生命周期](01-request-lifecycle.md)|
|3.3|模型骨架与核心算子|[张量怎样经过Qwen3](AIINFRAGUIDE_COMPANION.md#s33)|
|3.4|KV Cache与BlockManager|[页号、slot、引用与前缀](AIINFRAGUIDE_COMPANION.md#s34)|
|3.5|Scheduler与连续批处理|[separate与mixed的区别](AIINFRAGUIDE_COMPANION.md#s35)|
|3.6|ModelRunner、CUDA Graph与基准|[固定地址和图外工作](AIINFRAGUIDE_COMPANION.md#s36)|
|3.7|HTTP/SSE、取消、背压与TTL|[在线生命周期补充](AIINFRAGUIDE_COMPANION.md#s37)|
|3.8|ExecutionPlan与KV事务|[reserve、commit、rollback](AIINFRAGUIDE_COMPANION.md#s38)|
|3.9|独立draft、两侧KV与GPU候选|[接受/拒绝后的状态](AIINFRAGUIDE_COMPANION.md#s39)|
|3.10|随机推测概率校正|[p/q、残差、RNG与支持限制](AIINFRAGUIDE_COMPANION.md#s310)|
|3.11|GPU–CPU分层缓存|[event与传输lease](AIINFRAGUIDE_COMPANION.md#s311)|
|3.12|W4A16、Marlin及校准|[格式、执行和质量三层](AIINFRAGUIDE_COMPANION.md#s312)|
|3.13|BF16权重布局与可信评测|[最近实现与正负结果](AIINFRAGUIDE_COMPANION.md#s313)|

基础学习先使用普通separate、TP1、draft关闭、CPU KV层关闭的路径。无需同时理解所有实验开关。先能解释一个请求的状态，再看矩阵和kernel；每节均有代码入口、具体例子及自查练习。

CPU练习模板在[这里](exercises/01_cpu_exercises.py)，[核对答案](exercises/01_checks.md)单独保留。练习里的NotImplementedError是留给你的题目，不是待修运行时缺陷。GPU运行前仍需空闲卡guard。

原文与当前版本有差异：当前支持greedy、chunked prefill和条件性mixed路径；聊天模板、采样、Graph捕获范围也要按当前实际入口判断。配套讲义已逐项解释，不把原文的示意代码当成当前源码。

实验结论见[优化报告](../results/DRAFT_OPTIMIZATION.md)，支持范围见[配置说明](../../configs/README.md)。旧冻结交付的阅读路线（服务器归档：`/home/hrxu/nano-full-opt-20261024/learning/READING_GUIDE_ZH.md`）和面试题（服务器归档：`/home/hrxu/nano-full-opt-20261024/learning/INTERVIEW_20_ZH.md`）作为历史材料保留；旧路径可查[迁移表](../../archive/path_map.json)。

2026-10-26：运行时优化已冻结，当前进入学习阶段。本次更新仅整理文档，不改变代码、默认配置或原验收阈值。
