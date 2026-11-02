# 项目结构与代码阅读顺序

## 运行主线

```text
HTTP/SSE 或 Python API
  → LLMEngine
  → Scheduler / PlannedScheduler
  → BlockManager：reserve / commit / rollback
  → ExecutionPlan
  → ModelRunner / Graph
  → Qwen3 / layers
  → 验证、采样、提交、事件和清理
```

|目录|职责|优先入口|
|---|---|---|
|nanovllm/|运行时包|server.py、engine/llm_engine.py|
|nanovllm/engine/|请求状态、调度、KV、draft 和缓存|execution_plan.py、planned_scheduler.py、block_manager.py|
|nanovllm/layers/|Attention、Linear、采样与量化算子|attention.py、linear.py、awq_packed.py|
|nanovllm/models/|模型骨架|qwen3.py|
|nanovllm/utils/|加载与执行 Context|loader.py、context.py|
|examples/|小型可运行例子|example.py、chat_nano.py、inspect_runtime.py|
|configs/|独立参考配置|README.md|
|tests/|回归与故障检查|`python -m tests list`|
|benchmarks/|正式性能、质量、环境与 GPU guard|`python -m benchmarks list`|
|benchmarks/micro/|单算子和次要实验|通过完整 benchmark 索引选择|
|docs/learning/|源码讲义与练习|README.md|
|docs/results/|可读实验结果|README.md|
|docs/FULL_OPT_20261025/|旧冻结交付的配置索引|README.md|
|scripts/|仓库检查与辅助入口|check_repository.py|
|archive/|旧路径、备份和外部证据索引|README.md|

## 文件保留原则

运行代码、必要测试、复用基准、配置和可读报告进入 Git。测试继续集中在 `tests/regression/`，稳定模块名保持可导入；不为视觉上减少文件而删除状态机或异常回归。

模型权重、虚拟环境、编译扩展、性能 trace、原始输出和大型源码快照留在仓库外。`cuda-learning/` 是独立 CUDA 练习，上传此推理引擎时忽略；本地原件与历史快照保留。

## 文档路径调整

以下报告从 `docs/` 移到 `docs/results/`：DRAFT_OPTIMIZATION、CACHE_RUNTIME_OPTIMIZATION、CACHE_POLICY_FOLLOWUP、CACHE_JOINT_RESIDENCY、JOINT_MIXED_FIX。仓库内部引用随之更新。

历史冻结配置、原始日志和旧归档的内容指纹保持原样；本轮只整理可读源树和发布入口。原服务器的模型、环境、GPU 任务和旧实验目录不因 GitHub 整理而迁移。

## 仓库检查

```bash
python scripts/check_repository.py
python -B -m tests run core
```

前者仅使用 Python 标准库，检查公开源树中的文件大小、凭据特征和相对文档链接；后者需要已配置的项目依赖，使用 CPU 子进程，不启动真实 GPU 推理。
