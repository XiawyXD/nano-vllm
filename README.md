# Nano-vLLM Runtime Lab

基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 扩展的大模型推理运行时，面向 RTX 4090 / SM89 与 dense Qwen3。围绕请求生命周期、KV 缓存、独立 draft、CUDA Graph、W4A16/Marlin 和可复现实验开展优化。

## 从哪里开始

- **第一次阅读**：[请求生命周期](docs/learning/01-request-lifecycle.md) → [完整学习目录](docs/learning/README.md)。
- **运行项目**：[安装、服务与测试](docs/RUNNING.md)。
- **看优化结果**：[实验报告索引](docs/results/README.md)。
- **管理 GitHub**：[上传与后续开发](docs/GITHUB_UPLOAD.md)。
- **找具体文件**：[项目结构](docs/PROJECT_STRUCTURE.md)。

## 项目能力

|模块|本分支的扩展|
|---|---|
|服务|HTTP/SSE、OpenAI 风格接口、取消、deadline、慢消费者背压、TTL 与资源回收|
|调度与 KV|不可变 ExecutionPlan、mixed P/D 与验证计划、reserve/commit/rollback、页引用与异步租约|
|前缀缓存|GPU/CPU 分层、异步和批量传输、成本模型、联合驻留与逻辑页身份检查|
|推测解码|增量 n-gram、独立 Qwen3 draft、私有 KV、greedy 接受、temperature 随机校正|
|执行优化|Draft/verification/prefill/tail CUDA Graph、metadata 复用、GPU 候选直连、BF16 权重布局|
|量化|AWQ g128 packed loader、融合 QKV/gate-up 布局、Marlin 执行与质量评测|
|验证|共同历史 logits、有效 KV、跨进程、故障回滚、五轮交错性能及 HTTP 生命周期检查|

上游已有分页 KV、基础前缀缓存、chunked prefill、基础 CUDA Graph 和 TP。本分支的改动围绕运行时集成、执行优化及验证闭环；第三方实现来源见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

## 已记录的优化结果

各行是独立冻结实验的结果，模型、精度、输入及对照见对应报告。

|工作|结果|评测场景|
|---|---|---|
|GPU/CPU 联合驻留|相对 GPU-only **1.202×**，相对旧异步 mixed **1.087×**|1.7B BF16、TP1/C1、GPU KV 限制 12 页|
|BF16 物理权重布局|普通 consistent decode **1.066×**|8B、TP1/C1、同数值路径五轮对照|
|独立 draft K4|相对同样优化后的 consistent decode **1.765×**|8B target + 0.6B draft，吞吐导向|
|Draft prefill Graph|相对既有 draft **1.069×**|冷前缀执行增量|
|8B AWQ/Marlin target|稳态 **2.302×**；allocated 峰值 **15.51 → 5.95 GiB**|量化 target 的冻结负载|

普通推理配置完成过 30 分钟 HTTP/SSE 生命周期验收。Draft 的流式间隔、量化 draft、正常容量缓存负对照和各类未通过项都保留在[报告](docs/results/README.md)中。实验开关和五套配置分别验收；普通服务、缓存与 draft 使用各自的支持范围。

## 目录

```text
nanovllm/                 实际运行代码
  engine/                 调度、KV、Graph、draft、缓存
  layers/                 Attention、Linear、采样及量化执行
  models/                 Qwen3
  utils/                  权重加载、Context 与格式检查
serve_nano_vllm.py         普通 HTTP/SSE 服务入口
examples/                 离线、聊天与运行状态观察
configs/                  五套参考配置与实验开关
tests/                    CPU 统一入口及显式 GPU 回归
benchmarks/               可复用性能/质量工具；micro/ 为算子基准
docs/                     运行、学习、实验报告和 GitHub 指南
scripts/                  仓库检查、历史验收启动器及可选环境入口
licenses/                 第三方许可
archive/                  历史归档索引，不包含原始数据
```

## 最小运行方式

推理运行使用 Linux、NVIDIA GPU 和匹配的 PyTorch/CUDA/FlashAttention；Windows 本地副本可用于阅读和 Git 管理。已观测的依赖版本在 [requirements-release.txt](requirements-release.txt)，安装和 GPU 选择见[运行说明](docs/RUNNING.md)。

在依赖已配置的环境中：

```bash
python -m pip install -e .
python -B -m tests list
python -B -m tests run core
python -B -m benchmarks list

export MODEL_DIR=/path/to/Qwen3-1.7B
# 先按运行说明检查空闲 GPU，并设置对应 UUID / 物理 index。
python serve_nano_vllm.py --model "$MODEL_DIR" --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 8 \
  --max-num-batched-tokens 256 --prefill-chunk-size 128 --max-prefill-steps 1 \
  --batching-mode separate --speculative-policy off --metadata-backend reuse
```

`scripts/serve_validated.py` 是原服务器冻结验收的包装器，需要外部冻结目录。新克隆先使用上面的通用入口。模型权重、环境、原始 trace、二进制扩展和实验快照不放进 Git；各自获取与归档位置见[运行说明](docs/RUNNING.md)和[归档索引](archive/README.md)。

## 许可

保留上游 [MIT LICENSE](LICENSE)。AWQ/Marlin 相关来源与 Apache-2.0 许可见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
