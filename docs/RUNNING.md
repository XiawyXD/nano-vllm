# 运行、测试与复现

服务器工作目录：`/home/hrxu/nano-vllm`。已有 Python：`.venv/bin/python`。当前项目源码与整理前运行时逐字节一致；测试与 benchmark 路径已整理，历史实验仍绑定原指纹。

第一次动手运行、部署与调试，请看[逐 case 学习指南](learning/02-running-and-debugging.md)：每个示例都有运行命令、观察字段、预期结果和范围说明。

2026-10-28 新增默认关闭的[成本感知缓存和有界批量异步传输](results/CACHE_RUNTIME_OPTIMIZATION.md)，使用新的冻结指纹和独立校准。旧验收结果继续绑定原冻结源码；不能直接套用于新增实验配置。

后续[容量价值策略与空路径检查](results/CACHE_POLICY_FOLLOWUP.md)保留了限定收益、工作集切换回退和布局停止依据；两项独立实验均默认关闭，原服务启动器继续绑定原已验收版本。

## CPU 回归（不占 GPU）

```bash
cd /home/hrxu/nano-vllm
.venv/bin/python -B -m tests list
.venv/bin/python -B -m tests run core
.venv/bin/python -B -m tests run extended
.venv/bin/python -B -m tests run service-hardening
```

core 为整理前后相同的 9 套回归。extended 包括其余显式 CPU 入口，每套独立子进程并隐藏 GPU；不把真实 tokenizer、checkpoint、GPU 检查混入默认执行。原来的数值和概率断言不变。每次输出到唯一目录并保留逐套日志与 summary.json。

单项及 GPU 检查见 [tests/README.md](../tests/README.md)。旧统一命令 `python -m benchmarks run service-hardening` 等仍可分派到新位置；旧直接模块名 `benchmarks.check_*` 改为 `tests.regression.check_*`。一次性历史入口已从默认列表移除。

## 新克隆的安装与通用入口

运行使用 Linux 和兼容的 NVIDIA/CUDA 环境；本地 Windows 可用于代码与 Git 管理。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-release.txt
python -m pip install -e . --no-deps
```

requirements-release.txt记录已有环境的顶层版本；FlashAttention安装还需要匹配的编译环境或wheel。`uv.lock`保留原依赖锁定视图，本轮没有更换现有环境或重新解析依赖。Marlin的原生扩展另行获取，并受代码里的SHA/ABI检查约束。

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
export CUDA_VISIBLE_DEVICES=GPU-填写空闲卡完整UUID
export NANOVLLM_BENCH_GPU_INDEX=填写对应物理编号
python -B -m benchmarks.runtime_v2_gpu_guard
export MODEL_DIR=/path/to/Qwen3-1.7B
python serve_nano_vllm.py --model "$MODEL_DIR" --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 8 \
  --max-num-batched-tokens 256 --prefill-chunk-size 128 --max-prefill-steps 1 \
  --batching-mode separate --speculative-policy off --metadata-backend reuse
```

模型目录由用户提供；通用入口不依赖原服务器冻结归档。下节的严格启动器用于复现已有冻结服务。

## 原服务器冻结服务

```bash
.venv/bin/python scripts/serve_validated.py
```

上述只显示配置。真正启动前先检查稳定空闲卡，设置完整 GPU UUID 和对应物理 index，再执行：

```bash
export CUDA_VISIBLE_DEVICES=GPU-填写完整UUID
export NANOVLLM_BENCH_GPU_INDEX=填写对应物理编号
.venv/bin/python scripts/serve_validated.py --execute
```

包装器调用原最终交付的严格启动器，仍检查冻结源码、模型权重和 guard，默认端口 18784；不会悄悄改用目录整理后的评测源码。通用服务仍用根目录 `serve_nano_vllm.py`；更改其配置属于新的实验，不能直接沿用验收结论。

聊天客户端：`.venv/bin/python examples/chat_nano.py --help`。离线用法读 [examples/example.py](../examples/example.py)。

## 冻结的五套配置与正式实验

[configs/README.md](../configs/README.md) 是独立配置的索引，不自动启用失败/研究配置。原始最终复现命令和模型/依赖清单在 /home/hrxu/nano-full-opt-20261024/REPRODUCE_ZH.md（服务器归档：`/home/hrxu/nano-full-opt-20261024/REPRODUCE_ZH.md`）。它们针对冻结 candidate，保留旧路径即可复现，不必套用主工作树的新路径。

冷启动、稳态和 profiler 继续分开；目录整理没有产生新的 GPU 正确性或性能结论。

性能工具主线：`python -m benchmarks list` 仅显示 8 个入口；`python -m benchmarks list-all` 显示全部保留的兼容命令。新增缓存实验可用 `python -m benchmarks run cache-runtime -- --help` 查看。


联合驻留的独立开关、支持范围、原始证据和冻结复现命令见 [报告](results/CACHE_JOINT_RESIDENCY.md)。普通 GPU-only 和旧异步路径默认值不变。


joint+mixed 已修复延迟备份被 H2D 覆盖的顺序错误，启动限制已解除；当前严格验收为 TP1/Qwen3-1.7B BF16/C1/chunk128。新开关仍默认关闭，完整命令与负结果见 [JOINT_MIXED_FIX](results/JOINT_MIXED_FIX.md)。
