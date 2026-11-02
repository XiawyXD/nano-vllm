# Draft prefill CUDA Graph：实现、验收与复现

2026-10-25：本次增量通过原收益/正确性门槛后合入当前源码，新增 `draft_prefill_graph=False`，显式开启使用。冻结最终报告中的旧源码和历史结论不覆盖。

## 本次改动

- `nanovllm/engine/draft_runner.py`：启动时捕获有界的真实 Q/K 前缀形状。完整前缀块覆盖前四个 chunk，每个形状分别捕获“只写 KV”和“写 KV + 最后位置预测”；另覆盖 Q32/Q33 冷前缀。32-token 请求先由 target 生成首 token，draft 实际看到 Q33。匹配不到的形状保留原 eager fallback，不做 query padding。
- `nanovllm/engine/draft_graph.py`：复用 suffix 图执行真实多 token prefill；加入 K 上界、可选预测和顺序复用的 graph memory pool。CPU metadata 一次批量填充，替代逐 token 的 Torch 标量赋值。原异步拷贝的 event fence 保留。
- 中间前缀块省略后续必然覆盖的 LM head/argmax，保留 transformer/KV 写入；最后一块和后续 proposal token 正常计算。这个独立优化本身五轮仅约 1.0002×，不能单独宣传加速。
- 同一 runner 的共享池和私有 KV 绑定单个执行 stream；跨 stream 调用在修改静态输入前报错。close 等待已绑定 stream，覆盖异常清理。增加 Graph 命中/fallback、跳过预测等计数。
- `nanovllm/config.py`：独立布尔开关，拒绝无 draft、eager 或未开启 draft Graph 的组合；运行时摘要记录开关。没有改变其他默认配置。
- 回归测试：增加真实形状/fallback、KV-only、容量边界、异常清理、stream 生命周期与默认关闭检查；修复目录整理后测试 helper 的旧路径。跨进程扩展用例统一覆盖 eager/Graph 两侧，原数值断言不变。

代码阅读顺序：`Config.__post_init__` → `DraftRunner.__init__` → `DraftRunner._rollout` → `DraftSuffixGraph.run/_step` → `DraftRunner.stats/close`。

## 五轮端到端结果

物理 GPU0（UUID `GPU-2eb3dcc0-b069-2c12-1137-3cfcf70129f4`），RTX 4090；8B consistent BF16 target + 0.6B FP16 draft，TP1、单活跃请求、K4、最大上下文 1024、chunk 256。每轮中英/代码/重复文本各 32/768-token 两例，共 8 个请求，每例生成 64 tokens；五轮交错四臂，普通 decode 的 CUDA Graph 正常开启。

输入来自已有独立 DEV 的确定性 ID 规则，属于历史 DEV 的运行时回归，不是新独立 holdout；没有用结果挑 K、shape 或量化参数。每次 worker/trial 都复用既有 GPU guard，记录 GPU UUID、进程与主机负载；不停止其他任务。

| 指标 | 原 FP16 draft 路径 | 开启 prefill Graph |
|---|---:|---:|
| 每轮 512 输出 token，总耗时中位数 | 6.1979 s | 5.7977 s |
| 吞吐中位数 | 82.61 token/s | 88.31 token/s |
| TTFT p95（40 个稳态请求合并） | 264.47 ms | 265.35 ms |
| 实际事件间隔 p95 | 29.01 ms | 28.88 ms |
| 五轮合计接受/提议 token | 1710/3050 | 1710/3050 |
| 五轮合计 target verification | 795 | 795 |

逐轮旧/新耗时比为 **1.068965 / 1.068011 / 1.070011 / 1.069030 / 1.072163**，中位数 **1.06903×**：吞吐约增加 **6.9%**，耗时约减少 **6.46%**。五轮均更快，所有类别及两种长度分别满足原 `median >= 1.03`、全部五轮更快、TTFT/事件间隔 p95 比值 `<= 1.10` 门槛。32-token 组 1.04587×，768-token 组 1.08459×。

收益主要在第一次 draft 提案的冷前缀处理，不代表每个 decode token 都快 6.9%；TTFT 是 target 的首 token，新增图通常在它之后执行，因此不预期 TTFT 同步改善。

相对同 consistent 普通 decode 的本轮吞吐比为 1.86667×、40/40 输出相同，但事件间隔 p95 为其 **1.498×**，整套 draft 仍未通过普通服务的流式门槛。相对实际更快的普通 Torch decode 仅作不同数值后端的耗时比较：1.69405×、20/40 输出相同，TTFT/事件间隔均超门槛。**本次通过的是对既有 draft 路径的增量验收，不能升级为全局服务推荐。**

## 显存与启动代价

- 峰值 allocated：17,996,372,992 → 18,081,650,688 bytes，增加 **81.33 MiB**。
- 峰值 reserved：18,276,679,680 → 18,708,692,992 bytes，增加 **412 MiB**。这是 PyTorch allocator 保留量，不能只报告 allocated 隐去这个代价。
- draft Graph resident：17,046,528 → 102,324,224 bytes；Graph entries 2 → 12。新增池、静态缓冲区和激活存储在 target KV 容量预算之前分配。
- 新进程初始化时间中位数：约 6.55 → 7.67 s；配对启动增量中位数 **1.121 s**。文件系统/模型缓存已热，不是清空磁盘缓存后的冷启动；初始化不计入稳态吞吐。
- 上述是 max_model_len=1024/chunk=256 的实测值；其他形状/容量需重新预算，不按该常数外推。

## 正确性与同步定位

沿用 `atol=0.01, rtol=0.01`，未调整数值、质量或收益门槛。FP16 及同 checkpoint 的 Marlin draft 生命周期通过：255/256/257 边界、固定历史多 Q 形状、有效 KV、拒绝裁剪、请求隔离、EOS/stop/长度、取消和 planner/draft/target 异常。固定历史 target logits 最大绝对差 0，有效 KV bitwise 一致。FP16 额外跨进程 eager/Graph 比较覆盖相同扩展用例集。候选 CPU 25 套及 draft Graph 专项 17 项通过。

分开运行的 torch.profiler 诊断，769-token 冷 draft 前缀 + K4，同模型权重，开/关 prefill 图后 token 与有效 KV 哈希一致：

| CPU/CUDA 发射 | 关闭 | 开启 |
|---|---:|---:|
| 普通 cudaLaunchKernel | 914 | 233 |
| cudaGraphLaunch | 3 | 6 |
| eager draft forward | 4 | 1 |

| 拷贝/同步诊断 | 关闭 | 开启 |
|---|---:|---:|
| cudaMemcpyAsync 次数 | 15 | 9 |
| cudaStreamSynchronize 次数 | 14 | 8 |
| 每次提案结果回传计数 | 1 | 1 |

| 融合/复用机会结论 | 处理 |
|---|---|
| 冷前缀连续 CPU 发射 | 有界 prefill Graph 已合入 |
| 中间 chunk 的无用 LM head | 跳过，保持真实 Q/K 与最终预测 |
| K 个 Q1 replay 再合成一张大图 | 五轮约 1.000×，额外约 8.1 MiB，未合入 |

单次 profiler 的等待时长受捕获追踪和 CPU/GPU 重叠影响，不能把同步次数减少当作等待时间同比下降；性能结论只取不含 profiler 的五轮配对。服务器没有可用 nsys，本次使用 torch.profiler，不声称取得 Nsight Systems 数据。每轮末端传回小型 proposal 是当前 CPU 调度器真正需要的依赖；原代码已经不是逐 token `item()`，本次没有把它虚构成已消除的瓶颈。

## 支持范围与负结果

- 本次性能结论限 RTX 4090、dense Qwen3、TP1、单活跃请求、greedy、K4、上述 32/768-token 负载。native draft 原有 fixed-policy、缓存组合限制不变。
- Graph 按 `(Q, start+Q, 是否最终块)` 精确查找，只覆盖前四个完整 chunk 与冷 Q32/Q33；例如冷 Q255 会 fallback，不能把它算成 Graph 加速。
- `draft_async_metadata` 仍默认关闭，`draft_prefill_graph` 也默认关闭。随机 temperature 推测路径未取得本次加速验收，不据此宣传。
- Marlin draft 的本次通过只表明同 checkpoint 的运行时 Graph 一致性；旧量化 draft 的质量/接受率和 dense/Marlin 全词表差异遗留结论不变，不推荐其替代 FP16 draft。本次未重新量化。
- 没有新增 TP2/TP4、W4A4、HTTP/SSE 服务验收或并发支持。没有把本次结果写回旧冻结 release 的源指纹。

## 复现

以下命令在服务器执行。`R` 是不可覆盖的本轮证据目录。先只读检查空闲 GPU；若 GPU0 不空闲，选择另一张空闲卡的物理 index 和完整 UUID，两者必须对应。guard 发现外来计算进程即停止。

```bash
cd /home/hrxu/nano-vllm
PY=/home/hrxu/nano-vllm/.venv/bin/python
R=/home/hrxu/nano-vllm-archive/draft-sync-graph-20261025-133231
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
export CUDA_VISIBLE_DEVICES=GPU-2eb3dcc0-b069-2c12-1137-3cfcf70129f4
export NANOVLLM_BENCH_GPU_INDEX=0
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
$PY -B -m benchmarks.runtime_v2_gpu_guard

# CPU 回归，不使用 GPU。
CUDA_VISIBLE_DEVICES='' $PY -B -m tests run extended --output "$R/results/repro-cpu-$(date +%s)"
CUDA_VISIBLE_DEVICES='' $PY -B -m tests.regression.check_draft_graph

# 五轮四臂配对；冻结 baseline/candidate 源、fixture、原阈值不变。
$PY -B "$R/tools/measure_prefill.py" suite --output "$R/results/repro-paired-$(date +%s)"
```

当前源码启用方式（Python API；不虚构服务 CLI 开关）：

```python
import json
from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
ensure_gpu1_available()  # 先于 CUDA 初始化/模型加载；环境须设置完整 GPU UUID。
from nanovllm import LLM, SamplingParams

profile = json.load(open('/home/hrxu/nano-vllm/configs/fp16_draft.json'))
config = dict(profile['config'])
model = config.pop('model')
config['draft_prefill_graph'] = True  # 关闭它即同版无 prefill Graph 参考路径。
llm = LLM(model, **config)
try:
    sample = dict(profile['sampling'], ignore_eos=True)
    print(llm.generate(['请解释分页 KV 缓存。'], SamplingParams(**sample)))
    print(llm.runtime_metrics()['draft_model'])  # 查命中、fallback、图显存；任意提示不保证命中。
finally:
    llm.exit()
```

FP16 跨进程生命周期：使用两个新输出路径，先 eager，后 Graph 比较。同一命令仅替换 `--execution`，Graph 侧另加 `--prefill-graph --compare-report`。

```bash
RUN="$R/results/repro-lifecycle-$(date +%s)"
mkdir "$RUN"
$PY -B -m tests.regression.check_marlin_draft_lifecycle \
  --model /home/hrxu/huggingface/Qwen3-8B \
  --draft /home/hrxu/huggingface/Qwen3-0.6B --draft-quantization none \
  --execution eager --extended-numerics --gpu-memory-utilization .90 \
  --output "$RUN/eager.json"
$PY -B -m tests.regression.check_marlin_draft_lifecycle \
  --model /home/hrxu/huggingface/Qwen3-8B \
  --draft /home/hrxu/huggingface/Qwen3-0.6B --draft-quantization none \
  --execution graph --prefill-graph --extended-numerics --gpu-memory-utilization .90 \
  --compare-report "$RUN/eager.json" --output "$RUN/graph.json"
```

## 证据与贡献边界

完整目录：`/home/hrxu/nano-vllm-archive/draft-sync-graph-20261025-133231`；快捷指针 `LATEST_DRAFT_SYNC_AUDIT` 位于其父目录。

- `source-before.tar.gz/json`：修改前包括未跟踪源码的快照；`baseline/`、`candidate/`、`candidate-prefill/`：各轮冻结输入。
- `freeze-prefill.json`、`fixture.json`、`results/paired-prefill001/`：原始 token、事件、配置、环境、源码/模型身份和每轮负载；`results/prefill-final-analysis001.json`：门槛计算。
- `results/prefill-fp16-lifecycle001.json`、`prefill-marlin-lifecycle001.json`：同进程形状/KV/异常检查；`delivery-fp16-*-cross001.json`：统一用例后的跨进程检查。
- `results/prefill-profile001/`：CPU/CUDA 原始 trace 与诊断；`results/grouped-screen002/`、`results/paired001/`：未采用/未达收益门槛的实验也保留。
- `results/prefill-fp16-graph-cross001.json`：两侧 probe 列表不一致造成的首次跨进程测试失败。`delivery-test-amendment.json` 记录仅增强测试的修复，未修改已测运行时代码或阈值。
- `integration.json`：本次合入清单、前后 SHA、Git HEAD/index 保持和运行时与已测源码一致性；`results/integrated-cpu001/`：合入后回归。

贡献是 nano-vllm draft 运行时的启动期 Graph、metadata/LM-head 优化与可复现实验，复用 PyTorch CUDA Graph、现有 attention/Marlin kernel；不是自研新的 SGEMM/Marlin kernel。下一轮若扩展负载，先独立冻结输入和覆盖形状，复验显存/启动代价及流式门槛，不追加验证集调参。


## 流式事件间隔：后续五轮定位（2026-10-25）

结论：**吞吐通过，流式延迟仍未通过，不能同时达标。** 本轮没有合入新的运行时算法或更换数值后端；交付分段诊断工具、冻结 K1 实验和五轮负结果，原默认值与数值阈值保持不变。上文 prefill Graph 对既有 K4 draft 的增量收益仍成立，但不等于普通服务的流式门槛通过。

请求事件的实际调用顺序是：`LLMEngine._step_planned_impl` 中 `draft_propose` → CPU 规划与 KV 预留 → `packed_runner.run_plan` 的 target verification → 验证接受数、逐 token 提交 → 发布一个真实事件。候选 token 通过验证前不发布。本轮不将同一轮接受的多个 token 伪装成多个等间隔事件。

分段测量使用与正式输入 source ID 不重叠的 DEV 子集，分别测 K1/2/4/8，记录 CPU wall time（包含 GPU 完成等待，不是纯 kernel 时间）：

| 固定 K | 热路径 draft 中位数 | target 验证中位数 | DEV 实际事件间隔 p95 |
|---|---:|---:|---:|
| 1 | 2.25 ms | 18.89 ms | 22.70 ms |
| 2 | 4.24 ms | 18.87 ms | 24.84 ms |
| 4 | 8.18 ms | 18.91 ms | 28.98 ms |
| 8 | 15.93 ms | 18.98 ms | 37.15 ms |

target metadata 准备约 0.31–0.32 ms、发射约 0.11 ms；`sample_and_wait_ms` 约 18.3–18.5 ms 主要覆盖已发射 target 的执行完成，不能当成可删除的空同步。每次 proposal 只回传一次小型候选结果；CPU 调度器确实依赖候选和验证结果。当前主因是 draft 与 target 的串行计算依赖，不能靠去掉这次同步解决。

还有单独的冷前缀问题：768-token prompt 在首 token 后让 draft 看到 Q769，前三个 256-token chunk 命中 prefill Graph，末尾 Q1 仍走 eager fallback。正式数据中首 token 后第一次事件间隔远大于稳态 pooled p95，下面同时报告，避免隐藏这个停顿。本轮未扩展 Graph shape 或全局改写算子来掩盖稳态问题。

按测前 protocol 选择：DEV 若没有同时满足原 TTFT/事件间隔门槛的 K，冻结事件间隔最小的 K，仅作有界诊断。结果选 K1，然后使用固定的另一组历史 DEV 输入，五轮交错比较 K1、原 K4、普通 consistent、实际更快的普通 Torch。所有普通 decode CUDA Graph 保持开启；无正式计时 hook/profiler，初始化和 warmup 不计入稳态。

| 配置 | 吞吐中位数 | TTFT p95 | 实际事件间隔 p95 | 首 token 后第一次事件间隔 p95 |
|---|---:|---:|---:|---:|
| 普通 Torch | 52.09 token/s | 243.29 ms | 17.41 ms | 17.53 ms |
| 普通 consistent | 47.39 token/s | 300.38 ms | 19.30 ms | 19.35 ms |
| 原 FP16 draft K4 + prefill Graph | 88.14 token/s | 274.80 ms | 28.88 ms | 63.37 ms |
| FP16 draft K1 + prefill Graph | 67.48 token/s | 272.86 ms | 22.64 ms | 57.06 ms |

- K1 相对普通 consistent：配对吞吐比中位数 **1.42332×**，TTFT p95 比 **0.90839**，事件间隔 p95 比 **1.17316**。吞吐门槛 通过，流式门槛 未通过。
- K1 相对普通 Torch：配对吞吐比中位数 **1.29548×**，TTFT p95 比 **1.12151**，事件间隔 p95 比 **1.30081**。吞吐门槛 通过，流式门槛 未通过。
- K1 相对原 FP16 draft K4 + prefill Graph：配对吞吐比中位数 **0.76605×**，TTFT p95 比 **0.99295**，事件间隔 p95 比 **0.78414**。吞吐门槛 未通过，流式门槛 通过。

原门槛为配对吞吐中位数 >=1.03、五轮均更快、TTFT/事件间隔 p95 比 <=1.10。本轮 K1 相对最快普通 Torch 的事件间隔约增加 30.1%，相对同 consistent 普通路径约增加 17.3%，均失败。它相对 K4 的事件间隔减少约 21.6%，吞吐却减少约 23.4%，不能称为无代价加速。K4 继续作为吞吐实验配置，普通 decode 继续作为流式起点，K1 仅为诊断，不升级默认。

正确性：五轮稳态 40/40 请求及 warmup 中 K1、K4、同 consistent 普通路径输出逐 token 相同，所有请求结束时 KV blocks/refcount 为零。运行时、权重和采样算法均未改动，没有放宽原 `atol=0.01/rtol=0.01`，也没有把历史数值测试称为本轮重新运行。实际更快的 Torch 属于不同算术后端，只有20/40稳态输出相同，其速度比较不能宣称同分布或相同输出。

K1 的接受率从 K4 的 56.07% 变为 78.21%，这是只提议第一个位置后的统计口径变化，不代表 draft 质量提高。五轮 target verification 795 →1400 次，解释了吞吐回落。两种 draft 配置的峰值 allocated 约18.08GB、reserved约18.71GB，K1 并未省掉启动时已有 K<=8 Graph，不能声称降低了模型/Graph显存。

正式证据目录：`/home/hrxu/nano-vllm-archive/draft-stream-20261025-001`。`protocol.json`、`screen-fixture.json`、`freeze.json` 记录停止/选择规则；`results/segmentation002.json` 记录全量分段；`results/paired001/` 保存每轮原始输入来源、输出 token、事件、配置、源码/环境指纹、GPU guard、主机负载；`results/final-analysis001.json` 包含总体和逐类别门槛。`segmentation001.json` 的初稿使用耗时过滤作热路径摘要，已由基于输入历史和实际 K 的002替代；两者均保留，正式性能从未过滤慢事件。

本轮是历史 DEV 的运行时回归，不是新独立质量 holdout 或 HTTP/SSE 长稳验收。所有配对固定空闲物理 GPU0，worker/trial 边界检查外来计算任务；不停止任何其他任务。

复现五轮（在服务器执行；确认卡空闲，guard 发现占用即退出；输出目录必须新建）：

```bash
cd /home/hrxu/nano-vllm
export CUDA_VISIBLE_DEVICES=GPU-2eb3dcc0-b069-2c12-1137-3cfcf70129f4
export NANOVLLM_BENCH_GPU_INDEX=0
R=/home/hrxu/nano-vllm-archive/draft-stream-20261025-001
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python -B   "$R/tools/run_stream.py" suite --output "$R/results/repro-$(date +%s)"
```

单独复现分段：将最后一条命令的参数替换为 `worker --arm baseline --k 1 --screen --instrument --output "$R/results/repro-segments-$(date +%s)"`。这只用于诊断，不将带 hook 结果作为正式性能。当前 Python API 可沿用上文 `configs/fp16_draft.json`，仅将 sampling 的 `speculative_max_draft_tokens` 设为1、config 的 `draft_prefill_graph` 设为True；该配置仍是实验参考。

后续有意义的研究是减少真实 draft forward 成本或保持相同数值行为的 target 验证成本，以及消除冷前缀末块的 eager fallback；需要独立 DEV 和新的冻结实验。纯 CPU metadata 优化即使全部消除约0.4ms，也不足以填平当前相对最快普通 decode 的事件间隔差距。跨请求并发、未经验证的跨 stream 推测、量化 draft 质量回退、修改事件发送口径均不能替代原门槛验收。


## 2026-10-26：冷前缀尾块与 GPU 候选直连验证

本次代码经同 checkpoint 的数值、有效 KV、生命周期和五轮配对验收后合入当前源码，新增开关均默认关闭。结论是：**相对既有 draft 的增量通过原门槛；相对普通 decode，吞吐与流式延迟仍不能同时达标。** 普通服务的冻结启动器和既有30分钟HTTP/SSE结论未更新，本轮只验收原生 Python API。

### 改了什么、沿什么顺序读

1. `DraftRunner` 启动期补齐完整 chunk 后的真实 Q1 尾块 Graph：容量1024/chunk256时覆盖257/513/769。没有 padding，也不在请求里 capture。独立 `draft_prefill_tail_graph` 依赖已有 `draft_prefill_graph`。
2. 新增 `device_draft.py` 协议。`DeviceVerificationItem` 只包含请求、最后已提交 token、K、位置和物理页表；在实际结果返回前，访问 CPU 候选字段会报错。CPU 计划不伪造候选 token ID。
3. `PlannedScheduler.upgrade_device_verification` 在 GPU 工作前扩展真实 KV reservation；`DraftRunner.begin_device_proposal` 返回 GPU 候选及受 owner 保护的 pending 状态，不做前置 D2H。
4. `ModelRunner.run_device_verification` 把候选同 stream 复制到 target Graph 输入，复用已有 packed/pinned metadata 与 Graph。packet 中未使用的暂存 ID 会在 replay 前被真实 GPU 候选全部覆盖。
5. 验证后一次回传候选和 target 预测作为提交证据，`materialize_device_verification` 生成普通不可变执行项，再复用原 CPU greedy 接受、EOS/stop、逐 token 提交和 KV 裁剪。**接受逻辑仍在 CPU，未宣称全 GPU 校正或 draft/target 并行。**
6. pending 状态禁止另一次提案；异常分支先等待已入队 stream，再归还 target 页和清理私有 KV。planner 失败后可继续请求；draft/target 执行失败按原执行器约定 fail closed，需要重建引擎。

新 `draft_device_verification` 要求 TP1、单活跃请求、greedy、consistent target、native Graph、`draft_verify_metadata=True` 和 `draft_async_metadata=True`。不扩展 TP2/TP4、随机推测或量化质量支持范围。首次绑定的执行 stream 不可随后切换，取消沿用现有 step 边界语义。

状态例子：历史为 H，K2 候选为 a/b，draft 已写入 H+a 的 KV。target 首次拒绝 b 并给出 c 时，真实序列提交 H+a+c；私有有效 KV 保留 H+a，下一轮只补 c。全接受并给出 bonus d 时提交 H+a+b+d，下一轮补 b/d 两个输入。未接受尾部可以物理残留，但不会成为有效历史。

### 五轮正式结果

物理GPU3完整UUID `GPU-cffbe123-7f3a-fa27-4919-047ba4dd4790`；官方8B BF16 consistent target +0.6B FP16 draft，TP1/C1，固定最大上下文1024、chunk256。每轮4类文本各32/768两例，8请求×64输出tokens；6臂×5轮交错，共30worker。所有普通 decode Graph 保持开启。K4吞吐/K1延迟候选在与正式输入 source ID 不重叠的 DEV 上选择后冻结；正式输入仍属历史DEV运行时回归，不是新质量holdout。

| 配置 | 吞吐中位数 | TTFT p95 | 实际事件间隔 p95 | 首token后的首次事件间隔 p95 |
|---|---:|---:|---:|---:|
| 普通Torch | 51.78 token/s | 248.74 ms | 17.43 ms | 17.46 ms |
| 普通consistent | 47.29 token/s | 307.90 ms | 19.27 ms | 19.35 ms |
| 旧draft K4（prefill Graph开启） | 87.40 token/s | 282.35 ms | 28.78 ms | 64.02 ms |
| 新组合K4 | 90.11 token/s | 285.42 ms | 28.43 ms | 37.21 ms |
| 旧draft K1 | 66.67 token/s | 297.26 ms | 22.64 ms | 59.93 ms |
| 新组合K1 | 69.18 token/s | 290.41 ms | 22.23 ms | 30.99 ms |

- 同K4增量：配对加速中位数 **1.032494×**，五轮全部更快；TTFT比 1.010876、事件间隔比 0.987733，满足原>=1.03/全部五轮/<=1.10门槛。逐轮速度比分别为 1.035432 / 1.032494 / 1.036294 / 1.014251 / 1.030653。
- 同K1增量：配对加速中位数 **1.033856×**，五轮全部更快；TTFT比 0.976956、事件间隔比 0.982177，满足原>=1.03/全部五轮/<=1.10门槛。逐轮速度比分别为 1.037625 / 1.045141 / 1.029425 / 1.022504 / 1.033856。

K4对旧K4的首次事件间隔缩短约41.9%，K1对旧K1约48.3%。32-token组的增量不足3%，不宣传所有单请求都有稳定收益；768-token K4各类通过增量门槛，K1重复文本有一轮未更快。限定建议是为**已有 draft 吞吐实验**提供默认关闭的增量选项，普通流式服务仍使用原普通 decode 配置。

相对同consistent普通decode，新K4/K1的吞吐配对比为1.9071×/1.4643×，输出40/40请求一致，但事件间隔比为1.4750/1.1536，仍超1.10。相对实际更快的普通Torch，速度仅作为不同算术后端对照（20/40请求输出相同）；新K4/K1的TTFT、事件间隔也未同时通过原门槛。没有关闭普通Graph或修改事件定义。

同K的新旧接受数、提议数和target验证次数全部一致。五轮K4为1710/3050接受、795次验证，K1为1095/1400接受、1400次验证。K1较高的接受比例是K改变统计位置的结果，不是模型质量提高。

### 成本与贡献边界

- 相对已有prefill-enabled draft，峰值allocated增加约 **24.37MiB**，reserved增加 **66MiB**；这是两个不同口径，都保留。K4配对启动增量中位数约 **0.247s**，K1约 **0.311s**。新进程启动在文件系统/模型缓存已热的条件下测量，不能称为未缓存磁盘冷启动。
- 独立profiler的8轮验证+首token中，D2H由17次降为9次。共享相同packed/pinned/tail配置的控制组中，stream synchronize也为17→9；普通旧配置是25次，其中另有pageable metadata等待。profiler结束的device synchronize不计入推理收益。
- 组合收益包含新增尾块Graph、GPU提案协议，以及复用已有packed/pinned metadata；不能把约3.2%全部归为GPU直连一项。单独尾块Graph五轮仅1.02065×且一轮略慢，未通过总体吞吐门槛。
- target验证的profile中，stable BF16 linear约占GPU kernel时间95.51%。流水级数3→2更慢；小N的BN64/BN32微基准仅约1%–2%且是局部算子结果，因此未修改或宣传新的linear kernel。CPU发射减少也不等于消除了真实GPU计算依赖。
- ncu为2021.3且section/rules安装路径失效，未取得SOL计数器数据。权威证据是torch.profiler原始trace和分开的CUDA Graph/event微基准；不补造带宽或occupancy数字。

### 正确性与支持范围

沿用atol/rtol=.01/.01，不调整质量或收益阈值。25套CPU回归及27项专项（19项原Graph+8项新协议）通过。真实GPU覆盖K1/2/4/8、32/255/256/257/768长度、115轮；target logits、target有效KV、draft有效KV逐轮哈希在新旧路径和独立进程间一致。另有固定历史不同Q形状、同执行路径KV、EOS/stop、取消后新请求隔离、planner失败后恢复，以及draft/target入队后的故障fence与回滚。

正式计时全在GPU3完成且每worker/trial guard无外来进程。补充审计时GPU3、随后GPU2先后被其他任务占用，guard中止用例/拒绝启动，记录均保留。等待GPU2连续空闲后，逐轮审计在该卡配对补完；未把两张卡的时间拼成性能结论，也未停止其他任务。

只验收上述官方dense模型组合和Python入口。未新增HTTP/SSE长稳验收，未改变冻结的strict服务启动器，未宣称量化draft质量达标、Marlin新kernel、MTP或多卡支持。实验配置里的ignore_eos=True只为复现固定输出长度；真实EOS使用设为False，已有停止检查保持有效。

### 可运行入口

实验配置：`configs/experiments/fp16_draft_device_verification.json`，含config、sampling及失败/适用范围元数据，不能把整个JSON作为LLM kwargs。新增开关默认都为False。

```python
import json
from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
ensure_gpu1_available()  # 提前设置物理卡完整UUID和index；先于模型/CUDA初始化。
from nanovllm import LLM, SamplingParams
p = json.load(open('configs/experiments/fp16_draft_device_verification.json'))
c = dict(p['config']); model = c.pop('model')
llm = LLM(model, **c)
try:
    s = dict(p['sampling']); s['ignore_eos'] = False
    print(llm.generate(['请解释分页KV缓存。'], SamplingParams(**s)))
    print(llm.runtime_metrics()['decode_graph'])  # 检查device replay/合并回传实际计数。
finally:
    llm.exit()
```

CPU和当前源码GPU回归（模型路径可用--model/--draft覆盖；输出路径必须新建）：

```bash
cd /home/hrxu/nano-vllm
PY=/home/hrxu/nano-vllm/.venv/bin/python
R=/home/hrxu/nano-vllm-archive/draft-execution-20261026-001
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $PY -B -m tests.regression.check_device_draft
# 先选空闲卡。下面仅示例物理GPU2；guard会拒绝外来计算进程。
export CUDA_VISIBLE_DEVICES=GPU-fae44001-5277-3d60-bd3a-0e634ed0e6a8
export NANOVLLM_BENCH_GPU_INDEX=2
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
RUN="$R/results/repro-current-$(date +%s)"; mkdir "$RUN"
$PY -B -m tests.regression.check_device_draft_gpu --variant legacy --output "$RUN/legacy.json"
$PY -B -m tests.regression.check_device_draft_gpu --variant device --compare "$RUN/legacy.json" --output "$RUN/device.json"
```

六臂五轮复现冻结源码/输入（原测量用GPU3；换卡需完整同卡重跑，不能混合比值）：

```bash
export CUDA_VISIBLE_DEVICES=GPU-cffbe123-7f3a-fa27-4919-047ba4dd4790
export NANOVLLM_BENCH_GPU_INDEX=3
$PY -B "$R/tools/run_final.py" suite --output "$R/results/repro-paired-$(date +%s)"
```

### 证据、面试与下一轮Handoff

证据根目录为上面的R。`source-before.tar.gz/json`含未跟踪源码；`models-frozen.json`以真实文件SHA256绑定模型，不虚构HF revision；`dependencies.freeze.json`记录环境包版本。`freeze-device-formal.json`、`formal-fixture.json`、`device-paired001/`和`final-analysis001.json`保存选择、原始token/事件、源码指纹、主机负载和全部门槛。`strong-audit-complete001.json`指向逐轮KV/logits审计PASS报告；`integration.json`记录合入清单、旧修改保留及Git index/HEAD未变。源码快照、失败的测试脚本和GPU占用中止记录均保留，旧冻结报告不覆盖。

面试可按“KV预留→GPU提案→同stream验证→CPU物化→原提交/裁剪”讲这次工作。要能回答：为什么不能在验证前发送草稿；为什么减少同步仍不能达到普通decode事件间隔；为什么全接受会剩两个draft输入未缓存；为什么归还页前必须fence；为什么目标模型旧Torch与consistent不能声称输出等价。

简历可据证据写：基于Nano-vLLM扩展RTX4090单请求推测执行，实现显式GPU候选与KV预留/物化/提交协议；在固定TP1/C1/greedy负载五轮配对中，同K4 draft增量1.0325×，冷前缀首次事件间隔64.0→37.2ms，并建立跨进程逐轮logits/KV及异常回滚验收。不能写成“已解决普通服务流式延迟”或自研Marlin/GEMM kernel。

下一轮研究仍需独立DEV、冻结后验证：Q2 draft实际forward和target权重读取成本才是剩余主要问题。不要继续靠选择更短K、重复微小metadata参数网格或改变事件发送口径制造达标；改变推测架构、额外GPU、量化参考模型都应另立范围和公平基线。


## 2026-10-26（第二轮）：固定数值路径的 target 权重布局优化

本轮只保留一个执行优化：`stable_linear_packed=True`。默认关闭，已验证范围是 RTX4090、TP1、单活跃请求、官方 Qwen3-8B BF16 consistent target；可单独普通 decode，也可配既有 Qwen3-0.6B FP16 greedy draft。量化模型、TP2/TP4、新模型架构及随机推测没有扩展。普通 Torch 仍是不同算术后端，不能把二者说成逐 token 等价。

### 为什么改、具体改在哪里

上轮去掉候选的前置 D2H 后，继续减少 CPU 同步已经不足以解决主要成本。本轮事件测量中，K1 的 draft Q1/Q2 GPU 时间中位数约2.10/2.19ms；Q2本身的额外宽度不是最大的负担，执行一次draft仍需真实GPU时间。旧独立profiler显示target验证约95.5%的GPU kernel时间在线性层；这是时间占比，不是已达到显存带宽上限的证明。

1. `Config.stable_linear_packed` 是显式布尔开关，只允许 dense TP1 BF16 consistent target；默认False，非法组合在CUDA初始化前拒绝。
2. `ModelRunner._initialize` 在权重加载后、draft初始化/模型warmup/KV预算/Graph捕获之前调用 `pack_stable_linears`。对144个dense线性层逐个执行 `weight.T.contiguous().T`，逻辑形状仍为 `[N,K]`，物理stride从`[K,1]`变成`[1,N]`。替换原storage，不常驻两份整模型。Embedding和LM head不重排。
3. `LinearBase.project` 在该显式布局下调用新 `stable_layout.py`。kernel按`W[k*N+n]`读取物理权重，N方向连续；BM/BN/BK仍为16/128/64，warp数4、流水级数3，保持K分块和FP32累加/BF16输出。所有query形状使用同一算法，没有按M切换数值路径。
4. 原row-major stable kernel、Torch路径、draft权重/模型/KV、接受提交及取消回滚逻辑均保留。GPU回归增加 `--packed-linear` 开关以复现新旧比较。

从SGEMM角度看，输入为`X[M,K]`，权重逻辑为`W[N,K]`，输出为`X @ W.T`。改变的是W的物理地址映射，没有改变矩阵的数学维度。例如逻辑矩阵`[[a,b],[c,d],[e,f]]`，旧storage为`a,b,c,d,e,f`，新storage为`a,c,e,b,d,f`，仍按逻辑`W[row,col]`得到原值。

### 七臂、五轮的正式结果

筛选/正式输入使用历史DEV中此前未用的003–005/006–008 source IDs，四类文本各32/768两例，8请求×64输出；只用于运行时回归，不是新的模型质量holdout。候选和输入在正式结果产生前冻结，35个worker交错运行在同一物理GPU3；每个worker/trial通过GPU guard，普通decode Graph保持开启。另在GPU2做过独立小算子筛选，不把两卡计时拼接。

| 配置 | 吞吐中位数 token/s | TTFT p95 ms | 事件间隔 p95 ms |
|---|---:|---:|---:|
| 普通Torch | 51.92 | 239.42 | 17.39 |
| 原consistent普通decode | 47.15 | 288.60 | 19.23 |
| packed普通decode | 50.36 | 282.59 | 17.92 |
| 原draft K1 | 68.86 | 280.99 | 22.20 |
| packed target + draft K1 | 72.92 | 270.88 | 20.89 |
| 原draft K4 | 85.28 | 274.30 | 28.40 |
| packed target + draft K4 | 88.87 | 275.89 | 27.05 |

- packed普通decode 对 原consistent普通decode：配对速度中位数 **1.065853×**，五轮全部更快；TTFT比0.979146、事件间隔比0.931864，通过原增量门槛。
- packed target + draft K1 对 原draft K1：配对速度中位数 **1.056195×**，五轮全部更快；TTFT比0.964028、事件间隔比0.941031，通过原增量门槛。
- packed target + draft K4 对 原draft K4：配对速度中位数 **1.042090×**，五轮全部更快；TTFT比1.005823、事件间隔比0.952492，通过原增量门槛。
- packed target + draft K1 对公平的packed普通decode：配对吞吐1.4480×，事件间隔比1.1661，**流式门槛失败**。
- packed target + draft K4 对公平的packed普通decode：配对吞吐1.7649×，事件间隔比1.5098，**流式门槛失败**。
- 分组负结果保留：packed普通decode的英文768组TTFT比例1.1317，超过1.10；packed K4代码768组有一轮速度比0.9772且TTFT比例1.2016。总体混合负载通过不代表每个类别都通过，不能宣传全场景收益。K1各已测类别的同K增量门槛通过。

特别注意：新K1相对**旧**consistent普通decode的事件间隔比例已经低于1.10，但这个优化也能用于普通decode。以**新**packed普通decode作公平基线，K1/K4仍未通过流式门槛。不能保留较慢普通基线来宣称问题解决。吞吐中位数之比与配对加速中位数口径不同，结论使用后者。

- packed target + draft K1：五轮接受1090/提议1395，target验证1395次；与同K旧实现一致。
- packed target + draft K4：五轮接受1655/提议3355，target验证855次；与同K旧实现一致。

### 成本、正确性与服务稳定性

- packed普通decode 对 原consistent普通decode：稳态峰值allocated差+0.00MiB、reserved差+44.00MiB；配对启动时间增量中位数+0.109s。
- packed target + draft K1 对 原draft K1：稳态峰值allocated差+0.00MiB、reserved差-8.00MiB；配对启动时间增量中位数+0.091s。
- packed target + draft K4 对 原draft K4：稳态峰值allocated差+0.00MiB、reserved差-8.00MiB；配对启动时间增量中位数+0.022s。
144个权重共13,891,534,848 bytes仅改变物理布局；最大单个重排临时张量192MiB，这是张量体积，不作为全启动显存峰值实测。重排本身约0.109s；新进程启动在文件/模型缓存已热的条件下测量，不称为未缓存磁盘冷启动。未重排LM head，未增加常驻权重副本。

原25套CPU回归、配置支持检查通过；115轮真实GPU检查覆盖K1/2/4/8、32/255/256/257/768长度、固定共同历史的不同Q、Graph及eager路径、有效KV、EOS/stop、取消后新请求及三类故障回滚。与参考路径跨进程逐轮target logits/target有效KV/draft有效KV哈希一致，数值阈值仍为atol/rtol=.01/.01；没有降低质量或收益门槛。正式非Torch配置输出与consistent参考完全相同，同K接受/提议/验证次数相同。

当前源码的packed普通decode配置完成真实HTTP/SSE生命周期检查：容量、超时、慢消费者、steady 1800s和干净重启五阶段PASS；steady实际运行1800.24s，结果{'COMPLETED': 1127, 'EXPECTED_CANCEL': 375}。资源/guard持续采样，结束后KV引用、引擎记录、TTL记录、事件、端口和自有进程按原断言检查回收。输入和输出事件保留在`results/http-soak002/`。首轮http-soak001约27分钟时出现外来GPU进程，guard中止并正常清理；失败原记录保留，未把分段时间拼成30分钟，也未停止外来任务。这是该普通配置的稳定性验收，不是draft HTTP收益或新服务默认配置的证明。

### 哪些没有合入，为什么现在停止优化

- FP16 draft同类物理布局试验：Q1多数无收益或变慢；Q2的一个投影虽然局部更快，却未通过原allclose阈值（最大绝对差0.125），Q33也有失败形状。未修改draft权重布局，不拿局部收益换数值要求。
- 原布局warp=2中位速度比约0.942，warp=8约1.000；pipeline=4约1.022，均未形成值得加入的端到端候选。
- packed布局的独立筛选：pipeline=4/5的中位速度比约1.007，部分形状约0.915；warp=8约0.995，停止扩大网格。
- LM head转置的单算子收益约4%，只占整轮一部分，不据此宣称还能获得4%整模型收益；保持Embedding/LM head原布局，避免为很小预期收益引入tied-storage处理。
- 量化draft质量、蒸馏、动态K、跨stream推测重叠、更换模型/精度、MTP及多卡都需要新研究或扩大范围，当前证据不支持把它们作为下一项大概率收益。它们留在研究Handoff，不继续跑校准或参数网格。

结论是本轮有一个可复现的正收益，但现有限制下没有确认下一条大概率通过门槛的低风险执行优化。现在冻结实现，转向读懂和独立复现。此结论不表示数学上已达性能上限，也不把尚缺硬件计数器的诊断说成已完成的roofline证明。

### 复现与学习

两个独立配置位于 `configs/experiments/consistent_packed_linear.json` 和 `configs/experiments/fp16_draft_packed_linear.json`。JSON含config/sampling/decision，传给LLM的仅是config。先选空闲GPU，使用完整UUID和物理index，guard会拒绝其他计算任务。

```bash
cd /home/hrxu/nano-vllm
PY=/home/hrxu/nano-vllm/.venv/bin/python
R=/home/hrxu/nano-vllm-archive/draft-q2-linear-20261026-001
export CUDA_VISIBLE_DEVICES=GPU-cffbe123-7f3a-fa27-4919-047ba4dd4790
export NANOVLLM_BENCH_GPU_INDEX=3 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
D="$R/results/repro-gpu-$(date +%s)"; mkdir "$D"
$PY -B -m tests.regression.check_device_draft_gpu --variant legacy --output "$D/reference.json"
$PY -B -m tests.regression.check_device_draft_gpu --variant device --packed-linear --compare "$D/reference.json" --output "$D/packed.json"
# 七臂五轮：使用归档内冻结源码、输入、开关和原门槛，不重新选参数。
$PY -B "$R/tools/run_experiment.py" suite --phase formal --output "$R/results/repro-paired-$(date +%s)"
# HTTP/SSE：冻结源码对应的原生命周期工具；新输出路径，约30分钟以上。
cd "$R/candidate"
$PY -B -m benchmarks.runtime_v2_suite run --plan "$R/soak-plan.json" --output "$R/results/repro-http-$(date +%s)" --allow-gpu
```

模型使用真实文件SHA绑定，见 `models-frozen.json`；环境见 `dependencies.freeze.json`；源码前后快照包括未跟踪源码。筛选负结果、全部35个worker、token/事件、显存、主机负载与guard在归档中保留。新旧代码贡献边界是本轮target物理布局与对应kernel；不归为自研量化、Marlin或新的推测算法。

学习从 `docs/learning/01-request-lifecycle.md` 开始：先普通separate/draft关闭路径，做完该课1–3题，再学KV、Graph、draft，最后回来看本轮的物理stride与kernel。不要从所有实验开关开始。
