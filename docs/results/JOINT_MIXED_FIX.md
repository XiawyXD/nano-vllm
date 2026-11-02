# joint + mixed：KV 身份错误的定位、修复与五轮验收

本轮证据目录 `/home/hrxu/nano-vllm-archive/joint-mixed-20261029-001`；性能根目录 `/home/hrxu/nano-vllm-archive/joint-mixed-20261029-001/experiment`。生产源码先完整快照（包括未跟踪源码与既有修改），诊断和正式实验在隔离副本进行。修复的是实际状态错误，并恢复 mixed 的可运行路径；新功能仍需显式启用，不升级通用默认配置。

## 根因与修复

`PlannedScheduler` 将 `BlockManager.defer_host_stores` 设为 True，以保证 CPU planner 不启动 GPU 复制。joint 回载允许备份 GPU 受害页，但 `_allocate_block(offload=True)` 在 mixed 中只登记 `物理页号 → 旧前缀 key`，没有立刻备份；随后 H2D 把新前缀写进同一物理页。下一次 engine flush 读到的是新内容，却仍发布到旧 key 下。

```text
旧：GPU 页 Y → 登记待备份(key=Y, slot=p) → H2D 写入 X → 稍后 CPU[Y] = X
新：GPU 页 Y → 登记待备份(key=Y, slot=p) → 完成 Y 的备份 → H2D 写入 X
```

因此两端拷贝字节可以完全相同，逻辑身份却已错误。仅比较 D2H/H2D 两端字节，无法发现“旧 key 对应新字节”。第一次故障诊断的 15/20 次拷贝审计全部通过，但第 0 层、逻辑位置 0 的 KV 已不同，K 最大绝对差达到 257.5；共同历史下 logits 最大差为 25，不能归为轻微舍入差异。

修改点：`nanovllm/engine/block_manager.py::restore_host_prefix` 在目的页分配并加入临时 pin 后、任何 H2D 提交前，flush 待写回受害页。所有本轮 CPU 源在此前已经 pin，备份规划不会淘汰尚未提交的读取源；GPU 目的页 lease 和原有 consumer event 保持不变。该方法由 engine 在 planner 前的传输边界调用，不把 GPU 工作塞进 CPU planner。

没有引入全设备同步；legacy D2H 原本就使用完成 event，本次修复其发生顺序。没有改 BF16 精度、采样、模型算子、KV dtype 或联合策略参数，也没有关闭普通 decode Graph。`config.py` 移除临时的 joint+mixed 禁用检查，原 TP1/modelKV/dense/非推测/legacy async 限制保留。

个人修改的贡献边界：本轮新增的是覆盖前备份的顺序修复、mixed 启用与逻辑身份/异常回归，以及四路径独立冻结评测。后面的加速数字是“修复后的联合驻留整体”相对旧异步或 GPU-only 的结果；联合淘汰/准入算法和异步传输原先已经存在，不能把它们的全部收益归给这几行修复，也不能用输出错误的旧 joint 路径作为有效性能基线。

## 三条正确性轨道

|检查|修复前|修复后|
|---|---|---|
|原 DEV 自由生成反例|第 6 请求、第 4 输出 token 为 271，参考为 198|完整 48 请求自由生成与原 mixed 参考一致|
|固定共同历史有效 KV|第 0 层、位置 0 已不同|1024/1025/1028 有效长度全部逐位相等|
|共同历史 8 个输出位置 logits|最大绝对差 25，原阈值失败|每个位置最大差 0，argmax 一致|
|同 Q=128 的重复执行与 Graph/eager|独立对照|KV 和 logits 逐位相等，最大差 0|
|无 CPU 缓存，Q=256 或 Q=1 对 Q=128|独立数值边界|仍有约 0.19～0.375 logits 差异，原 0.01/0.01 阈值失败，保留 REVIEW_REQUIRED|

最后一项不能被前面的修复掩盖：该对照没有 CPU 回载，使用相同 token 历史、有效长度与权重，差异关联 BF16 跨 query 执行形状；本轮没有修改全局算子来消除它。即使本例 argmax 相同，也不能替代全词表 logits 验收。严格结论只覆盖本次固定的 C1、chunk128 与对应执行路径，不宣称任意 batch/query 形状均一致。

新增逻辑身份回归：CPU 多页回载前必须保存旧逻辑页的原始 sentinel 字节；备份故障必须先于 H2D，既有源 pin、先前提交的页 lease、engine discard/reap 必须正确清理。两个 CPU 用例在旧源码失败；真实 GPU 的旧代码也在“CPU key 对应另一逻辑页内容”断言失败，修复后通过。新增 mixed Config 正向用例，CPU 联合驻留共 19 项通过。保留旧负例、最初诊断和测试工具搭建失败，不把它们覆盖为 PASS。

真实模型保留 255/256/257/513/1025-token 边界、有效 KV 逐位一致、严格 logits、跨进程、EOS、取消与异常清理。另保留原回载非 deferred 路径回归。对应证据见 `model-fixed-{mixed,separate}001.json`、`retained-full001.json`、`fixed-common-comparison001.json`、`graph-eager-comparison001.json` 与 GPU/CPU 原始日志。

## 数据、基线与冻结

新 TRAIN/DEV/final 种子 2026102901/02/03，各 100 请求、48 个不同 prompt，使用不同原创结构化中英/代码材料。完整 prompt、首个 256-token 页及各级前缀链，与彼此及前 3 轮输入哈希重叠为零。这是合成缓存轨迹，不是生产流量。

联合策略不调参：TRAIN 只按同一公式确定扫描上限 16，历史容量 4096；同 GPU/模型的原实测成本表不变。DEV 检查后冻结源码、模型文件、依赖、输入、配置、工具及物理 UUID，再进入最终验证。最终结果不反馈参数选择。

四条路径：`gpu_only` 为原 separate 普通路径；`legacy_async` 为原 separate 旧异步；`legacy_mixed` 为受控 mixed 旧异步；`joint` 为修复后的 mixed 联合驻留。前三条都使用保留的基线源码。所有路径 decode Graph 开启，TP1、单活跃请求、输入 1025、输出 8 token、chunk128、预算256。压力组 GPU 12 页、CPU 8×28 MiB；正常容量组保持相同自动显存预算。两组各五轮交错、每条路径独立进程，共 40 个正式 worker。

原门槛：logits atol=rtol=0.01；同路径有效 KV 逐位一致；输出 token 与对应参考完全相同。性能要求配对中位数 ≥1.03×、五轮全快、TTFT 与真实流式事件间隔 p95 比 ≤1.10；绝对 SLO TTFT1000ms/TPOT30ms 另报。端到端包含真实排队/调度/forward/采样/事件和尾部传输 drain，正式计时没有 profiler 或逐 step 钩子。

DEV 整体状态 `DEV_SCREEN_FAIL_DIAGNOSTIC_ONLY`，仍按冻结规则保留诊断标签，不因个别子项收益而升级默认配置。每个子项按下表分别判断。

## 容量压力：五轮

|负载|相对旧异步 separate|相对旧异步 mixed|相对 GPU-only separate|原门槛：separate / mixed / GPU-only|
|---|---:|---:|---:|---|
|pooled|1.1009×|1.0867×|1.2024×|PASS / PASS / PASS|
|phase_shift|1.1224×|1.1081×|1.1832×|PASS / PASS / PASS|
|first_cpu_reuse|1.3003×|1.2776×|2.0279×|PASS / PASS / PASS|
|mixed_hot_cold|1.0644×|1.0467×|1.0473×|PASS / PASS / PASS|
|unique|1.0185×|1.0072×|1.0082×|FAIL / FAIL / FAIL|
|shared_branches|1.0029×|1.0076×|1.0034×|FAIL / FAIL / FAIL|

|对照|负载|TTFT p95 比|真实事件间隔 p95 比|五轮均快|输出一致|
|---|---|---:|---:|---|---|
|legacy_async|pooled|0.8938|1.0124|True|True|
|legacy_async|phase_shift|0.8865|1.0134|True|True|
|legacy_async|first_cpu_reuse|0.7743|1.0074|True|True|
|legacy_mixed|pooled|0.8967|0.9997|True|True|
|legacy_mixed|phase_shift|0.9020|1.0000|True|True|
|legacy_mixed|first_cpu_reuse|0.7802|0.9944|True|True|
|gpu_only|pooled|0.8223|1.0121|True|True|
|gpu_only|phase_shift|0.8423|1.0126|True|True|
|gpu_only|first_cpu_reuse|0.4924|1.0047|True|True|

|路径|stores / loads|prefill token / forward 次数|allocated / reserved 峰值 MiB|构造中位数 s|绝对 SLO 请求|
|---|---:|---:|---:|---:|---:|
|gpu_only|0 / 0|407540 / 7180|3664.53 / 3744.00|2.188|97 / 500|
|legacy_async|1200 / 200|356340 / 6780|3682.53 / 3800.00|2.197|100 / 500|
|legacy_mixed|1200 / 200|356340 / 6780|3682.54 / 3800.00|2.199|99 / 500|
|joint|1320 / 360|315380 / 6460|3682.54 / 3800.00|2.199|100 / 500|

五轮合计，相比旧异步 separate，joint 减少 40960 个实际 prefill token 和 320 次 forward；D2H stores 从 1200 到 1320，H2D loads 从 200 到 360。以实际少算的工作和额外复制共同解释收益；不能把更高命中率直接等同于更快。

## 正常显存：五轮

|负载|相对旧异步 separate|相对旧异步 mixed|相对 GPU-only separate|原门槛：separate / mixed / GPU-only|
|---|---:|---:|---:|---|
|pooled|1.0021×|1.0075×|1.0041×|FAIL / FAIL / FAIL|
|phase_shift|1.0081×|1.0092×|1.0060×|FAIL / FAIL / FAIL|
|first_cpu_reuse|0.9996×|1.0058×|0.9981×|FAIL / FAIL / FAIL|
|mixed_hot_cold|1.0027×|1.0077×|0.9918×|FAIL / FAIL / FAIL|
|unique|1.0106×|1.0054×|1.0079×|FAIL / FAIL / FAIL|
|shared_branches|1.0065×|1.0093×|1.0013×|FAIL / FAIL / FAIL|

|对照|负载|TTFT p95 比|真实事件间隔 p95 比|五轮均快|输出一致|
|---|---|---:|---:|---|---|
|legacy_async|pooled|0.9822|1.0119|True|True|
|legacy_async|phase_shift|0.9855|1.0119|False|True|
|legacy_async|first_cpu_reuse|0.9885|1.0079|False|True|
|legacy_mixed|pooled|0.9862|0.9987|False|True|
|legacy_mixed|phase_shift|0.9894|0.9968|False|True|
|legacy_mixed|first_cpu_reuse|0.9874|0.9994|False|True|
|gpu_only|pooled|0.9862|1.0145|False|True|
|gpu_only|phase_shift|0.9908|1.0126|False|True|
|gpu_only|first_cpu_reuse|0.9901|1.0154|False|True|

|路径|stores / loads|prefill token / forward 次数|allocated / reserved 峰值 MiB|构造中位数 s|绝对 SLO 请求|
|---|---:|---:|---:|---:|---:|
|gpu_only|0 / 0|238580 / 5860|19904.53 / 19984.00|2.206|99 / 500|
|legacy_async|0 / 0|238580 / 5860|19904.53 / 19984.00|2.216|99 / 500|
|legacy_mixed|0 / 0|238580 / 5860|19904.54 / 19984.00|2.232|99 / 500|
|joint|0 / 0|238580 / 5860|19904.54 / 19984.00|2.205|100 / 500|

五轮合计，相比旧异步 separate，joint 减少 0 个实际 prefill token 和 0 次 forward；D2H stores 从 0 到 0，H2D loads 从 0 到 0。以实际少算的工作和额外复制共同解释收益；不能把更高命中率直接等同于更快。

各加速比为“每轮参考总耗时 / joint 总耗时”的五轮中位数，不是各 case 比值的平均值。显存是 Torch allocator 峰值，启动只计 LLM 构造，系统文件/JIT 缓存可能为热态；不称为冷机启动。CPU 缓存预算是额外资源，不作为免费容量。真实物理 KV 页数、每请求 token、全部事件、源指纹及主机并行负载保留在原始 JSON。

## 复现

只在物理 GPU3 稳定空闲时运行，不停止他人任务。换卡或模型须重新校准并建立独立协议。所有输出使用新名称：

```bash
export R=/home/hrxu/nano-vllm-archive/joint-mixed-20261029-001
export E="$R/experiment"
export PY=/home/hrxu/nano-vllm/.venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-cffbe123-7f3a-fa27-4919-047ba4dd4790
export NANOVLLM_BENCH_GPU_INDEX=3
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "$E/candidate"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run cache-joint --output "$R/repro-cpu001"
"$PY" -B -m tests.regression.check_joint_cache_gpu --deferred \
  --policy "$E/calibration-joint.json" --output "$R/repro-gpu001.json"
"$PY" -B -m tests.regression.check_cache_runtime_gpu --backend legacy --batching-mode mixed \
  --joint-policy "$E/calibration-joint.json" --compare "$R/model-reference001.json" \
  --output "$R/repro-model001.json"
B="$E/candidate/benchmarks/joint_mixed_experiment.py"
"$PY" -B "$B" suite --split final --freeze "$E/freeze001.json" --output "$E/repro-pressure001"
"$PY" -B "$B" analyze --index "$E/repro-pressure001/index.json" --output "$E/repro-pressure001-analysis.json"
"$PY" -B "$B" suite --split final --freeze "$E/freeze001.json" --normal-capacity --output "$E/repro-normal001"
"$PY" -B "$B" analyze --index "$E/repro-normal001/index.json" --output "$E/repro-normal001-analysis.json"

# 原错误源码仍保留，可复现 matches_retained_reference=false
"$PY" -B "$R/diagnose.py" --source-root "$R/diagnostic-unfixed" \
  --variant joint --mode free --output "$R/repro-original-negative001.json"
# 修复版的共同历史 / KV 检查；原反例的前6请求，不用于性能宣称
"$PY" -B "$R/diagnose.py" --variant joint --mode common --kv --copy-audit \
  --output "$R/repro-fixed-common001.json"
```

本轮没有 HTTP/SSE 30 分钟服务验收，也没有 MTP/TP2/TP4/量化组合扩展。mixed 的旧状态错误已修复；跨 query 数值边界、没有稳定收益的场景继续保留，不对已看过的最终集追加调参。
