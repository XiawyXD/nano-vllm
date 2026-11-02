# 缓存后续优化：容量价值、空路径与停止条件

本轮按“改善成本决策 → 清理无传输开销 → 判断布局/预取是否值得继续”的顺序推进。服务器证据目录为 `/home/hrxu/nano-vllm-archive/cache-policy-v2-20261028-001`，以下简写 `$R`。新增功能仍为显式实验，不改变普通服务、旧同步/异步缓存和已有验收配置的默认值。

## 范围与验收口径

RTX 4090 GPU3，Qwen3-1.7B BF16，TP1，model dtype KV，推测关闭。正式性能为单活跃请求，decode CUDA Graph 开启；mixed 额外检查真实回载正确性。端到端是引擎排队、调度、forward、采样和真实输出事件，不包含 HTTP 网络。

沿用 logits `atol=rtol=0.01`、同路径有效 KV 逐位相等、对应参考路径输出 token 完全相同。正式收益要求五轮配对中位数至少 1.03×、五轮都更快，TTFT/真实事件间隔 p95 比均不超过 1.10。另报告 TTFT 1000 ms、TPOT 30 ms 的绝对 SLO；不以增量检查通过替代服务验收。

开始前保存了包含未跟踪源码的 `source-before.tar.gz` 和 6490 个文件的 `snapshot.json`，保留已有修改。三份隔离代码分别为 `baseline`（上轮版本）、`candidate`（仅容量策略）、`candidate-fastpath`（容量策略加可关闭空路径）。`freeze-policy-and-idle.json` 绑定三份运行时、模型文件、输入和校准；空路径开关的同源码对照另由 `freeze-idle.json` 绑定。源码不会根据最终结果修改参数。

## 第一项：把 CPU 容量的机会成本计入决策

旧规则只比较新页的预计重算节省、H2D 和 D2H 成本，CPU 满时继续按 LRU 淘汰。本轮新增 `capacity_value` 模式，计算新页净价值与待淘汰页的保留价值：

```text
已有页保留价值 = 最近窗口内预计复用次数 × (每页重算节省 − 回载成本)
新页净价值     = 同类保留价值 − 本次写回成本
```

空间有富余时保留首次复用机会；空间不足时，新页净价值必须高于可替换页的保留价值。已有页的写回成本已经发生，不重复扣除。候选只扫描最旧的最多 64 项；窗口内记录有界，每种长度的只读成本计算最多缓存 128 项。待完成 D2H 和 H2D 正在读取的 CPU 页不能作为淘汰对象。先完成整个接纳规划，再修改 slab 和索引。

访问窗口由 TRAIN 的有限复用距离 p90 决定，固定公式 `clip(2*p90,4,64)`，本次得到 **12 个请求**；首次复用先验为 **53/163**。只拟合这一套配方，没有 LFU/TTL 网格，也没有读取 DEV/最终数据来选窗口。

数据包含冷热混合、工作集切换、第一次复用、全唯一请求和共享前缀分支。GPU KV 固定 12 页，CPU 8 页，各种长度在热点与冷请求中平衡；TRAIN/DEV/final 的完整 prompt 和首个 256-token 页与彼此、旧数据的哈希重叠均为零。这是同构合成轨迹，背景材料共享，不能称为真实生产分布的独立采样。

覆盖审计发现，原 `first_reuse` 的混合长度前缀能放入 GPU，实际 CPU 存/载均为零。因此最终主矩阵完成后，另外使用四个 1025-token 的 DEV 前缀循环补充真实首次 CPU 回载诊断，原路径回载 30 页，新策略 24 页，单轮耗时比仅 **0.8436×**，TTFT p95 比 1.1803。这项补充结果只作为覆盖性反例，不作为五轮正向收益或参数选择依据；策略、数据主矩阵和阈值没有修改。它说明本轮容量策略仍会牺牲部分首次 CPU 复用，不能替代原通用路径。

DEV 同步路径总体 1.0340×，进入最终验证；批量路径总体 1.0159×，按预登记规则停止该后端候选。最终同步路径五轮：

|对照|总体加速比|冷热混合|工作集切换|结论|
|---|---:|---:|---:|---|
|容量策略 / 原同步成本策略|1.0113×|1.1172×|0.9087×|总体未达门槛|
|容量策略 / 最快旧异步路径|1.0017×|1.0980×|0.8985×|总体未达门槛|

冷热混合的 1.0980× 为五轮均快，TTFT p95 比 0.9060、事件间隔比 1.0014，输出一致，限定增量检查通过。工作集切换的 TTFT p95 比为 1.1141，超过 1.10，不能推广。

五轮总写回 **700 → 205 页**，减少约 70.7%；容量拒绝 405 次，真正替换 85 次。这里不能只看字节：

|场景|旧路径 prefill token|新策略 prefill token|原因边界|
|---|---:|---:|---|
|冷热混合|91,040|75,680|保留热点，减少重算|
|工作集切换|66,720|83,360|近期旧热点仍获保护，新热点适应较慢|

工作集切换多计算约 25% 的 prefill token。历史策略无法在第一次遇见新前缀时知道后面是否发生复用；保护旧热点和接纳新热点存在取舍。本轮保留这一负结果，停止调参。限定冷热混合的研究价值成立，通用服务配置不通过。

入口：`nanovllm/engine/cache_cost.py` 的 `retention_value/plan_admission`；同步/异步 `tiered_cache.py` 和批量 `cache_transfer.py` 负责排除受保护页及执行规划。旧校准文件未指定 replacement 时继续使用旧策略。新校准必须与本轮源码及物理 GPU 绑定，不能把文件交给不实现该策略的旧运行时并声称功能已启用。

## 第二项：没有传输时少做 CPU 工作

独立开关 `host_kv_transfer_idle_fastpath=False`，CLI 为 `--host-kv-transfer-idle-fastpath`。只在 batched 后端启用，默认关闭。源码中真正的复制、CUDA Graph、采样和数值算子保持原执行路径。

启用后，separate 和 mixed 在没有待写回页面时跳过空 flush；没有 CPU 可用前缀且没有待完成写回时跳过无意义的预取扫描；没有在途 H2D 时不展开消费页表；没有未记账消费页时不构造集合；已经观察过的请求不重复进入热度更新；没有待完成或待回收的复制时跳过 reap。

`needs_reap` 同时检查 `pending_stores`、`inflight` 和 `completed_loads`。GPU 复制完成但完成队列尚未交给 manager 时，目的页 lease 仍需释放，不能误判为空闲。每步保留 failed/原始 stream 检查，复制中的源页、目的页引用、取消、故障与结束时的 drain 依赖也保留。

同源码 flag-off/flag-on 对照由 `benchmarks/cache_idle_experiment.py` 执行；最快旧异步路径另作对照。四个真实模型运行覆盖 separate/mixed 的实际 CPU 回载，输出与存/载字节相同；255/256/257/513/1025 边界、固定历史 logits、有效 KV 和跨进程检查通过。

五轮正式检查完成，均保持原门槛：

|对照|配对中位数|更快轮数|TTFT p95 比|事件间隔 p95 比|相对门槛|
|---|---:|---:|---:|---:|---|
|快路径 / 同源码关开关|1.0068×|5/5|0.9861|0.9964|失败|
|快路径 / 最快旧异步|0.9990×|2/5|1.0002|1.0017|失败|
|自然容量：快路径 / 同源码关开关|1.0065×|5/5|0.9905|0.9955|失败|

压力场景实测资源：

|路径|Torch allocated 峰值 MiB|Torch reserved 峰值 MiB|LLM 构造中位数 s|绝对 SLO 请求|
|---|---:|---:|---:|---:|
|idle_control|3888.53|3968.00|2.432|105/560|
|idle_fastpath|3888.53|3968.00|2.428|105/560|
|legacy_async|3682.53|3800.00|2.252|105/560|

启动只计 LLM 构造，编译/文件页缓存为热态；不称为完整冷机启动。显存为 Torch allocator 实测峰值，CUDA context 等开销不在此列。快路径没有减少固定 staging，也不因此获得绝对 SLO/HTTP 验收。

## 第三项：布局与预取的继续/停止依据

独立 `profile_idle.py` 仅使用 DEV，正式性能进程不安装计时钩子。它分别检查空路径调用数，以及完整轨迹中的 gather、D2H/H2D stream span 和提交开销。H2D span 已包含 scatter，不重复相加；将 host 提交时间与全部设备传输时间相加时会重复计算重叠，所以只用作乐观串行预算，不能称为实测关键路径或普适数学上限。

实测调用数（独立 profiler，12 个 DEV 请求、每个输出 2 token，自然容量）：

|开关|flush|put_many|reap|消费等待|消费记账|热度观察|owner/failed 检查|
|---|---:|---:|---:|---:|---:|---:|---:|
|关|46|46|92|46|46|46|92|
|开|0|0|0|0|0|12|46|

压力轨迹的独立事件测量：全程 7564.19 ms；gather + D2H/H2D 设备 span 合计 78.78 ms，约 1.04%；提交 host wall 合计 10.99 ms。

把所有传输 span 和提交时间都假设可以完全删除，乐观串行估计也只有 **1.0120×**，低于 1.03×。这个估计重复计算了重叠，不是关键路径实测或不可能性定理；它用于判断本轮是否有足够依据增加布局复杂度。

当前 C1 调度没有更早拿到下一请求的可用预取窗口；H2D 数据被消费前仍需真实依赖，D2H 已异步。固定字节量没有减少。没有找到足以支撑原收益门槛的布局/预取证据，按预登记规则停止第三项，不新增 stream、重排 Attention KV 布局或宣传零拷贝加速。

详细三表（kernel/overlap/fuse）与传输预算见 `$R/PROFILE_IDLE_ANALYSIS.md`、`profile-*/report.json`、`profile-*/transfer-budget.json`。

## 正确性、已知边界与证据

- `cpu-policy-extended/summary.json`：26 套 CPU 回归；`cpu-policy002` 含 18 个成本用例。
- `cpu-fast-core`、`cpu-fast-planned`、`cpu-fast-idle`：快路径核心、真实 CPU planner 和空路径状态检查；复制依赖与完成队列不能被跳过。
- `model-candidate-reference.json`、`model-policy-{legacy,batched}002.json`：同源码旧路径参考和容量策略的全词表/有效 KV/跨进程严格对照。
- `model-fast-{reference,enabled}.json`、`idle-{separate,mixed}-{control,fast}.json`：快路径模型边界与真实回载联动。
- `formal-policy-analysis.json`、`policy-work-counters.json`：上述五轮结果与实际计算量。
- 启动时保存的模型权重 SHA、校准原始计时和每进程环境/主机并行负载均归档。`dependencies-current.txt` 为本轮实际依赖。

首次 CPU 用例因小张量预算按错误 dtype 大小编写而失败，修正测试预算后通过；首次模型跨版本比较被原工具的源码绑定检查拒绝，随后建立同一候选源码内的旧路径参考，未改数值阈值。空路径首个用例还发现 Scheduler 仍在每步 reap，补齐短路后保持原断言通过。失败文件保留。

旧 DEV 的跨 batch/query 数值反例仍是 `REVIEW_REQUIRED`，不能因为本轮 C1 输入恰好相同就宣称已解决所有数值路径。新配置不获得旧普通服务的 HTTP/30 分钟生命周期验收结论。当前只对本轮明确列出的模型、后端、内存预算和负载解释结果。

## 复现

先确认本轮物理 GPU3 稳定空闲；不要停止他人任务。更换模型/后端/GPU 需要独立校准，不伪造 UUID。所有输出用新名称：

```bash
export R=/home/hrxu/nano-vllm-archive/cache-policy-v2-20261028-001
export PY=/home/hrxu/nano-vllm/.venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-cffbe123-7f3a-fa27-4919-047ba4dd4790
export NANOVLLM_BENCH_GPU_INDEX=3
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

cd "$R/candidate"
"$PY" -B -m benchmarks.cache_policy_experiment suite --root "$R" --inputs "$R/inputs.json" \
  --freeze "$R/freeze-policy-and-idle.json" --split final --rounds 5 \
  --variants v1_legacy v2_legacy legacy_async --output "$R/repro-policy-001"
"$PY" -B -m benchmarks.cache_policy_experiment analyze --root "$R" \
  --inputs "$R/repro-policy-001/index.json" --output "$R/repro-policy-001-analysis.json"

cd "$R/candidate-fastpath"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run cache-cost --output "$R/repro-cost-cpu001"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run cache-idle --output "$R/repro-idle-cpu001"
"$PY" -B -m benchmarks.cache_idle_experiment suite --root "$R" --inputs "$R/inputs.json" \
  --freeze "$R/freeze-idle.json" --split final --rounds 5 --output "$R/repro-idle001"
"$PY" -B -m benchmarks.cache_idle_experiment analyze --root "$R" \
  --inputs "$R/repro-idle001/index.json" --output "$R/repro-idle001-analysis.json"

# 自动 GPU KV 容量下的同源码开关对照
"$PY" -B -m benchmarks.cache_idle_experiment suite --root "$R" --inputs "$R/inputs.json" \
  --freeze "$R/freeze-idle.json" --split final --rounds 5 --normal-capacity --output "$R/repro-idle-normal001"
"$PY" -B -m benchmarks.cache_idle_experiment analyze --root "$R" \
  --inputs "$R/repro-idle-normal001/index.json" --output "$R/repro-idle-normal001-analysis.json"
```

本轮停止：容量策略在固定冷热混合轨迹改善了热点污染，但工作集切换回退使通用配置失败；空路径清理已正确执行，是否达性能门槛按上表判断；传输布局缺乏支撑门槛的证据，不继续堆功能。两项功能默认关闭，普通服务和原缓存配置不升级。结果各自绑定其冻结源码；集成后不将不同版本的增量相乘，也不声称组合性能已经验证。已知跨形状数值反例仍由下一份独立协议处理，本轮不放宽阈值或重写全局数值算子。
