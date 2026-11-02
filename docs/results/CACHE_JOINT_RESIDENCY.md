> 历史冻结报告：以下 mixed 限制和负例记录原样保留；后续已定位并修复 KV 身份时序错误，最新代码与验收见 [JOINT_MIXED_FIX](JOINT_MIXED_FIX.md)。

# GPU/CPU 联合前缀驻留：冻结结果与复现

证据目录：`/home/hrxu/nano-vllm-archive/cache-joint-20261028-001`。本轮只实现一个算法候选，没有参数网格。先独立 TRAIN 拟合有界扫描长度，再做 DEV，冻结后完成两组各五轮的 GPU-only / 最快旧 legacy_async / joint 对照。补充 DEV mixed 检查发现反例后，仅新增 Config 范围检查；用同一冻结数据和算法重新执行 DEV、冻结和两组五轮，最终交付版本另有完整 30 个独立正式 worker。两版结果全部保留。新增开关默认 `None`；普通服务与旧缓存配置不变。

## 判断口径

RTX 4090 物理 GPU3，Qwen3-1.7B BF16，TP1，model KV，C1，1025-token 输入、8-token 输出，chunk128，预算256。decode CUDA Graph 正常开启。对照同物理卡、相同模型/输入/实际 GPU KV 容量；CPU 缓存两条路径均为 8×28 MiB。压力组固定 GPU 12 页；自然容量组使用相同自动显存预算，不人为制造容量不足。两个组不能混成一个加速数字。

数值沿用 logits `atol=rtol=0.01`、同路径有效 KV 逐位一致；正式自由生成 token 对各自参考完全相同。收益要求配对中位数 ≥1.03×、五轮全部更快、TTFT 与真实输出事件间隔 p95 比均 ≤1.10。TTFT 1000 ms、TPOT 30 ms 的绝对 SLO 另报，增量通过不等于服务 SLO 通过。

DEV 的目标场景改善，但全部场景对两个基线的筛选未通过，原始状态 `DEV_SCREEN_FAIL_DIAGNOSTIC_ONLY`。按预登记规则停止调参，冻结为 `FROZEN_DIAGNOSTIC_ONLY_DEV_FAILED`。以下正式结果用于检验作用范围和负结果；不能事后删除负对照、修改 DEV 规则或升级默认配置。各子项增量门槛与整体配置决策分开解释。

## 具体修改

1. `nanovllm/engine/joint_cache.py`：新增独立 `JointResidencyPolicy`。GPU 受害页只从无引用页中选择，按最近实际请求排序；同次请求先选 CPU 已有副本的页，再选更深的页，保留可直接使用的前导前缀。CPU 替换先回收已有 GPU 副本，再处理祖先缺失的页，最后按近期请求与深度排序。首次出现的已观察前缀不因旧热点累计频次低而被拒绝。
2. `block_manager.py`：一次请求同时记录 CPU 链式 key 与 GPU hash 的映射；GPU victim 选择接入策略。回载前 pin 所有候选 CPU 源，才允许备份 GPU 受害页。旧路径继续保留原回载行为。
3. `tiered_cache.py`：除已提交 H2D 的源外，替换规划还排除本次尚未提交的回载源。保持已有同步 D2H、异步 H2D、consumer event 与 GPU 页租约；没有增加 stream、放松 fence 或改变算术。
4. Config/CLI：`host_kv_joint_residency_path` / `--host-kv-joint-residency-path`，默认关闭。首版只允许 TP1、dense、model KV、prefix cache、legacy async、非推测、**separate** 调度，与旧 cost-policy/batched 路径互斥；性能只验收 C1。mixed+joint 在加载模型前明确报错。
5. 测试与实验：新增联合驻留 CPU 状态检查、真实 GPU 备份/取消/部分提交故障检查；扩展原模型检查的 `--joint-policy`。新实验复用旧 guard、run_trial、SLO 定义，额外将尾部真实传输 drain 计入总耗时，不安装 profiler/逐 step hooks。

这是一套联合元数据策略，CPU 在 promotion 后仍可暂存 GPU 副本，待容量压力时优先回收；不是完整 exclusive-cache 实现。父页可用性只做一阶判断，不是整条前缀闭包最优规划；回载门槛沿用旧 H2D 成本比较，没有完整预测本次回载会引发的 GPU 受害页 D2H 和替换机会成本。组件共同构成单一候选，本轮没有把总收益分别归功于某一个组件，也不称为完整联合成本最优算法。

## 数据与冻结

新种子 TRAIN/DEV/final 为 2026102811/12/13，各 100 条请求、48 个不同 prompt，包含热点切换、首次 CPU 复用、冷热混合、全唯一与共享分支。三个 split 使用不同原创结构化材料，保留全部文本与 token。对彼此及两轮旧输入，完整 prompt、首个 256-token 页和各级前缀链哈希重叠均为零；这是合成缓存轨迹，不是生产流量或独立主题抽样。

首次 CPU 复用为四个不同 1025-token 前缀循环四次，16 个完整前缀页超过 12 页 GPU 容量，原路径确实发生 CPU 回载。扫描上限只用 TRAIN 的前缀页数 p90 套用 `clip(4*p90,8,64)`，得到 **16**；历史容量固定 4096。这里的拟合是轨迹统计，不是训练神经网络预测器。没有用 DEV/final 拟合频次、窗口或其他参数。成本表复用先前同模型、同 GPU、同后端的实测表，未伪造新校准数值。

`protocol.json` 在新数据前记录；`harness-protocol.json` 在分词准备阶段绑定工具及口径；`freeze001.json` 绑定首版。额外 DEV mixed 生命周期发现一个请求第 4 个输出 token 不同后，按 `protocol-scope-guard.json` 只在 Config 拒绝 mixed，未改变 separate 执行、算法、扫描上限、成本表、数据或阈值；原首版完整保存在 `candidate-before-scope-guard`。新 `freeze002.json` 在重跑任何最终 GPU 执行前绑定最终源码与全部配置，`correctness-scope-before-final.json` 绑定该版本原数值边界检查。最终结果不进入调参，也不能通过限制范围删除已有负例。

压力组首次执行在第四轮 worker 的 case 前检测到外部 GPU 任务，守卫退出；没有停止外部进程。GPU3 连续三次恢复空闲后，前三轮全部 9 个完整 worker 的源码、原始 trial 和 guard 哈希核对通过，原样保留；仅补齐后两轮，结果索引为 `formal002/index.json`。中断 worker 整体不计入有效计时，其失败记录仍保留在 `formal001`。恢复协议 `protocol-resume.json` 在补齐前写入，没有按速度筛选或修改冻结内容。

## 五轮：容量压力

|负载|相对最快旧异步|相对 GPU-only|对旧异步 TTFT / 事件 p95 比|五轮增量门槛|
|---|---:|---:|---:|---|
|pooled|1.0419×|1.1331×|0.9782 / 0.9994|旧异步 PASS；GPU-only PASS|
|phase_shift|1.0132×|1.0753×|0.9793 / 1.0003|旧异步 FAIL；GPU-only PASS|
|first_cpu_reuse|1.3033×|1.9966×|0.7648 / 0.9939|旧异步 PASS；GPU-only PASS|
|mixed_hot_cold|1.0023×|0.9917×|0.9828 / 0.9968|旧异步 FAIL；GPU-only FAIL|
|unique|1.0048×|0.9867×|0.9939 / 0.9998|旧异步 FAIL；GPU-only FAIL|
|shared_branches|1.0005×|1.0014×|1.0014 / 0.9992|旧异步 FAIL；GPU-only FAIL|

|路径|stores / loads|prefill token / forward 次数|allocated / reserved 峰值 MiB|构造中位数 s|绝对 SLO 请求|
|---|---:|---:|---:|---:|---:|
|gpu_only|0 / 0|407540 / 7180|3664.53 / 3744.00|2.244|95 / 500|
|legacy_async|1200 / 200|356340 / 6780|3682.53 / 3800.00|2.243|94 / 500|
|joint|1320 / 280|335860 / 6620|3682.53 / 3800.00|2.237|95 / 500|

从实际计算量解释结果（五轮计数一致，下面按一轮列出）：

|负载|每轮 prefill token：旧 → 新|每轮 forward：旧 → 新|每轮 stores / loads：旧 → 新|
|---|---:|---:|---:|
|first_cpu_reuse|8720 → 5648|196 → 172|35 / 30 → 53 / 42|
|phase_shift|30240 → 29216|492 → 484|111 / 10 → 117 / 14|
|mixed_hot_cold|17432 → 17432|328 → 328|57 / 0 → 57 / 0|

首次 CPU 复用的 prefill 重算减少约 35.2%，即使存载次数增加，仍得到稳定收益，说明优化目标应是净执行成本而非最少传输字节。热点切换只少算约 3.39% 的 prefill token，写回与回载反而增加，净收益余量小。首版五轮第一轮相对旧异步约 0.9979×，未满足五轮全快；这个反例保留，不能因为范围修复后重跑得到另一组数值就删除。最终版本按上表原门槛独立判断；五轮状态计数一致，未发现这组不稳定来自策略状态随机漂移。执行耗时有波动，但不能仅凭主机负载记录就把失败归因于噪声。

当前冷热混合中两条 CPU 路径都没有回载，prefill 计算量相同，joint 没有减少实际工作；无复用与能放入 GPU 的共享分支也不应预设 CPU 层有收益。本轮数据与上一轮不同，不能将上一轮的退化比值与本轮正结果直接相减或相乘来宣称改善幅度。

## 五轮：正常自动显存容量

|负载|相对最快旧异步|相对 GPU-only|对旧异步 TTFT / 事件 p95 比|五轮增量门槛|
|---|---:|---:|---:|---|
|pooled|1.0050×|1.0076×|0.9925 / 1.0003|旧异步 FAIL；GPU-only FAIL|
|phase_shift|1.0123×|1.0031×|0.9874 / 1.0030|旧异步 FAIL；GPU-only FAIL|
|first_cpu_reuse|1.0003×|1.0052×|1.0003 / 0.9993|旧异步 FAIL；GPU-only FAIL|
|mixed_hot_cold|1.0045×|1.0055×|0.9893 / 0.9983|旧异步 FAIL；GPU-only FAIL|
|unique|1.0223×|1.0135×|0.9836 / 0.9989|旧异步 FAIL；GPU-only FAIL|
|shared_branches|0.9851×|1.0034×|1.0238 / 1.0020|旧异步 FAIL；GPU-only FAIL|

|路径|stores / loads|prefill token / forward 次数|allocated / reserved 峰值 MiB|构造中位数 s|绝对 SLO 请求|
|---|---:|---:|---:|---:|---:|
|gpu_only|0 / 0|238580 / 5860|19904.53 / 19984.00|2.236|94 / 500|
|legacy_async|0 / 0|238580 / 5860|19904.53 / 19984.00|2.247|97 / 500|
|joint|0 / 0|238580 / 5860|19904.53 / 19984.00|2.253|95 / 500|

正常容量各路径实际均为 592 个 GPU KV 页，所有 CPU 存/载均为零，各路径计算量一致。这里的 `first_cpu_reuse` 只是沿用轨迹名，实际发生的是 GPU 复用；真实 CPU 首次回载的覆盖来自压力组，不能混淆。这个负对照没有达到稳定收益门槛，普通单卡正常显存场景继续优先 GPU-only。

速度为每一轮基线总耗时 / joint 总耗时，再取五轮中位数；不是平均各 case 加速比。全部计入完整请求循环及尾部复制 drain。延迟来自真实 token/text 事件，不为批量事件补造时间戳。逐轮比值、每请求 token、环境与主机并行负载均在对应原始 JSON。

显存为 Torch allocator 峰值，不包含全部 CUDA context；启动只计 LLM 构造，文件缓存/JIT 可为热态，不能称为完整冷机时间。CPU 缓存配置 224 MiB；legacy 路径逐页分配 pinned tensor，实际 resident/pending 字节另有原始记录。不能把 CPU 容量算作免费资源。

## 正确性及边界

新增 16 项 CPU 检查、原核心与缓存/调度相关回归通过。真实 GPU 检查验证回载期间备份、源 pin、取消后的目的页 lease、第二次 H2D 提交异常及回收。最终 separate 版本真实模型覆盖 255/256/257/513/1025-token 边界，固定历史 logits、有效 KV 逐位一致及跨进程比较通过；EOS、取消与异常清理保持原断言。

mixed 的直接固定历史回载检查曾通过，但补充的真实 mixed planner 热点切换 DEV 轨迹有一个请求第 4 个生成 token 不同，`mixed-lifecycle-comparison.json` 明确标记 `REVIEW_REQUIRED_MIXED_TOKEN_MISMATCH`。当前证据没有确认是 BF16 跨 query 形状差异还是其他问题，不能宣称已经定位根因。首版的正向检查不能覆盖这个反例；最终交付直接拒绝 joint+mixed，不放宽阈值、不全局改算子。

旧跨 batch/query BF16 数值反例仍为 `REVIEW_REQUIRED`，本轮没有修复全局算术。性能脚本的 exact-token 检查不能替代固定历史 logits/KV 检查。新配置没有新增 HTTP/SSE 30 分钟服务验收；只支持本报告所列范围，默认关闭。

## 复现

必须先确认物理 GPU3 稳定空闲，不停止其他任务。换 GPU/模型/后端须重新校准并建立独立协议，不伪造 UUID。以下复用已冻结输入和源码，输出改为全新名称：

```bash
export R=/home/hrxu/nano-vllm-archive/cache-joint-20261028-001
export PY=/home/hrxu/nano-vllm/.venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-cffbe123-7f3a-fa27-4919-047ba4dd4790
export NANOVLLM_BENCH_GPU_INDEX=3
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "$R/candidate"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run cache-joint --output "$R/repro-cpu001"
"$PY" -B -m tests.regression.check_joint_cache_gpu \
  --policy "$R/calibration-joint.json" --output "$R/repro-copy001.json"
"$PY" -B -m tests.regression.check_cache_runtime_gpu --backend legacy \
  --joint-policy "$R/calibration-joint.json" --compare "$R/model-reference004.json" \
  --output "$R/repro-model001.json"
B="$R/candidate/benchmarks/joint_cache_experiment.py"
"$PY" -B "$B" suite --split final --freeze "$R/freeze002.json" --output "$R/repro-pressure001"
"$PY" -B "$B" analyze --index "$R/repro-pressure001/index.json" --output "$R/repro-pressure001-analysis.json"
"$PY" -B "$B" suite --split final --freeze "$R/freeze002.json" --normal-capacity --output "$R/repro-normal001"
"$PY" -B "$B" analyze --index "$R/repro-normal001/index.json" --output "$R/repro-normal001-analysis.json"
```

原始负结果与首次工具失败保留；本轮不继续对已看过的 DEV/final 调参。是否值得下一候选，需要新的独立数据与预登记规则。普通 GPU-only 与原异步路径继续可用。
