# 最终验收与学习交付：2026-10-25

本轮实现、测量与有界研究已收尾。**通过了服务生命周期和多项正确性检查，但并非所有优化都可推荐。** 五套原配置独立保留，缓存与 draft 不组合，全部全局默认值不变。完整机器可读决定见 [CONFIG_DECISIONS.json](CONFIG_DECISIONS.json)。

## 先看配置结论

|配置／改动|实际结果|最终结论|
|---|---|---|
|普通1.7B服务|当前源码 TP1/C8、1800.68秒 HTTP/SSE；接纳16,341，完成12,256，预期取消4,085；五阶段PASS|可运行已验收配置，使用 serve_validated.py；TTL5为本次明确测试设置|
|GPU–CPU前缀缓存|固定336MiB GPU KV＋1GiB CPU，重复前缀五轮中位1.3032×；无复用无稳定收益|仅容量压力吞吐证据；绝对SLO30/180，不能宣传为生产SLO达标，默认关闭|
|FP16独立draft|同consistent target下1.7930×且全部token一致；比更快Torch普通decode1.6661×，但后者输出不同|事件间隔p95退步，不全局推荐；保留为明确数值后端的研究配置|
|8B AWQ/Marlin target|稳态2.3020×，峰值allocated15.51→5.95GiB；质量通过，Q1边界/Graph/跨进程PASS|目标模型分布改变；只对本次重复文本类别通过全部对应门槛；中文/英文/代码TTFT回退，不推广全负载|
|旧MSE量化draft|相对FP16仅1.0043×，接受率46.98%→44.99%|未达固定1.03收益门槛，且dense/Marlin整模型logits仍REVIEW_REQUIRED|
|新MSE checkpoint|DEV NLL/KL通过；英文接受率改善，中文下降、验证次数增加|按冻结停止规则拒绝；未进入最终质量或正式性能验证，不追加调参|
|异步draft metadata|输出/计数完全一致；正式中位1.0003×，一轮更慢|无稳定收益，独立开关保留但默认OFF|
|随机推测|CPU穷举与统计、GPU每例100,000样本、真实8B Graph/eager生命周期通过|temperature-only、显式启用；未宣称端到端加速，不代表修复BF16数值分歧|

## 正式速度和延迟：不要把吞吐当作所有延迟都改善

五轮交错、同一物理GPU0，每组48个冻结prompt、每prompt64输出；每worker48 warmup＋48 steady。每variant240条steady，共1680条，未从中选择最快trial。普通decode Graph正常开启。短prompt不足256token整页，无前缀复用；缓存另做独立容量测试。

|配置|相对更快普通decode的配对中位吞吐比|TTFT p95 ms|真实事件间隔p95 ms|稳态峰值allocated GiB|
|---|---:|---:|---:|---:|
|普通 BF16 decode|1.0000×|56.35|17.38|15.51|
|consistent 普通 decode|0.9301×|63.55|18.65|15.50|
|FP16 draft K4|1.6661×|60.62|27.47|16.75|
|FP16 draft＋异步 metadata|1.6676×|60.46|27.48|16.75|
|旧 MSE 量化 draft|1.6756×|59.23|26.22|16.15|
|8B AWQ/Marlin target|2.3020×|62.08|7.02|5.95|

Torch普通decode与consistent的输出不同；1.6661×属于跨数值后端计时，不能称为原Torch输出的无损加速。native相对自己的consistent基线是精确token对照。AWQ改变参考模型，不要求逐token等于BF16。所有variant在自己的五轮中输出稳定。

AWQ pooled TTFT比值为 **1.1016675 > 1.10**，代码类别为 **1.1007387 > 1.10**，没有四舍五入成通过。native pooled事件间隔比值1.5804，也未放宽门槛。绝对1000ms TTFT／30ms摊销TPOT和相对尾延迟是不同判定，详见 完整SLO报告（服务器归档：`/home/hrxu/nano-full-opt-20261024/service_tools/tail_gate/formal_slo_actual001/formal_slo_report.json`）。

稳态显存是Torch实际allocated峰值，包含模型、KV和Graph驻留，不能称为总NVML或冷加载最大值。新进程engine初始化分别约4.10s普通、6.52s FP16 draft、7.26s旧MSE、4.28s AWQ；checkpoint预先哈希使文件缓存已热，未声称冷磁盘加载。TTFT不包括模型初始化。详见 正式原始分析（服务器归档：`/home/hrxu/nano-full-opt-20261024/formal_analysis001.json`）。

## 本轮具体修改了什么

服务器 `/home/hrxu/nano-vllm` 已同步8个文件，其他源码、Git HEAD/index和已有修改保留；合入前后源码快照及9套共享目录CPU回归见 合入凭据（服务器归档：`/home/hrxu/nano-full-opt-20261024/runtime_tools/shared_integration001/FINAL_RECEIPT.json`）。冻结实验源码为 `candidate/`；本地可读副本为 `final-source/`。

|文件|原因与实现|
|---|---|
|nanovllm/config.py|新增独立 draft_async_metadata=False、校验及有效配置记录；TP1、fixed policy等限制保留|
|engine/draft_graph.py|固定pinned host状态，non_blocking H2D；每owner CUDA event保护复用，异常时用原stream fence，关闭前等待；默认阻塞参考路径保留|
|engine/draft_runner.py|把开关贯穿Q1 token／Q2 suffix Graph与清理，复用已有缓冲区；不改模型数学或KV语义|
|benchmarks/check_draft_graph.py|增加异步开关的真实GPU验证入口，保留原断言|
|benchmarks/check_marlin_draft_lifecycle.py|增加257边界、Q1/2/4/8及target有效KV覆盖；原 .01/.01 不变|
|engine/speculative_sampling.py|普通FP32 softmax权重不变，在producer显式FP64归一化并保留FP64 p/q；严格validator逐字不变|
|engine/stochastic_draft.py|q的ownership、空proposal dtype及统计同步为FP64，workspace warmup包含真实概率驻留|
|benchmarks/check_speculative_sampling.py|两项期望独立计算同一FP32权重的FP64归一化；仍精确torch.equal，未换成宽松allclose|

概率故障实际为prefix256的target p总和 **1.0000010285328311**；draft Graph/eager相同，全部有限非负。修复producer后仍用FP64的 **1e-12** 检查；外来非法概率继续拒绝。CPU原20项＋9项新增边界检查、真实8B两路径、GPU概率统计均通过。不能将此修复描述成RoPE或BF16数值问题的修复。

实验工具另修复了AutoRound报告torch.dtype序列化及原生即时packing的I32/F16 buffer审计；没有改MSE训练目标、seed、数据或200迭代配方。001/002工程失败全部保留，003是同一固定配方唯一完成导出的候选。

## 为什么量化draft没有成为推荐配置

共同历史诊断没有发现packed布局加载错误：局部同输入投影通过；权重量化的dense参考已经出现主要NLL损失。但Marlin/dense完整logits最大差0.24609375、首个严格超限在layer3 k_norm，仍保留REVIEW_REQUIRED，不能断言已完全排除kernel问题。首差异证据（服务器归档：`/home/hrxu/nano-full-opt-20261024/diagnosis/FD002_FINDINGS_ZH.md`）。

新配方从官方FP16起点，全28层、W4A16/g128/asym/200iters；TRAIN256×256，DEV64×256，独立最终集64×256。ID、整行与32-gram隔离审计通过。新候选DEV整体NLL增量0.057433、KL0.078226低于旧MSE0.090005，英文KL也下降；但中文1200/4131 <1218/4062，验证3195>3189，依冻结规则停止。

实际当前DEV6诊断中，新旧MSE在两条中文上首候选即拒绝 **24→32**，verification **62→66**；英文则首拒 **14→9**、verification **46→42**。相同已提交历史能观察到候选由对变错。未保存原始top2/margin，不能臆断全部由低margin或某层造成。六条总verification139→142；这增加了执行工作量，不只是单轮kernel变慢。instrumented成本与历史DEV另列，均不是正式速度：完整剖析（服务器归档：`/home/hrxu/nano-full-opt-20261024/runtime_tools/PROFILE_FINDINGS_ZH.md`）。

NLL/KL是分布平均指标，不保证argmax和连续接受前缀。更接近0.6B FP16也不等于更接近8B target的决定。最终新候选没有再看holdout或选择新参数。

## 如何学习和复现

从 本地七模块源码入口（服务器归档：`/home/hrxu/nano-full-opt-20261024/learning/LOCAL_SOURCE_ENTRY_ZH.md`） 开始，再读 完整路线（服务器归档：`/home/hrxu/nano-full-opt-20261024/learning/READING_GUIDE_ZH.md`）、20个面试问题（服务器归档：`/home/hrxu/nano-full-opt-20261024/learning/INTERVIEW_20_ZH.md`）。顺序是请求生命周期→KV reserve/commit/rollback→Graph→独立draft→概率校正→Marlin→校准评测。每一部分先画状态变化，再复现测试，最后独立做一个小改动并解释结果。

本轮每variant15360输出：普通target forward15360；FP16 draft target5615＋draft20745，接受9745；旧MSE target5775＋draft21305，接受9585。这解释了接受率降低如何抵消低bit计算收益。此处target调用＋接受数=输出数依赖短prompt单次prefill，不能推广到chunked prefill。

[运行与复现命令](REPRODUCE_ZH.md) · [简历版本](RESUME_FINAL_ZH.md) · [已知问题/Handoff](HANDOFF_FULL_OPT_20261025.md)

贡献边界：基于已有nano-vllm工程；分页KV、TP和基础Graph不是本轮原创，缓存/draft集成来自前序迭代。本轮新增上述8文件修复及可复现实验闭环。Marlin复用vLLM0.17.1已验证kernel，AutoRound负责校准算法，不能写成自研Marlin CUDA内核。模型revision、依赖和完整权重哈希见 模型provenance（服务器归档：`/home/hrxu/nano-full-opt-20261024/reproduction/model_provenance_v1.json`） 与 环境清单（服务器归档：`/home/hrxu/nano-full-opt-20261024/environment_final001.json`）；缺失upstream revision明确标UNKNOWN_LOCAL_CHECKPOINT_CONTENT_HASHED。
