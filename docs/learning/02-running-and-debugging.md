# 动手运行与调试：从一次请求到完整验收

这份指南针对服务器 `/home/hrxu/nano-vllm` 的当前分支，目标是让你能自己启动、观察、断点和复现。每个 case 说明做什么、怎么运行、看哪里、预期是什么。教学 demo 只验证短请求的执行路径，性能与质量结论仍以各自冻结的正式实验为准。

建议顺序：环境与 CPU → 单步和 prefix → HTTP/SSE → Graph → draft → 量化 → 专项回归 → 正式实验。每次只改变一个开关。

## 0. 先进入服务器，确认环境

下面终端均指服务器的 bash。Windows 上先执行：

```powershell
ssh hrxu@10.29.98.230
```

服务器上：

```bash
cd /home/hrxu/nano-vllm
export PY=/home/hrxu/nano-vllm/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
"$PY" -V
"$PY" -B -m tests list
"$PY" -B -m benchmarks list
"$PY" -B examples/inspect_runtime.py --case step --show
```

`--show` 只打印本 case 的配置，不初始化 CUDA。`configs/*.json` 外层包含 status/decision，只有 `config` 是 LLM 初始化参数；draft 的 `sampling` 单独传给 `SamplingParams`。

选卡前连续观察至少两次，并查看计算进程，不能只看利用率：

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
sleep 5
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

仅当同一张卡稳定空闲时，把下面示例换成它的**完整 UUID 和对应物理 index**：

```bash
export CUDA_VISIBLE_DEVICES=GPU-cffbe123-7f3a-fa27-4919-047ba4dd4790
export NANOVLLM_BENCH_GPU_INDEX=3
"$PY" -B -m benchmarks.runtime_v2_gpu_guard
```

预期 JSON 的 `status` 是 `AVAILABLE`。guard 拒绝外来计算进程、错误 UUID/index，但不是独占锁；检查到其他人占用时停止新实验，不停止对方任务。启动前与每次独立 case 都重新检查。一个脚本正在用 GPU 时，不要同时启动第二个 case。

建立唯一输出根目录；下面所有 `$RUN` 都指当前服务器终端中的这个目录：

```bash
RUN="/home/hrxu/nano-vllm-archive/learning-runs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$RUN"
printf '%s\n' "$RUN"
```

每个 demo 的 `--output` 必须是尚不存在的目录；重新运行用新的目录名。日志和报告不会覆盖旧实验。

## 1. CPU 回归：调度、资源和采样规则

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run core --output "$RUN/cpu-core"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run extended --output "$RUN/cpu-extended"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests.regression.check_stochastic_engine \
  --cpu --output "$RUN/stochastic-cpu.json"
```

预期 core 为 9 套 PASS；extended 为 25 套 CPU 检查 PASS；随机推测 CPU 测试输出 `status: PASS`。看 `$RUN/cpu-core/summary.json`、逐套 `.log` 和 `$RUN/stochastic-cpu.json`。

这些测试覆盖调度、KV 账本、缓存/draft、概率采样、Graph 配置和服务状态等 CPU 逻辑。它们不能证明真实 GPU 输出、量化质量或速度。若失败先读对应 log，保留报错；不要修改断言让它通过。

想只调一组：

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run execution-plan --output "$RUN/cpu-plan"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run tiered-cache --output "$RUN/cpu-tiered"
CUDA_VISIBLE_DEVICES='' "$PY" -B -m tests run speculative-sampling --output "$RUN/cpu-sampling"
```

## 2. 真实 GPU 单步：chunked prefill 和 decode

```bash
"$PY" -B -u examples/inspect_runtime.py --case step --output "$RUN/step"
```

做什么：加载 1.7B，构造 7 个固定 token ID，关闭前缀复用、chunk=3、最多生成 3 个 token；逐次调用真实 `LLMEngine.step()`。这些固定 ID 用来观察状态，不以生成文字质量为目标。

预期共五轮：

|轮次|`scheduled_prefill`|`query_decode`|`generated_tokens`|`cached_after`|
|---|---:|---:|---:|---:|
|1|3|0|0|3|
|2|3|0|0|6|
|3|1|0|1|7|
|4|0|1|1|8|
|5|0|1|1|0|

最后一轮请求结束并释放页，所以 `cached_after=0`，不是第 5 轮没有计算 KV。`scheduled_after` 是 postprocess 后的值，已经清零。最终报告应显示 `DEMO_COMPLETED_NOT_ACCEPTANCE`，资源计数为 0：

```bash
cat "$RUN/step/report.json"
```

从 `scheduler.py:92`、`:141`、`:210` 和 `model_runner.py:352` 追踪这些字段。

## 3. 前缀复用：第二次少算哪些输入

```bash
"$PY" -B -u examples/inspect_runtime.py --case prefix --max-tokens 8 --output "$RUN/prefix"
```

做什么：在同一个引擎中，对同样 300 个 token 连续发两次请求。当前页大小 256，第一轮创建完整前缀页，第二轮复用它。

预期第一请求 `prefill_tokens=300`、`prefix_hits=0`；第二请求 `prefill_tokens=44`、`prefix_hits=1`。完整页以外的剩余 token 仍要计算。查看 `records`、`scheduler.prefix_cache_*` 和输出 token；这是 GPU 前缀复用的学习例子，不是 CPU 分层缓存命中或吞吐结论。

## 4. 普通生成、Graph/eager 与取消

```bash
"$PY" -B -u examples/inspect_runtime.py --case ordinary --max-tokens 8 --output "$RUN/ordinary"
"$PY" -B -u examples/inspect_runtime.py --case ordinary-eager --max-tokens 8 --output "$RUN/ordinary-eager"
"$PY" -B -u examples/inspect_runtime.py --case cancel --max-tokens 8 --output "$RUN/cancel"
"$PY" -B -u examples/inspect_runtime.py --case stop --max-tokens 8 --output "$RUN/stop"
```

前两条使用相同 prompt 和 greedy；`ordinary-eager` 只把 `enforce_eager` 改为 True。预期正常结束、各有 8 个事件；在 `runtime_metrics.decode_graph` 中，普通配置有 `decode_replays`，eager 会记录 `decode_fallback_disabled`。可检查输出 ID 是否一致，但单次一致不是所有形状的一致性证明。

`cancel` 在第一个事件后关闭 generator，观察 `resource_state.used_blocks`、`reservations`、`host_load_leases`、`engine_requests` 均为 0。它模拟 Python 流消费者退出；HTTP 断连及慢消费者由下面服务检查和专项回归覆盖。`stop` 先生成一次，取已生成文本的首字符作为停止串，再发起同一个 greedy 请求，预期最终事件的 `finish_reason` 为 `stop`，停止串不会作为可见输出返回。EOS 和长度限制另由生命周期回归覆盖。

报告中的 `elapsed_ms` 是教学演示的本机事件时间，包含调用开销，不能当成正式 TTFT/TPOT 测量。draft 一次事件可以提交多个 token，`event.token_id` 是该事件最后一个 token，不能把事件数量当作输出 token 数。

脚本在请求尚未注销时另外保存完整 `completion_token_ids`。可以用下面的命令比较 Graph/eager 的整个输出，而不只看每个事件最后一个 token：

```bash
"$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); print("equal:", a["completion_token_ids"] == b["completion_token_ids"]); print("graph:", a["completion_token_ids"]); print("eager:", b["completion_token_ids"])' "$RUN/ordinary/report.json" "$RUN/ordinary-eager/report.json"
```

## 5. 启动 HTTP/SSE 服务

先退出已运行的 GPU demo；本节使用原冻结验收源码与配置，端口默认 18784：

```bash
"$PY" -B scripts/serve_validated.py
"$PY" -B scripts/serve_validated.py --execute --host 127.0.0.1 --port 18784
```

第一条只展示配置；第二条校验冻结源码与模型、执行 guard 后启动并阻塞。它是 1.7B TP1/C8 普通服务，draft 和 CPU 分层缓存关闭；严格启动器运行的是历史冻结源码，不等同于当前工作树。启动失败先读实际异常，不绕过哈希或 guard。

在服务器的**第二个终端**：

```bash
curl -s http://127.0.0.1:18784/health
curl -s http://127.0.0.1:18784/v1/models
curl -s http://127.0.0.1:18784/metrics

curl -s http://127.0.0.1:18784/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/hrxu/huggingface/Qwen3-1.7B","prompt":"Explain CUDA in one sentence.","max_tokens":16,"temperature":0,"stream":false}'

curl -N http://127.0.0.1:18784/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/home/hrxu/huggingface/Qwen3-1.7B","prompt":"Explain CUDA in one sentence.","max_tokens":16,"temperature":0,"stream":true}'
```

预期 `/health` 返回 JSON；普通请求返回 JSON completion；流式请求返回多个 SSE `data:` 事件并结束。模型输出文字不固定。观察 `/metrics` 中请求与运行时计数。客户端中断一次 `curl -N` 可观察取消，但服务清理仍以对应生命周期回归为准。

服务只绑定服务器回环地址。若要从自己的 Windows 访问，在 Windows 另开终端建立 SSH 隧道：

```powershell
ssh -N -L 18784:127.0.0.1:18784 hrxu@10.29.98.230
```

然后访问本机 `http://127.0.0.1:18784`。实验结束在服务终端 Ctrl-C；只关闭你启动的服务。

如果要断点调试**当前工作树服务**，先重新检查空闲卡，然后在项目目录执行下面独立实验配置：

```bash
"$PY" -B -m benchmarks.runtime_v2_gpu_guard && \
"$PY" -B serve_nano_vllm.py --model /home/hrxu/huggingface/Qwen3-1.7B \
  --host 127.0.0.1 --port 18785 --tensor-parallel-size 1 \
  --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 256 \
  --prefill-chunk-size 128 --batching-mode separate --speculative-policy off
```

此命令和冻结服务是不同配置，不继承历史 30 分钟验收结论。想先调 Python 控制流，可加 `--enforce-eager`；该模式不用于比较 Graph 性能。

## 6. 独立 draft 和 Graph 的实际路径

先体验无需加载小模型的 n-gram proposer：

```bash
"$PY" -B -u examples/inspect_runtime.py --case ngram --max-tokens 16 --output "$RUN/ngram"
```

这个 case 用重复 token 构造历史，并启用 fixed n-gram 推测。查看 `runtime_metrics.speculative`；候选数取决于生成历史是否匹配，候选为空或回退普通 decode 也可能是正常结果。它与下面的独立 draft 是两种 proposer。

以下命令每次加载 8B target 和 0.6B draft，TP1、单请求；显存与启动开销明显高于 1.7B 普通 case。每条运行前重查 GPU。

```bash
"$PY" -B -u examples/inspect_runtime.py --case draft --max-tokens 16 --output "$RUN/draft"
"$PY" -B -u examples/inspect_runtime.py --case draft-prefill --max-tokens 16 --output "$RUN/draft-prefill"
"$PY" -B -u examples/inspect_runtime.py --case draft-device --max-tokens 16 --output "$RUN/draft-device"
"$PY" -B -u examples/inspect_runtime.py --case draft-packed --max-tokens 16 --output "$RUN/draft-packed"
```

|case|在做什么|重点看什么|
|---|---|---|
|`draft`|原 FP16 draft K4＋target 验证|`runtime_metrics.speculative`、`draft_model.proposer` 的提议/接受/验证计数|
|`draft-prefill`|在 draft 上显式开启 prefill Graph|`prefill_graph_replays`、`prefill_graph_fallbacks`；具体短 prompt 可能回退|
|`draft-device`|加入尾块 Graph、GPU 候选及依赖的 metadata 配置|`device_proposal_enqueues`、`device_proposal_completions`、验证图计数|
|`draft-packed`|在前一配置上加入 target 权重物理布局|`effective`、输出 token 和两侧 KV 回归；不据单次耗时宣称加速|

这些 case 使用不同进程和相同默认 prompt/greedy，请检查完整报告的 `runtime_metrics`。优化后应保持同一 consistent target 的输出与接受/验证语义；一次 smoke 不足以证明所有边界。实际事件的 `token_index` 可能跨多个位置，体现一轮提交多个 token。

## 7. 随机推测、分层缓存与量化

```bash
"$PY" -B -u examples/inspect_runtime.py --case stochastic --max-tokens 16 --output "$RUN/stochastic"
"$PY" -B -u examples/inspect_runtime.py --case tiered --max-tokens 16 --output "$RUN/tiered"
"$PY" -B -u examples/inspect_runtime.py --case quant-target --max-tokens 16 --output "$RUN/quant-target"
```

- `stochastic` 显式启用 `rejection`、temperature=0.8；看随机 proposal 与验证执行。这里只是能跑通，**不**以同 seed 文本相同证明分布正确。CPU 概率回归及原 GPU 统计验证负责分布检查；top-k/top-p/penalty 组合不在该路径支持范围。
- `tiered` 加载 GPU–CPU 缓存配置。短 prompt 常见 `host_cache.entries=0`，属于没有容量压力的负对照；不能把它说成发生了 CPU 命中。回收、在途复制和页复用先看 CPU `tiered-cache` 回归；真实容量对照看冻结实验。
- `quant-target` 加载 8B AWQ checkpoint 和 Marlin 共享库，查看 `runtime_metrics.target_quantization` 和显存。输出完成只说明执行链路跑通；它改变 target 分布，不能要求与 BF16 文本逐 token 相同，也不能据此推断质量或加速。

若 `quant-target` 报找不到 `_C.abi3.so`，检查 `configs/quantized_target.json` 的 `marlin_library` 与文件是否存在；不要把 dense 反量化参考冒充 Marlin 执行。

## 8. 真实 GPU 边界回归（比 smoke 耗时更长）

下列命令各占一张卡。先确认空闲，使用相同 UUID/index，同一轮每个测试都建立新文件；不要和上面的服务并行运行。

```bash
"$PY" -B -m tests.regression.check_device_draft_gpu \
  --variant legacy --output "$RUN/legacy.json"
"$PY" -B -m tests.regression.check_device_draft_gpu \
  --variant device --compare "$RUN/legacy.json" --output "$RUN/device.json"
"$PY" -B -m tests.regression.check_device_draft_gpu \
  --variant device --packed-linear --compare "$RUN/legacy.json" --output "$RUN/packed.json"
```

做什么：先建立参考，再对比 GPU 候选和 packed target，检查各轮 logits、有效 KV、输出与回滚等。检查 JSON 的 `status`、逐项结果；只要任一异常就保留报告定位，不能放宽阈值。还可以查看：

真实模型分层缓存回载：测试先生成并缓存两个完整页，人工施加页压力、清零该测试进程自己的 GPU KV，再从 CPU 回载并检查 token、页内容与资源回收。它不是性能测试：

```bash
"$PY" -B -m tests.regression.check_tiered_model \
  --model /home/hrxu/huggingface/Qwen3-1.7B --output "$RUN/tiered-roundtrip"
"$PY" -B -m tests.regression.check_tiered_model \
  --model /home/hrxu/huggingface/Qwen3-1.7B --async-prefetch --output "$RUN/tiered-async"
```

预期两个 `report.json` 为 `PASS`、`restored_blocks >= 2`、`exact_prefix_kv=true`、`clean=true`；`--async-prefetch` 是异步 H2D 路径，不能据此宣称全部传输都异步或更快。

随机接受/拒绝的 GPU 小词表分布回归：

```bash
"$PY" -B -m tests.regression.check_stochastic_gpu --gpu --output "$RUN/stochastic-distribution.json"
```

预期 `status=PASS`、`gpu_tests_run=true`；原规则是每例 100000 次采样、最大单元误差阈值 0.01。它检验概率算法而不是整模型速度。真实 draft wrapper 和 engine 的扩展测试分别使用 `--model /home/hrxu/huggingface/Qwen3-0.6B`、`check_stochastic_engine --gpu`，具体参数先查看 `--help`。

FP16 draft 的 255/256/257 边界及 eager/Graph 跨进程比较：

```bash
"$PY" -B -m tests.regression.check_marlin_draft_lifecycle \
  --model /home/hrxu/huggingface/Qwen3-8B --draft /home/hrxu/huggingface/Qwen3-0.6B \
  --draft-quantization none --execution eager --extended-numerics \
  --gpu-memory-utilization .90 --output "$RUN/draft-eager.json"
"$PY" -B -m tests.regression.check_marlin_draft_lifecycle \
  --model /home/hrxu/huggingface/Qwen3-8B --draft /home/hrxu/huggingface/Qwen3-0.6B \
  --draft-quantization none --execution graph --prefill-graph --extended-numerics \
  --gpu-memory-utilization .90 --compare-report "$RUN/draft-eager.json" --output "$RUN/draft-graph.json"
```

预期报告为 PASS；比较的是同 checkpoint 固定历史 logits、有效 KV 和生命周期，原数值阈值不变。请保持两次运行的源码与模型不变；不要把两个不同配置的 token 直接当作所有数值一致性的证据。

其他入口可先查看参数：

```bash
"$PY" -B -m tests.regression.check_marlin_draft_lifecycle --help
"$PY" -B -m tests.regression.check_stochastic_engine --help
"$PY" -B -m tests.regression.check_marlin_gpu --help
```

FP16 draft 的 eager/Graph 跨进程边界命令、固定模型与输出路径详见 `docs/DRAFT_OPTIMIZATION.md` 的“复现”。Marlin draft 通过生命周期检查也不代表量化 draft 质量达标。

特别注意：旧 `check_marlin_gpu` 微基准硬编码物理 GPU 1 和对应 UUID，并要求 `--authorize-gpu-window`；它不是任意空闲卡通用命令。只有该卡稳定空闲时才能运行，不能改硬编码或冒用别人的 GPU 使用权。当前 `quant-target` demo 可在 guard 通过的其他空闲卡运行。

## 9. 在源码里打断点

运行教学脚本前仍需正确设置空闲 GPU；用一个新的输出路径：

```bash
"$PY" -B -m pdb examples/inspect_runtime.py --case step --output "$RUN/pdb-step"
```

在 `(Pdb)` 提示符依次输入：

```text
b /home/hrxu/nano-vllm/nanovllm/engine/scheduler.py:141
b /home/hrxu/nano-vllm/nanovllm/engine/scheduler.py:223
c
p (seq.num_tokens, seq.num_cached_tokens, request_tokens, remaining)
n
p seq.num_scheduled_tokens
c
p (is_prefill, seq.num_cached_tokens, seq.num_tokens)
c
```

第一个断点在本轮输入数写入之前；`n` 后看本轮输入数。第二个断点在 KV 进度结算之后、判断是否跳过中间 chunk 之前。之后 `c` 会再次命中断点；`p` 看变量，`n` 单步，`l` 看源码，`q` 退出。断点会暂停你的 GPU 进程，调试完及时退出释放自己的资源。
`q` 提前终止 demo 时，报告可能记为 `DEMO_FAILED`，这是未完成演示的标记；用新的输出目录重新运行即可。

其他断点入口：

|目标|入口|检查什么|
|---|---|---|
|构造 prefill|`model_runner.py:352`|`start/end`、当前 query 与已有 KV|
|draft 提案|`draft_runner.py:205`、`:283`|suffix、Graph 命中、GPU pending|
|GPU 验证计划|`planned_scheduler.py:339`、`:365`|reservation、物化、commit/rollback|
|target GPU 验证|`model_runner.py:551`|候选与静态输入缓冲|
|前缀回载|`block_manager.py:408`、`:107`|目标页引用、传输 lease 和完成 event|
|量化执行|`layers/awq_marlin_sidecar.py`|packed 权重元数据与后端选择|
|物理布局|`layers/stable_layout.py`|逻辑维度与物理 stride|

普通 `pdb` 不能逐指令调试 GPU kernel，也会显著改变时序；正确性调试与性能测量应分开。

## 10. 正式性能和质量复现

日常 smoke 不统计加速。正式对照必须使用冻结输入、同一物理卡、五轮交错和不带 profiler 的计时，同时保留真实普通 decode Graph 基线。以下是现成最终交付的七组五轮原始流程，运行时间较长，会占用 GPU；先独立检查空闲，输出目录不能已存在：

```bash
cd /home/hrxu/nano-full-opt-20261024
# 此份 freeze 绑定物理 GPU0；只有 GPU0 稳定空闲时才运行。
export CUDA_VISIBLE_DEVICES=GPU-2eb3dcc0-b069-2c12-1137-3cfcf70129f4
export NANOVLLM_BENCH_GPU_INDEX=0
PYTHONPATH=/home/hrxu/nano-vllm "$PY" -B -m benchmarks.runtime_v2_gpu_guard
FORMAL_RUN="/home/hrxu/nano-full-opt-20261024/learner-formal-$(date +%Y%m%d-%H%M%S)-$$"
"$PY" -B performance.py suite --freeze "$PWD/formal_freeze001.json" --output "$FORMAL_RUN"
"$PY" -B analyze_formal.py --index "$FORMAL_RUN/index.json" --output "$FORMAL_RUN/analysis.json"
```

worker 要求输出位于 `/home/hrxu/nano-full-opt-20261024/` 下，且检查 freeze 中的 GPU UUID；不能直接改用 `$RUN` 或另一张卡。若 GPU0 不空闲，等待；不要修改 freeze 来伪装原始复现。

W4A16 target 的既有固定验证集质量复现（同样只用于复现，不用于调参；输出需位于冻结根目录内且不存在）：

```bash
QUALITY_RUN="/home/hrxu/nano-full-opt-20261024/learner-quality-$(date +%Y%m%d-%H%M%S)-$$"
"$PY" -B quality.py evaluate --family target --split holdout \
  --model /home/hrxu/huggingface/Qwen3-8B-AWQ --packed \
  --reference-manifest /home/hrxu/nano-full-opt-20261024/reference/target/holdout/manifest.json \
  --output "$QUALITY_RUN"
```

该程序会对照冻结参考计算完整词表质量指标，运行时间与磁盘输出较大；`status=PASS` 首先表示评测执行完成，是否通过质量/分场景门槛仍要看最终分析和配置结论。执行前仍需完整 UUID、物理 index 和空闲卡 guard。

新一轮 target 布局的七臂五轮、对应设备回归和 30 分钟 HTTP/SSE 原始命令见 `/home/hrxu/nano-vllm/docs/DRAFT_OPTIMIZATION.md` 的“2026-10-26（第二轮）→复现与学习”。这些实验引用各自冻结代码，不因为当前分支新增教学文件而自动重新验收。历史输入含 DEV 回归，不应作为新质量 holdout。

正式记录至少包含原始输入、输出 token、源码/模型身份、配置、各轮吞吐、TTFT、真实事件间隔、接受/验证次数、显存、启动成本和 guard/主机负载。量化还要用独立冻结质量数据，分别看 NLL/KL、各类别接受率与实际执行成本；不改变阈值或用最终验证集挑参数。分层缓存需同时保留重复前缀和无复用负对照。

## 11. 常见现象

|现象|先检查什么|
|---|---|
|guard 报 occupied/UUID 不匹配|核对物理 index、完整 UUID、计算进程；等待空闲，不停止他人任务|
|`output` 已存在|换一个新目录；保留旧报告|
|服务一直未就绪|先看启动终端的模型路径、哈希、显存和端口错误；`--execute` 会先校验冻结文件|
|`prompt + max_tokens exceeds max_model_len`|减少 token 或在新的实验配置里重新评估上下文和 KV 预算|
|Graph fallback 计数增加|先看真实 Q/K、bucket/容量、是否 eager；fallback 是覆盖范围，不自动是错误|
|draft 输入正常但事件数小于 token 数|一次事件可能提交多个 token；看 `token_index` 和生成计数|
|两个 greedy 路径 token 不同|先区分 backend/精度/历史，再做固定共同历史的 logits 与有效 KV 对比|
|量化输出与 BF16 不同|先核对执行后端和固定质量指标；W4A16 是有损量化|
|短 tiered demo 未命中 CPU|检查是否存在完整前缀、GPU 容量压力及 host_cache 计数；短 demo 本身不保证回载|
|GPU 内存没降到 0|确认本进程退出和 `nvidia-smi` 进程列表；allocator 保留量与请求 KV 引用是不同概念|

目前实际运行过的短案例和输出位于服务器 `/home/hrxu/nano-vllm-archive/learning-smoke-20261027/`。它们只供学习和排错，不替代正式五轮/质量/30 分钟验收。
