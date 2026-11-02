# 直接运行与复现

服务器实际修复目录 `/home/hrxu/nano-vllm`；冻结实验根 `/home/hrxu/nano-full-opt-20261024`。源码快照包含未跟踪源码；不重置原工作区，不自动执行训练或留下服务。

```bash
ssh hrxu@10.29.98.230
cd /home/hrxu/nano-full-opt-20261024
export PY=/home/hrxu/nano-vllm/.venv/bin/python
export CUDA_VISIBLE_DEVICES=GPU-2eb3dcc0-b069-2c12-1137-3cfcf70129f4
export NANOVLLM_BENCH_GPU_INDEX=0
nvidia-smi
```

所有GPU入口均复用guard；若不是空闲卡，退出并等待，不停止别人的任务。配对仍用同一物理UUID。不要改冻结源码、原profiles或阈值后继续引用旧结果。

## 查看配置及运行已验收服务

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -B learning/demo_cli.py --profiles release_profiles_final.json --profile ordinary_service show
CUDA_VISIBLE_DEVICES='' "$PY" -B serve_validated.py
"$PY" -B serve_validated.py --execute --host 127.0.0.1 --port 18784
```

第三条阻塞运行。它使用原样通过30分钟验收的1.7B/TP1/C8配置、Graph ON、host cache/draft OFF和明确TTL5限制，启动前校验源码与checkpoint。Ctrl-C通过已有RequestManager清理；模型初始化不计入稳态TTFT。

```bash
curl -s http://127.0.0.1:18784/health
curl -N http://127.0.0.1:18784/v1/completions -H 'Content-Type: application/json' -d '{"model":"/home/hrxu/huggingface/Qwen3-1.7B","prompt":"Explain CUDA in one sentence.","max_tokens":32,"temperature":0,"stream":true}'
```

## 学习时先跑CPU，GPU保持显式启用

```bash
cd /home/hrxu/nano-vllm
CUDA_VISIBLE_DEVICES='' "$PY" -B -m benchmarks.check_speculative_sampling
CUDA_VISIBLE_DEVICES='' "$PY" -B -m benchmarks.check_stochastic_engine --cpu --output /home/hrxu/nano-full-opt-20261024/validation/learner_stochastic_cpu_NEW.json
cd /home/hrxu/nano-full-opt-20261024
CUDA_VISIBLE_DEVICES='' "$PY" -B learning/demo_cli.py --profiles release_profiles_final.json --profile quantized_target show
```

文件名中的NEW应换成从未用过的名字。demo默认show；明确执行前先读最终decision，不能把REVIEW配置当作推荐。具体功能演示使用 `stream` 子命令：`"$PY" -B learning/demo_cli.py --profiles release_profiles_final.json --profile quantized_target stream --allow-gpu --gpu-index 0 --gpu-uuid "$CUDA_VISIBLE_DEVICES" --output learning/demo_awq_NEW`。它不是正式测速。

## 完整正式对照：保持冻结配置，另建输出

```bash
"$PY" -B performance.py suite --freeze /home/hrxu/nano-full-opt-20261024/formal_freeze001.json --output /home/hrxu/nano-full-opt-20261024/replay_formal_NEW
"$PY" -B analyze_formal.py --index /home/hrxu/nano-full-opt-20261024/replay_formal_NEW/index.json --output /home/hrxu/nano-full-opt-20261024/replay_analysis_NEW.json
"$PY" -B service_tools/driver.py execute --manifest /home/hrxu/nano-full-opt-20261024/service_tools/frozen002/manifest.json --phase soak --attempt learner_soak_NEW --allow-gpu
```

suite自动5轮、七组、每case warmup和steady。速度与profiler分开。30分钟soak不能用短测替代。完整复现花费较长，日常阅读不必反复跑完整缓存矩阵。

## 质量与校准复现

`quality.py`使用原FP16 reference，完整词表KL，不以seed文本相同代替概率检验。冻结最终集仅用于复现，不参与新的调参。现有reference和权重保持原位置。

```bash
"$PY" -B quality.py evaluate --family target --split holdout --model /home/hrxu/huggingface/Qwen3-8B-AWQ --packed --reference-manifest /home/hrxu/nano-full-opt-20261024/reference/target/holdout/manifest.json --output /home/hrxu/nano-full-opt-20261024/replay_quality_NEW
CUDA_VISIBLE_DEVICES='' /home/hrxu/nano-quality-20261024/quant_env/bin/python -B reproduction/prepare_learner_v2.py --output-root /home/hrxu/learner_full_opt_NEW --gpu-index 0
```

最后一条只准备新的学习目录、执行CPU准备并打印guarded训练命令，不会自动派发GPU，也不复制DEV/HOLDOUT。MSE配方和once-completed-candidate guard保留。详见 `reproduction/FINAL_SOURCE_SNAPSHOT_V1.md`。

环境：服务Python3.12.13、Torch2.10.0+cu128、Triton3.6.0、Transformers5.14.1、FlashAttention2.8.3、驱动570.211.01；AutoRound使用独立quant_env0.16.0及其冻结依赖清单。Marlin库SHA为10ec98d19907ec8d31cc4425064c41da4a704789196020bd3c0f36de581d82b9，不安装整个vLLM。模型路径、revision/未知项和所有文件SHA见reproduction/model_provenance_v1.json。
