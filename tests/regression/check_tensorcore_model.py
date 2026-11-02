"""真实单卡模型的MLP后端验收；每个后端/Graph组合必须使用独立进程。

保存首次prefill完整词表logits和最终tokens，跨报告数值比较由调用者另行执行。
调试hooks和CPU tensor复制会同步/影响时序，耗时不是正式性能基准。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from benchmarks.tensorcore_utils import compare, fp32_reference_mode, provenance, save_report


PROMPTS = [
    "Explain why a language model stores keys and values during autoregressive generation. "
    "Compare the work performed while reading a complete prompt with the work performed "
    "when generating one additional token. Describe what happens when several requests "
    "have different context lengths. Use precise language and distinguish memory capacity "
    "from memory bandwidth. Give a short examples.example involving a shared conversation prefix.",
    "Complete the following Python function and explain its boundary conditions. "
    "The input may contain an empty list or repeated values. Do not mutate the input.\n"
    "def merge_sorted(left: list[int], right: list[int]) -> list[int]:\n"
    "    i = 0\n    j = 0\n    result = []\n"
    "    while i < len(left) and j < len(right):\n"
    "        # Append the smaller value and advance the corresponding index.\n",
]


def layer_unit_check(mlp, args):
    """使用第0层真实权重、固定x比较gate/up融合及完整MLP，隔离自回归历史分岔。"""
    from nanovllm.layers.tensorcore_mlp import fused_swiglu
    gate_weight, down_weight = mlp.gate_up_proj.weight, mlp.down_proj.weight
    generator = torch.Generator(device=gate_weight.device).manual_seed(args.seed)
    x = torch.randn((128, gate_weight.shape[1]), generator=generator,
                   device=gate_weight.device, dtype=torch.bfloat16)
    torch_middle = mlp.act_fn(F.linear(x, gate_weight))
    fused_middle = fused_swiglu(x, gate_weight)
    torch_output = F.linear(torch_middle, down_weight)
    fused_output = F.linear(fused_middle, down_weight)
    with fp32_reference_mode():
        projected = F.linear(x.float(), gate_weight.float())
        gate, up = projected.chunk(2, -1)
        fp32_middle = F.silu(gate)*up
        fp32_output = F.linear(fp32_middle, down_weight.float())
        # 额外单列融合算子的舍入约定，不能与纯FP32 accuracy reference混为一谈。
        rounded_gate, rounded_up = projected.to(torch.bfloat16).float().chunk(2, -1)
        rounded_contract = (F.silu(rounded_gate)*rounded_up).to(torch.bfloat16)
    rows = dict(fused_middle_vs_torch=compare(fused_middle, torch_middle, args.atol, args.rtol),
        fused_middle_vs_rounded_contract=compare(fused_middle, rounded_contract, args.atol, args.rtol),
        fused_middle_vs_fp32=compare(fused_middle, fp32_middle, args.atol, args.rtol),
        full_mlp_vs_torch=compare(fused_output, torch_output, args.atol, args.rtol),
        full_mlp_vs_fp32=compare(fused_output, fp32_output, args.atol, args.rtol))
    return dict(x_shape=list(x.shape), gate_up_shape=list(gate_weight.shape),
                down_shape=list(down_weight.shape), dtype=str(gate_weight.dtype),
                comparisons=rows, passed=all(row['allclose'] for row in rows.values()))


def main():
    """保存完整诊断证据；始终关闭本次引擎，不改变请求采样结果或后端实现。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='/home/hrxu/huggingface/Qwen3-1.7B')
    ap.add_argument('--backend', choices=('torch', 'triton', 'auto'), required=True)
    ap.add_argument('--requests', type=int, choices=(1, 2), default=2,
                    help='one request exercises M=1 auto dispatch during decode')
    ap.add_argument('--min-tokens', type=int, default=128)
    ap.add_argument('--graph', action='store_true')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.25)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--atol', type=float, default=0.02)
    ap.add_argument('--rtol', type=float, default=0.02)
    args = ap.parse_args()
    if args.min_tokens < 1 or not 0 < args.gpu_memory_utilization <= 1:
        ap.error('min-tokens must be positive; gpu-memory-utilization must be in (0,1]')
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        ap.error('tolerances must be finite and nonnegative')
    output = Path(args.output)
    tensors_path = output.with_suffix('.pt')
    if output.exists() or tensors_path.exists():
        ap.error('choose a new output filename; existing JSON/tensor evidence is never overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', args=vars(args), provenance=provenance(),
        comparison_note='Full-model logits and final token equality across backend reports have NOT been checked by this script.',
        timing_note='Debug hooks and CPU copies perturb execution; elapsed time is not a performance benchmark.')
    root = Path(__file__).resolve().parents[2]
    for relative in ('tests/regression/check_tensorcore_model.py', 'nanovllm/config.py',
                     'nanovllm/models/qwen3.py', 'nanovllm/engine/model_runner.py'):
        p = root/relative
        report['provenance']['source_sha256'][relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    llm, hooks, runner, old_run, had_override = None, [], None, None, False
    tensor_evidence = {}
    began = time.perf_counter()
    try:
        from nanovllm import LLM, SamplingParams
        from nanovllm.layers.tensorcore_mlp import should_use_fused_swiglu
        from nanovllm.utils.context import get_context
        with torch.inference_mode():
            llm = LLM(args.model, tensor_parallel_size=1, max_model_len=512,
                max_num_batched_tokens=256, prefill_chunk_size=128, max_num_seqs=4,
                gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
                enforce_eager=not args.graph, sampler_backend='torch',
                mlp_backend=args.backend, mlp_min_tokens=args.min_tokens)
            runner = llm.model_runner
            mlps = [layer.mlp for layer in runner.model.model.layers]
            if not all(layer.mlp_backend == args.backend and layer.mlp_min_tokens == args.min_tokens for layer in mlps):
                raise AssertionError('requested MLP settings did not reach every model layer')
            report['effective_config'] = {name: getattr(llm.config, name) for name in (
                'model', 'tensor_parallel_size', 'max_model_len', 'max_num_batched_tokens',
                'prefill_chunk_size', 'max_num_seqs', 'gpu_memory_utilization',
                'enable_prefix_caching', 'enforce_eager', 'sampler_backend', 'mlp_backend', 'mlp_min_tokens')}
            report['model'] = dict(dtype=str(next(runner.model.parameters()).dtype),
                mlp_dtype=str(mlps[0].gate_up_proj.weight.dtype), layer_count=len(mlps),
                hidden_size=llm.config.hf_config.hidden_size,
                intermediate_size=llm.config.hf_config.intermediate_size)
            if mlps[0].gate_up_proj.weight.dtype != torch.bfloat16:
                raise ValueError('this experiment requires a BF16 checkpoint/model')
            report['layer0_unit'] = layer_unit_check(mlps[0], args)
            routing = dict(python_forward_hooks=dict(fused=0, fallback=0, token_rows={}),
                           graph_replay_routes=dict(replays=0, fused=0, fallback=0, buckets={}),
                note='Python hooks observe actual eager forwards. Graph replay bypasses Python hooks; graph counts infer fixed captured routes from bucket shape/config, not kernel profiler samples. Constructor warmup/capture and isolated unit check are excluded.')
            report['routing'] = routing

            def count_forward(module, inputs):
                """仅读取shape/dtype等CPU元信息，统计实际执行到的Python MLP路由。"""
                x = inputs[0]
                fused = should_use_fused_swiglu(x, module.gate_up_proj.weight, module.mlp_backend,
                                              module.mlp_min_tokens, module.mlp_device_capability)
                row = routing['python_forward_hooks']
                row['fused' if fused else 'fallback'] += 1
                key = str(x.shape[0])
                row['token_rows'][key] = row['token_rows'].get(key, 0)+1

            for mlp in mlps:
                hooks.append(mlp.register_forward_pre_hook(count_forward))
            old_run = runner.run_model
            had_override = 'run_model' in runner.__dict__

            def observed_run(input_ids, positions, is_prefill):
                """调用原forward并原样返回logits；仅第一次prefill额外保存CPU证据。"""
                context = get_context()
                bucket = None
                if not is_prefill and not runner.enforce_eager and input_ids.numel() <= 512:
                    bucket = next((b for b in runner.graph_bs if b >= input_ids.numel() and b in runner.graphs), None)
                    if bucket is not None and context.block_tables.shape[1] > runner.graph_vars['block_tables'].shape[1]:
                        bucket = None
                logits = old_run(input_ids, positions, is_prefill)
                # 保存每一步，不能只验prefill而漏掉auto在decode M=1才启用的路径。
                # 这些CPU复制仅用于小回归，会影响时间，所以本脚本不报告性能收益。
                record = dict(is_prefill=is_prefill, logits=logits.detach().cpu().clone(),
                              input_ids=input_ids.detach().cpu().clone(),
                              positions=positions.detach().cpu().clone())
                for key in ('cu_seqlens_q', 'cu_seqlens_k', 'context_lens'):
                    value = getattr(context, key, None)
                    if value is not None:
                        record[key] = value.detach().cpu().clone()
                tensor_evidence.setdefault('forward_records', []).append(record)
                if bucket is not None:
                    row = routing['graph_replay_routes']
                    row['replays'] += 1
                    row['buckets'][str(bucket)] = row['buckets'].get(str(bucket), 0)+1
                    proxy_hidden = runner.graph_vars['outputs'][:bucket]
                    for mlp in mlps:
                        fused = should_use_fused_swiglu(proxy_hidden, mlp.gate_up_proj.weight, mlp.mlp_backend,
                                                       mlp.mlp_min_tokens, mlp.mlp_device_capability)
                        row['fused' if fused else 'fallback'] += 1
                if is_prefill and 'first_prefill_logits' not in tensor_evidence:
                    tensor_evidence.update(first_prefill_logits=logits.detach().cpu().clone(),
                        input_ids=input_ids.detach().cpu().clone(), positions=positions.detach().cpu().clone())
                    for key in ('cu_seqlens_q', 'cu_seqlens_k'):
                        value = getattr(context, key, None)
                        if value is not None:
                            tensor_evidence[key] = value.detach().cpu().clone()
                    report['first_prefill'] = dict(logits_shape=list(logits.shape),
                        dtype=str(logits.dtype), token_rows=input_ids.numel(),
                        finite=bool(torch.isfinite(logits).all()), argmax=logits.argmax(-1).cpu().tolist(),
                        note='Full vocabulary vector for every row returned by normal prefill LM head; not logits for every prompt position.')
                return logits

            runner.run_model = observed_run
            prompts = PROMPTS[:args.requests]
            encoded = [llm.tokenizer.encode(prompt) for prompt in prompts]
            report['prompts'] = [dict(text=text, token_ids=ids) for text, ids in zip(prompts, encoded)]
            report['outputs'] = llm.generate(encoded,
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)
            manager = llm.scheduler.block_manager
            report['resources_after'] = dict(used_blocks=len(manager.used_block_ids),
                kv_ref_total=sum(block.ref_count for block in manager.blocks),
                engine_request_records=len(llm._requests), scheduler_finished=llm.is_finished())
            if not tensor_evidence or not report['first_prefill']['finite']:
                raise AssertionError('missing/non-finite first prefill logits')
            if any(report['resources_after'][name] for name in ('used_blocks', 'kv_ref_total', 'engine_request_records')):
                raise AssertionError('request or KV references remained after generate')
            if any(len(row['token_ids']) != 8 for row in report['outputs']):
                raise AssertionError('generation did not produce exactly 8 tokens')
            report['status'] = 'PASS_LOCAL_CHECKS_CROSS_REPORT_PENDING' if report['layer0_unit']['passed'] else 'REVIEW_REQUIRED'
    except Exception as exc:
        report['status'], report['error'] = 'FAIL', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if runner is not None and old_run is not None:
            if had_override:
                runner.run_model = old_run
            else:
                del runner.run_model
        try:
            if llm is not None:
                llm.exit()
        except Exception as exc:
            report['status'] = 'FAIL'
            report['cleanup_error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            report['elapsed_seconds_not_benchmark'] = time.perf_counter()-began
            if tensor_evidence:
                torch.save(tensor_evidence, tensors_path)
                report['tensor_evidence'] = str(tensors_path)
            save_report(output, report)
            print(json.dumps(dict(status=report['status'], output=str(output))), flush=True)
    if report['status'] == 'REVIEW_REQUIRED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
