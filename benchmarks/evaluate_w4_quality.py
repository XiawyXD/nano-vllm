"""固定 teacher-forced WikiText 行，区分 BF16->FP16 与 W4->FP16 的质量差异。

只评估预先冻结的 manifest；不训练、不调整阈值。GPU 上计算 NLL，不下载整词表。
"""
import argparse
import json
import math
from pathlib import Path
import time

from benchmarks.cache_draft_experiment import digest, write_new, sources, file_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--dtype', choices=['checkpoint','float16','bfloat16'], default='checkpoint')
    p.add_argument('--quantization', choices=['awq'])
    p.add_argument('--quant-backend', choices=['triton','marlin'], default='triton')
    p.add_argument('--marlin-library')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--split', choices=['calibration','validation'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    if digest({k:v for k,v in manifest.items() if k!='sha256'}) != manifest['sha256']:
        raise ValueError('quality manifest was modified')
    report = dict(status='NOT_RUN', quality_manifest_sha256=manifest['sha256'],
                  source_sha256=sources(), model=args.model, dtype=args.dtype,
                  quantization=args.quantization, quant_backend=args.quant_backend, split=args.split, rows=[])
    engine = None
    try:
        from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
        from benchmarks.runtime_v2_performance import environment
        report['gpu'] = ensure_gpu1_available(evidence_path=args.output/'gpu_guard.jsonl')
        report['environment'] = environment()
        report['checkpoint_files'] = {p.name:file_record(p) for p in sorted(Path(args.model).iterdir())
            if p.is_file() and (p.suffix=='.safetensors' or p.name in
                ('config.json','tokenizer.json','tokenizer_config.json','download_manifest.json'))}
        from transformers import AutoTokenizer
        frozen_tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer_path'],local_files_only=True)
        current_tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        if (frozen_tokenizer.get_vocab()!=current_tokenizer.get_vocab()
                or frozen_tokenizer.all_special_ids!=current_tokenizer.all_special_ids):
            raise ValueError('quality input token IDs belong to a different tokenizer mapping')
        report['tokenizer_mapping_and_special_ids_equal'] = True
        import torch
        from nanovllm import LLM, SamplingParams
        from nanovllm.engine.sequence import Sequence
        from nanovllm.layers.awq_packed import AWQPackedLinear
        started = time.perf_counter()
        engine = LLM(args.model, quantization=args.quantization, model_dtype=args.dtype,
            quant_backend=args.quant_backend, marlin_library=args.marlin_library,
            tensor_parallel_size=1, max_num_seqs=1, max_model_len=512, max_num_batched_tokens=128,
            max_kvcache_blocks=3, enable_prefix_caching=False, gpu_memory_utilization=.90,
            rmsnorm_backend='triton', enforce_eager=False)
        torch.cuda.synchronize()
        report['startup_seconds'] = time.perf_counter()-started
        report['weight_load_seconds'] = engine.model_runner.weight_load_seconds
        report['config'] = engine.config.runtime_summary()
        model = engine.model_runner.model
        from nanovllm.utils.weight_memory import weight_storage_bytes
        report['memory'] = dict(parameter_bytes=weight_storage_bytes(model), allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
            kv_bytes=engine.model_runner.kv_cache.numel()*engine.model_runner.kv_cache.element_size())
        report['packed_modules'] = {name:m.stats() for name,m in model.named_modules() if isinstance(m,AWQPackedLinear)}
        manager = engine.scheduler.block_manager
        total_nll, correct, finite, count = 0., 0, 0, 0
        for row in manifest['workloads'][args.split]:
            ensure_gpu1_available(evidence_path=args.output/'gpu_guard.jsonl')
            tokens = row['tokens']
            seq = Sequence(tokens[:-1], SamplingParams(temperature=0, max_tokens=1))
            try:
                reservation = manager.reserve_to(seq,len(seq))
                started = time.perf_counter()
                logits = engine.model_runner.run_speculative([seq]).float()
                labels = torch.tensor(tokens[1:],device=logits.device,dtype=torch.long)
                losses = torch.nn.functional.cross_entropy(logits,labels,reduction='none')
                prediction = logits.argmax(-1)
                finite_logits = torch.isfinite(logits).all(-1)
                # 只有每位置标量、预测 token 回传，不复制完整词表。
                host = torch.stack((losses,prediction.float(),finite_logits.float()),-1).cpu().tolist()
                elapsed = time.perf_counter()-started
                manager.commit(reservation,len(seq))
                nll = [v[0] for v in host]
                predicted = [int(v[1]) for v in host]
                nfinite = sum(int(v[2]) for v in host)
                total_nll += sum(nll)
                correct += sum(a==b for a,b in zip(predicted,tokens[1:]))
                finite += nfinite
                count += len(nll)
                report['rows'].append(dict(row=row['row'],tokens=tokens,nll=nll,
                    predicted_tokens=predicted,finite_positions=nfinite,forward_and_metric_seconds=elapsed))
            finally:
                if seq.seq_id in manager._reservations:
                    manager.rollback(manager._reservations[seq.seq_id])
                manager.deallocate(seq)
        mean = total_nll/count
        report.update(status='PASS_MEASUREMENT' if finite==count and math.isfinite(mean) else 'FAIL_NONFINITE',
            mean_teacher_forced_nll=mean, perplexity=math.exp(mean) if mean<700 else float('inf'),
            top1_next_token_accuracy=correct/count, finite_positions=finite, label_tokens=count,
            quality_decision='Compare paired reports against frozen manifest; this status alone is NOT quality acceptance')
    except Exception as error:
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if engine is not None:
            engine.exit()
        write_new(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','mean_teacher_forced_nll','perplexity','label_tokens')}))


if __name__=='__main__':
    main()
