"""8B单卡适配验收；nano/HF分进程加载，避免两份权重同时占用24GB。

普通BF16路径优先；HF使用nano生成的同一历史做teacher-forced对照，
不把不同历史的logits拿来比较，也不修改阈值或tie-break。
"""
import argparse
import json
from pathlib import Path
import time
import hashlib
from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available

MODEL = '/home/hrxu/huggingface/Qwen3-8B'
PROMPTS = ['用两句话解释为什么推理时需要KV Cache。', 'Write a Python function that adds two integers.']

def verify_snapshot():
    """防止分片未下载齐就启动；原loader逐文件加载，不能只依赖目录存在。"""
    folder = Path(MODEL)
    index = json.loads((folder/'model.safetensors.index.json').read_text())
    manifest = json.loads((folder/'download_manifest.json').read_text())
    required = set(index['weight_map'].values()) | {
        'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json',
        'generation_config.json','vocab.json','merges.txt','LICENSE'}
    for name in sorted(required):
        entry = manifest['files'][name]
        path = folder/name
        assert path.stat().st_size == entry['bytes'], name
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for chunk in iter(lambda:file.read(8*1024*1024),b''): digest.update(chunk)
        assert digest.hexdigest() == entry['sha256'], name
    return dict(revision=manifest['hf_revision'], checked_files=len(required), status='VERIFIED')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['nano','hf','verify','structure'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--graph', action='store_true')
    args = ap.parse_args()
    snapshot = verify_snapshot()
    if args.mode == 'verify':
        print(json.dumps(snapshot))
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'structure':
        # 只在meta设备上构造HF结构，读取safetensors的shape，不加载GPU或实权重。
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from safetensors import safe_open
        config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
        with torch.device('meta'):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
                                                    attn_implementation='eager')
        expected = {k:tuple(v.shape) for k,v in model.state_dict().items()}
        index = json.loads((Path(MODEL)/'model.safetensors.index.json').read_text())['weight_map']
        actual = {}
        for shard in sorted(set(index.values())):
            with safe_open(str(Path(MODEL)/shard),framework='pt',device='cpu') as file:
                for key in file.keys():
                    assert key not in actual and index[key] == shard
                    actual[key] = tuple(file.get_slice(key).get_shape())
        assert actual == expected, 'checkpoint key/shape mismatch'
        result = dict(status='PASS_STRUCTURE_ONLY', snapshot=snapshot,
                      tensors=len(actual), parameters=sum(p.numel() for p in model.parameters()),
                      gpu_inference='NOT_RUN', note='meta结构匹配不是实际加载/数值/性能验收')
        (root/'structure.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    ensure_gpu1_available(evidence_path=root/'gpu_guard.jsonl')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if args.mode == 'hf':
        report = json.loads((root/'nano.json').read_text())
        expected = torch.load(root/'nano_logits.pt', map_location='cpu', weights_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='eager').eval().to('cuda')
        checks = []
        with torch.inference_mode():
            for sample, tensors in zip(report['samples'], expected):
                for index, ref in enumerate(tensors):
                    history = sample['prompt_ids'] + sample['token_ids'][:index]
                    logits = model(torch.tensor([history], device='cuda')).logits[0,-1].float().cpu()
                    diff = (logits-ref).abs()
                    checks.append(dict(prompt=sample['prompt'], step=index, max_abs=diff.max().item(),
                        mean_abs=diff.mean().item(), argmax_equal=logits.argmax().item()==ref.argmax().item(),
                        strict_close=bool(torch.allclose(logits,ref,atol=.01,rtol=.01))))
        result = dict(checks=checks, status='PASS' if all(x['strict_close'] and x['argmax_equal'] for x in checks) else 'REVIEW_REQUIRED')
        (root/'hf_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    from nanovllm import LLM, SamplingParams
    begun = time.perf_counter()
    llm = LLM(MODEL, runtime_preset='baseline', max_model_len=2048,
        max_num_batched_tokens=256, max_num_seqs=2, gpu_memory_utilization=.85,
        enforce_eager=not args.graph, enable_prefix_caching=False,
        mlp_backend='torch', sampler_backend='torch')
    result = dict(model=MODEL, snapshot=snapshot, graph=args.graph, startup_seconds=time.perf_counter()-begun,
                  kv_blocks=llm.config.num_kvcache_blocks, samples=[],
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  parameter_count=sum(p.numel() for p in llm.model_runner.model.parameters()),
                  dtype=str(next(llm.model_runner.model.parameters()).dtype))
    captures, all_logits = [], []
    runner = llm.model_runner
    original = runner.run_model
    def observed(*a,**kw):
        logits = original(*a,**kw)
        if len(captures)<4: captures.append(logits[-1].detach().float().cpu().clone())
        return logits
    try:
        for prompt in PROMPTS:
            captures.clear()
            ids = tokenizer.apply_chat_template([dict(role='user',content=prompt)],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=False)  # 新版 transformers 默认可能返回 BatchEncoding，显式要 ID 列表。
            runner.run_model = observed
            output = llm.generate([ids], SamplingParams(temperature=0,max_tokens=96), use_tqdm=False)[0]
            runner.run_model = original
            all_logits.append(list(captures))
            result['samples'].append(dict(prompt=prompt,prompt_ids=ids,**output))
            assert not llm.scheduler.block_manager.used_block_ids
            for rid in list(llm._requests): llm.forget_request(rid)
        # 长输入仅作容量/页边界冒烟，不作文本质量或吞吐结论。
        long_ids = tokenizer.encode('Explain key value caching. '*500)[:1800]
        llm.generate([long_ids], SamplingParams(temperature=0,max_tokens=8),use_tqdm=False)
        for rid in list(llm._requests): llm.forget_request(rid)
        # 最大并发配置为2，实际跑一批2请求，不能只用单请求推断并发可用。
        batch_inputs = [sample['prompt_ids'] for sample in result['samples']]
        batch_outputs = llm.generate(batch_inputs, SamplingParams(temperature=0,max_tokens=16),use_tqdm=False)
        result['batch2_outputs'] = batch_outputs
        for rid in list(llm._requests): llm.forget_request(rid)
        seq = llm.add_request('Explain attention.', SamplingParams(temperature=0,max_tokens=32))
        llm.step()
        llm.cancel_request(seq.seq_id)
        llm.forget_request(seq.seq_id)
        assert not llm.scheduler.block_manager.used_block_ids
        # 本机临时HTTP端口：验证chat模板和SSE，退出后不留常驻服务。
        import threading, urllib.request
        from nanovllm.server import create_http_server
        server,manager = create_http_server(llm,'127.0.0.1',0,model_name=MODEL)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            payload = dict(model=MODEL,messages=[dict(role='user',content='/no_think\n请回答：1+1等于几？')],
                           temperature=0,max_tokens=32,stream=True)
            req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as response: stream=response.read().decode()
            assert '[DONE]' in stream
            (root/'http_sse.txt').write_text(stream)
        finally:
            server.shutdown(); manager.close(); server.server_close(); thread.join(5)
        assert not llm.scheduler.block_manager.used_block_ids
        result.update(status='PASS_SMOKE', used_blocks=0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            long_prompt_tokens=len(long_ids),http_sse=True,cancel=True)
        torch.save(all_logits,root/'nano_logits.pt')
        (root/'nano.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        runner.run_model = original
        llm.exit()

if __name__ == '__main__': main()
