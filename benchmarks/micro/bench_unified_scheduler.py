"""统一调度验收：真实模型 greedy 对照、动态加入、取消、预算和 block 回收。

运行：CUDA_VISIBLE_DEVICES=0 ... python benchmarks/micro/bench_unified_scheduler.py
使用低显存预算；不连接或重启已经运行的 HTTP 服务。
"""
import json
import argparse
from pathlib import Path
from time import perf_counter
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from nanovllm import LLM, SamplingParams
from nanovllm.server import _openai_params, create_http_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-0.6B")
    parser.add_argument("--memory", type=float, default=0.18)
    args = parser.parse_args()
    llm = LLM(args.model, max_model_len=512,
              max_num_batched_tokens=32, max_num_seqs=4, prefill_chunk_size=32,
              max_prefill_steps=1, gpu_memory_utilization=args.memory,
              enforce_eager=not args.graph)
    report = {}
    def params(spec=False, **kwargs):
        return SamplingParams(temperature=0, max_tokens=16,
                              speculative_ngram_size=3 if spec else 0, **kwargs)
    try:
        # 255-token prompt 使生成跨过 256-token block 边界。
        prompts = [[1, 2, 3] * 85, [4, 5, 6] * 24, "The capital of France is"]
        baseline = llm.generate(prompts, params(), use_tqdm=False)
        start = perf_counter()
        mixed = llm.generate(prompts, [params(True), params(), params(True)], use_tqdm=False)
        assert [x["token_ids"] for x in mixed] == [x["token_ids"] for x in baseline]
        report["mixed_greedy_equal"] = True
        report["mixed_seconds"] = perf_counter() - start
        report["speculative_stats"] = llm.speculative_stats()
        assert report["speculative_stats"]["target_verification_calls"] > 0
        assert not llm.scheduler.block_manager.used_block_ids

        # 动态加入 + waiting/running 取消 + 增量文本完整性。
        first = llm.add_request(prompts[0], params(True))
        cancelled_wait = llm.add_request(prompts[1], params(True))
        assert llm.cancel_request(cancelled_wait.seq_id)
        records, texts = [first], {first.seq_id: ""}
        cancelled_running = None
        for step in range(100):
            if step == 2:
                late = llm.add_request(prompts[2], params())
                records.append(late)
                texts[late.seq_id] = ""
            llm.step()
            for event in llm.step_events():
                if event.request_id in texts:
                    texts[event.request_id] += event.text
            if first.num_completion_tokens > 0 and cancelled_running is None:
                cancelled_running = llm.add_request([7, 8, 9], params(True))
            if cancelled_running is not None and cancelled_running.num_completion_tokens > 0 and not cancelled_running.is_finished:
                assert llm.cancel_request(cancelled_running.seq_id)
            if llm.is_finished():
                break
        assert llm.is_finished()
        assert cancelled_running.finish_reason == "cancelled"
        assert not cancelled_running.block_table and not cancelled_wait.block_table
        report["dynamic_token_matches_static_batch"] = []
        for seq, expected in zip(records, (baseline[0], baseline[2])):
            report["dynamic_token_matches_static_batch"].append(seq.completion_token_ids == expected["token_ids"])
            assert texts[seq.seq_id] == llm.tokenizer.decode(seq.completion_token_ids)
        # 独立检查普通 greedy 改变 batch 形状是否也出现数值分岔，保留证据，
        # 不能把它混同于 speculative 接受规则正确性的证明。
        alone = llm.generate([prompts[2]], params(), use_tqdm=False)[0]
        report["ordinary_single_equals_static_batch"] = alone["token_ids"] == baseline[2]["token_ids"]
        report["ordinary_single_tokens"] = alone["token_ids"]
        report["ordinary_static_tokens"] = baseline[2]["token_ids"]
        report["ordinary_dynamic_tokens"] = records[1].completion_token_ids
        assert not llm.scheduler.block_manager.used_block_ids
        report["dynamic_cancel_stream"] = True

        # 不支持的 speculative 随机采样必须在入队前拒绝，不能静默变成 greedy。
        submitted = llm.scheduler.total_requests_submitted
        try:
            llm.add_request("hello", SamplingParams(temperature=0.7, speculative_ngram_size=3))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid speculative sampling accepted")
        assert submitted == llm.scheduler.total_requests_submitted
        assert _openai_params({"temperature": 0, "speculative_ngram_size": 3}).use_speculative
        report["validation_and_api_params"] = True
        # 使用真实模型和临时本机端口验证 HTTP 参数确实触发 speculative；
        # 两条并发连接分别请求 speculative SSE 和普通 JSON。
        server, manager = create_http_server(llm, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/completions"
        previous_calls = llm.speculative_stats().get("target_verification_calls", 0)
        def request(spec):
            payload = dict(prompt=prompts[0], temperature=0, max_tokens=16,
                           stream=spec, speculative_ngram_size=3 if spec else 0)
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read().decode()
        try:
            with ThreadPoolExecutor(2) as pool:
                streaming, normal = list(pool.map(request, (True, False)))
            assert "data: [DONE]" in streaming
            chunks = [json.loads(line[6:]) for line in streaming.splitlines()
                      if line.startswith("data: {")]
            text = "".join(c["choices"][0].get("text", "") for c in chunks)
            assert text == json.loads(normal)["choices"][0]["text"]
            assert llm.speculative_stats()["target_verification_calls"] > previous_calls
            assert not llm.scheduler.block_manager.used_block_ids
            report["real_http_mixed_sse"] = True
        finally:
            server.shutdown()
            manager.close()
            server.server_close()
            thread.join()
        report["scheduler"] = llm.scheduler.stats()
        report["status"] = "PASS"
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
