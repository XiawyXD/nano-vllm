"""复用既有到达表，比较相同调度策略下 verification eager/graph。

保留动态形状造成的缓存满回退，不能只报告固定batch理想场景。
这是引擎级实验，不含网络开销；mixed也只是合成文本，并非真实业务数据集。
"""
import argparse
import json
from pathlib import Path
from nanovllm import LLM
from benchmarks.micro.bench_serving_matrix import workload, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="/home/hrxu/huggingface/Qwen3-1.7B")
    args = ap.parse_args()
    llm = LLM(args.model, max_model_len=1536, max_num_batched_tokens=512,
              max_num_seqs=16, gpu_memory_utilization=.5, enable_prefix_caching=False)
    result = dict(model=args.model, trials=[])
    modes = ["baseline", "chunk256", "spec_eager", "spec_graph"]
    def trial(name, mode):
        runner = llm.model_runner
        runner.enable_verification_graph = mode == "spec_graph"
        before = dict(runner.verification_graphs.stats)
        item = run(llm, workload(llm.tokenizer, name), "spec4" if mode.startswith("spec_") else mode)
        item.update(mode=mode, workload=name, graph_delta={k:v-before[k] for k,v in runner.verification_graphs.stats.items()})
        return item
    try:
        for name in ("mixed", "repeat"):
            llm.model_runner.verification_graphs.entries.clear()
            for mode in modes:
                trial(name, mode)
            for repeat in range(5):
                for mode in modes[repeat%4:] + modes[:repeat%4]:
                    item = trial(name, mode)
                    item["repeat"] = repeat
                    result["trials"].append(item)
                    print(name, mode, item["tokens_per_second"], item["graph_delta"], flush=True)
                    Path(args.output).write_text(json.dumps(result, indent=2))
    finally:
        llm.exit()

if __name__ == "__main__":
    main()
