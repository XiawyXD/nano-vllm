"""无 GPU 的等价性检查；微基准不等于端到端吞吐收益。"""
import json
import random
import time
from nanovllm.speculative import NGramProposer
from nanovllm.incremental_ngram import IncrementalNGramProposer


def main():
    rng = random.Random(17)
    checks = 0
    for n in (1, 2, 3, 5):
        for trial in range(50):
            tokens = [rng.randrange(8) for _ in range(trial)]
            inc = IncrementalNGramProposer(n, 4)
            ref = NGramProposer(n, 4)
            for _ in range(30):
                limit = rng.randrange(1, 5)
                assert inc.propose(tokens, limit) == ref.propose(tokens, limit)
                # 包含不接受草稿、部分接受、跳过多步的 committed 历史变化。
                tokens.extend(rng.randrange(8) for _ in range(rng.randrange(1, 7)))
                checks += 1
            tokens = tokens[:n+2]
            assert inc.propose(tokens) == ref.propose(tokens)
            inc.reset()
            tokens = [3, 2, 1] * 30
            assert inc.propose(tokens) == ref.propose(tokens)
            assert inc.propose(tokens, 0) == []
    rows = []
    for length in (256, 4096, 16384):
        times = {}
        results = {}
        for name, cls in (("reference", NGramProposer), ("incremental", IncrementalNGramProposer)):
            tokens = ([1, 2, 3] * (length//3+1))[:length]
            proposer = cls(3, 4)
            start = time.perf_counter()
            outputs = []
            # 计时包含初次建索引，不用只测缓存命中的方式夸大收益。
            for _ in range(128):
                outputs.append(proposer.propose(tokens))
                tokens.append((tokens[-1] % 3) + 1)
            times[name] = (time.perf_counter()-start)*1000
            results[name] = outputs
        assert results["reference"] == results["incremental"]
        rows.append(dict(context=length, rounds=128, milliseconds=times))
    print(json.dumps(dict(checks=checks, status="PASS", microbench=rows), indent=2))

if __name__ == "__main__":
    main()
