"""FlashInfer 0.5.3 随机采样适配；默认后端不变，也不捕获到CUDA Graph。

优先保持现有请求合同：seed、repetition penalty显式回退，截断歧义批量修复。
这不是一个包办所有工作的单kernel：批量温度/softmax、FlashInfer过滤与抽样、
边界检查共同构成管线。检查会同步一次，必须把这部分也计入端到端耗时。
"""
from collections import Counter
from importlib.metadata import version

import torch


class FlashInferSampler:
    def __init__(self, min_batch_size=32):
        if type(min_batch_size) is not int or min_batch_size < 1:
            raise ValueError("flashinfer_min_batch_size must be a positive integer")
        try:
            installed = version("flashinfer-python")
            if installed != "0.5.3":
                raise RuntimeError(f"validated FlashInfer version is 0.5.3, got {installed}")
            from flashinfer import sampling
        except ImportError as error:
            raise RuntimeError("FlashInfer backend requires the documented .venv-flashinfer environment") from error
        self.ops = sampling
        self.stats = Counter()
        # 微基准和真实模型C8均观察到额外开销，保守默认仅候选行>=32时启用。
        # 这是本机校准阈值，不是所有GPU通用最优点；显式1可做消融。
        self.min_batch_size = min_batch_size

    def filtered_probs(self, logits, seqs):
        """返回候选分布和需参考回退的行。先top-k，再在其分布上做top-p。

        FlashInfer的阈值过滤可能保留一整个并列分数组；原Torch排序top-p只保留
        其中一部分。对并列边界与接近累计概率阈值的行，不擅自改变旧语义。
        """
        vocab = logits.shape[1]
        temperatures = torch.tensor([s.temperature for s in seqs], device=logits.device,
                                    dtype=torch.float32)
        # 转换/除法产生新张量，不原地修改来自模型的logits。
        probs = torch.softmax(logits.float() / temperatures[:, None], dim=-1)
        if not torch.isfinite(probs).all().item():
            raise ValueError("sampling requires a finite nonempty probability distribution")
        bad = torch.zeros(len(seqs), dtype=torch.bool, device=logits.device)
        if any(0 < s.top_k < vocab for s in seqs):
            ks = torch.tensor([min(s.top_k or vocab, vocab) for s in seqs],
                              device=logits.device, dtype=torch.int32)
            kept = self.ops.top_k_renorm_probs(probs, ks)
            # 原实现以第k大值为阈值，全部边界并列项都保留。若库行为不同则回退。
            boundary = probs.masked_fill(kept <= 0, float("inf")).amin(dim=-1)
            bad |= ((probs >= boundary[:, None]) != (kept > 0)).any(dim=-1)
            probs = kept
        if any(s.top_p < 1 for s in seqs):
            ps = torch.tensor([s.top_p for s in seqs], device=logits.device, dtype=torch.float32)
            kept = self.ops.top_p_renorm_probs(probs, ps)
            boundary = probs.masked_fill(kept <= 0, float("inf")).amin(dim=-1)
            equal = probs == boundary[:, None]
            higher = (probs * (probs > boundary[:, None])).sum(dim=-1)
            inclusive = (probs * (probs >= boundary[:, None])).sum(dim=-1)
            # 2e-6是选择保守回退的边界带，不是放宽数值测试的精度容差。
            # 只有“排序会截断并列组的一部分”才需修复；整个并列组都需保留时无需回退。
            ambiguous = ((equal.sum(dim=-1) > 1) & ((inclusive-boundary) > ps)) | ((higher-ps).abs() < 2e-6) | ((inclusive-ps).abs() < 2e-6) | ((inclusive-boundary-ps).abs() < 2e-6)
            bad |= (ps < 1) & (ambiguous | (higher > ps) | (inclusive < ps))
            probs = kept
        bad |= ~torch.isfinite(probs).all(dim=-1) | (probs.sum(dim=-1) <= 0)
        return probs, bad

    @staticmethod
    def reference_filter_batch(logits, seqs):
        """并列边界使用Torch原排序语义，但按batch执行，不逐请求启动全部算子。

        这里确实仍有sort，不能称为所有行都sorting-free；计数器单独记录修复行。
        sorted logits中先应用top-k阈值，再计算top-p，顺序与原参考完全相同。
        """
        device, vocab = logits.device, logits.shape[1]
        temps = torch.tensor([s.temperature for s in seqs], dtype=torch.float32, device=device)
        values, indices = (logits.float() / temps[:, None]).sort(dim=-1, descending=True)
        ks = torch.tensor([min(s.top_k or vocab, vocab)-1 for s in seqs], dtype=torch.long, device=device)
        values = values.masked_fill(values < values.gather(1, ks[:, None]), -float("inf"))
        ps = torch.tensor([s.top_p for s in seqs], dtype=torch.float32, device=device)
        remove = values.softmax(-1).cumsum(-1) > ps[:, None]
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        values = values.masked_fill(remove & (ps[:, None] < 1), -float("inf"))
        # 回到词表原顺序再softmax，和参考路径一样，不在排序顺序上偷换浮点归约。
        filtered = torch.empty_like(values).scatter_(1, indices, values)
        return filtered.softmax(-1)

    def __call__(self, logits, seqs, reference):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FlashInfer sampler is host-driven; keep it outside CUDA Graph capture")
        # 0.5.3抽样接口使用batch RNG；为保留请求级seed+生成位置、重排/取消独立性，
        # 有seed的行仍调用原Torch实现。不要假装deterministic=True代表跨后端同seed等价。
        fast, fallback = [], []
        for i, seq in enumerate(seqs):
            if seq.temperature <= 1e-10:
                reason = "greedy_reference"
            elif seq.seed is not None:
                reason = "seed_reference"
            elif seq.repetition_penalty != 1.0:
                reason = "penalty_reference"
            else:
                fast.append(i)
                continue
            self.stats[reason] += 1
            fallback.append(i)
        if len(fast) < self.min_batch_size:
            self.stats["small_batch_reference_rows"] += len(fast)
            return reference(logits, seqs)
        output = torch.empty(len(seqs), dtype=torch.long, device=logits.device)
        if fast:
            indices = torch.tensor(fast, device=logits.device, dtype=torch.long)
            selected_logits = logits.index_select(0, indices)
            selected_seqs = [seqs[i] for i in fast]
            probs, bad = self.filtered_probs(selected_logits, selected_seqs)
            # 一次D2H边界判定；低并发可能比参考更慢，所以不是自动选择的默认后端。
            bad_rows = bad.tolist()
            repair = [j for j, value in enumerate(bad_rows) if value]
            if repair:
                repair_indices = torch.tensor(repair, dtype=torch.long, device=logits.device)
                corrected = self.reference_filter_batch(selected_logits.index_select(0, repair_indices),
                                                        [selected_seqs[j] for j in repair])
                probs.index_copy_(0, repair_indices, corrected)
            self.stats["reference_filter_rows"] += len(repair)
            self.stats["sorting_free_rows"] += len(fast)-len(repair)
            # 必须传持续推进的默认generator；0.5.3内部在generator=None时新建
            # generator，不能依赖它提供跨调用不断变化的随机流。
            generator = torch.cuda.default_generators[logits.device.index]
            # 引擎构造阶段会把默认device切成CUDA；0.5.3内部创建RNG状态tensor时
            # 未显式指定CPU，因此局部覆盖factory默认device，防止生成CUDA ByteTensor。
            # probs和采样输出仍在显式CUDA设备；不改进程全局默认device，也不patch上游包。
            with torch.device("cpu"):
                sampled = self.ops.sampling_from_probs(probs, deterministic=True, generator=generator)
            output.index_copy_(0, indices, sampled.long())
            self.stats["flashinfer_rows"] += len(fast)
            self.stats["flashinfer_calls"] += 1
        if fallback:
            fallback.sort()
            indices = torch.tensor(fallback, device=logits.device, dtype=torch.long)
            output.index_copy_(0, indices, reference(logits.index_select(0, indices), [seqs[i] for i in fallback]))
        return output
