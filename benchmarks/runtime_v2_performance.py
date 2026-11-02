"""Runtime V2 七组合性能验收；固定输入、轮次交错、原始事件与失败证据全部保留。

正式：python -m benchmarks.runtime_v2_performance run --model MODEL --output NEW_DIR
冒烟：上述命令加 --quick（小样本仅说明能运行，不会声称收益门槛通过）。
CPU：python -m benchmarks.runtime_v2_performance self-test
仅冻结输入：python -m benchmarks.runtime_v2_performance prepare --model MODEL --output NEW_DIR
prepare只分词、检查输入唯一性与记录哈希，不启动GPU，也不代表性能验收。
正式长测应由runtime_v2_suite启动，以获得整个Linux进程组的回收保证；本脚本单独运行
只负责直接模型worker，不能保证回收第三方编译器另外创建的所有孙进程。

这是engine-only评测，不含HTTP/TLS/网络。一个事件可能提交多个token，本脚本只记录
真实事件时间，不线性插值“每token时间”。固定L2门控和可变L8分开；普通decode Graph
在所有组合开启，只有最后一组启用mixed Graph。没有读取质量holdout来调阈值。
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

from benchmarks.release_matrix import NATURAL_TASKS, CODE_TASKS, summarize, provenance

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("baseline", "mixed", "reuse", "fixedDV", "adaptive_gate", "adaptive_length", "graph")
THRESHOLDS = dict(benefit_throughput_ratio=1.05, nonregression_throughput_ratio=0.95,
                  maximum_p95_ratio=1.10, minimum_rounds=5, minimum_p95_requests=30,
                  minimum_p95_intervals=100)
# 原创性能输入，与质量holdout无关。长输入仅拼接不同段落，不重复同段填充。
CHINESE_TASKS = (
    "旧城的公交总站要调整夜班发车表。最后一班火车常晚点，但末班公交不能无限等待。请设计一个兼顾换乘乘客、司机工时与车辆回库的规则。讨论信息更新不及时、天气变化和临时封路时如何降级，分别给出可观测指标和不能仅靠平均数解释的现象。",
    "一家社区厨房每天接收不同数量的蔬菜捐赠。收货记录可能重复，冷藏柜容量有限，不同蔬菜的保质期也不同。请设计从验收、分类、入库到分配的工作流程。说明怎样追踪批次、处理重复登记、避免过期，并让后来接班的人理解当天决策。",
    "山地测绘队沿两条不同路径测量一座桥的位置。两组记录中的时间与坐标略有差异，不能直接平均。请解释应先检查哪些条件，怎样保存原始测量和校正结果，以及如何区分仪器误差、坐标系差异和真正的地形变化。",
    "图书馆准备修复一批边缘破损的地图。部分地名可以从其他版本推断，但推断不能混进原始文字。请给修复人员写一份操作说明：如何标注确定内容、模糊内容和猜测，如何保留修订历史，以及如何让读者知道哪些信息仍存在争议。",
    "一个小型剧团要在三间排练室之间安排演员。某些场景需要同一位演员同时练习唱歌和走位，而道具运输也占用时间。请建立可解释的排程办法，指出哪些约束不能违反，哪些目标可以折中，并讨论临时缺席时怎样局部调整。",
    "海边学校要建立雨水收集系统。屋顶面积、降雨时段、水箱容量和用水需求都在变化。请提出一套从小规模试验开始的方案，解释为什么年平均降雨量不足以决定水箱大小，并给出评估溢流、缺水与维护成本的方法。",
    "手工陶瓷作坊发现同一批釉料在不同窑位呈现不同颜色。请设计不浪费大量材料的排查实验，区分温度曲线、摆放位置、坯体含水率和釉层厚度的影响。说明应记录哪些原始数据，以及为什么一次成功烧制不能证明工艺已经稳定。",
    "维修部门想按紧急程度而非报修先后安排任务。电梯停运、路灯损坏和办公室漏水的影响不同，还可能存在重复报修。请设计优先级和合并规则，解释如何防止低优先级任务永久等待，并给出向使用者说明延期原因的方式。",
    "一个自然保护区计划开放新的步道。管理者同时关心游客安全、动植物栖息地和志愿者巡查时间。请比较固定关闭部分路段与按季节开放两种方案，列出需要收集的证据，并说明哪些数据不足时不应急于作出永久决定。",
    "实验室保存了十年的气象观测表，其中有缺失值、错误单位和仪器更换记录。请写一份清洗规范，说明哪些情况可以自动修复，哪些必须人工审核。特别解释为什么把缺失值全部填成零会破坏后续分析。",
    "自行车店准备提供预约维修。检查故障只需几分钟，真正修复却可能等待配件数日。请把接单、诊断、报价、等件和交车设计成清楚的状态流程，讨论顾客取消、同意新报价和配件延迟时如何保持记录一致。",
    "博物馆用多个传感器记录展柜湿度。网络中断后设备会补传旧数据。请说明怎样区分事件发生时间与接收时间，如何发现重复数据，以及实时报警和历史报表为什么可能需要不同的处理规则。",
    "一支登山队在没有网络的地区使用两台设备记录路线。两名队员可能同时修改同一个路标的名字和坐标。请设计重新联网时的合并办法，保留各自原始记录，并区分真正的重复记录和位置相近但不同的路标。说明发生冲突时哪些决定不能自动完成。",
    "工厂准备更换包装线上的条码阅读器。新设备读得更快，但偶尔会把磨损标签识别成另一个有效号码。请设计覆盖正常、磨损和反光包装的验收实验，保留每次识别的原始图像，并说明为什么整体准确率提高不一定意味着每种包装都改善。",
    "档案室要把纸质借阅记录迁移到新的系统。记录中可能同时存在曾用名、重复编号和未归还的材料。请制定分批迁移方案，说明如何维持迁移期间的借还业务，怎样检查余额与实际库存，以及失败时如何回到可核对的状态而不丢失新记录。",
    "一家面包店希望预测第二天的需求。节假日、天气和临时团购都会影响销售，但过量生产和缺货的代价不同。请提出从简单规则开始的预测与验证流程，分别记录销售量、未满足的需求和剩余库存，并说明为什么不能只用卖出的数量代表全部需求。",
)
ENGLISH_QUESTIONS = (
    "Identify the most important invariant in these cases and explain how a test could falsify it.",
    "Separate assumptions from observations in the following material, then suggest one controlled experiment.",
    "Describe a plausible failure sequence and a recovery procedure that preserves the original evidence.",
    "Compare a simple implementation with a more complex one, including the cost of maintaining each.",
    "Explain which work can proceed independently and which work requires an explicit ordering constraint.",
    "Choose three useful measurements and explain what each measurement cannot tell us.",
    "Propose a minimal prototype for the following cases and a clear criterion for stopping the experiment.",
    "List the resources that could become exhausted and suggest a bounded policy for each one.",
    "Explain how a newcomer could reproduce the main observation without relying on unstated knowledge.",
    "Distinguish persistent state from temporary state and describe when each can safely be discarded.",
    "Give a counterexample to an attractive but incorrect shortcut in the following material.",
    "Describe what should be recorded before making a change and what should be compared afterwards.",
    "Explain how to handle an interrupted operation without either losing work or applying it twice.",
    "Compare the needs of a single user with those of several simultaneous users in these cases.",
    "Describe a boundary case that ordinary examples overlook and a test that exposes it clearly.",
    "Explain which conclusions would change if the input size grew substantially, and justify the distinction.",
)
CHINESE_QUESTIONS = (
    "请找出材料中最重要的不变量，并说明什么样的反例能够推翻当前方案。",
    "请区分已经观测到的事实和仍待验证的假设，再设计一个控制变量实验。",
    "请描述一种可信的失败过程，以及保留原始证据的恢复办法。",
    "请比较简单方案和复杂方案的优缺点，特别说明后续维护需要付出的代价。",
    "请说明哪些工作能够独立开展，哪些工作必须保持明确的先后顺序。",
    "请选择三个有用的测量指标，并指出每个指标不能解释的问题。",
    "请为材料中的问题设计最小可行实验，并给出停止实验的明确条件。",
    "请列出可能耗尽的资源，分别设计容量限制与超限时的处理办法。",
    "请说明一位新成员如何复现关键观察，避免依赖未写出的隐含经验。",
    "请区分需要长期保存的状态和临时状态，解释何时可以安全丢弃它们。",
    "请举例说明一个看似省事却不正确的捷径，以及它为何会造成误判。",
    "请解释实施变更前应记录什么，变更后应如何进行公平的比较。",
    "请设计操作被中断后的处理流程，同时避免丢失工作和重复执行。",
    "请比较单人使用和多人同时使用时的需求，解释可能出现的新矛盾。",
    "请指出常规示例容易遗漏的边界情况，并设计能稳定暴露问题的测试。",
    "如果输入规模显著增长，请说明哪些结论会改变，哪些原则仍然成立。",
)

# 最终 holdout 的问题前缀在选定 varlen 候选之后才编写；背景段落仍与
# calibration/validation 共享，因此只能称为 token 序列独立，不能称为
# 独立抽样的真实流量。完整输入及 SHA 在 prepare 阶段冻结。
FINAL_ENGLISH_QUESTIONS = (
    "A team receives inconsistent records from three clocks. State which timestamps can be trusted and how you would verify the ordering.",
    "The system appears faster after a change, but some users wait longer. Design a measurement that could expose the tradeoff.",
    "A service restart happens between writing data and acknowledging it. Describe the state that must survive and a safe retry rule.",
    "Two implementations agree on ordinary inputs but disagree near a capacity limit. Give a boundary test and explain the expected invariant.",
    "A cached result is reused after its source changes. Specify when the cache must be invalidated and what evidence would reveal stale data.",
    "A worker receives jobs in bursts while another stays idle. Propose a balancing policy and explain how to detect unfairness.",
    "The first attempt of an operation succeeds but its response is lost. Describe how to avoid applying the operation twice.",
    "A benchmark has excellent average throughput and poor tail latency. Explain which raw events you would inspect before recommending it.",
)
FINAL_CHINESE_QUESTIONS = (
    "三台时钟给出不一致的记录。请说明哪些时间戳可以信任，并设计核对事件先后的办法。",
    "系统改动后平均速度提高，但部分用户等待更久。请设计能暴露这种代价的测量方法。",
    "服务在写入数据与确认之间重启。请说明必须保留哪些状态，以及如何安全重试。",
    "两种实现对常规输入一致，却在容量边界分歧。请给出边界测试和应保持的不变量。",
    "缓存结果被复用时原始数据已经改变。请说明何时失效，并给出发现陈旧结果的证据。",
    "任务成批到来，一名工作者忙碌而另一名空闲。请提出分配策略并说明如何发现不公平。",
    "一次操作已成功，但确认消息丢失。请解释如何避免重试时重复执行。",
    "基准测试平均吞吐很好，尾部延迟却很差。请说明推荐前应核对哪些原始事件。",
)


def sha(path: Path) -> str:
    """散列文件作为证据身份；大文件分块读取。"""
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def object_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_new(path: Path, value) -> None:
    """不覆盖旧报告；每trial一个新文件，失败也保留其已经观测到的事件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def environment() -> dict:
    """固定显式环境与软件版本，不采集密码或任意环境变量。worker再补实际GPU信息。"""
    versions = {}
    for package in ("torch", "triton", "transformers", "flash-attn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
        variables={name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "NANOVLLM_SPEC_GRAPH",
            "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTORCH_CUDA_ALLOC_CONF",
            "NANOVLLM_BENCH_GPU_INDEX")})


def variant_config(name: str, controls: dict) -> tuple[dict, dict]:
    """输出引擎Config和请求SamplingParams；后者决定显式请求spec及最大候选长度。"""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name}")
    rank = VARIANTS.index(name)
    config = dict(controls, runtime_preset="baseline", enforce_eager=False,
        batching_mode="separate" if rank == 0 else "mixed",
        metadata_backend="dynamic" if rank < 2 else "reuse",
        speculative_policy="off" if rank < 3 else "fixed" if rank == 3 else "adaptive",
        enable_mixed_graph=name == "graph")
    sampling = dict(temperature=0, ignore_eos=True, speculative_ngram_size=0 if rank < 3 else 3,
                    speculative_max_draft_tokens=2 if rank < 5 else 8)
    return config, sampling


def build_workloads(tokenizer, args) -> list[dict]:
    """冻结分词后的prompt及到达表；自然组不重复填充，高重复组单独明确标识。"""
    cases = []
    sources = dict(english=NATURAL_TASKS, chinese=CHINESE_TASKS, code=CODE_TASKS)
    for dataset in args.datasets:
        for scenario in args.scenarios:
            for concurrency in args.concurrencies:
                jobs = []
                count = args.requests_per_case
                for index in range(count):
                    length = args.long_tokens if scenario == "long" or (scenario in ("dynamic", "staggered") and index % 4 == 0) else args.short_tokens
                    if dataset == "repeat":
                        split = getattr(args, "input_split", "validation")
                        repeat_text = {
                            "calibration": "red blue green. ",
                            "validation": "oak pine birch. ",
                            "final_holdout": "river stone willow. ",
                        }[split]
                        unit = tokenizer.encode(repeat_text, add_special_tokens=False)
                        if not unit:
                            raise ValueError("repeat fixture did not tokenize")
                        prompt = (unit * math.ceil(length / len(unit)))[:length]
                        text = repeat_text * math.ceil(length / len(unit))
                    else:
                        paragraphs = sources[dataset]
                        split = getattr(args, "input_split", "validation")
                        questions = CHINESE_QUESTIONS if dataset == "chinese" else ENGLISH_QUESTIONS
                        # 两个split使用不同问题前缀，分词后冻结且交叉检查不相同。
                        # 背景材料主题仍有共享，不把它称为真实流量的独立随机采样。
                        middle = len(questions) // 2
                        questions = (
                            questions[:middle] if split == "calibration"
                            else questions[middle:] if split == "validation"
                            else FINAL_CHINESE_QUESTIONS if dataset == "chinese"
                            else FINAL_ENGLISH_QUESTIONS
                        )
                        if count > len(paragraphs) * len(questions):
                            raise ValueError(f"{dataset}: requested more distinct combinations than authored materials support")
                        offset = (index // len(questions)) % len(paragraphs)
                        text = questions[index % len(questions)] + "\n\n" + "\n\n".join(paragraphs[offset:] + paragraphs[:offset])
                        encoded = tokenizer.encode(text, add_special_tokens=False)
                        if len(encoded) < length:
                            raise ValueError(f"{dataset} distinct text has {len(encoded)} tokens, needs {length}; do not pad")
                        prompt = encoded[:length]
                    # 首波同时到达，之后才动态补入，避免把C16写成事实上仅C1的串行低负载。
                    arrival = (.005 + (index - concurrency) * args.arrival_interval
                               if scenario == "dynamic" and index >= concurrency else 0.0)
                    jobs.append(dict(job=index, arrival_s=arrival, prompt_token_ids=prompt,
                        prompt_sha256=object_sha(prompt), prompt_text=tokenizer.decode(prompt, skip_special_tokens=False),
                        source_text=text, source_text_sha256=object_sha(text), max_tokens=args.output_tokens))
                if scenario == "staggered":
                    # Supplemental overlap probe, not part of the registered release matrix:
                    # two short requests reach Decode, then a long prompt arrives while
                    # they are still active. Dynamic replenishment alone does not force
                    # this P-vs-D collision because its first wave fills all slots.
                    if concurrency < 3 or count < 3:
                        raise ValueError("staggered requires concurrency and request count >= 3")
                    order = [1, 2, 0] + [index for index in range(count) if index not in (0, 1, 2)]
                    jobs = [jobs[index] for index in order]
                    for index, job in enumerate(jobs):
                        job["job"] = index
                        job["arrival_s"] = 0.0 if index < 2 else .05 + (index - 2) * args.arrival_interval
                unique = len({job["prompt_sha256"] for job in jobs})
                unique_texts = len({job["prompt_text"] for job in jobs})
                if dataset != "repeat" and (unique != count or unique_texts != count):
                    raise ValueError(f"{dataset}/{scenario}: token truncation produced repeated prompts; do not relabel repeated inputs as independent")
                cases.append(dict(id=f"{dataset}-{scenario}-c{concurrency}", dataset=dataset, scenario=scenario,
                    concurrency=concurrency, jobs=jobs, unique_prompt_count=unique,
                    input_note="Distinct question+passage combinations per case; paragraphs occur once within each natural/code prompt. Same frozen inputs are reused across variants, warmups and rounds."))
    return cases


def verify_input_split(tokenizer, args, cases) -> dict:
    """对三个 split 逐 token 哈希查重；背景主题共享不构成统计独立。"""
    chosen = getattr(args, "input_split", "validation")
    own_hashes = {job["prompt_sha256"] for case in cases for job in case["jobs"]}
    others = {}
    for split in ("calibration", "validation", "final_holdout"):
        if split == chosen:
            continue
        other_args = copy.copy(args)
        other_args.input_split = split
        other = build_workloads(tokenizer, other_args)
        other_hashes = {job["prompt_sha256"] for case in other for job in case["jobs"]}
        overlap = own_hashes & other_hashes
        if overlap:
            raise ValueError(f"{chosen}/{split} share {len(overlap)} tokenized prompts")
        others[split] = dict(unique_prompts=len(other_hashes),
                             workload_sha256=object_sha(other), overlap_count=0)
    return dict(status="PASS_DISJOINT_INPUTS_ONLY", chosen_split=chosen,
                chosen_unique_prompts=len(own_hashes), other_splits=others, overlap_count=0,
                note="Same shape controls, disjoint prompt token sequences; authored background topics can overlap.")


def snapshot(llm) -> dict:
    """保存真实runtime_metrics原始结构，缺失接口立即失败，不冒充已有观测性。"""
    metrics = llm.runtime_metrics()
    return asdict(metrics) if is_dataclass(metrics) else copy.deepcopy(metrics)


def cpu_resources() -> dict:
    """读取当前RSS和进程生存期high-water，不把它冒称为本trial的采样峰值。"""
    result = dict(rss_bytes=None, process_high_water_rss_bytes=None, threads=None, file_descriptors=None,
                  scope="Point-in-time RSS; high-water is entire process lifetime, not this trial's peak.")
    # 同物理 GPU 配对仍可能受其他卡的 CPU 调度/加载影响；记录主机并行负载。
    result["host_loadavg"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    for source, name in (("/proc/stat", "host_cpu_stat"), ("/proc/loadavg", "host_loadavg_raw"),
                         ("/proc/pressure/cpu", "host_cpu_pressure")):
        try:
            text = Path(source).read_text()
            result[name] = text.splitlines()[0] if source == "/proc/stat" else text.strip()
        except OSError:
            result[name] = None
    status = Path("/proc/self/status")
    if status.is_file():
        rows = {}
        for line in status.read_text().splitlines():
            key, _, value = line.partition(":")
            rows[key] = value.strip().split()
        for source, destination in (("VmRSS", "rss_bytes"), ("VmHWM", "process_high_water_rss_bytes")):
            if source in rows:
                result[destination] = int(rows[source][0]) * 1024
        if "Threads" in rows:
            result["threads"] = int(rows["Threads"][0])
        try:
            result["file_descriptors"] = len(list(Path("/proc/self/fd").iterdir()))
        except OSError:
            pass
        result["backend"] = "Linux /proc"
    else:
        try:
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["process_high_water_rss_bytes"] = value if sys.platform == "darwin" else value * 1024
            result["backend"] = "resource; current RSS unavailable"
        except ImportError:
            result["backend"] = "unavailable on this platform"
    return result


def metric_deltas(before: dict, after: dict) -> dict:
    """累计计数器作差后再计算比率；零候选/零Graph尝试返回None而非伪造0%或100%。"""
    output = {}
    for section, names in (("speculative", ("proposed_tokens", "accepted_tokens", "target_verification_calls",
                                            "draft_rounds", "fallback_decode_calls")),
                           ("packed_graph", ("captures", "replays", "fallbacks"))):
        left, right = before.get(section, {}), after.get(section, {})
        keys = set(names) | {key for key in right if key.startswith("fallback_")}
        output[section] = {key: right.get(key, 0) - left.get(key, 0) for key in keys}
    left, right = before.get("decode_graph", {}), after.get("decode_graph", {})
    decoded = {key: right.get(key, 0) - left.get(key, 0) for key in right
               if key == "decode_replays" or key.startswith("decode_fallback_")}
    output["decode_graph"] = dict(decoded, replays=decoded.get("decode_replays", 0),
        fallbacks=sum(value for key,value in decoded.items() if key.startswith("decode_fallback_")))
    spec = output["speculative"]
    spec["acceptance_rate"] = spec["accepted_tokens"] / spec["proposed_tokens"] if spec["proposed_tokens"] > 0 else None
    for section in ("packed_graph", "decode_graph"):
        values = output[section]
        attempts = values["replays"] + values["fallbacks"]
        values["replay_fraction"] = values["replays"] / attempts if attempts > 0 else None
    return output


def finalize_request(record: dict) -> None:
    """TTFT含入口排队；事件gap与摊销TPOT分开，绝不为一个burst补造token时间。"""
    events = [event for event in record["events"] if event["cumulative_tokens"] > 0]
    gaps = [(right["time_s"] - left["time_s"]) * 1000 for left, right in zip(events, events[1:])]
    count = record.get("output_tokens", 0)
    visible = next((event for event in record["events"] if event.get("text")), None)
    record.update(event_gap_ms=gaps, event_count=len(events),
        entry_queue_ms=(record["submit_s"] - record["arrival_s"]) * 1000,
        first_visible_text_ms=(visible["time_s"] - record["arrival_s"]) * 1000 if visible is not None else None,
        ttft_ms=(events[0]["time_s"] - record["arrival_s"]) * 1000 if events else None,
        submit_to_first_event_ms=(events[0]["time_s"] - record["submit_s"]) * 1000 if events else None,
        tpot_ms=(events[-1]["time_s"] - events[0]["time_s"]) * 1000 / (count - 1) if events and count > 1 else None,
        latency_ms=(events[-1]["time_s"] - record["arrival_s"]) * 1000 if events else None)


def run_trial(llm, case, sampling, timeout, *, clock=time.perf_counter, sleep=time.sleep,
              params_factory=None, device=None, trace_steps=False) -> dict:
    """运行一个真实到达trace；device是可注入同步/显存接口，供纯CPU fake验证。"""
    if params_factory is None:
        from nanovllm import SamplingParams
        params_factory = SamplingParams
    if device is None:
        import torch
        device = torch.cuda
    before = snapshot(llm)
    cpu_before = cpu_resources()
    device.synchronize()
    device.reset_peak_memory_stats()
    records, active, pending, error, step_wall_ms, max_active = [], {}, 0, None, [], 0
    # 仅诊断运行启用；正式计时默认不记录逐 step 轨迹，避免改变热路径。
    step_trace = [] if trace_steps else None
    observed_paths = dict(pd_steps=0, dv_mixed_steps=0, verification_steps=0)
    started = clock()
    try:
        while pending < len(case["jobs"]) or active:
            now = clock() - started
            if now > timeout:
                raise TimeoutError(f"trial exceeded {timeout}s")
            while pending < len(case["jobs"]) and len(active) < case["concurrency"] and case["jobs"][pending]["arrival_s"] <= now:
                job = case["jobs"][pending]
                seq = llm.add_request(job["prompt_token_ids"], params_factory(**sampling, max_tokens=job["max_tokens"]))
                record = dict(job=job["job"], request_id=seq.seq_id, arrival_s=job["arrival_s"], expected_tokens=job["max_tokens"],
                    submit_s=clock() - started, input_tokens=len(job["prompt_token_ids"]), events=[], failed=False)
                records.append(record)
                active[seq.seq_id] = seq, record
                pending += 1
                max_active = max(max_active, len(active))
            if not active:
                sleep(min(.001, max(0.0, case["jobs"][pending]["arrival_s"] - now)))
                continue
            step_started = clock()
            if step_trace is not None:
                before_waiting = len(llm.scheduler.waiting)
                before_running = len(llm.scheduler.running)
            step_result = llm.step()
            stats = (asdict(step_result[1]) if isinstance(step_result, tuple)
                     and len(step_result) == 2 and is_dataclass(step_result[1]) else None)
            if stats is not None:
                observed_paths["pd_steps"] += int(stats.get("scheduled_prefill",0) > 0 and stats.get("query_decode",0) > 0)
                observed_paths["dv_mixed_steps"] += int(stats.get("query_verify",0) > 0 and stats.get("query_decode",0) > 0)
                observed_paths["verification_steps"] += int(stats.get("query_verify",0) > 0)
            # 同一step可返回多个request事件，它们共享这次观测时间；无虚构GPU token时间。
            observed = clock() - started
            step_wall_ms.append((started + observed - step_started) * 1000)
            if step_trace is not None:
                if stats is None:
                    raise RuntimeError("step trace requires structured StepStats")
                step_trace.append(dict(elapsed_ms=step_wall_ms[-1],
                    time_s=observed, waiting_before=before_waiting,
                    running_before=before_running,
                    prefill_tokens=stats.get("scheduled_prefill", 0),
                    decode_queries=stats.get("query_decode", 0),
                    verify_queries=stats.get("query_verify", 0),
                    generated_tokens=stats.get("generated_tokens", 0)))
            for event in llm.step_events():
                if event.request_id not in active:
                    raise RuntimeError("event belongs to unknown request")
                seq, record = active[event.request_id]
                progress = event.token_index + 1 if event.token_id >= 0 else seq.num_completion_tokens
                previous = record["events"][-1]["cumulative_tokens"] if record["events"] else 0
                if progress < previous:
                    raise AssertionError("cumulative token progress decreased")
                record["events"].append(dict(time_s=observed, cumulative_tokens=progress,
                    committed_in_event=progress - previous, token_id=event.token_id,
                    text=event.text, finished=event.finished, finish_reason=event.finish_reason))
                if event.finished:
                    record.update(output_token_ids=list(seq.completion_token_ids), output_tokens=seq.num_completion_tokens,
                                  finish_reason=event.finish_reason)
                    if progress != seq.num_completion_tokens:
                        raise AssertionError("terminal cumulative count differs from sequence")
                    if seq.num_completion_tokens != record["expected_tokens"] or event.finish_reason != "length":
                        raise AssertionError("ignore_eos request terminated early or with an unexpected finish reason")
                    llm.forget_request(seq.seq_id)
                    del active[seq.seq_id]
        device.synchronize()
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        for request_id, (seq, record) in list(active.items()):
            record.update(failed=True, error=error, output_token_ids=list(seq.completion_token_ids),
                          output_tokens=seq.num_completion_tokens)
            try:
                llm.cancel_request(request_id)
                llm.forget_request(request_id)
            except Exception as cleanup_error:
                record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        # 不在CUDA错误后继续下一trial；worker保存本记录后退出。
    seconds = max(clock() - started, 1e-12)
    cpu_after = cpu_resources()
    for record in records:
        finalize_request(record)
    manager = llm.scheduler.block_manager
    after = snapshot(llm)
    kv_refs = sum(block.ref_count for block in manager.blocks)
    output_tokens = sum(record.get("output_tokens", 0) for record in records)
    memory = {}
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        try:
            memory[name] = getattr(device, name)()
        except Exception:
            memory[name] = None
    return dict(status="FAIL" if error or manager.used_block_ids or kv_refs else "PASS", error=error,
        case=case["id"], dataset=case["dataset"], scenario=case["scenario"], concurrency=case["concurrency"],
        max_active_observed=max_active,
        seconds=seconds, output_tokens=output_tokens, tokens_per_second=output_tokens / seconds,
        requests=records, missing_requests=len(case["jobs"]) - len(records),
        failures=sum(record["failed"] for record in records) + len(case["jobs"]) - len(records),
        used_blocks_after=len(manager.used_block_ids), kv_ref_total_after=kv_refs,
        runtime_before=before, runtime_after=after, memory=memory,
        runtime_deltas=metric_deltas(before, after), cpu_resources_before=cpu_before, cpu_resources_after=cpu_after,
        full_step_wall_ms=step_wall_ms,
        step_trace=step_trace,
        observed_paths=observed_paths,
        entry_queue_ms=summarize([record["entry_queue_ms"] for record in records]),
        ttft_ms=summarize([record["ttft_ms"] for record in records]),
        first_visible_text_ms=summarize([record["first_visible_text_ms"] for record in records]),
        tpot_ms=summarize([record["tpot_ms"] for record in records]),
        event_gap_ms=summarize([gap for record in records for gap in record["event_gap_ms"]]))


def prepare(args) -> Path:
    """CPU分词并冻结配置/源文件/输入/environment哈希，尚不加载GPU模型。"""
    from transformers import AutoTokenizer
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cases = build_workloads(tokenizer, args)
    split_check = verify_input_split(tokenizer, args, cases)
    controls = dict(max_model_len=max(args.short_tokens, args.long_tokens) + args.output_tokens,
        max_num_batched_tokens=args.token_budget, prefill_chunk_size=args.chunk_size,
        max_prefill_steps=1, max_num_seqs=max(args.concurrencies), speculative_max_batch_size=0,
        adaptive_max_batch_size=max(args.concurrencies), tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
        sampler_backend="torch", mlp_backend="torch", rmsnorm_backend="torch",
        decode_attention_backend="flash", kv_cache_dtype="model",
        mixed_attention_backend=args.mixed_attention_backend,
        mixed_min_active_decodes=args.mixed_min_active_decodes,
        mixed_low_load_mode=args.mixed_low_load_mode)
    configs = {name: dict(zip(("engine", "sampling"), variant_config(name, controls))) for name in args.variants}
    env = environment()
    model = Path(args.model).resolve()
    model_files = {path.name: sha(path) for path in sorted(model.iterdir()) if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".txt", ".tiktoken"}}
    manifest = dict(schema=1, model=str(model), model_sha256=model_files,
        provenance=provenance(), environment=env, environment_sha256=object_sha(env),
        config_sha256=object_sha(configs), configs=configs, workload_sha256=object_sha(cases),
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        thresholds=THRESHOLDS, input_split_check=split_check, phase_note="Independent process per variant/round. First use and extra warmups separated from one steady trial per case. Process-wide kernel caches may already be warm; not a fresh machine cold start.",
        ordering_note="Five round-level interleavings, rotating variant order. Cases within a worker share the engine; case order is identical across variants in a round.",
        workload_note="Distinct authored question+passage combinations within each natural/code case, not production distribution or independent topic draws. Calibration and validation use disjoint question prefixes and different repeat units; background topics can overlap. Prefix cache disabled in all variants. Repeat group alone repeats text within prompts. Identical frozen cases intentionally recur across variants/warmup/rounds. Only calibration may inform choices; freeze before validation.",
        metric_note="engine-only, arrival-to-first-token-event TTFT; first_visible_text_ms uses first nonempty text event, None when no visible text. Amortized TPOT=(last event-first event)/(completion tokens-1), not individual-token latency. Event gaps are independently reported; burst token timestamps are not invented. CPU high-water is process lifetime, not a trial-local sampled peak.",
        variant_note="fixedDV=L2, not old fixed4. adaptive_gate is the same controller capped at2 (0/2 decisions), not a separate simplified gate-only implementation. adaptive_length changes cap to8. Ordinary decode Graph stays enabled in every variant; graph adds mixed Graph only.")
    output.mkdir(parents=True)
    write_new(output / "workloads.json", cases)
    write_new(output / "performance_manifest.json", manifest)
    return output


def worker(directory: Path, variant: str, round_id: int) -> None:
    """每个worker只加载一个模型；case初始reset，预热后保留成本样本与Graph缓存。"""
    manifest = load(directory / "performance_manifest.json")
    cases = load(directory / "workloads.json")
    if manifest["provenance"]["source_sha256"] != provenance()["source_sha256"] or object_sha(cases) != manifest["workload_sha256"]:
        raise RuntimeError("frozen source/workload changed")
    if object_sha(manifest["configs"]) != manifest["config_sha256"] or object_sha(environment()) != manifest["environment_sha256"]:
        raise RuntimeError("frozen config/environment changed")
    for filename, expected in manifest["model_sha256"].items():
        if sha(Path(manifest["model"]) / filename) != expected:
            raise RuntimeError(f"model/tokenizer changed: {filename}")
    work = directory / f"round_{round_id:02d}" / variant
    work.mkdir(parents=True, exist_ok=False)
    write_new(work / "started.json", dict(event="WORKER_STARTED", pid=os.getpid(), variant=variant, round=round_id))
    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
    ensure_gpu1_available(evidence_path=work / "gpu_observations.jsonl")
    import torch
    from nanovllm import LLM
    settings = manifest["configs"][variant]
    report = dict(status="RUNNING", variant=variant, round=round_id, trials=[],
        manifest_sha256=sha(directory / "performance_manifest.json"), environment=environment(),
        device=dict(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                    total_memory=torch.cuda.get_device_properties(0).total_memory,
                    cuda_runtime=torch.version.cuda))
    report["worker_environment_sha256"] = object_sha(dict(environment=report["environment"], device=report["device"]))
    llm = None
    try:
        with torch.inference_mode():
            started = time.perf_counter()
            llm = LLM(manifest["model"], **settings["engine"])
            report["model_initialization_seconds"] = time.perf_counter() - started
            report["effective_runtime"] = llm.config.runtime_summary()
            if not callable(getattr(llm, "runtime_metrics", None)):
                raise RuntimeError("runtime_metrics() integration is required for this benchmark")
            for case_index, case in enumerate(cases):
                # 隔离case的有界mixed图缓存，避免前case占满4项令后case始终fallback。
                # 普通decode Graph仍保留；模型JIT也不会假装回到fresh-machine状态。
                torch.cuda.synchronize()
                llm.model_runner.packed_graphs.entries.clear()
                llm.model_runner.packed_graphs.seen.clear()
                controller = getattr(llm, "adaptive_controller", None)
                if controller is not None:
                    controller.reset()
                times = getattr(llm, "_event_times", None)
                if times is not None:
                    times.clear()
                phases = ["first_use"] + [f"warmup_{number}" for number in range(manifest["args"]["warmups"])] + ["steady"]
                for phase in phases:
                    # 只允许 first_use/warmup 捕获；steady 陌生 shape 必须 eager。
                    # 因而稳态 captures 应为 0，覆盖不足会显示为 fallback，而非延迟尖峰。
                    llm.set_graph_capture_enabled(phase != "steady")
                    from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
                    ensure_gpu1_available(evidence_path=work / "gpu_observations.jsonl")
                    trial = run_trial(llm, case, settings["sampling"], manifest["args"]["trial_timeout"],
                                      trace_steps=manifest["args"].get("trace_steps", False))
                    trial.update(variant=variant, round=round_id, phase=phase)
                    path = work / f"{case_index:03d}_{case['id']}_{phase}.json"
                    write_new(path, trial)
                    report["trials"].append(dict(path=path.relative_to(directory).as_posix(), sha256=sha(path), status=trial["status"]))
                    print(json.dumps(dict(variant=variant, round=round_id, case=case["id"], phase=phase,
                                         status=trial["status"], tokens_s=trial["tokens_per_second"])), flush=True)
                    if trial["status"] != "PASS":
                        raise RuntimeError(trial["error"] or "KV resource leak")
            report["status"] = "PASS_EXECUTION_ONLY"
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if llm is not None:
                llm.exit()
        except Exception as error:
            report.update(status="FAIL", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_new(work / "report.json", report)


def aggregate(trials: list[dict]) -> dict:
    """只聚合steady，保留每轮throughput和全部request/interval样本量。"""
    groups = {}
    for trial in trials:
        if trial["phase"] == "steady":
            groups.setdefault((trial["case"], trial["variant"]), []).append(trial)
    rows = {}
    for (case, variant), items in groups.items():
        requests = [request for trial in items for request in trial["requests"]]
        rates = [trial["tokens_per_second"] for trial in items]
        rows[f"{case}/{variant}"] = dict(case=case, variant=variant, rounds=len(items),
            throughput=dict(summarize(rates), median=statistics.median(rates), minimum=min(rates), maximum=max(rates)),
            ttft_ms=summarize([r["ttft_ms"] for r in requests]), tpot_ms=summarize([r["tpot_ms"] for r in requests]),
            first_visible_text_ms=summarize([r["first_visible_text_ms"] for r in requests]),
            submit_to_first_event_ms=summarize([r["submit_to_first_event_ms"] for r in requests]),
            entry_queue_ms=summarize([r["entry_queue_ms"] for r in requests]),
            event_gap_ms=summarize([gap for r in requests for gap in r["event_gap_ms"]]),
            failures=sum(trial["failures"] for trial in items),
            execution_pass=all(trial["status"] == "PASS" for trial in items))
    return rows


def compare_group(candidate, baseline) -> dict:
    """预先固定5%吞吐和10%P95门槛；小样本不能凭一个p95就声称无退化。"""
    ratio = candidate["throughput"]["median"] / baseline["throughput"]["median"]
    enough = min(candidate["rounds"], baseline["rounds"]) >= THRESHOLDS["minimum_rounds"]
    latencies = {}
    for metric in ("ttft_ms", "tpot_ms", "event_gap_ms"):
        required = THRESHOLDS["minimum_p95_intervals"] if metric == "event_gap_ms" else THRESHOLDS["minimum_p95_requests"]
        left, right = candidate[metric], baseline[metric]
        samples = min(left["n"], right["n"]) >= required
        value = left["p95"] / right["p95"] if right["p95"] not in (None, 0) and left["p95"] is not None else None
        enough &= samples and value is not None
        latencies[metric] = dict(candidate_n=left["n"], baseline_n=right["n"], required_n=required,
                                p95_ratio=value, sufficient=samples)
    safe = ratio >= THRESHOLDS["nonregression_throughput_ratio"] and all(
        item["p95_ratio"] is not None and item["p95_ratio"] <= THRESHOLDS["maximum_p95_ratio"] for item in latencies.values())
    return dict(status="NOT_EVALUATED_INSUFFICIENT_SAMPLES" if not enough else "PASS" if safe else "REGRESSION",
        throughput_ratio=ratio, benefit_at_least_5pct=ratio >= THRESHOLDS["benefit_throughput_ratio"],
        nonregression_pass=safe if enough else None, p95=latencies)


def is_full_matrix(settings: dict) -> bool:
    """用户可缩小矩阵排错，但缩小后的成功不能升级为完整投递验收成功。"""
    variants = set(settings["variants"])
    split = settings.get("input_split")
    approved_scope = (
        (split == "validation" and variants == set(VARIANTS))
        or (split == "final_holdout" and variants in (set(VARIANTS), {"baseline", "mixed"}))
    )
    return (not settings["quick"] and approved_scope
        and settings.get("rounds", 5) >= THRESHOLDS["minimum_rounds"]
        and settings.get("requests_per_case", 32) >= 32
        and set(settings["datasets"]) == {"english", "chinese", "code", "repeat"}
        and set(settings["scenarios"]) == {"short", "long", "dynamic"}
        and set(settings["concurrencies"]) == {1, 4, 8, 16})


def path_coverage(trials: list[dict]) -> dict:
    """实际执行证据门槛：配置名不能代替PD、DV、复用、控制器或Graph重放真的发生。"""
    rows = {}
    for trial in trials:
        if trial["phase"] != "steady":
            continue
        row = rows.setdefault(trial["variant"], dict(pd_steps=0,dv_mixed_steps=0,verification_steps=0,
            metadata_reuse_calls=0,packed_graph_replays=0,decisions_zero=0,decisions_two=0,decisions_long=0))
        for key, amount in trial.get("observed_paths", {}).items():
            row[key] += amount
        before, after = trial["runtime_before"], trial["runtime_after"]
        def delta(section, key):
            return max(0, after.get(section, {}).get(key, 0) - before.get(section, {}).get(key, 0))
        row["metadata_reuse_calls"] += delta("metadata", "reuse_calls")
        row["packed_graph_replays"] += delta("packed_graph", "replays")
        left, right = before.get("adaptive", {}).get("counters", {}), after.get("adaptive", {}).get("counters", {})
        for key, lengths in (("decisions_zero", (0,)), ("decisions_two", (2,)), ("decisions_long", (4,8))):
            row[key] += sum(max(0,right.get(f"decisions_length_{length}",0)-left.get(f"decisions_length_{length}",0)) for length in lengths)
    for variant, row in rows.items():
        rank = VARIANTS.index(variant)
        missing = []
        if rank >= 1 and not row["pd_steps"]: missing.append("no_actual_P_plus_D_step")
        if rank >= 2 and not row["metadata_reuse_calls"]: missing.append("no_metadata_reuse")
        if rank >= 3 and not row["verification_steps"]: missing.append("no_actual_verification_step")
        if rank >= 3 and not row["dv_mixed_steps"]: missing.append("no_actual_D_plus_V_step")
        if rank >= 4 and not row["decisions_zero"]: missing.append("no_controller_gate_off_decision")
        if rank >= 4 and not row["decisions_two"]: missing.append("no_controller_length2_decision")
        if rank >= 5 and not row["decisions_long"]: missing.append("no_controller_length4_or8_decision")
        if variant == "graph" and not row["packed_graph_replays"]: missing.append("no_mixed_graph_replay")
        row.update(status="REVIEW_REQUIRED" if missing else "PASS", missing=missing)
    return rows


def collect(directory: Path, failures: list[dict]) -> dict:
    """失败trial也进入summary；最终组合与baseline验收，其他组合保留消融/负收益。"""
    manifest = load(directory / "performance_manifest.json")
    trials, worker_reports = [], []
    for path in sorted(directory.glob("round_*/*/report.json")):
        report = load(path)
        worker_reports.append(dict(path=path.relative_to(directory).as_posix(), sha256=sha(path), status=report["status"]))
        if report["manifest_sha256"] != sha(directory / "performance_manifest.json"):
            raise RuntimeError("worker manifest changed")
        for entry in report["trials"]:
            trial_path = directory / entry["path"]
            if sha(trial_path) != entry["sha256"]:
                raise RuntimeError("raw trial changed")
            trials.append(load(trial_path))
    groups = aggregate(trials)
    comparisons, adjacent, differences = {}, {}, []
    for key, group in groups.items():
        base = groups.get(f"{group['case']}/baseline")
        if base and group["variant"] != "baseline":
            comparisons[key] = compare_group(group, base)
            previous_variant = VARIANTS[VARIANTS.index(group["variant"]) - 1]
            previous = groups.get(f"{group['case']}/{previous_variant}")
            if previous:
                adjacent[key] = dict(reference_variant=previous_variant, **compare_group(group, previous))
    # 生成历史分岔仅触发数值复核，不能靠改tie-break掩盖，也不把它直接说成模型错误。
    baseline_trials = {(t["case"], t["round"]): t for t in trials if t["phase"] == "steady" and t["variant"] == "baseline"}
    for trial in trials:
        if trial["phase"] != "steady" or trial["variant"] == "baseline":
            continue
        reference = baseline_trials.get((trial["case"], trial["round"]))
        if reference:
            expected = {r["job"]: r.get("output_token_ids") for r in reference["requests"]}
            changed = [r["job"] for r in trial["requests"] if r.get("output_token_ids") != expected.get(r["job"])]
            if changed:
                differences.append(dict(case=trial["case"], round=trial["round"], variant=trial["variant"], jobs=changed))
    target_variant = manifest["args"]["variants"][-1]
    final = [value for key, value in comparisons.items() if key.endswith("/" + target_variant)]
    cases = load(directory / "workloads.json")
    enough = bool(final) and len(final) == len(cases) and all(
        row["status"] != "NOT_EVALUATED_INSUFFICIENT_SAMPLES" for row in final)
    complete_matrix = is_full_matrix(manifest["args"])
    coverage = path_coverage(trials)
    paths_verified = set(coverage) == set(manifest["args"]["variants"]) and all(row["status"] == "PASS" for row in coverage.values())
    final_safe = enough and all(row["status"] == "PASS" for row in final)
    benefit = any(row["benefit_at_least_5pct"] for row in final)
    performance_status = "NOT_EVALUATED" if not enough or not complete_matrix else "PASS" if final_safe and benefit and paths_verified else "REVIEW_REQUIRED"
    expected_workers = manifest["args"]["rounds"] * len(manifest["args"]["variants"])
    steady_count = sum(t["phase"] == "steady" for t in trials)
    execution_ok = (not failures and len(worker_reports) == expected_workers and steady_count == expected_workers * len(cases)
        and all(report["status"] == "PASS_EXECUTION_ONLY" for report in worker_reports)
        and all(t["status"] == "PASS" for t in trials))
    status = "FAIL" if not execution_ok else "REVIEW_REQUIRED" if differences else "PASS" if performance_status == "PASS" else "NOT_RUN" if performance_status == "NOT_EVALUATED" else "REVIEW_REQUIRED"
    return dict(status=status, execution_status="PASS" if execution_ok else "FAIL", performance_status=performance_status,
        trajectory_status="REVIEW_REQUIRED" if differences else "PASS_OBSERVED_GENERATION_ONLY",
        thresholds=THRESHOLDS, aggregate=groups, comparisons_to_baseline=comparisons,
        comparisons_to_previous_variant=adjacent,
        path_coverage=coverage,
        workload_inputs=[dict(case=case["id"], requests=len(case["jobs"]), unique_inputs=case["unique_prompt_count"],
            measured_repetitions=manifest["args"]["rounds"], warmup_repetitions_per_round=1+manifest["args"]["warmups"],
            prefix_cache_enabled=False) for case in cases],
        greedy_trajectory_differences=differences, worker_reports=worker_reports, failures=failures,
        expected_workers=expected_workers, observed_workers=len(worker_reports), complete_release_matrix=complete_matrix,
        steady_trials=steady_count, cold_and_warmup_trials=sum(t["phase"] != "steady" for t in trials),
        quantile_note="Nearest-rank quantiles include n. P99 with fewer than100 requests is a sparse order statistic, not reliable tail estimation; no P99 release gate is claimed.",
        acceptance_note="Final graph combination must avoid >5% throughput loss and >10% p95 growth in every selected natural/repeat case and show >=5% throughput benefit somewhere. Other ablations retain regressions without hiding them. Quick/small matrices do not establish the full release gate.")


def terminate_child(process) -> None:
    """只回收本Popen子进程；继承上层suite进程组，让上层还能回收整个工作树。"""
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(args) -> int:
    directory = prepare(args)
    failures, process = [], None
    try:
        for round_id in range(args.rounds):
            order = args.variants[round_id % len(args.variants):] + args.variants[:round_id % len(args.variants)]
            for variant in order:
                from benchmarks.runtime_v2_gpu_guard import ensure_gpu1_available
                ensure_gpu1_available(evidence_path=directory / "gpu_observations.jsonl")
                command = [sys.executable, "-m", "benchmarks.runtime_v2_performance", "worker",
                           "--prepared", str(directory), "--variant", variant, "--round", str(round_id)]
                with (directory / f"round_{round_id:02d}_{variant}.log").open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    write_new(directory / f"round_{round_id:02d}_{variant}_process.json", dict(pid=process.pid, command=command, event="STARTED"))
                    try:
                        code = process.wait(timeout=args.worker_timeout)
                    except subprocess.TimeoutExpired:
                        terminate_child(process)
                        code = 124
                    if code != 0:
                        failures.append(dict(round=round_id, variant=variant, exit_code=code))
                        # CUDA失败后不开后续worker，现场保留；新进程复跑由使用者显式发起。
                        raise RuntimeError(f"worker failed: {variant}, round {round_id}, exit {code}")
                process = None
    except BaseException as error:
        failures.append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        if process is not None:
            terminate_child(process)
        try:
            report = collect(directory, failures)
        except Exception as error:
            report = dict(status="FAIL", execution_status="FAIL", performance_status="NOT_EVALUATED",
                aggregation_error=f"{type(error).__name__}: {error}", failures=failures,
                raw_evidence_note="Logs and per-trial files remain unchanged in this directory.")
        write_new(directory / "summary.json", report)
        print(json.dumps(dict(status=report["status"], output=str(directory), execution_status=report["execution_status"],
                              performance_status=report["performance_status"])))
    return 0 if report["status"] == "PASS" else 1 if report["status"] == "FAIL" else 2


def self_test() -> None:
    """CPU fake覆盖burst事件、排队、动态到达、失败保留和冻结门槛，不模拟GPU性能。"""
    from types import SimpleNamespace
    import unittest
    class Tests(unittest.TestCase):
        def test_variants(self):
            for name in VARIANTS:
                config, params = variant_config(name, {})
                self.assertFalse(config["enforce_eager"])
                self.assertEqual(config["enable_mixed_graph"], name == "graph")
            self.assertEqual(variant_config("fixedDV", {})[1]["speculative_max_draft_tokens"], 2)
            self.assertEqual(variant_config("adaptive_gate", {})[1]["speculative_max_draft_tokens"], 2)
            self.assertEqual(variant_config("adaptive_length", {})[1]["speculative_max_draft_tokens"], 8)
        def test_burst_is_not_expanded(self):
            record = dict(arrival_s=0, submit_s=.01, output_tokens=5,
                events=[dict(time_s=.1, cumulative_tokens=3), dict(time_s=.2, cumulative_tokens=5)])
            finalize_request(record)
            self.assertEqual(record["event_count"], 2)
            self.assertEqual(record["event_gap_ms"], [100])
            self.assertAlmostEqual(record["tpot_ms"], 25)
        def test_visible_text_is_not_ttft_or_zero_when_missing(self):
            record = dict(arrival_s=0,submit_s=0,output_tokens=2,
                events=[dict(time_s=.1,cumulative_tokens=1,text=""),dict(time_s=.2,cumulative_tokens=2,text="字")])
            finalize_request(record)
            self.assertEqual(record["ttft_ms"],100)
            self.assertEqual(record["first_visible_text_ms"],200)
            record["events"][1]["text"]=""
            finalize_request(record)
            self.assertIsNone(record["first_visible_text_ms"])
        def test_ratios_use_counter_deltas(self):
            before=dict(speculative=dict(proposed_tokens=100,accepted_tokens=90),decode_graph=dict(decode_replays=10,decode_fallback_eager=5))
            after=dict(speculative=dict(proposed_tokens=110,accepted_tokens=95),decode_graph=dict(decode_replays=13,decode_fallback_eager=6))
            result=metric_deltas(before,after)
            self.assertEqual(result["speculative"]["acceptance_rate"],.5)
            self.assertEqual(result["decode_graph"]["replay_fraction"],.75)
            self.assertIsNone(metric_deltas({}, {})["packed_graph"]["replay_fraction"])
        def test_sample_and_regression_gates(self):
            def group(rate, n=40, p95=1):
                return dict(rounds=5, throughput=dict(median=rate), ttft_ms=dict(n=n,p95=p95),
                            tpot_ms=dict(n=n,p95=p95), event_gap_ms=dict(n=n*30,p95=p95))
            self.assertEqual(compare_group(group(106), group(100))["status"], "PASS")
            self.assertEqual(compare_group(group(94), group(100))["status"], "REGRESSION")
            self.assertEqual(compare_group(group(106,p95=1.11), group(100))["status"], "REGRESSION")
            self.assertEqual(compare_group(group(106,n=4), group(100))["status"], "NOT_EVALUATED_INSUFFICIENT_SAMPLES")
        def test_quick_or_partial_matrix_is_not_full_acceptance(self):
            settings = dict(quick=False,input_split="validation",variants=list(VARIANTS),datasets=["english","chinese","code","repeat"],
                            scenarios=["short","long","dynamic"],concurrencies=[1,4,8,16])
            self.assertTrue(is_full_matrix(settings))
            self.assertFalse(is_full_matrix(dict(settings,quick=True)))
            self.assertFalse(is_full_matrix(dict(settings,input_split="calibration")))
            self.assertFalse(is_full_matrix(dict(settings,concurrencies=[1,4])))
            holdout = dict(settings, input_split="final_holdout",
                           variants=["baseline", "mixed"], rounds=5,
                           requests_per_case=32)
            self.assertTrue(is_full_matrix(holdout))
            self.assertFalse(is_full_matrix(dict(holdout, rounds=3)))
            self.assertFalse(is_full_matrix(dict(holdout, requests_per_case=16)))
            self.assertFalse(is_full_matrix(dict(holdout, input_split="calibration")))
        def test_output_is_immutable(self):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"result.json"
                write_new(path,dict(status="FAIL"))
                with self.assertRaises(FileExistsError):
                    write_new(path,dict(status="PASS"))
                self.assertEqual(load(path)["status"],"FAIL")
        def test_fake_trial_and_failure(self):
            for should_fail in (False, "injected_error", "early_terminal"):
                class Clock:
                    value = 0.0
                    def now(self):
                        self.value += .001
                        return self.value
                    def sleep(self, value):
                        self.value += value
                clock = Clock()
                class Engine:
                    def __init__(self):
                        self.requests, self.events, self.next_id = {}, [], 0
                        self.scheduler = SimpleNamespace(block_manager=SimpleNamespace(blocks=[], used_block_ids=set()))
                    def runtime_metrics(self): return dict(fake_cpu_only=True)
                    def add_request(self, ids, params):
                        seq = SimpleNamespace(seq_id=self.next_id, completion_token_ids=[], num_completion_tokens=0)
                        self.next_id += 1
                        self.requests[seq.seq_id] = seq
                        return seq
                    def step(self):
                        if should_fail == "injected_error": raise RuntimeError("injected failure")
                        self.events = []
                        for seq in self.requests.values():
                            seq.completion_token_ids += [1, 2]
                            seq.num_completion_tokens = len(seq.completion_token_ids)
                            self.events.append(SimpleNamespace(request_id=seq.seq_id, token_index=seq.num_completion_tokens-1,
                                token_id=2, text="ab", finished=should_fail == "early_terminal" or seq.num_completion_tokens == 4,
                                finish_reason="length"))
                    def step_events(self): return self.events
                    def forget_request(self, rid): self.requests.pop(rid, None)
                    def cancel_request(self, rid): pass
                device = SimpleNamespace(synchronize=lambda:None, reset_peak_memory_stats=lambda:None)
                case = dict(id="fake",dataset="fake",scenario="dynamic",concurrency=1,
                    jobs=[dict(job=i,arrival_s=i*.02,prompt_token_ids=[1],max_tokens=4) for i in range(2)])
                engine = Engine()
                result = run_trial(engine, case, {}, 5, clock=clock.now,sleep=clock.sleep,params_factory=lambda **kw:kw,device=device)
                self.assertEqual(result["status"], "FAIL" if should_fail else "PASS")
                self.assertFalse(engine.requests)
                if not should_fail:
                    self.assertEqual(result["output_tokens"],8)
                    self.assertEqual([r["event_count"] for r in result["requests"]],[2,2])
                    self.assertEqual(result["missing_requests"],0)
                else:
                    self.assertEqual(result["missing_requests"],1)
                    self.assertEqual(result["failures"],2)
        def test_distinct_natural_prompts_and_repeated_synthetic_are_explicit(self):
            class Tokenizer:
                def encode(self,text,**kwargs): return list(text.encode("utf-8"))
                def decode(self,values,**kwargs): return bytes(values).decode("utf-8",errors="replace")
            args = SimpleNamespace(datasets=["english","chinese","code","repeat"],scenarios=["short"],
                concurrencies=[1,4], requests_per_case=32, short_tokens=384,long_tokens=768,output_tokens=8,arrival_interval=.015)
            cases = build_workloads(Tokenizer(),args)
            for case in cases:
                self.assertEqual(case["unique_prompt_count"],1 if case["dataset"] == "repeat" else 32)
                self.assertTrue(all("source_text" in job and "prompt_text" in job for job in case["jobs"]))
            self.assertEqual(verify_input_split(Tokenizer(), args, cases)["overlap_count"], 0)
            for split in ("calibration", "validation", "final_holdout"):
                args.input_split = split
                split_cases = build_workloads(Tokenizer(), args)
                report = verify_input_split(Tokenizer(), args, split_cases)
                self.assertEqual(set(report["other_splits"]),
                                 {"calibration", "validation", "final_holdout"} - {split})
                self.assertEqual(report["overlap_count"], 0)
            args.short_tokens=8
            with self.assertRaisesRegex(ValueError,"repeated prompts"):
                build_workloads(Tokenizer(),args)
        def test_staggered_arrival_forces_long_prefill_into_active_decode(self):
            class Tokenizer:
                def encode(self,text,**kwargs): return list(text.encode("utf-8"))
                def decode(self,values,**kwargs): return bytes(values).decode("utf-8",errors="replace")
            args = SimpleNamespace(datasets=["repeat"],scenarios=["staggered"],
                concurrencies=[4],requests_per_case=8,short_tokens=8,long_tokens=16,
                output_tokens=4,arrival_interval=.015,input_split="calibration")
            case = build_workloads(Tokenizer(),args)[0]
            jobs = case["jobs"]
            self.assertEqual([job["job"] for job in jobs],list(range(8)))
            self.assertEqual([len(job["prompt_token_ids"]) for job in jobs[:3]],[8,8,16])
            self.assertEqual([job["arrival_s"] for job in jobs[:4]],[0.0,0.0,.05,.065])
        def test_path_names_alone_cannot_pass(self):
            trial = dict(phase="steady",variant="graph",observed_paths={},runtime_before={},runtime_after={})
            report = path_coverage([trial])["graph"]
            self.assertEqual(report["status"],"REVIEW_REQUIRED")
            self.assertIn("no_mixed_graph_replay",report["missing"])
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
                         scope="CPU mock only; no tokenizer or GPU throughput measurement")))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run", aliases=["prepare"])
    command.add_argument("--model", required=True)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--quick", action="store_true")
    command.add_argument("--input-split", choices=("calibration", "validation", "final_holdout"), default=None)
    command.add_argument("--variants", default=",".join(VARIANTS))
    command.add_argument("--datasets", default=None)
    command.add_argument("--scenarios", default=None)
    command.add_argument("--concurrencies", default=None)
    command.add_argument("--rounds", type=int, default=None)
    command.add_argument("--requests-per-case", type=int, default=None)
    command.add_argument("--short-tokens", type=int, default=None)
    command.add_argument("--long-tokens", type=int, default=None)
    command.add_argument("--output-tokens", type=int, default=None)
    command.add_argument("--warmups", type=int, default=1)
    command.add_argument("--token-budget", type=int, default=512)
    command.add_argument("--chunk-size", type=int, default=128)
    command.add_argument("--arrival-interval", type=float, default=.015)
    command.add_argument("--gpu-memory-utilization", type=float, default=.4)
    command.add_argument("--mixed-attention-backend", choices=("split","varlen"), default="split")
    command.add_argument("--mixed-min-active-decodes", type=int, default=0)
    command.add_argument("--mixed-low-load-mode", choices=("separate", "micro", "legacy_fifo"), default="separate")
    command.add_argument("--trace-steps", action="store_true",
                         help="diagnostic per-step timing/plan trace; never use for release timing")
    command.add_argument("--trial-timeout", type=float, default=120)
    command.add_argument("--worker-timeout", type=float, default=7200)
    command = sub.add_parser("worker")
    command.add_argument("--prepared", required=True, type=Path)
    command.add_argument("--variant", choices=VARIANTS, required=True)
    command.add_argument("--round", type=int, required=True)
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return
    if args.command == "worker":
        worker(args.prepared.resolve(), args.variant, args.round)
        return
    args.variants = args.variants.split(",")
    args.input_split = args.input_split or ("calibration" if args.quick else "validation")
    args.datasets = (args.datasets or ("english,repeat" if args.quick else "english,chinese,code,repeat")).split(",")
    args.scenarios = (args.scenarios or ("short,dynamic" if args.quick else "short,long,dynamic")).split(",")
    args.concurrencies = [int(value) for value in (args.concurrencies or ("1,4" if args.quick else "1,4,8,16")).split(",")]
    defaults = dict(rounds=1 if args.quick else 5, requests_per_case=8 if args.quick else 32,
                    short_tokens=48 if args.quick else 96, long_tokens=96 if args.quick else 768,
                    output_tokens=8 if args.quick else 32)
    for key, value in defaults.items():
        if getattr(args, key) is None: setattr(args, key, value)
    if (not args.variants or not set(args.variants) <= set(VARIANTS) or len(set(args.variants)) != len(args.variants)
            or not set(args.datasets) <= {"english","chinese","code","repeat"}
            or not set(args.scenarios) <= {"short","long","dynamic","staggered"}
            or any(len(values) != len(set(values)) for values in (args.datasets,args.scenarios,args.concurrencies))
            or min(args.concurrencies + [args.rounds,args.requests_per_case,args.short_tokens,args.long_tokens]) <= 0
            or args.output_tokens < 2 or args.warmups < 0 or not 0 < args.chunk_size <= args.token_budget
            or not math.isfinite(args.arrival_interval) or args.arrival_interval < 0
            or not 0 < args.gpu_memory_utilization <= 1
            or args.mixed_min_active_decodes < 0
            or not all(math.isfinite(value) and value > 0 for value in (args.trial_timeout,args.worker_timeout))):
        parser.error("invalid matrix/shape/timeout controls")
    if args.command == "prepare":
        try:
            directory = prepare(args)
        except Exception as error:
            # 参数/语料检查失败也是证据；已有目录绝不修改。
            if not args.output.exists():
                import traceback
                args.output.mkdir(parents=True)
                write_new(args.output / "prepare_failure.json", dict(status="FAIL", gpu_status="NOT_RUN",
                    error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                    arguments={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
                    provenance=provenance()))
            raise
        cases = load(directory / "workloads.json")
        print(json.dumps(dict(status="PREPARED", performance_status="NOT_RUN",
            output=str(directory), cases=len(cases),
            requests=sum(len(case["jobs"]) for case in cases),
            scope="CPU tokenizer and immutable input/config manifest only")))
        return
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
