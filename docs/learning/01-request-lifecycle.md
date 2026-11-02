# 第1课：沿一次请求的真实调用顺序读代码

这份逐行讲义对应[AIInfraGuide对照课程3.2](AIINFRAGUIDE_COMPANION.md#s32)，在线部分补充到[3.7](AIINFRAGUIDE_COMPANION.md#s37)。建议先看[3.1架构图](AIINFRAGUIDE_COMPANION.md#s31)，确认Engine、Scheduler和ModelRunner分别负责什么，再顺着下面的实际调用阅读。

本课沿当前主工作树的普通服务路径阅读。旧冻结 candidate 及验收数字保留在历史交付中；当前源码已加入默认关闭的执行优化，不能再视为与旧冻结版本逐字节一致。选择普通服务：Qwen3-1.7B、TP1、`batching_mode="separate"`、draft关闭、CPU KV缓存关闭。这里走 `Scheduler` / `_step_legacy`。mixed路径的 `PlannedScheduler`、`reserve_to/commit/rollback` 留到后面，不混进今天的调用链。

所有位置从0开始。例子中的token编号和物理页号是示意，不代表某句话的实际分词结果。

## 先记住三个对象

|对象|作用|
|---|---|
|`RequestRecord`|HTTP侧：公开ID、完成信号、事件队列、结果和保留时间|
|`Sequence`|推理侧：token IDs、waiting/running/finished状态、KV有效长度、页表、采样参数|
|`BlockManager`|分配/归还KV物理页使用权，管理页号、引用计数和缓存索引|

权重是模型固定参数。KV是每层Attention为已处理token算出的K/V激活；它们是不同的东西。

## 0．请求到来之前，模型与KV大显存已经准备好

入口：`ModelRunner.allocate_kv_cache`，在 [model_runner.py:254](../../nanovllm/engine/model_runner.py#L254)。

主要张量形状：

```text
kv_cache[2, 层数, 物理页数, block_size, KV头数, head_dim]
         K/V
```

每层Attention拿到自己的K/V view。正式配置block_size=256。请求来了以后，`BlockManager.allocate`主要分配页号，通常不再为每条请求重新`torch.empty`一整块KV显存。

## 1．HTTP线程解析请求并提交

从 [server.py:465](../../nanovllm/server.py#L465) 的 `_Handler._handle_openai` 开始。

```text
POST /v1/completions 或 /v1/chat/completions
    → 解析prompt/messages和SamplingParams
    → RequestManager.submit
```

chat输入会先通过tokenizer的chat template变成prompt。普通completions直接接收字符串或token ID列表。

`RequestManager.submit` 在 [server.py:108](../../nanovllm/server.py#L108)：检查服务健康、未完成请求容量，调用 `engine.add_request`，再建立HTTP侧的 `RequestRecord`。流式请求额外有有界 `events` 队列。

HTTP公开ID可以是`cmpl-...`；引擎内部用整数`seq_id`关联请求，不要把两种ID混为一谈。

## 2．变成Sequence并进入waiting

[llm_engine.py:215](../../nanovllm/engine/llm_engine.py#L215)：

```python
prompt = self.tokenizer.encode(prompt)  # 输入是字符串时
seq = Sequence(prompt, sampling_params)
self.scheduler.add(seq)
self._requests[seq.seq_id] = seq
```

入口会先检查token是否合法、总长度和物理KV容量。容量上限使用：

```python
required_tokens = len(prompt) + max_tokens - 1
```

最后一个输出不会再被送入模型，因此不需要为它建立KV。先理解这个`-1`，后面的状态变化会清楚很多。

新 `Sequence` 在 [sequence.py:35](../../nanovllm/engine/sequence.py#L35) 初始化：

```text
status = WAITING
token_ids = prompt的拷贝
num_cached_tokens = 0
num_scheduled_tokens = 0
block_table = []
```

[Scheduler.add:60](../../nanovllm/engine/scheduler.py#L60) 只是把对象放进 `waiting`。此时还没有开始这条请求的GPU forward，也没有给它分配KV页。

## 3．后台worker推进引擎

[RequestManager._run:248](../../nanovllm/server.py#L248) 循环执行：

```python
with self.lock:
    self._expire()
    if not self.engine.is_finished():
        self.engine.step()
        for event in self.engine.step_events():
            self._complete_event(event)
```

HTTP线程负责接入/读取事件，一个后台worker统一推进引擎。多个HTTP请求可以被scheduler合成batch；它们并非各自开一个线程同时调用同一个模型。

`submit/cancel/step`由同一把锁保护。TP1时ModelRunner就在本进程；这里的worker首先是线程概念。

## 4．本轮调度：选谁、处理多少token、用哪些KV页

[LLMEngine.step:358](../../nanovllm/engine/llm_engine.py#L358) 根据配置选择路径。今天进入 `_step_legacy`：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

实际还包围了统计、stop处理和事件生成，完整函数从 [llm_engine.py:369](../../nanovllm/engine/llm_engine.py#L369) 阅读。

[Scheduler.schedule:92](../../nanovllm/engine/scheduler.py#L92) 对新请求先尝试prefill：

1. 看waiting队首、batch容量和本轮token budget。
2. 探测前缀命中；本课无命中，所以cached blocks=0。
3. `BlockManager.allocate`为prompt所需页建立`seq.block_table`。
4. 设置`num_scheduled_tokens`：本轮真正处理多少prompt token。

普通路径第一次会按prompt所需页建立页表，chunk大小决定本轮写入多少KV，不等于每次只分配一个chunk的页。

正式普通配置prefill chunk为128。长prompt会分多轮；中间chunk不提交completion。若本轮将处理完整prompt，scheduler会在调用GPU之前把请求标成RUNNING并放进running——这是预先安排下一阶段，KV此刻尚未完成写入。

separate路径每轮选prefill或decode batch；`max_prefill_steps`用于避免持续prefill把decode饿住。

## 5．ModelRunner准备输入，Attention实际写KV

[ModelRunner.run:534](../../nanovllm/engine/model_runner.py#L534)：

```python
input_ids, positions = prepare_prefill(...) 或 prepare_decode(...)
logits = self.run_model(input_ids, positions, is_prefill)
return self.sampler(logits, seqs).tolist()
```

`prepare_prefill`取区间 `[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)`，构造位置及slot映射，然后把本轮输入送到GPU。多个请求的prompt段可以拼成一批。

继续往模型里跟，实际顺序是：

```text
ModelRunner.run_model
  → Qwen3ForCausalLM.forward
    → Qwen3Model.forward
      → embedding：token ID变成hidden向量
      → 逐层Qwen3DecoderLayer.forward
        → RMSNorm
        → Qwen3Attention.forward
          → qkv_proj：一次投影得到Q/K/V
          → Q/K归一化与RoPE
          → Attention.forward：写KV，再计算Attention
          → o_proj
        → residual/RMSNorm与MLP
      → 最后一次residual/RMSNorm
  → compute_logits：hidden投影到词表
  → sampler：从logits选择下一个token ID
```

对应入口：[模型主体](../../nanovllm/models/qwen3.py#L256)、[每层](../../nanovllm/models/qwen3.py#L218)、[QKV与RoPE](../../nanovllm/models/qwen3.py#L97)。先沿函数调用读，暂时不展开每个算子的CUDA实现。Graph路径会重放已经捕获的模型计算，不会每轮重新执行以上所有Python层调用。

假设逻辑第0页分到了物理页7，block_size=256，则该请求前几个位置映射为：

```text
逻辑token位置     0      1      2      3      4
物理KV slot    1792   1793   1794   1795   1796
                  7 × 256 + 页内偏移
```

`block_table`在CPU上记录页号；ModelRunner把页表和slot映射做成GPU元数据。

每层模型算出当前输入的Q/K/V后，在 [Attention.forward:133](../../nanovllm/layers/attention.py#L133) 先 `store_kvcache`，再做Attention。真正的写入在 [attention.py:20](../../nanovllm/layers/attention.py#L20)：

```python
slot = slot_mapping[idx]
cache_offsets = slot * D + channel_offset
store(K_cache[cache_offsets], key)
store(V_cache[cache_offsets], value)
```

这是解释用的伪代码；实际为Triton kernel。D是这一层一个token的全部KV heads展开宽度，一个slot会存很多浮点数。

`BlockManager.allocate`负责使用权/页表，Attention负责填入真实K/V数值。

## 6．完整prefill产生第一个输出token

prompt假设有5个token：`[p0,p1,p2,p3,p4]`。模型处理它们后，LM Head选最后位置的hidden state预测`t1`，不是给用户输出5个completion。

代码在 [embed_head.py:73](../../nanovllm/layers/embed_head.py#L73)：prefill取每条序列最后一个query位置。greedy场景采样选出一个token ID。

[Scheduler.postprocess:210](../../nanovllm/engine/scheduler.py#L210) 随后更新：

```python
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
seq.append_token(token_id)
```

这时：

```text
token_ids = [p0,p1,p2,p3,p4,t1]
总长度 = 6
有效KV长度 = 5
```

**t1只是刚预测出的ID，尚未经过模型，当然没有它的K/V。**

## 7．decode输入上一个新token，预测下一个

下一轮从running选请求。`can_append/may_append`保证当前输入位置有物理页，跨页时按需增页。

[prepare_decode:430](../../nanovllm/engine/model_runner.py#L430) 只准备：

```text
input_ids = [t1]
positions = [5]
context_lens = [6]
slot_mapping = [1797]   # 物理页7、页内偏移5
```

模型计算t1的K/V并写入slot1797，用当前Q读取历史KV（包含刚写入的t1），再预测t2。所以context_len可以是6，而进入本轮前cached是5。

decode的 [run_model:485](../../nanovllm/engine/model_runner.py#L485) 可以重放CUDA Graph得到hidden states，再算logits与采样。CPU调度和HTTP并没有被放进同一张Graph。

KV缓存省去了过去token的重复forward；当前token的Attention仍需要读取历史KV。

## 8．返回事件、结束与清理

`postprocess`检查EOS和max_tokens；engine还处理stop字符串。完成时先归还KV页使用权并从running移除。

engine的 `_make_step_events` 生成 `GenerationEvent`，RequestManager把它放进该请求的事件队列；HTTP线程在 [server.py:573](../../nanovllm/server.py#L573) 消费队列、写SSE，最后发`[DONE]`。非流式等待record.done后一次返回JSON。

token ID经tokenizer转成文本增量；一个token不等于一个汉字，一个事件也未必有可见文本。

[BlockManager.deallocate:531](../../nanovllm/engine/block_manager.py#L531) 减引用计数、归还页，清空该Sequence页表及cached计数。KV大tensor仍属于常驻引擎，不需每次cudaFree或清零。

随后RequestManager `_finish` 保存结果并 `engine.forget_request`；HTTP完成记录按TTL有界保留。因此“归还KV”“删除引擎请求”“删除HTTP结果”有不同时间点。

客户端断开、取消或超时会收口到 `_terminate → engine.cancel_request → scheduler.cancel → deallocate`。取消在调度step之间安全处理，不代表强行打断正在执行的CUDA kernel。

## 用一张表串起来

假设prompt5token、max_tokens=3、无EOS/stop、无前缀命中，分到物理页7：

|时刻|总token数|输出数|有效KV数|页表|
|---|---:|---:|---:|---|
|刚入waiting|5|0|0|[]|
|完整prefill并提交t1|6|1|5|[7]|
|输入t1，提交t2|7|2|6|[7]|
|输入t2，提交t3并结束|8|3|0（释放前7）|[]|

三次模型调用处理的query token分别是5、1、1；最终t3没有再输入模型。总token数、有效KV数、query token数与模型调用次数是四个不同量。

## 独立练习

1. **纸上状态追踪。** prompt7token、max_tokens=2、prefill_chunk_size=3，无前缀/EOS/stop。逐轮写出prefill/decode、scheduled、总长度、cached、输出数和状态。哪轮第一次能向客户端发token？
2. **纯CPU地址映射。** 完成`exercises/01_cpu_exercises.py` 的physical_slot。B=256，block_table=[7,2]，求位置0/255/256/257的slot。
3. **容量边界。** 完成minimum_kv_blocks。教学B=4：7个prompt＋2个输出为什么只需2页？最后token存在于token_ids却没有KV，是否矛盾？不要修改正式配置的256。
4. **给学习副本加日志。** 在schedule与postprocess前后打印seq_id、status、len、cached、scheduled、block_table，运行一个max_tokens=4的小请求。预期看到进入decode时通常`cached=len-1`。带日志运行仅供理解，不作为性能数据；GPU先通过空闲卡guard。
5. **取消推演。** 分别在waiting和第一次输出之后取消：哪些队列、页表、引用、引擎请求和HTTP结果会变化？对照 `Scheduler.cancel`、`RequestManager._terminate/_finish`，再看已有CPU测试：`tests.regression.check_service_hardening`、`tests.regression.check_tokenizer_stream`。

建议先做1～3，再读 [核对答案](exercises/01_checks.md)。本课没有要求你重新实现模型或Attention kernel。
