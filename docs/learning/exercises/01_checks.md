# 第1课练习核对

先独立做，再看这里。本文件不是对正式引擎做新的GPU验收。

## 练习1

单请求、7个prompt、最多2个输出、chunk=3，无EOS/stop/prefix：

|step|本轮|输入query数|结束后总长度|结束后cached|输出数|
|---|---|---:|---:|---:|---:|
|1|prefill|3|7|3|0|
|2|prefill|3|7|6|0|
|3|prefill|1|8|7|1|
|4|decode|1|9|0（释放前8）|2|

前两轮仍在waiting；第3轮scheduler知道会处理完prompt，会在调用GPU前移入running。第4轮完成后FINISHED，页表被清空。

部分prefill虽然计算了模型，但postprocess不向completion追加token。第3轮才有第一个completion事件。

## 练习2

```text
逻辑页号 = position // B
页内偏移 = position % B
物理slot = block_table[逻辑页号] * B + 页内偏移
```

对应结果1792、2047、512、513。逻辑位置连续，物理地址可以跳跃；页表负责连接它们。

## 练习3

最多需要进入模型的token数 = prompt_tokens + max_new_tokens - 1。

再对block_size向上取整。P=7/T=2/B=4，KV最多8个位置，2页；最后生成的第9个序列token不再forward，不需要第3页。

## 练习4、5的核对标准

- 普通decode开始前，末尾刚预测token尚无KV，通常cached=len-1。
- 完成后cached归零与页表清空是归还使用权，不是KV从未计算。
- waiting阶段尚未调度的请求通常无页可归还；prefill后的取消需归还已持有页。
- cancel先从waiting/running移除并归还页，标记CANCELLED；HTTP终止收口再forget引擎请求，结果记录可按TTL保留。
- 常驻服务的模型权重和KV池仍在GPU；不能用nvidia-smi是否恢复零显存判断单请求是否清理。
