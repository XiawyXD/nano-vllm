# 简历项目版本（按实际参与范围使用）

**基于nano-vllm的GPU推理、推测解码与量化实验平台**  
技术栈：Python、PyTorch、CUDA Graph、Triton、FlashAttention、AWQ/Marlin、AutoRound、HTTP/SSE；RTX4090/SM89、dense Qwen3、TP1。

- 基于nano-vllm扩展并验证分页KV生命周期、GPU–CPU前缀缓存及独立draft路径；在336MiB GPU KV＋1GiB CPU的重复前缀容量压力负载中，五轮交错对照吞吐中位1.303×。保留无复用负对照与绝对SLO未达标结果。
- 为独立draft构建Graph执行和验证闭环；Qwen3-8B consistent target＋FP16 .6B draft、K4，在冻结短上下文C1负载中相对同数值后端普通decode达1.793×，最终token精确一致；同时报告流式事件间隔退步，不包装成全场景服务加速。
- 接入真实AWQ/Marlin W4A16执行与融合QKV/gate-up布局；8B量化target五轮稳态吞吐为BF16普通decode的2.302×，实际稳态峰值allocated从15.51降至5.95GiB。目标分布改变、TTFT回退及仅重复文本类别通过全部对应门槛均单独披露。
- 实现标准概率接受/拒绝校正并修复大词表FP32概率归一化边界问题；建立TRAIN/DEV/最终验证隔离、完整词表KL、接受率及执行成本验收，GPU每例100,000样本统计通过。当前版本30分钟TP1/C8 HTTP/SSE验收接纳16,341请求，其中4,085次预期取消，清理与重启通过。

面试需能解释：reserve/commit/rollback；Graph固定地址与host buffer寿命；draft减少target次数却增加自身forward；p/q必须对应实际proposal；NLL/KL改善不保证贪心接受率；Marlin是复用成熟kernel而不是自行实现；所有性能数字的模型、精度、负载和局限。

本轮新量化draft并未整体达标，不能写“成功解决量化draft加速”。异步metadata和旧MSE分别仅0.03%／0.43%计时差异，未达3%门槛。不要写TP4 draft、W4A4、MTP、MoE或多机已支持。

以上数字逐项链接 `CONFIG_DECISIONS.json`、`formal_analysis001.json`、`formal_execution_summary001.json` 和 `service_tools/frozen002/soak002/http-raw/summary.json`。学习并独立复现后，再按实际负责的代码、测试与分析调整个人贡献描述。
