# Handoff：本轮优化与有界验收收尾

## 固定入口

- 共享项目：`/home/hrxu/nano-vllm`，只合入8个授权文件，HEAD/index与已有dirty工作保留；源码before/after在runtime_tools/shared_integration001。
- 冻结canonical source：`/home/hrxu/nano-full-opt-20261024/candidate`。327文件完整源码快照及SHA在reproduction/final_source_frozen001；本地阅读副本final-source。
- 最终决定：CONFIG_DECISIONS.json；最终元数据profiles：release_profiles_final.json。原release_profiles.json、protocol、performance/review/analyzer及模型不能改后继续引用旧结果。
- 五轮35worker已完成；服务1800.68秒五阶段PASS；GPU温度采样概率、真实8B随机Graph/eager、目标target002三进程Q1边界完成。检查各原始status，不把退出码0或局部通过替代完整数值验收。

## 已停止／仍有限制

1. 新全模型MSE003仅一个固定配方。DEV NLL/KL通过，但CN接受率下降、总verify增加，已STOP；没有新候选holdout或formal，不再扩展LFQ网格，不复活该候选。
2. 旧MSE dense/Marlin完整logits仍REVIEW_REQUIRED。fd002首bit差layer0 qkv，首严格超限layer3 k_norm，全模型max0.24609375；224同输入局部投影过门槛不证明全模型无误。
3. native相对consistent基线输出精确，但与普通Torch backend自由token不同；event-gap tail退步。只能保留明确数值后端的研究结果，不称原默认BF16输出无损加速。
4. AWQ target改变参考分布；pooled TTFT1.1016675、code1.1007387均高于1.10。仅本次重复文本负载通过对应质量、速度与尾延迟门槛，不能全局推广。
5. 缓存重复前缀容量吞吐1.303×，相对tail通过；绝对SLO30/180，unique无稳定收益。缓存和native draft禁止组合。本轮不重跑无关完整缓存矩阵，概率代码变更后的cache组件等价与当前生命周期复验见runtime_tools/source_equivalence_frozen002_001.json。
6. 随机推测仅TP1/C1、temperature-only、FP16 native draft；p/q producer FP64归一化，检查器未改；无端到端加速结论，不覆盖量化draft随机组合。
7. 目标Marlin边界检查仅同历史Q1 Graph/eager与跨进程，不宣称所有query-shape通过。probe v1的32/128错误假设及v2误借PlannedScheduler资源字段均为测试工具问题，旧文件/失败保留；v3按绑定36层144模块及typed Scheduler严格检查。

## 贡献与后续学习边界

本轮运行时代码8文件；Marlin和AutoRound复用成熟实现，先前KV/服务/draft功能保留并复验。详见FINAL_REPORT_ZH和学习路线。学习后如研究新优化，应另建源码/数据角色，重新冻结DEV选择和未使用的最终验证；不能用本轮最终结果调K、配方或阈值再称独立验证。

未来可以研究prefill kernel选择、流式draft目标、target决策边界保护等，但本轮没有实现或承诺这些结果。TP2/TP4 draft、W4A4、MTP、EAGLE/Medusa、MoE和多机不在交付支持范围，不新增宣传开关。

模型、reference logits和raw tensor payload仍保留在原路径，最终bundle仅含源码/配置/原始JSON/日志/文档，二进制payload通过SHA locators单独记录。未知upstream revision明确未知；不要用当前HF分支猜旧文件revision。最终归档成功以外部bundle receipt及哈希为准。
