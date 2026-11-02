# 启动辅助

serve_validated.py 包装原最终交付的严格服务启动器；无参数仅显示配置，--execute 才运行 GPU。with_flashinfer.sh 是可选 FlashInfer 环境入口，先读脚本再使用。旧 serve_qwen8.sh 已归档，其旧 GPU/环境默认值不作为当前推荐。


`python scripts/check_repository.py`：仅标准库的源码大小、凭据特征与相对文档链接检查；`--history` 额外检查当前分支可达历史。GPU、模型和性能验收仍使用 tests/benchmarks 的独立入口。
