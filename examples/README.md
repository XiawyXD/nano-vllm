# 使用示例

- example.py：离线 LLM.generate 用法，先修改本地模型路径。
- chat_nano.py：已启动 OpenAI HTTP 服务的聊天客户端；用 --help 查看参数。

通用服务入口在根目录 serve_nano_vllm.py，严格已验收启动入口在 scripts/serve_validated.py。微基准在 benchmarks/micro/，不属于最小使用流程。
