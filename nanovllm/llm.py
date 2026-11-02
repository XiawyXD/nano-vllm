"""对外暴露的 LLM 接口。

`LLM` 暂时没有新增逻辑，只是给 `LLMEngine` 提供更简洁的公共名称。
"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    # 继承引擎的初始化、请求调度和 generate 方法。
    pass
