"""经过验收后才能提升的运行时预设，不影响用户显式指定的参数。

开发中的组合应通过各独立开关运行消融。阶段六完成后，用冻结验证结果选择
OPTIMIZED_PROFILE 的内容；PENDING 状态绝不能暗示新路径已验证。
"""

BASELINE_PROFILE = {
    "batching_mode": "separate",
    "speculative_policy": "fixed",
    "metadata_backend": "dynamic",
    "enable_mixed_graph": False,
}
OPTIMIZED_PROFILE = dict(BASELINE_PROFILE)
OPTIMIZED_PROFILE_STATUS = "PENDING_VALIDATION"


def resolve_runtime_profile(name, overrides):
    """None 表示未显式设置；用户参数始终覆盖预设值。"""
    if name not in {"baseline", "optimized"}:
        raise ValueError("runtime_preset must be baseline or optimized")
    result = dict(BASELINE_PROFILE if name == "baseline" else OPTIMIZED_PROFILE)
    result.update({key: value for key, value in overrides.items() if value is not None})
    return result
