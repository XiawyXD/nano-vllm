"""仅运行评测工具的 CPU 自测；不接受任意子命令或 GPU 参数。

这是“测量工具自身”的检查，不是模型性能、数值或30分钟稳定性验收。
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    """运行固定CPU入口，保留各自输出；任一失败即返回非零退出码。"""
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args()
    checks = (
        ("performance", ["-m", "benchmarks.runtime_v2_performance", "self-test"]),
        ("profile", ["-m", "benchmarks.runtime_v2_profile", "self-test"]),
        ("soak", ["-m", "tests.regression.check_runtime_v2_soak"]),
        ("engine_utils", ["-m", "tests.regression.check_runtime_v2_engine_utils"]),
    )
    results = []
    for name, arguments in checks:
        result = subprocess.run([sys.executable, *arguments], cwd=Path(__file__).resolve().parents[2])
        results.append(dict(check=name, exit_code=result.returncode))
    passed = all(row["exit_code"] == 0 for row in results)
    print(json.dumps(dict(status="PASS" if passed else "FAIL", checks=results,
        scope="CPU harness self-tests only; no actual model/GPU inference")))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
