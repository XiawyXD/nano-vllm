"""记录复现所需的明确版本和文件哈希，不收集凭据或完整环境变量。"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def command(args):
    result = subprocess.run(args, text=True, capture_output=True)
    return {"args": args, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    packages = ["torch", "triton", "transformers", "flash-attn", "numpy", "xxhash",
                "safetensors", "huggingface-hub", "tokenizers", "regex", "tqdm", "setuptools"]
    versions = {name: metadata.version(name) for name in packages}
    root = Path(__file__).resolve().parents[1]
    files = list((root / "nanovllm").rglob("*.py")) + list((root / "benchmarks").glob("*.py"))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    model = Path(args.model)
    model_hashes = {name: hashlib.sha256((model/name).read_bytes()).hexdigest()
                    for name in ("config.json", "tokenizer_config.json", "generation_config.json")
                    if (model/name).exists()}
    import torch
    report = dict(python=sys.version, platform=platform.platform(), versions=versions,
                  torch_cuda=torch.version.cuda, torch_cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                  git_head=command(["git", "rev-parse", "HEAD"]),
                  git_status=command(["git", "status", "--short"]),
                  gpus=command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]),
                  nvcc=command(["nvcc", "--version"]), source_sha256=hashes,
                  model=str(model), model_metadata_sha256=model_hashes)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"python": platform.python_version(), "versions": versions,
                      "torch_cuda": report["torch_cuda"], "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
