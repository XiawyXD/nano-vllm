#!/usr/bin/env bash
# 本机可复现入口：不修改exp，只给当前子进程提供CUDA12.8/ninja与独立依赖。
# 例：bash scripts/with_flashinfer.sh -m benchmarks run flashinfer-sampling check --output artifacts/NEW_check
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/home/hrxu/cuda128-toolkit}"
export PATH="$project_dir/.venv-flashinfer/bin:$CUDA_HOME/bin:/home/hrxu/miniconda3/envs/exp/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
cd "$project_dir"
exec "$project_dir/.venv-flashinfer/bin/python" "$@"
