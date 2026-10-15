#!/usr/bin/env bash
set -euo pipefail
campaign=$1
variant=$2
seconds=$3
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
cd "$campaign/source"
sha256sum --quiet -c "$campaign/eval-source-sha256.txt"
export PYTHONPATH="/storage/ice1/9/8/yluo432/vlm-benchmark/runs/pgm-coverage-eval-2026-10-09/runtime:$campaign/source"
export HF_HOME=/storage/ice1/9/8/yluo432/vlm-cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python=/storage/ice1/9/8/yluo432/envs/vlm-bench/bin/python
"$python" "$campaign/source/cluster/evaluate_relational_frozen.py" --campaign "$campaign/$variant" --mode pin
exec "$python" -u "$campaign/source/cluster/evaluate_relational_frozen.py" \
  --campaign "$campaign/$variant" --seconds "$((seconds-240))"
