#!/usr/bin/env bash
set -euo pipefail
campaign=$1
model=$2
task=$3
manifest=$4
digest=$5
count=$6
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
export PYTHONPATH="$campaign/runtime:$campaign/source"
export HF_HOME=/storage/ice1/9/8/yluo432/vlm-cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /storage/ice1/9/8/yluo432/envs/vlm-bench/bin/python -u "$campaign/source/cluster/evaluate_adaptation.py" \
 --campaign "$campaign/$model/$task" --model "$model" --task "$task" \
 --manifest "$manifest" --manifest-sha256 "$digest" --count "$count" --output "$campaign/$model/$task/test"
