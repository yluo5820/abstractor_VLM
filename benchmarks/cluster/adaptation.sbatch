#!/usr/bin/env bash
set -euo pipefail
campaign=$1
model=$2
task=$3
mode=$4
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
export PYTHONPATH="$campaign/runtime:$campaign/source"
export HF_HOME=/storage/ice1/9/8/yluo432/vlm-cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python=/storage/ice1/9/8/yluo432/envs/vlm-bench/bin/python
root=/storage/ice1/9/8/yluo432/vlm-benchmark/data/clevr-train-seed1
if [ "$task" = pgm ]; then root=/storage/ice1/9/8/yluo432/vlm-benchmark/data/pgm-adaptation-2026-10-07; fi
output="$campaign/$model/$task"
if [ "$mode" = pilot ]; then
 exec "$python" -u -m vlm_bench.adaptation --model "$model" --task "$task" --data-root "$root" --output "$output/pilot" --pilot --seconds 1000
elif [ "$mode" = train ]; then
 exec "$python" -u -m vlm_bench.adaptation --model "$model" --task "$task" --data-root "$root" --output "$output/train" --pilot-result "$output/pilot/pilot-result.json" --seconds 6900
fi
