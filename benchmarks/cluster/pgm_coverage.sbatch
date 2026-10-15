#!/usr/bin/env bash
set -euo pipefail
campaign=$1
model=$2
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
export PYTHONPATH="$campaign/runtime:$campaign/source"
export HF_HOME=/storage/ice1/9/8/yluo432/vlm-cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python=/storage/ice1/9/8/yluo432/envs/vlm-bench/bin/python
# Include imports, model loading and the correctness pilot in the four-hour budget.
deadline=$("$python" -c 'import time; print(time.time()+14220)')
exec "$python" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m vlm_bench.parallel_adaptation --model "$model" \
  --data-root /storage/ice1/9/8/yluo432/vlm-benchmark/data/pgm-coverage-20pct-2026-10-08 \
  --output "$campaign/$model/pgm/train" --deadline "$deadline"
