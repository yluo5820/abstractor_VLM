#!/usr/bin/env bash
set -euo pipefail
campaign=$1
variant=$2
mode=$3
seconds=$4
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
cd "$campaign/source"
sha256sum --quiet -c "$campaign/source-sha256.txt"
export PYTHONPATH="/storage/ice1/9/8/yluo432/vlm-benchmark/runs/pgm-coverage-eval-2026-10-09/runtime:$campaign/source"
export HF_HOME=/storage/ice1/9/8/yluo432/vlm-cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python=/storage/ice1/9/8/yluo432/envs/vlm-bench/bin/python
deadline=$("$python" -c 'import time,sys; print(time.time()+int(sys.argv[1])-120)' "$seconds")
exec "$python" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m vlm_bench.relational_frozen_train --variant "$variant" --mode "$mode" \
  --data-root /storage/ice1/9/8/yluo432/vlm-benchmark/data/pgm-coverage-20pct-2026-10-08 \
  --output "$campaign/$variant" --deadline "$deadline"
