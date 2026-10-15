#!/usr/bin/env bash
# Resources and dependencies are explicit in the saved submission receipts.
set -euo pipefail
campaign=$1
mode=$2
project=/storage/ice1/9/8/yluo432/vlm-benchmark
environment=/storage/ice1/9/8/yluo432/envs/vlm-bench
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
cd "$campaign/source"
export PYTHONPATH="$campaign/runtime:$campaign/source"
export HF_HOME=/storage/ice1/9/8/yluo432/vlm-cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
if [ "$mode" = pilot ]; then
  "$environment/bin/python" -m pip install --no-index --no-deps --target "$campaign/runtime" "$campaign"/wheels/*.whl
  exec "$environment/bin/python" -u -m vlm_bench.finetune \
    --data-root "$project/data/clevr-train-seed1" --output "$campaign/pilot" --pilot --seconds 1000
elif [ "$mode" = train ]; then
  exec "$environment/bin/python" -u -m vlm_bench.finetune \
    --data-root "$project/data/clevr-train-seed1" --output "$campaign/train" \
    --pilot-result "$campaign/pilot/pilot-result.json" --seconds 6900
else
  printf 'Unknown mode: %s\n' "$mode" >&2
  exit 2
fi
