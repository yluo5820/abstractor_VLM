#!/usr/bin/env bash
set -euo pipefail
campaign=$1
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
export PYTHONPATH="$campaign/runtime:$campaign/source"
export HF_HOME=/storage/ice1/9/8/yluo432/vlm-cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /storage/ice1/9/8/yluo432/envs/vlm-bench/bin/python -u "$campaign/evaluation-source/evaluate_lora.py" \
 --campaign "$campaign" \
 --manifest /storage/ice1/9/8/yluo432/vlm-benchmark/data/clevr-full-seed1/manifests-v2/clevr-art-idrules-test-full.jsonl \
 --output "$campaign/test-ood"
