#!/usr/bin/env bash
# Submit with one H200, 2 CPUs, 16 GiB, <=10h and --signal=B:TERM@300.
# Identical resubmissions resume; pipeline-status.json distinguishes complete/resumable.
# This wrapper does not submit continuation jobs or change resource requests.
set -euo pipefail
source_root=$1
environment=$2
cache=$3
model=$4
revision=$5
preflight=$6
manifest=$7
output=$8
seconds=${9:-35400}
source "$source_root/cluster/guard.sh"
require_gpu_allocation
cd "$source_root"
test -x "$environment/bin/python" || { echo 'Verified Python environment is missing.' >&2; exit 1; }
export HF_HOME="$cache/huggingface"
export PIP_CACHE_DIR="$cache/pip"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
exec "$environment/bin/python" cluster/compare_pgm.py \
  --source "$source_root" --model "$model" --revision "$revision" \
  --preflight-manifest "$preflight" --full-manifest "$manifest" --output "$output" \
  --segment-seconds "$seconds" --cpu-threads "${SLURM_CPUS_PER_TASK:-2}"
