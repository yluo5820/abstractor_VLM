#!/usr/bin/env bash
# One H200, two CPUs, 16 GiB; submit with --signal=B:TERM@300.
set -euo pipefail
project=$1
environment=$2
cache=$3
model=$4
revision=$5
manifest=$6
output=$7
task=$8
render_digest=$9
seconds=${10:-6600}
source "$project/cluster/guard.sh"
require_gpu_allocation
cd "$project"
export HF_HOME="$cache/huggingface"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
"$environment/bin/python" -m vlm_bench.clevr_prepare \
  --root "$(dirname "$(dirname "$manifest")")" --output "$(dirname "$manifest")" \
  --render-config-sha256 "$render_digest" --require-ready
exec "$environment/bin/python" cluster/compare_pgm.py \
  --source "$project" --model "$model" --revision "$revision" \
  --preflight-manifest "$manifest" --full-manifest "$manifest" --output "$output" \
  --segment-seconds "$seconds" --cpu-threads "${SLURM_CPUS_PER_TASK:-2}" \
  --full-count 10000 --expected-dataset clevr-art --expected-task "$task"
