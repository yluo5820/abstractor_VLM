#!/usr/bin/env bash
# Resource directives are supplied by ice.py from the verified profile.
set -euo pipefail
project=$1
environment=$2
cache=$3
manifest=$4
output=$5
model=$6
revision=$7
max_samples=$8
source "$project/cluster/guard.sh"
require_gpu_allocation
cd "$project"
test -x "$environment/bin/python" || { printf 'Verified Python environment is missing.\n' >&2; exit 1; }
export HF_HOME="$cache/huggingface"
export PIP_CACHE_DIR="$cache/pip"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
args=(--manifest "$manifest" --output "$output" --model "$model" --revision "$revision")
if [ "$max_samples" = all ]; then max_samples=0; fi
args+=(--max-samples "$max_samples")
args+=(--cpu-threads "${SLURM_CPUS_PER_TASK:-2}" --interop-threads 1)
exec "$environment/bin/python" -m vlm_bench.run "${args[@]}"
