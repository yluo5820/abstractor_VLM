#!/usr/bin/env bash
# External sbatch/srun flags choose the verified GPU and bounded allocation.
set -euo pipefail
source_root=$1
environment=$2
cache=$3
data_root=$4
output=$5
expected_gpu=$6
max_batch=${7:-8}
budget_seconds=${8:-720}
source "$source_root/cluster/guard.sh"
require_gpu_allocation
cd "$source_root"
export HF_HOME="$cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
set +e
"$environment/bin/python" -u -m vlm_bench.perf_audit \
  --pgm-manifest "$data_root/pgm-neutral-hf-pilot512.jsonl" \
  --clevr-manifest "$data_root/clevr-rmts-smoke.jsonl" \
  --output "$output" --require-gpu-name "$expected_gpu" \
  --dtype bfloat16 --threads 1 2 4 --max-batch "$max_batch" \
  --pgm-samples 32 --clevr-samples 16 \
  --time-budget-seconds "$budget_seconds"
audit_status=$?
set -e
if [ -f "$output" ]; then
  "$environment/bin/python" "$source_root/cluster/summarize_audit.py" "$output" "${output%.json}.md"
fi
exit "$audit_status"
