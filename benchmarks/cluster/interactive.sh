#!/usr/bin/env bash
set -euo pipefail
cluster_dir=$(cd "$(dirname "$0")" && pwd)
source "$cluster_dir/guard.sh"
require_gpu_allocation
scontrol show job "$SLURM_JOB_ID"
project=$1
environment=$2
cache=$3
cd "$project"
export HF_HOME="$cache/huggingface"
export PIP_CACHE_DIR="$cache/pip"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
if [ -f "$environment/bin/activate" ]; then source "$environment/bin/activate"; fi
printf '\nAllocation ready. Exit this shell when finished to release this interactive allocation.\n'
# The outer srun shell already initialized modules. A second login shell can
# replace HF_HOME and activate the user's default conda environment again.
exec bash --noprofile --norc -i
