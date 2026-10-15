#!/usr/bin/env bash
# Explicit sbatch resource flags are required. Each invocation owns its GPU.
set -euo pipefail
campaign=$1
environment=$2
shift 2
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
cd "$campaign/source"
export PYTHONPATH="$campaign/runtime:$campaign/source"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec "$environment/bin/python" -u -m slot_abstractor.evaluate "$@"
