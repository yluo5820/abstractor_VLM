#!/usr/bin/env bash
# Submit with the verified ICE debug profile: one H200, 2 CPUs, 16 GiB, 20 minutes.
set -euo pipefail
campaign=$1
project=$2
environment=$3
shift 3
source "$campaign/source/cluster/guard.sh"
require_gpu_allocation
cd "$campaign/source"
export PYTHONPATH="$campaign/runtime:$campaign/source"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec "$environment/bin/python" -u -m slot_abstractor.pilot \
  --upstream "$campaign/upstream" \
  --clevr-encoder "$campaign/weights/clevr.pth.tar" \
  --pgm-encoder "$campaign/weights/pgm.pth.tar" \
  --clevr-root "$project/data/clevr-full-seed1" \
  --pgm-shard "$project/data/pgm-mirror-full/data/test-00000-of-00009.parquet" \
  --output "$campaign/pilot.json" --seconds 900 "$@"
