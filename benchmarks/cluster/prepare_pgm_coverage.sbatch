#!/usr/bin/env bash
set -euo pipefail
campaign=$1
export PYTHONPATH="$campaign/runtime:$campaign/source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/storage/ice1/9/8/yluo432/envs/vlm-bench/bin/python
"$python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  "$campaign/source/cluster/test_parallel_gradients.py"
exec "$python" -u "$campaign/source/cluster/prepare_pgm_coverage.py" \
  --existing /storage/ice1/9/8/yluo432/vlm-benchmark/data/pgm-adaptation-2026-10-07 \
  --output /storage/ice1/9/8/yluo432/vlm-benchmark/data/pgm-coverage-20pct-2026-10-08 --workers 8
