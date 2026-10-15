#!/usr/bin/env bash
set -euo pipefail
campaign=$1
export PYTHONPATH="$campaign/source"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /storage/ice1/9/8/yluo432/envs/vlm-bench/bin/python -u "$campaign/source/cluster/prepare_pgm_adaptation.py" \
 --output /storage/ice1/9/8/yluo432/vlm-benchmark/data/pgm-adaptation-2026-10-07
