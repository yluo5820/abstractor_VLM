#!/usr/bin/env bash
# CPU-only preparation; submit the GPU evaluation with afterok on this job.
set -euo pipefail
project=$1
environment=$2
data=$3
manifest=$4
test -n "${SLURM_JOB_ID:-}" || { echo 'Preparation requires Slurm.' >&2; exit 2; }
case "$(hostname -s)" in *login*|*Login*) echo 'Refusing login-node preparation.' >&2; exit 2;; esac
cd "$project"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export ARROW_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
scontrol show job "$SLURM_JOB_ID"
exec "$environment/bin/python" -m vlm_bench.pgm_full \
  --parquet "$data"/test-*-of-00009.parquet \
  --output "$manifest" --seed 0 --batch-size 16 --resume
