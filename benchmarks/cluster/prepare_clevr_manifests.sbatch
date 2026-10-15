#!/usr/bin/env bash
# CPU-only completeness gate after all CLEVR rendering chunks succeed.
set -euo pipefail
project=$1
environment=$2
assets=$3
root=$4
test -n "${SLURM_JOB_ID:-}" || { echo 'Preparation requires Slurm.' >&2; exit 2; }
case "$(hostname -s)" in *login*|*Login*) echo 'Refusing login-node preparation.' >&2; exit 2;; esac
cd "$project"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
exec "$environment/bin/python" cluster/clevr_full.py \
  --assets "$assets" --root "$root" \
  --blender /storage/ice1/9/8/yluo432/tools/blender-2.79b/blender \
  --workers "${SLURM_CPUS_PER_TASK:-2}" --build-manifests
