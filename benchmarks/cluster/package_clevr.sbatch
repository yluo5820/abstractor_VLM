#!/usr/bin/env bash
# CPU-only archive creation; resource flags are recorded with each submission.
set -euo pipefail
environment=$1
source=$2
root=$3
assets=$4
support=$5
destination=$6
test -n "${SLURM_JOB_ID:-}" || { echo 'Packaging requires Slurm.' >&2; exit 2; }
case "$(hostname -s)" in *login*|*Login*) echo 'Refusing login-node packaging.' >&2; exit 2;; esac
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "$environment/bin/python" "$source/package_clevr.py" \
  --root "$root" --assets "$assets" --support "$support" --destination "$destination"
