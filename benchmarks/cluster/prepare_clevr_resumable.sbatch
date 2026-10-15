#!/usr/bin/env bash
# CPU-only; one-hour segments retain durable contact sheets and manifest rows.
set -euo pipefail
project=$1
environment=$2
root=$3
output=$4
render_digest=$5
mode=${6:-prepare}
test -n "${SLURM_JOB_ID:-}" || { echo 'Preparation requires Slurm.' >&2; exit 2; }
case "$(hostname -s)" in *login*|*Login*) echo 'Refusing login-node preparation.' >&2; exit 2;; esac
cd "$project"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
args=(--root "$root" --output "$output" --render-config-sha256 "$render_digest")
if [ "$mode" = verify ]; then
  args+=(--require-ready)
elif [ "$mode" != prepare ]; then
  echo 'Mode must be prepare or verify.' >&2; exit 2
fi
exec "$environment/bin/python" -m vlm_bench.clevr_prepare "${args[@]}"
