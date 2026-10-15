#!/usr/bin/env bash
# CPU-only, checkpointed CLEVR rendering. All resources come from sbatch flags.
set -euo pipefail
project=$1
environment=$2
assets=$3
root=$4
seconds=$5
new_groups=${6:-all}
split=${7:-test}
shard_index=${8:-0}
shards=${9:-1}
test -n "${SLURM_JOB_ID:-}" || { echo 'Rendering requires Slurm.' >&2; exit 2; }
case "$(hostname -s)" in *login*|*Login*) echo 'Refusing login-node rendering.' >&2; exit 2;; esac
cd "$project"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
scontrol show job "$SLURM_JOB_ID"
args=(--assets "$assets" --root "$root"
  --blender /storage/ice1/9/8/yluo432/tools/blender-2.79b/blender
  --workers "${SLURM_CPUS_PER_TASK:-2}" --max-seconds "$seconds" --run)
args+=(--split "$split" --shard-index "$shard_index" --shards "$shards")
if [ "$new_groups" != all ]; then
  args+=(--max-new-groups-per-task "$new_groups")
fi
exec "$environment/bin/python" cluster/clevr_full.py "${args[@]}"
