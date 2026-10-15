#!/usr/bin/env bash
# Supply verified CPU resources to sbatch; this job does not request a GPU.
set -euo pipefail
project=$1
environment=$2
root=$3
task=$4
count=$5
manifest=$6
shift 6
test -n "${SLURM_JOB_ID:-}" || { echo 'Rendering requires Slurm.' >&2; exit 2; }
case "$(hostname -s)" in *login*) echo 'Refusing login-node rendering.' >&2; exit 2;; esac
cd "$project"
scontrol show job "$SLURM_JOB_ID"
"$environment/bin/python" cluster/prepare_data.py clevr-render \
  --root "$root" --task "$task" --split test --limit "$count" \
  --seed 0 --render-seed 1 --threads "${SLURM_CPUS_PER_TASK:-1}" \
  --blender /storage/ice1/9/8/yluo432/tools/blender-2.79b/blender --run "$@"
"$environment/bin/python" -m vlm_bench.data clevr-art \
  --root "$root" --task "$task" --regime ood --split test \
  --limit "$count" --seed 0 --output "$manifest"
