#!/usr/bin/env bash
# Run only after inspecting existing environments and confirming installation policy.
set -euo pipefail
if [ "$#" -ne 4 ] || [ "$1" != --policy-verified ]; then
  printf 'Usage: setup_environment.sh --policy-verified /absolute/new/venv /absolute/cache /absolute/requirements.txt\n' >&2
  exit 2
fi
cluster_dir=$(cd "$(dirname "$0")" && pwd)
source "$cluster_dir/guard.sh"
require_gpu_allocation
environment=$2
cache=$3
requirements=$4
for path in "$environment" "$cache" "$requirements"; do
  case "$path" in /*) ;; *) printf 'Paths must be absolute.\n' >&2; exit 2;; esac
done
if [ -e "$environment" ]; then
  printf 'Environment path already exists; inspect and reuse it rather than overwrite.\n' >&2
  exit 2
fi
test -f "$requirements"
export PIP_CACHE_DIR="$cache/pip"
python3 -m venv "$environment"
"$environment/bin/python" -m pip install -r "$requirements"
"$environment/bin/python" -m pip install --no-deps -e "$cluster_dir/.."
"$environment/bin/python" -m pip freeze > "$environment/requirements.resolved.txt"
