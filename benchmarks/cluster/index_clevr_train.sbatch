#!/usr/bin/env bash
set -euo pipefail
source_root=$1
environment=$2
data_root=$3
shards=$4
cd "$source_root"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec "$environment/bin/python" -u cluster/index_clevr_train.py --root "$data_root" --shards "$shards"
