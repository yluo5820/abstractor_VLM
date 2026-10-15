#!/usr/bin/env bash
# Original OCRA rendering requires the legacy Blender Python API.
set -euo pipefail
cluster_dir=$(cd "$(dirname "$0")" && pwd)
source "$cluster_dir/guard.sh"
require_gpu_allocation
if [ "$#" -ne 1 ]; then echo 'Usage: install_blender.sh /absolute/new/prefix' >&2; exit 2; fi
prefix=$1
case "$prefix" in /*) ;; *) echo 'Use an absolute installation path.' >&2; exit 2;; esac
test ! -e "$prefix" || { echo 'Installation path already exists.' >&2; exit 2; }
test "$(uname -m)" = x86_64 || { echo 'This official Blender build requires x86_64.' >&2; exit 2; }
parent=$(dirname "$prefix")
mkdir -p "$parent"
temporary=$(mktemp -d "$parent/.blender-download.XXXXXX")
curl --fail --location --connect-timeout 15 --max-time 300 \
  https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2 \
  -o "$temporary/blender.tar.bz2"
printf '%s  %s\n' 43824a4e0b0c6de6fa34ff224eec44c1cc9f26a95f6f3c8c2558d1c05704183c \
  "$temporary/blender.tar.bz2" | sha256sum -c -
mkdir "$temporary/package"
tar --no-same-owner --strip-components=1 -xjf "$temporary/blender.tar.bz2" -C "$temporary/package"
"$temporary/package/blender" --version
mv -T -n "$temporary/package" "$prefix"
printf 'Installed original-renderer dependency at %s\n' "$prefix"
