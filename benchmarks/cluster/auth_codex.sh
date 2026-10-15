#!/usr/bin/env bash
# User runs this on the Mac. This performs login only, not remote agent work.
set -euo pipefail
bench_dir=$(cd "$(dirname "$0")/.." && pwd)
ssh -S "$bench_dir/.local/ice.sock" -O check ice-login
exec ssh -tt -S "$bench_dir/.local/ice.sock" -o BatchMode=yes ice-login \
  /storage/ice1/9/8/yluo432/tools/codex-0.156.1/bin/codex login --device-auth
