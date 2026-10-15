#!/usr/bin/env bash
# Run this on your Mac; enter passwords and Duo responses only in the terminal.
set -euo pipefail
bench_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
socket_dir="$bench_dir/.local"
mkdir -p "$socket_dir"
chmod 700 "$socket_dir"
socket="$socket_dir/ice.sock"
if ssh -S "$socket" -O check ice-login 2>/dev/null; then
  echo 'An authenticated ICE connection is already available.'
  exit 0
fi
ssh -M -S "$socket" -o ControlPersist=1h -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 -o ConnectTimeout=15 -fN ice-login
ssh -S "$socket" -O check ice-login
echo 'ICE authentication complete. Return to Codex and say: connected.'
