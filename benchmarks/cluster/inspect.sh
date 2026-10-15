#!/usr/bin/env bash
# Read-only discovery through the separately authenticated SSH master.
set -euo pipefail
bench_dir=$(cd "$(dirname "$0")/.." && pwd)
socket="$bench_dir/.local/ice.sock"
mkdir -p "$bench_dir/.local"
chmod 700 "$bench_dir/.local"
report="$bench_dir/.local/ice-inspect-$(date -u +%Y%m%dT%H%M%SZ).txt"
ssh -S "$socket" -o BatchMode=yes -o ConnectTimeout=15 ice-login 'bash -l -s' <<'REMOTE' | tee "$report"
set -u
section() { printf '\n### %s\n' "$1"; }
probe() {
  if command -v timeout >/dev/null 2>&1; then timeout 20 "$@"; else "$@"; fi
  rc=$?
  if [ "$rc" -ne 0 ]; then printf '(unavailable or not permitted; exit %s)\n' "$rc"; fi
  return 0
}
section 'Identity and scheduler tools'
date -u
hostname -f
id -un
printf 'HOME=%s\nSCRATCH=%s\n' "$HOME" "${SCRATCH:-unset}"
command -v squeue scontrol sinfo sacctmgr sbatch srun || true
section 'Existing user jobs (read only; none cancelled)'
probe squeue -u "$USER" -o '%.20i %.24j %.12T %.20P %.24a %.20q %.12l %.25R'
job_ids=$(squeue -h -u "$USER" -o '%A' 2>/dev/null | sort -u | head -n 10)
for job in $job_ids; do
  case "$job" in *[!0-9]*|'') continue;; esac
  probe scontrol show job "$job"
done
section 'User associations and QoS'
probe sacctmgr -n -P show assoc where user="$USER" format=Cluster,Account,User,Partition,QOS,DefaultQOS
probe sacctmgr -n -P show qos format=Name,MaxWall,MaxTRESPerJob,MaxTRESPU
section 'Partitions and GPU request evidence'
probe sinfo -o '%P|%a|%l|%D|%G'
probe scontrol show partition
section 'Existing software and environments'
command -v python python3 conda mamba micromamba module node npm codex || true
python3 --version 2>/dev/null || true
if type module >/dev/null 2>&1; then
  module -t list 2>&1 || true
  module -t avail python cuda anaconda 2>&1 | head -n 100 || true
fi
if command -v conda >/dev/null 2>&1; then probe conda env list; fi
printf 'Shallow project/environment directory names only:\n'
for base in "$HOME" "${SCRATCH:-}" "$HOME/scratch" "$HOME/projects" "$HOME/Projects"; do
  [ -n "$base" ] && [ -d "$base" ] || continue
  printf '%s\n' "$base"
  if command -v timeout >/dev/null 2>&1; then
    timeout 15 find "$base" -maxdepth 2 -type d \( -name '*VLM*' -o -name '*vlm*' -o -name '*abstract*' -o -name '.venv' -o -name 'envs' -o -name '*benchmark*' \) -print 2>/dev/null | head -n 60 || true
  fi
done
section 'Storage capacity and quota'
probe df -h "$HOME"
if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH" ]; then probe df -h "$SCRATCH"; fi
if command -v quota >/dev/null 2>&1; then probe quota -s; fi
section 'Lightweight outbound HTTPS HEAD checks (no downloads or auth)'
if command -v curl >/dev/null 2>&1; then
  for url in https://huggingface.co https://pypi.org/simple/ https://registry.npmjs.org/ https://auth.openai.com/; do
    printf '%s ' "$url"
    curl --head --silent --show-error --max-time 10 --output /dev/null --write-out 'HTTP %{http_code}\n' "$url" || true
  done
fi
section 'Policy status'
printf '%s\n' 'No installation or compute work performed.' 'Successful network checks do not establish permission to install software.' 'Verify PACE user-local installation and external-service policy before installing or authenticating remote Codex.'
REMOTE
printf '\nSaved discovery report: %s\n' "$report"
