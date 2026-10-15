#!/usr/bin/env bash
# Source before installing dependencies or running GPU inference.
require_gpu_allocation() {
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    printf 'Refusing compute work: SLURM_JOB_ID is absent. Use an allocation.\n' >&2
    return 1
  fi
  case "$(hostname -s)" in
    *login*|*Login*) printf 'Refusing compute work on a login host.\n' >&2; return 1;;
  esac
  if [ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ]; then
    printf 'Refusing compute work: no Slurm GPU allocation recorded.\n' >&2
    return 1
  fi
  command -v nvidia-smi >/dev/null 2>&1 || {
    printf 'nvidia-smi is unavailable in this allocation.\n' >&2; return 1;
  }
  printf 'Compute host: %s\nSlurm job: %s\nAllocated GPUs: %s\n' "$(hostname -f)" "$SLURM_JOB_ID" "${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || return 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  require_gpu_allocation
fi
