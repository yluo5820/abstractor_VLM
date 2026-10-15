#!/usr/bin/env bash
# CPU-only, pinned weight downloads. Positional pairs: REPOSITORY COMMIT ...
set -euo pipefail
environment=$1
cache=$2
receipt=$3
shift 3
test -n "${SLURM_JOB_ID:-}" || { echo 'Prefetch requires Slurm.' >&2; exit 2; }
case "$(hostname -s)" in *login*|*Login*) echo 'Refusing login-node preparation.' >&2; exit 2;; esac
export HF_HOME="$cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS=2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1
"$environment/bin/python" - "$receipt" "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from huggingface_hub import snapshot_download

receipt = Path(sys.argv[1])
arguments = sys.argv[2:]
if not arguments or len(arguments) % 2:
    raise ValueError('Pass repository/revision pairs')
models = []
for repository, revision in zip(arguments[::2], arguments[1::2]):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Each model requires a pinned commit')
    print(f'Downloading {repository}@{revision}', flush=True)
    snapshot = Path(snapshot_download(
        repository, revision=revision, token=False, max_workers=2,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', 'README.md', 'LICENSE*'],
    ))
    if snapshot.name != revision:
        raise ValueError('Downloaded snapshot does not match requested commit')
    files = {}
    for path in sorted(snapshot.rglob('*')):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        files[str(path.relative_to(snapshot))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Snapshot is missing model configuration or weights')
    models.append({'repository': repository, 'revision': revision, 'snapshot': str(snapshot), 'files': files})
    print(f'Ready: {repository}, {sum(item["bytes"] for item in files.values())} bytes', flush=True)
receipt.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt.with_suffix('.tmp')
with temporary.open('w') as stream:
    json.dump({'completed_utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'models': models}, stream, indent=2)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(receipt)
PY
