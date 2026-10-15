"""Evaluate pinned PGM adapters on disjoint shards, then audit and merge predictions."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_partition(full, shards):
    expected = {r['id']: r for r in full}
    if len(expected) != len(full):
        raise ValueError('Duplicate original test IDs')
    observed = set()
    for rows in shards:
        for row in rows:
            if row['id'] in observed or expected.get(row['id']) != row:
                raise ValueError('Duplicate, foreign or changed question in test shards')
            observed.add(row['id'])
    if observed != set(expected):
        raise ValueError('Shards do not cover the complete test split')


def check_adapter(spec):
    train = Path(spec['train'])
    if digest(train / 'config.json') != spec['training_config_sha256']:
        raise ValueError('Training configuration changed')
    if json.loads((train / 'best.json').read_text()) != spec['checkpoint']:
        raise ValueError('Selected checkpoint changed')
    adapter = train / spec['checkpoint']['directory'] / 'adapter'
    if {p.name: digest(p) for p in adapter.iterdir() if p.is_file()} != spec['adapter_sha256']:
        raise ValueError('Adapter changed')
    return adapter


def worker(plan, model, rank, campaign):
    from peft import PeftModel
    from vlm_bench import run, models
    run.require_allocation()
    shard, spec = plan['shards'][rank], plan['models'][model]
    if digest(shard['manifest']) != shard['manifest_sha256']:
        raise ValueError('Test shard changed')
    adapter = check_adapter(spec)
    output = campaign / model / 'shards' / str(rank)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'protocol': 'validation-selected-lora-pgm-neutral-test-v1',
                  **spec, 'model': model, 'task': 'pgm', 'shard': rank,
                  'original_manifest_sha256': plan['manifest_sha256'],
                  'evaluator_sha256': digest(__file__)}
    receipt = output / 'adapter-provenance.json'
    if receipt.exists() and json.loads(receipt.read_text()) != provenance:
        raise ValueError('Evaluation provenance changed on resume')
    run.atomic_json(receipt, provenance)
    original = models.load_backend

    def load(*args, **kwargs):
        # The unchanged runner hashes every actual image before loading the model.
        config = json.loads((output / 'config.json').read_text())
        if config['images_sha256'] != shard['images_sha256']:
            raise ValueError('Actual images differ from the original benchmark')
        backend = original(*args, **kwargs)
        backend.model = PeftModel.from_pretrained(backend.model, str(adapter), is_trainable=False)
        backend.model.eval()
        backend.metadata['adaptation'] = provenance
        return backend

    models.load_backend = load
    result = run.run(argparse.Namespace(manifest=shard['manifest'], output=str(output), model=model,
        revision=spec['revision'], max_samples=0, max_new_tokens=16, seed=0, dtype='bfloat16',
        cpu_threads=2, interop_threads=1, summary_every=100, dry_run=False))
    if not result['complete'] or result['evaluated'] != shard['count']:
        raise RuntimeError('Test shard incomplete')


def merge(plan, model, campaign):
    from vlm_bench import run
    if digest(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('Original test manifest changed')
    full = read_rows(plan['manifest'])
    shard_rows = [read_rows(s['manifest']) for s in plan['shards']]
    validate_partition(full, shard_rows)
    records = {}
    receipts = []
    for rank, (shard, rows) in enumerate(zip(plan['shards'], shard_rows)):
        if digest(shard['manifest']) != shard['manifest_sha256']:
            raise ValueError('Test shard changed')
        folder = campaign / model / 'shards' / str(rank)
        current = run.read_records(folder / 'predictions.jsonl')
        if current:
            config = json.loads((folder / 'config.json').read_text())
            if config['manifest_sha256'] != shard['manifest_sha256'] or config['images_sha256'] != shard['images_sha256']:
                raise ValueError('Shard image or manifest provenance mismatch')
            run.validate_records(current, rows)
        for row in current:
            if row['id'] in records:
                raise ValueError('Duplicate saved test prediction')
            records[row['id']] = row
        receipts.append({'rank': rank, 'evaluated': len(current), 'planned': len(rows),
                         'predictions_sha256': digest(folder / 'predictions.jsonl') if current else None})
    ordered = [records[r['id']] for r in full if r['id'] in records]
    summary = run.summarize(ordered, full) if ordered else {
        'planned': len(full), 'evaluated': 0, 'complete': False, 'groups': []}
    out = campaign / model / 'test'
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / 'predictions.jsonl.tmp'
    with temporary.open('w') as stream:
        for record in ordered:
            stream.write(json.dumps(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(out / 'predictions.jsonl')
    run.atomic_json(out / 'config.json', {'model': model, 'manifest_sha256': plan['manifest_sha256'],
        'images_sha256': plan['images_sha256'], 'revision': plan['models'][model]['revision'],
        'max_new_tokens': 16, 'dtype': 'bfloat16', 'seed': 0, 'parser': run.PARSER_VERSION,
        'shards': plan['shards'], 'example_ids': [r['id'] for r in full]})
    run.atomic_json(out / 'adapter-provenance.json', plan['models'][model])
    run.atomic_json(out / 'merge-audit.json', {'shards': receipts, 'exact_partition': True,
        'original_order_restored': True, 'duplicate_ids': 0, 'missing': len(full)-len(ordered),
        'predictions_sha256': digest(out / 'predictions.jsonl')})
    run.atomic_json(out / 'summary.json', summary)
    return summary


def launch(plan, model, campaign, seconds, worker_script=None, checkpoint_validator=None):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 4 or not all(visible) or len(set(visible)) != 4:
        raise ValueError('Expected exactly four allocated, visible GPUs')
    (checkpoint_validator or check_adapter)(plan['models'][model])
    folder = campaign / model
    folder.mkdir(exist_ok=True)
    lock = (folder / '.launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stopped.append(number))
    children, logs = [], []
    started = time.monotonic()
    try:
        for rank, gpu in enumerate(visible):
            log = (folder / f'shard-{rank}.log').open('a')
            logs.append(log)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
            args = [sys.executable, '-u', worker_script or __file__, '--campaign', str(campaign), '--model', model,
                    '--mode', 'worker', '--rank', str(rank)]
            children.append(subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if stopped or time.monotonic()-started >= seconds or any(p.poll() not in (None, 0) for p in children):
                break
            time.sleep(1)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        grace = time.monotonic() + 45
        for child in children:
            try:
                child.wait(timeout=max(0.1, grace-time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
    summary = merge(plan, model, campaign)
    from vlm_bench.run import atomic_json
    atomic_json(folder / 'launcher-summary.json', {'job_id': os.environ.get('SLURM_JOB_ID'),
        'worker_exit_codes': [p.returncode for p in children], 'elapsed_seconds': time.monotonic()-started,
        'complete': summary['complete'], 'evaluated': summary['evaluated']})
    if not summary['complete'] or any(p.returncode for p in children):
        raise SystemExit('Evaluation incomplete; committed predictions can be resumed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--model', choices=['lfm25', 'smolvlm2', 'spatialevo'], required=True)
    p.add_argument('--mode', choices=['launch', 'worker', 'merge'], default='launch')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--seconds', type=int, default=14220)
    a = p.parse_args()
    plan = json.loads((a.campaign / 'plan.json').read_text())
    if a.mode == 'worker':
        worker(plan, a.model, a.rank, a.campaign)
    elif a.mode == 'merge':
        print(json.dumps(merge(plan, a.model, a.campaign)))
    else:
        launch(plan, a.model, a.campaign, a.seconds)


if __name__ == '__main__':
    main()
