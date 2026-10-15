"""Build a nested 20% PGM training pool from pinned, cached official-train shards."""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from vlm_bench.pgm_mirror import REVISION, REPOSITORY, decode_panels
from vlm_bench.data import render_pgm, PGM_PROMPT, sha256_file
from vlm_bench.run import atomic_json


def shard_rows(job):
    spec, cache, root, selected = job
    path = Path(cache) / Path(spec['path']).name
    if sha256_file(path) != spec['sha256']:
        raise ValueError(f'Shard checksum mismatch: {path}')
    root = Path(root)
    records = []
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16, use_threads=False):
        for raw in batch.to_pylist():
            index = offset
            offset += 1
            original = raw['id']
            if original not in selected:
                continue
            identifier = f'pgm/neutral/train/{original}'
            relative = f'images/train/{hashlib.sha256(identifier.encode()).hexdigest()[:24]}.png'
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Each shard owns disjoint IDs. Atomic publication permits a safe CPU retry.
            image = render_pgm(decode_panels(raw))
            temporary = dest.with_suffix('.partial')
            image.save(temporary, format='PNG')
            temporary.replace(dest)
            records.append({'id': identifier, 'dataset': 'pgm', 'task': 'matrix-completion',
                'regime': 'neutral', 'split': 'train', 'prompt': PGM_PROMPT,
                'choices': list('ABCDEFGH'), 'answer': chr(65 + int(raw['target'])),
                'images': [relative], 'metadata': {'contact_sheet_sha256': sha256_file(dest),
                    'mirror_repository': REPOSITORY, 'mirror_revision': REVISION,
                    'source_shard': path.name, 'source_sha256': spec['sha256'],
                    'source_row_index': index, 'source_original_id': original}})
    print(json.dumps({'shard': path.name, 'rendered': len(records)}), flush=True)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--existing', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(__file__).with_name('pgm-adaptation-shards.json').read_text())
    shards = [s for s in spec if Path(s['path']).name.startswith('train-')]
    refs = []
    for s in shards:
        path = a.existing / 'parquet' / Path(s['path']).name
        for original in pq.read_table(path, columns=['id']).column('id').to_pylist():
            key = hashlib.sha256(f'20261007\0train\0{original}'.encode()).digest()
            refs.append((key, original, s['path']))
    assert len(refs) == 1200000 and len({r[1] for r in refs}) == 1200000
    chosen = sorted(refs)[:240000]
    by_shard = {}
    for _, original, shard in chosen:
        by_shard.setdefault(shard, set()).add(original)
    jobs = [(s, str(a.existing / 'parquet'), str(a.output), by_shard[s['path']]) for s in shards]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        records = {r['id']: r for group in pool.map(shard_rows, jobs) for r in group}
    assert len(records) == 240000
    old = [json.loads(line) for line in (a.existing / 'train.jsonl').open()]
    assert {r['id'] for r in old}.issubset(records)
    for row in old:
        assert row['answer'] == records[row['id']]['answer']
    train = a.output / 'train.jsonl'
    with train.open('w') as out:
        for _, original, _ in chosen:
            out.write(json.dumps(records[f'pgm/neutral/train/{original}'], sort_keys=True) + '\n')
    # Preserve exactly the earlier held-out official validation questions/images.
    validation = [json.loads(line) for line in (a.existing / 'validation.jsonl').open()]
    assert len(validation) == 1000 and all(r['split'] == 'validation' for r in validation)
    for row in validation:
        source = a.existing / row['images'][0]
        assert sha256_file(source) == row['metadata']['contact_sheet_sha256']
        target = a.output / row['images'][0]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(source.resolve())
    (a.output / 'validation.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in validation))
    atomic_json(a.output / 'READY.json', {'revision': REVISION, 'train_count': 240000,
        'validation_count': 1000, 'official_train_count': 1200000, 'fraction': 0.2,
        'manifests': {n: sha256_file(a.output/n) for n in ['train.jsonl', 'validation.jsonl']},
        'selection': 'nested smallest SHA256(seed=20261007, official train split, original ID); test never read',
        'sources': spec, 'prior_9000_nested': True})


if __name__ == '__main__':
    main()
