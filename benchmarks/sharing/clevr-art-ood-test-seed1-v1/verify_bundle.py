#!/usr/bin/env python3
"""Verify an extracted bundle using only the Python standard library."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f'Unsafe bundle path: {name}')
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or root not in path.resolve().parents:
        raise ValueError(f'Non-local bundle path: {name}')
    return path


def main():
    root = Path(__file__).resolve().parent
    hashes = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name in hashes:
            raise ValueError(f'Duplicate checksum entry: {name}')
        digest = hashlib.sha256()
        with safe_path(root, name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
        hashes[name] = expected
        if len(hashes) % 2500 == 0:
            print(f'Verified {len(hashes):,} files', flush=True)
    spec = json.loads((root / 'provenance/bundle-spec.json').read_text())
    images = set()
    for task, count in spec['counts'].items():
        name = f'clevr-art-{task}-test-full.jsonl'
        if hashes.get(name) != spec['manifests'][name]:
            raise ValueError(f'Original manifest hash differs: {name}')
        ids = set()
        choices = list('AB' if task == 'rmts' else 'ABCD')
        rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
        for row in rows:
            assert row['dataset'] == 'clevr-art' and row['task'] == task
            assert row['split'] == 'test' and row['regime'] == 'ood'
            assert row['choices'] == choices and row['answer'] in choices
            assert isinstance(row['prompt'], str) and row['prompt']
            assert len(row['images']) == 1
            image = row['images'][0]
            assert image not in images
            assert hashes[image] == row['metadata']['contact_sheet_sha256']
            images.add(image)
            assert row['id'] not in ids
            ids.add(row['id'])
        assert len(rows) == count
        assert ids == {f'clevr-art/{task}/ood/test/{i}' for i in range(count)}
        print(f'{task}: {count:,} complete questions verified', flush=True)
    assert len(images) == sum(spec['counts'].values())
    print(f'PASS: {len(hashes):,} files; {len(images):,} images; original manifests preserved.')


if __name__ == '__main__':
    main()
