#!/usr/bin/env python3
"""Package the pinned CLEVR test inputs without changing source data."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def collect(root, assets, support):
    spec_bytes = (support / 'bundle-spec.json').read_bytes()
    spec = json.loads(spec_bytes)
    manifests = root / 'manifests-v2'
    ready_bytes = (manifests / 'READY.json').read_bytes()
    ready = json.loads(ready_bytes)
    for key in ('counts', 'manifests', 'render_config_sha256'):
        if spec[key] != ready[key]:
            raise ValueError(f'Pinned readiness mismatch: {key}')
    render_config = (root / 'render-config.json').read_bytes()
    if sha(render_config) != spec['render_config_sha256']:
        raise ValueError('Render configuration changed')
    small = {
        'README.md': (support / 'README.md').read_bytes(),
        'verify_bundle.py': (support / 'verify_bundle.py').read_bytes(),
        'provenance/bundle-spec.json': spec_bytes,
        'provenance/READY.json': ready_bytes,
        'provenance/render-config.json': render_config,
    }
    for name, digest in spec['labels_sha256'].items():
        content = (root / name).read_bytes()
        if sha(content) != digest:
            raise ValueError(f'Symbolic labels changed: {name}')
        small[name] = content
    for name, digest in spec['license_git_blobs'].items():
        content = (assets / name).read_bytes()
        blob = b'blob ' + str(len(content)).encode() + b'\0' + content
        if hashlib.sha1(blob).hexdigest() != digest:
            raise ValueError(f'Upstream notice differs: {name}')
        small['licenses/' + ('OCRA_LICENSE.md' if name == 'LICENSE.md' else name)] = content
    images, seen, prompts, totals = [], set(), {}, {}
    for task, count in spec['counts'].items():
        name = f'clevr-art-{task}-test-full.jsonl'
        content = (manifests / name).read_bytes()
        if sha(content) != spec['manifests'][name]:
            raise ValueError(f'Original manifest changed: {name}')
        rows = [json.loads(line) for line in content.splitlines()]
        ids = [row['id'] for row in rows]
        if len(rows) != count or set(ids) != {f'clevr-art/{task}/ood/test/{i}' for i in range(count)}:
            raise ValueError(f'Incomplete or duplicated examples: {task}')
        choices = list('AB' if task == 'rmts' else 'ABCD')
        task_prompts = set()
        for row in rows:
            if (row['dataset'], row['task'], row['regime'], row['split']) != ('clevr-art', task, 'ood', 'test'):
                raise ValueError(f'Wrong task or split: {row["id"]}')
            if row['choices'] != choices or row['answer'] not in choices or len(row['images']) != 1:
                raise ValueError(f'Wrong answer layout: {row["id"]}')
            relative = row['images'][0]
            parts = PurePosixPath(relative).parts
            folder = name[:-6] + '_images'
            if (len(parts) != 2 or parts[0] != folder or '..' in parts or '\\' in relative
                    or not parts[1].endswith('.png') or relative in seen):
                raise ValueError(f'Unsafe or duplicated image: {relative}')
            seen.add(relative)
            images.append((relative, manifests / relative, row['metadata']['contact_sheet_sha256']))
            task_prompts.add(row['prompt'])
        if len(task_prompts) != 1 or not next(iter(task_prompts)):
            raise ValueError(f'Inconsistent prompt: {task}')
        prompts[task] = next(iter(task_prompts))
        totals[task] = dict(sorted(Counter(row['answer'] for row in rows).items()))
        small[name] = content
        sidecar = name + '.provenance.json'
        small[sidecar] = (manifests / sidecar).read_bytes()
    small['prompts.json'] = json_bytes(prompts)
    small['dataset-description.json'] = json_bytes({
        'name': spec['name'], 'scope': 'Two complete CLEVR-ART OOD test tasks; no training data',
        'counts': spec['counts'], 'answer_counts': totals, 'render_seed': 1, 'selection_seed': 0,
        'inputs': 'Unmodified RGB composite PNG and exact prompt from each manifest row',
        'original_manifest_sha256': spec['manifests'], 'original_render_config_sha256': spec['render_config_sha256'],
    })
    return spec, small, images


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        content = self.stream.read(size)
        self.digest.update(content)
        return content


def entry(name, size):
    item = tarfile.TarInfo(name)
    item.size, item.mode, item.mtime = size, 0o644, 0
    return item


def build(root, assets, support, destination):
    started = time.monotonic()
    spec, small, images = collect(root, assets, support)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (spec['name'] + '.tar.gz')
    partial = archive.with_suffix(archive.suffix + '.partial')
    if archive.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite an existing or partial archive; use a fresh destination')
    hashes, total_image_bytes = {}, 0
    status = destination / 'package-progress.json'

    def progress(state, completed):
        value = {'status': state, 'images_completed': completed, 'images_planned': len(images),
                 'elapsed_seconds': time.monotonic() - started, 'image_bytes': total_image_bytes,
                 'archive_bytes_so_far': partial.stat().st_size if partial.exists() else archive.stat().st_size,
                 'job_id': os.environ.get('SLURM_JOB_ID')}
        atomic_json(status, value)
        print(json.dumps(value), flush=True)

    try:
        with partial.open('xb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as bundle:
                    for name, content in sorted(small.items()):
                        bundle.addfile(entry(spec['name'] + '/' + name, len(content)), io.BytesIO(content))
                        hashes[name] = sha(content)
                    progress('packaging', 0)
                    for i, (name, path, expected) in enumerate(images, 1):
                        if path.parent.is_symlink():
                            raise ValueError(f'Image directory is a symlink: {path.parent}')
                        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(descriptor, 'rb') as stream:
                            info = os.fstat(stream.fileno())
                            if not stat.S_ISREG(info.st_mode):
                                raise ValueError(f'Not a regular PNG file: {path}')
                            reader = HashingReader(stream)
                            bundle.addfile(entry(spec['name'] + '/' + name, info.st_size), reader)
                            if reader.digest.hexdigest() != expected:
                                raise ValueError(f'Image bytes changed: {name}')
                            total_image_bytes += info.st_size
                        hashes[name] = expected
                        if i % 250 == 0:
                            progress('packaging', i)
                    checksums = ''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items())).encode()
                    bundle.addfile(entry(spec['name'] + '/SHA256SUMS', len(checksums)), io.BytesIO(checksums))
            raw.flush()
            os.fsync(raw.fileno())
        progress('hashing_archive', len(images))
        digest = hashlib.sha256()
        with partial.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        archive_hash = digest.hexdigest()
        partial.rename(archive)
        (destination / (archive.name + '.sha256')).write_text(f'{archive_hash}  {archive.name}\n')
        receipt = {'status': 'complete', 'archive': archive.name, 'sha256': archive_hash,
                   'archive_bytes': archive.stat().st_size, 'image_bytes': total_image_bytes,
                   'counts': spec['counts'], 'files': len(hashes) + 1,
                   'manifest_sha256': spec['manifests'], 'bundle_spec_sha256': sha(json_bytes(spec)),
                   'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'job_id': os.environ.get('SLURM_JOB_ID'), 'elapsed_seconds': time.monotonic() - started}
        atomic_json(destination / 'COMPLETE.json', receipt)
        progress('complete', len(images))
        return receipt
    except BaseException as error:
        atomic_json(destination / 'PACKAGE_ERROR.json', {'error': str(error), 'type': type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'assets', 'support', 'destination'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run the full packager inside a CPU Slurm allocation')
    if 'login' in os.uname().nodename.lower():
        parser.error('Refusing login-node packaging')
    print(json.dumps(build(args.root, args.assets, args.support, args.destination), indent=2), flush=True)


if __name__ == '__main__':
    main()
