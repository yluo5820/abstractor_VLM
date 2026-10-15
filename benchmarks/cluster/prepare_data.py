#!/usr/bin/env python3
"""Stage pinned CLEVR-ART assets and create a bounded original-renderer job.

No rendering occurs unless --run is given inside a Slurm allocation. Blender
2.78/2.79 must already be installed; this helper does not install software.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import re
import uuid
from urllib.request import Request, urlopen


OCRA_COMMIT = "f15fa625d7e5b4458d82c24108b26419037d940d"
CLEVR_COMMIT = "f0ce2c81750bfae09b5bf94d009f42e055f2cb3a"
# Pinned Git blob IDs verify the exact bytes, including the small Blender assets.
OCRA_FILES = {
    "LICENSE.md": "27cba578407f199eefd500ccf98f44d30d6f3b01",
    "README.md": "57eae363a4e86c02be1535d862ed2552119bd787",
    "render_images_rmts.py": "1fea12f378c4ee00ef2fb162c1da452525d74ae8",
    "render_images_idrules.py": "399b91aa8f43817dbbda84947d512fc33aaf39ee",
    "RMTS_ood_train.npz": "9a67eb0744a0a0bf6fd318bd2183eaaf9cdbf4e5",
    "RMTS_ood_test.npz": "9e7416f08229bf2fdec412444e77159ab5d7eb8e",
    "identity_rules_ood_train.npz": "4911264f6c889eb9f6e871332f804a9c9d093316",
    "identity_rules_ood_test.npz": "a392883a4b6f034e36e03689ff423f94055d9a8f",
    "data/base_scene.blend": "e2fa2eab7a0bd14e3ea93c6a0b7ba1439221b31f",
    "data/properties.json": "ee86d6b7e04eef86e358c6e118a3c592222beb76",
    "data/materials/MyMetal.blend": "4e6a5b810440c58199e8e7d56bfdb6ab38925dea",
    "data/materials/Rubber.blend": "b02790c22254e7659d35d0c157a0cad51f023e65",
    "data/shapes/SmoothCube_v2.blend": "b88a02aa3bbd8b6ea1972cf00b90f5005221d044",
    "data/shapes/SmoothCylinder.blend": "9ea5e0d081c3e58546ce80a798e19742b223485c",
    "data/shapes/Sphere.blend": "4d9189011e4c0bf1248d40af2ec55c681c67218f",
}
CLEVR_FILES = {
    "utils.py": ("image_generation/utils.py", "93d2dc93f75f25ba0427283ff50149f4fe90dbbf"),
    "CLEVR_LICENSE": ("LICENSE", "691f2ca2ad79755975fc23f3175e21a738040982"),
    "CLEVR_PATENTS": ("PATENTS", "cd2a8c69b5a153412fa71d9490484b7a05a4cdb8"),
}


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch_one(root: Path, name: str, url: str, expected_blob: str):
    path = root / name
    if path.exists():
        data = path.read_bytes()
    else:
        print(f"Fetching {name}", flush=True)
        request = Request(url, headers={"User-Agent": "vlm-bench-data-preparation/1"})
        with urlopen(request, timeout=60) as response:
            # Every pinned asset is <4 MB. A wrong endpoint must not fill storage.
            data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large asset: {name}")
    if git_blob(data) != expected_blob:
        raise ValueError(f"Pinned checksum mismatch: {path}; existing files are never replaced")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publication; failures leave no corrupt final filename.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
            staged.write(data)
        Path(staged.name).rename(path)
    return {"path": name, "url": url, "bytes": len(data),
            "git_blob": expected_blob, "sha256": hashlib.sha256(data).hexdigest()}


def fetch_clevr(destination: Path):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in OCRA_FILES.items():
        url = f"https://raw.githubusercontent.com/Shanka123/OCRA/{OCRA_COMMIT}/{name}"
        records.append(fetch_one(destination, name, url, expected))
    for name, (source, expected) in CLEVR_FILES.items():
        url = f"https://raw.githubusercontent.com/facebookresearch/clevr-dataset-gen/{CLEVR_COMMIT}/{source}"
        records.append(fetch_one(destination, name, url, expected))
    payload = {"ocra_commit": OCRA_COMMIT, "clevr_commit": CLEVR_COMMIT, "files": records}
    (destination / "benchmark-source-provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Verified {len(records)} assets ({sum(r['bytes'] for r in records):,} bytes): {destination}")


def _checkpoint_source(protocol: str, candidates: int) -> str:
    """Python 3.5 code executed by Blender; one atomic receipt per full group."""
    return '''
import hashlib as _benchmark_hashlib
import json as _benchmark_json
import os as _benchmark_os
_BENCHMARK_PROTOCOL = %r
_BENCHMARK_CHOICES = %r

def _benchmark_file_hash(path):
  with open(path, 'rb') as handle:
    return _benchmark_hashlib.sha256(handle.read()).hexdigest()

def _benchmark_start_group(base, index):
  directory = _benchmark_os.path.join(base, 'prob_{}'.format(index))
  if not _benchmark_os.path.exists(directory):
    return True
  marker = _benchmark_os.path.join(directory, '.benchmark-complete.json')
  if not _benchmark_os.path.isfile(marker):
    raise RuntimeError('Incomplete or untracked render group: ' + directory)
  with open(marker) as handle:
    record = _benchmark_json.load(handle)
  names = ['CLEVR_{}.png'.format(i) for i in range(_BENCHMARK_CHOICES)]
  if record.get('protocol_sha256') != _BENCHMARK_PROTOCOL or record.get('index') != index:
    raise RuntimeError('Render protocol mismatch: ' + directory)
  if set(_benchmark_os.listdir(directory)) != set(names + ['.benchmark-complete.json']):
    raise RuntimeError('Incomplete or foreign files in render group: ' + directory)
  if set(record.get('files', {})) != set(names):
    raise RuntimeError('Incomplete render checkpoint: ' + directory)
  for name in names:
    if _benchmark_file_hash(_benchmark_os.path.join(directory, name)) != record['files'][name]:
      raise RuntimeError('Modified render image: ' + directory)
  return False

def _benchmark_finish_group(base, index):
  directory = _benchmark_os.path.join(base, 'prob_{}'.format(index))
  names = ['CLEVR_{}.png'.format(i) for i in range(_BENCHMARK_CHOICES)]
  if set(_benchmark_os.listdir(directory)) != set(names):
    raise RuntimeError('Unexpected files before render checkpoint: ' + directory)
  record = {'schema_version': 2, 'index': index,
            'protocol_sha256': _BENCHMARK_PROTOCOL,
            'blender_version': 'Blender ' + bpy.app.version_string,
            'files': {name: _benchmark_file_hash(_benchmark_os.path.join(directory, name)) for name in names}}
  marker = _benchmark_os.path.join(directory, '.benchmark-complete.json')
  temporary = marker + '.tmp'
  with open(temporary, 'w') as handle:
    _benchmark_json.dump(record, handle, sort_keys=True)
    handle.flush()
    _benchmark_os.fsync(handle.fileno())
  _benchmark_os.replace(temporary, marker)

''' % (protocol, candidates)


def patch_renderer(source: str, task: str, split: str, indices: list[int], render_seed: int,
                   root: Path, protocol: str | None = None) -> str:
    """Change only split, bounded indices, deterministic RNG, and import search path."""
    prefix = "RMTS" if task == "rmts" else "identity_rules"
    replacements = {
        f'np.load("{prefix}_ood_train.npz")': f'np.load("{prefix}_ood_{split}.npz", allow_pickle=False)',
        "  for num_ex in range(10000):": (
            "  for num_ex in " + repr(indices) + ":\n"
            "    random.seed(" + str(render_seed) + " + num_ex)\n"
            "    np.random.seed((" + str(render_seed) + " + num_ex) % (2 ** 32))"
        ),
        "from __future__ import print_function": (
            "from __future__ import print_function\n"
            "import sys\nsys.path.insert(0, " + repr(str(root)) + ")"
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"Original renderer structure changed: expected one occurrence of {old!r}")
        source = source.replace(old, new, 1)
    if protocol is not None:
        marker = "\ndef render_scene(args,angle,all_feats,"
        if source.count(marker) != 1:
            raise ValueError("Cannot locate the end of the original rendering loop")
        source = source.replace(marker, "\n    _benchmark_finish_group(args.output_image_dir, num_ex)\n" + marker, 1)
        loop = "  for num_ex in " + repr(indices) + ":\n"
        source = source.replace(loop, loop + "    if not _benchmark_start_group(args.output_image_dir, num_ex):\n      continue\n", 1)
        future = "from __future__ import print_function\n"
        source = source.replace(future, future + _checkpoint_source(protocol, 2 if task == "rmts" else 4), 1)
    # The upstream Blender runtime uses Python 3.5; generated code must parse there.
    ast.parse(source, feature_version=(3, 5))
    return source


def _atomic_json(path: Path, record: dict):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as staged:
        json.dump(record, staged, indent=2, sort_keys=True)
        staged.write("\n")
        staged.flush()
        os.fsync(staged.fileno())
    Path(staged.name).replace(path)


def _inspect_groups(scenes: Path, indices: list[int], candidates: int, protocol: str,
                    legacy: dict[int, dict] | None = None):
    """Validate every existing group before reusing any; never repair/delete it."""
    from PIL import Image
    from vlm_bench.data import sha256_file

    known, adoptions, versions = set(), {}, set()
    names = {f"CLEVR_{i}.png" for i in range(candidates)}
    allowed = set(indices)
    for directory in sorted(scenes.iterdir()) if scenes.exists() else []:
        match = re.fullmatch(r"prob_(0|[1-9]\d*)", directory.name)
        if directory.is_symlink() or not directory.is_dir() or not match:
            raise ValueError(f"Foreign entry in render output: {directory}")
        index = int(match[1])
        if index not in allowed:
            raise ValueError(f"Existing problem outside requested deterministic subset: {directory}")
        marker = directory / ".benchmark-complete.json"
        checkpoint = None
        if marker.is_file() and not marker.is_symlink():
            checkpoint = json.loads(marker.read_text())
            expected = names | {marker.name}
        elif legacy and index in legacy:
            checkpoint = dict(legacy[index])
            expected = names
        else:
            raise ValueError(f"Incomplete or untracked render group: {directory}; no files were changed")
        if set(p.name for p in directory.iterdir()) != expected:
            raise ValueError(f"Incomplete or foreign files in render group: {directory}")
        if checkpoint.get("protocol_sha256") != protocol or checkpoint.get("index") != index:
            raise ValueError(f"Render protocol mismatch: {directory}")
        actual = {}
        for name in sorted(names):
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Nonregular render image: {path}")
            with Image.open(path) as image:
                if image.format != "PNG" or image.size != (320, 240):
                    raise ValueError(f"Unexpected PNG format/resolution: {path}")
                image.verify()
            actual[name] = sha256_file(path)
        if marker.is_file():
            if checkpoint.get("files") != actual:
                raise ValueError(f"Modified render image or checkpoint: {directory}")
        else:
            checkpoint["files"] = actual
            adoptions[index] = checkpoint
        version = checkpoint.get("blender_version")
        if not isinstance(version, str) or not version.startswith(("Blender 2.78", "Blender 2.79")):
            raise ValueError(f"Missing verified Blender version in checkpoint: {directory}")
        versions.add(version)
        known.add(index)
    if len(versions) > 1:
        raise ValueError("Existing groups were rendered with different Blender versions")
    return [i for i in indices if i not in known], adoptions, sorted(versions)


def _legacy_checkpoints(root: Path, task: str, split: str, seed: int, render_seed: int,
                        protocol: str, expected: dict):
    """Explicitly import old successful records; original PNG hashes did not exist."""
    from vlm_bench.data import sha256_file

    records = {}
    for path in sorted((root / "benchmark-render-plans").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema_version") or record.get("task") != task or record.get("split") != split:
            continue
        required = dict(expected, selection_seed=seed, render_seed=render_seed,
                        status="complete", exit_code=0, missing_files=[])
        if any(record.get(key) != value for key, value in required.items()):
            continue
        script = path.with_suffix(".py")
        if not script.is_file() or sha256_file(script) != record.get("patched_renderer_sha256"):
            raise ValueError(f"Legacy completed renderer script was modified: {script}")
        generated = patch_renderer((root / f"render_images_{task}.py").read_text(),
                                   task, split, record["indices"], render_seed, root)
        if script.read_text() != generated:
            raise ValueError(f"Legacy renderer was not generated by the verified helper: {script}")
        for index in record["indices"]:
            if not isinstance(index, int) or index < 0:
                raise ValueError(f"Invalid legacy problem index in {path}")
            entry = {"schema_version": 2, "protocol_sha256": protocol, "index": index,
                     "blender_version": record.get("blender_version"),
                     "legacy_adoption": {"record": path.name, "record_sha256": sha256_file(path),
                                         "historical_png_hashes_available": False}}
            if index in records and records[index]["blender_version"] != entry["blender_version"]:
                raise ValueError(f"Conflicting legacy render records for problem {index}")
            records[index] = entry
    return records


@contextmanager
def _render_lock(root: Path, task: str, split: str, run: bool):
    if not run:
        yield
        return
    directory = root / "benchmark-render-plans"
    directory.mkdir(exist_ok=True)
    with (directory / f".{task}-{split}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another helper is rendering this task/split; wait for that process to finish") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def plan_render(root: Path, task: str, split: str, limit: int, seed: int,
                render_seed: int, blender: str, threads: int, run: bool,
                resume: bool = False, adopt_legacy: bool = False):
    with _render_lock(root.resolve(), task, split, run):
        return _plan_render(root, task, split, limit, seed, render_seed, blender,
                            threads, run, resume, adopt_legacy)


def _plan_render(root: Path, task: str, split: str, limit: int, seed: int,
                 render_seed: int, blender: str, threads: int, run: bool,
                 resume: bool, adopt_legacy: bool):
    import numpy as np
    from vlm_bench.data import OFFICIAL_CLEVR_SHA256, sha256_file, select_subset

    root = root.resolve()
    if limit < 1 or threads < 1 or render_seed < 0:
        raise ValueError("limit and threads must be positive; render-seed must be nonnegative")
    if adopt_legacy and not resume:
        raise ValueError("--adopt-legacy requires --resume")
    for name, expected in OCRA_FILES.items():
        path = root / name
        if not path.is_file() or git_blob(path.read_bytes()) != expected:
            raise ValueError(f"Missing or modified pinned asset: {path}; run clevr-assets first")
    for name, (_, expected) in CLEVR_FILES.items():
        path = root / name
        if not path.is_file() or git_blob(path.read_bytes()) != expected:
            raise ValueError(f"Missing or modified CLEVR utility/license: {path}")
    prefix = "RMTS" if task == "rmts" else "identity_rules"
    labels = root / f"{prefix}_ood_{split}.npz"
    if sha256_file(labels) != OFFICIAL_CLEVR_SHA256[labels.name]:
        raise ValueError("Labels differ from the verified official split")
    with np.load(labels, allow_pickle=False) as data:
        count = len(data["y"])
    if limit > count:
        raise ValueError(f"Requested {limit} examples but the official split contains {count}")
    items = [(f"clevr-art/{task}/ood/{split}/{i}", Path(str(i))) for i in range(count)]
    selected, _ = select_subset(items, limit, seed)
    indices = [int(identifier.rsplit("/", 1)[1]) for identifier, _ in selected]
    scenes = root / "output" / f"{task}_images" / f"{split}_ood"
    source_path = root / f"render_images_{task}.py"
    source_identity = {"ocra_commit": OCRA_COMMIT, "clevr_commit": CLEVR_COMMIT,
                       "source_renderer_sha256": sha256_file(source_path),
                       "labels_sha256": sha256_file(labels), "render_mode": "CPU", "resolution": [320, 240]}
    protocol_settings = dict(source_identity, task=task, split=split, selection_seed=seed,
                             render_seed=render_seed, render_samples=64,
                             asset_git_blobs=dict(OCRA_FILES, **{k: v[1] for k, v in CLEVR_FILES.items()}))
    protocol = hashlib.sha256(json.dumps(protocol_settings, sort_keys=True).encode()).hexdigest()
    if scenes.exists() and any(scenes.iterdir()) and not resume:
        raise FileExistsError(f"Render destination is nonempty: {scenes}; use --resume to verify completed groups")
    legacy = _legacy_checkpoints(root, task, split, seed, render_seed, protocol, source_identity) if adopt_legacy else None
    pending, adoptions, versions = _inspect_groups(scenes, indices, 2 if task == "rmts" else 4, protocol, legacy)
    source = patch_renderer(source_path.read_text(), task, split, pending, render_seed, root, protocol)
    plan_dir = root / "benchmark-render-plans"
    plan_dir.mkdir(exist_ok=True)
    slug = f"{task}-{split}-n{limit}-seed{seed}-renderseed{render_seed}"
    slug += "-" + uuid.uuid4().hex[:12]
    script = plan_dir / f"{slug}.py"
    script.write_text(source)
    command = [blender, "--background", "--threads", str(threads), "--python", str(script),
               "--", "--use_gpu", "0", "--output_image_dir", str(scenes) + "/"]
    record = {"schema_version": 2, "task": task, "split": split, "regime": "ood", "selection_seed": seed,
              "render_seed": render_seed, "count": len(indices), "indices": indices,
              **source_identity,
              "protocol": protocol_settings, "protocol_sha256": protocol,
              "pending_indices": pending, "reused_count": len(indices) - len(pending),
              "legacy_adopted_indices": sorted(adoptions), "completed_blender_versions": versions,
              "patched_renderer_sha256": sha256_file(script),
              "argv": command, "cwd": str(root),
              "status": "planned"}
    record_path = plan_dir / f"{slug}.json"
    _atomic_json(record_path, record)
    print(f"Verified {len(indices) - len(pending)} complete groups; {len(pending)} remain to render.")
    if adoptions:
        print(f"Explicit legacy adoption: {len(adoptions)} groups lack historical PNG hashes; hashes are sealed when --run starts.")
    print("Run from an allocated Slurm compute shell:")
    print("cd " + shlex.quote(str(root)))
    print(shlex.join(command))
    if not run:
        return
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("--run requires an active Slurm compute allocation; never render on a login node")
    version = subprocess.check_output([blender, "--version"], text=True).splitlines()[0]
    if not version.startswith(("Blender 2.78", "Blender 2.79")):
        raise RuntimeError(f"Original renderer requires Blender 2.78/2.79, found {version}")
    if versions and versions != [version]:
        raise ValueError(f"Blender version differs from completed groups: {version!r} versus {versions}")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if threads > allocated:
        raise ValueError(f"Requested {threads} renderer threads but only {allocated} CPUs per task allocated")
    for index, checkpoint in adoptions.items():
        _atomic_json(scenes / f"prob_{index}" / ".benchmark-complete.json", checkpoint)
    record.update({"blender_version": version, "slurm_job_id": os.environ["SLURM_JOB_ID"], "status": "running"})
    _atomic_json(record_path, record)
    completed = subprocess.run(command, cwd=root) if pending else subprocess.CompletedProcess(command, 0)
    candidates = 2 if task == "rmts" else 4
    missing = [str(scenes / f"prob_{i}" / f"CLEVR_{j}.png") for i in indices
               for j in range(candidates) if not (scenes / f"prob_{i}" / f"CLEVR_{j}.png").is_file()]
    validation_error = None
    try:
        remaining, _, _ = _inspect_groups(scenes, indices, candidates, protocol)
        if remaining:
            validation_error = f"Unrendered indices: {remaining}"
    except (ValueError, OSError) as error:
        validation_error = str(error)
    record.update({"exit_code": completed.returncode, "missing_files": missing,
                   "validation_error": validation_error,
                   "status": "complete" if completed.returncode == 0 and not missing and not validation_error else "failed"})
    _atomic_json(record_path, record)
    if completed.returncode != 0 or missing or validation_error:
        raise RuntimeError(f"Render incomplete. Inspect {record_path}; existing partial files were preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("clevr-assets", help="Fetch only pinned labels, renderers, Blender assets and utilities (~19 MB)")
    assets.add_argument("--destination", type=Path, required=True)
    render = commands.add_parser("clevr-render", help="Prepare a deterministic bounded renderer, optionally run in Slurm")
    render.add_argument("--root", type=Path, required=True)
    render.add_argument("--task", choices=("rmts", "idrules"), required=True)
    render.add_argument("--split", choices=("train", "test"), required=True)
    render.add_argument("--limit", type=int, default=16)
    render.add_argument("--seed", type=int, default=0)
    render.add_argument("--render-seed", type=int, default=0)
    render.add_argument("--blender", default="blender")
    render.add_argument("--threads", type=int, default=1)
    render.add_argument("--run", action="store_true")
    render.add_argument("--resume", action="store_true", help="Verify and reuse checkpointed complete groups; reject incomplete or changed groups")
    render.add_argument("--adopt-legacy", action="store_true", help="With --resume, explicitly seal old completed runs that have no historical PNG hashes")
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        (fetch_clevr if command == "clevr-assets" else plan_render)(**args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
