#!/usr/bin/env python3
"""Render CLEVR-ART train or test splits, resuming atomic problem checkpoints.

Original Blender 2.79b CPU rendering, 64 samples, seed 1 + problem index.
Each worker owns one complete problem (2/4 scenes), in a fresh attempt directory.
Only verified complete groups enter output/. Interrupted attempts are retained;
the next invocation rerenders that problem, never deletes or replaces its files.

--max-seconds stops the workers and exits successfully with incomplete progress.
Run --build-manifests separately: it refuses until BOTH 10,000-problem splits are
complete. No scheduler jobs, downloads, installations, or GPU work occur here.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_data as helper
from vlm_bench.data import OFFICIAL_CLEVR_SHA256, sha256_file

TASKS = {"rmts": 2, "idrules": 4}
TOTAL = 10000
MARKER = ".benchmark-complete.json"


def allocation_guard(workers: int):
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        raise RuntimeError("Rendering requires a Slurm compute allocation; never use a login node")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if workers < 1 or workers > allocated:
        raise ValueError(f"Requested {workers} workers, but allocation has {allocated} CPUs per task")


def assigned_indices(total: int, shard_index: int = 0, shards: int = 1) -> list[int]:
    if not 1 <= shards <= total or not 0 <= shard_index < shards:
        raise ValueError("Shard index/count must partition the official problem indices")
    return list(range(shard_index, total, shards))


def settings(assets: Path, blender: Path, split: str = "test",
             shard_index: int = 0, shards: int = 1) -> dict:
    import numpy as np
    if split not in ("train", "test"):
        raise ValueError("Only the separate official train/test splits are supported")
    indices = assigned_indices(TOTAL, shard_index, shards)
    blobs = dict(helper.OCRA_FILES, **{key: value[1] for key, value in helper.CLEVR_FILES.items()})
    for name, expected in blobs.items():
        path = assets / name
        if not path.is_file() or helper.git_blob(path.read_bytes()) != expected:
            raise ValueError(f"Missing or modified pinned asset: {path}")
    version = subprocess.check_output([str(blender), "--version"], text=True, timeout=30).splitlines()[0]
    if version != "Blender 2.79 (sub 0)":
        raise ValueError(f"Expected installed Blender 2.79b ('Blender 2.79 (sub 0)'), found {version!r}")
    tasks = {}
    for task, choices in TASKS.items():
        label = ("RMTS" if task == "rmts" else "identity_rules") + f"_ood_{split}.npz"
        digest = sha256_file(assets / label)
        if digest != OFFICIAL_CLEVR_SHA256[label]:
            raise ValueError(f"Labels differ from verified official split: {label}")
        with np.load(assets / label, allow_pickle=False) as data:
            if len(data["y"]) != TOTAL:
                raise ValueError(f"Expected {TOTAL} official labels: {label}")
        identity = {"ocra_commit": helper.OCRA_COMMIT, "clevr_commit": helper.CLEVR_COMMIT,
                    "source_renderer_sha256": sha256_file(assets / f"render_images_{task}.py"),
                    "labels_sha256": digest, "render_mode": "CPU", "resolution": [320, 240]}
        protocol = dict(identity, task=task, split=split, selection_seed=0, render_seed=1,
                        render_samples=64, asset_git_blobs=blobs)
        tasks[task] = {"count": len(indices), "indices": indices, "candidates": choices, "labels": label,
                       "source_identity": identity, "protocol": protocol,
                       "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}
    return {"schema_version": 1, "assets": str(assets), "blender": str(blender),
            "split": split, "shard_index": shard_index, "shards": shards,
            "blender_version": version, "blender_sha256": sha256_file(blender),
            "renderer_helper_sha256": sha256_file(Path(helper.__file__)),
            "orchestrator_sha256": sha256_file(Path(__file__)),
            "method": "original-renderer-one-process-per-problem-v1", "tasks": tasks}


@contextmanager
def root_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".render.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another renderer owns this output root") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def initialize(root: Path, config: dict):
    path = root / "render-config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Render configuration changed; use a separate output root")
    else:
        if set(p.name for p in root.iterdir()) - {".render.lock"}:
            raise ValueError("Refusing to adopt a nonempty root without render-config.json")
        helper._atomic_json(path, config)
    for task, spec in config["tasks"].items():
        scenes_path(root, task, config.get("split", "test")).mkdir(parents=True, exist_ok=True)
        label = root / spec["labels"]
        if not label.exists():
            staged = root / (spec["labels"] + "." + uuid.uuid4().hex + ".tmp")
            shutil.copyfile(Path(config["assets"]) / spec["labels"], staged)
            staged.rename(label)
        if sha256_file(label) != spec["source_identity"]["labels_sha256"]:
            raise ValueError(f"Modified copied labels: {label}")
    (root / "sessions").mkdir(exist_ok=True)
    (root / "attempts").mkdir(exist_ok=True)


def scenes_path(root: Path, task: str, split: str = "test") -> Path:
    if split not in ("train", "test"):
        raise ValueError("Invalid split")
    return root / "output" / f"{task}_images" / f"{split}_ood"


def inspect(root: Path, config: dict) -> dict[str, list[int]]:
    pending = {}
    for task, spec in config["tasks"].items():
        indices, _, versions = helper._inspect_groups(scenes_path(root, task, config.get("split", "test")),
                                                      spec.get("indices", list(range(spec["count"]))),
                                                      spec["candidates"], spec["protocol_sha256"])
        if versions and versions != [config["blender_version"]]:
            raise ValueError(f"Completed {task} Blender version differs: {versions}")
        pending[task] = indices
    return pending


def publish(root: Path, config: dict, task: str, index: int, attempt: Path):
    spec = config["tasks"][task]
    images = attempt / "images"
    missing, _, versions = helper._inspect_groups(images, [index], spec["candidates"], spec["protocol_sha256"])
    if missing or versions != [config["blender_version"]]:
        raise ValueError(f"Attempt did not complete a verified problem: {attempt}")
    destination = scenes_path(root, task, config.get("split", "test")) / f"prob_{index}"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace completed group: {destination}")
    (images / f"prob_{index}").rename(destination)
    # Persist the directory entry before advertising this checkpoint in progress.
    handle = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def recover(root: Path, config: dict, pending: dict[str, list[int]]):
    """Publish receipts completed just before a previous coordinator was killed."""
    for task, indices in pending.items():
        recovered = set()
        for index in indices:
            base = root / "attempts" / task / str(index)
            for marker in sorted(base.glob(f"*/images/prob_{index}/{MARKER}")):
                attempt = marker.parents[2]
                record = json.loads((attempt / "attempt.json").read_text())
                if record.get("protocol_sha256") != config["tasks"][task]["protocol_sha256"]:
                    raise ValueError(f"Attempt protocol mismatch: {attempt}")
                publish(root, config, task, index, attempt)
                helper._atomic_json(attempt / "attempt.json", dict(record, status="recovered"))
                recovered.add(index)
                break
        pending[task] = [index for index in indices if index not in recovered]


def import_pilot(root: Path, config: dict, pilot: Path, adopt_legacy: bool):
    """Copy verified pilot groups; never seal, modify or delete their originals."""
    if pilot == root:
        raise ValueError("Pilot and full output roots must differ")
    for task, spec in config["tasks"].items():
        split = config.get("split", "test")
        assigned = spec.get("indices", list(range(spec["count"])))
        legacy = helper._legacy_checkpoints(pilot, task, split, 0, 1, spec["protocol_sha256"],
                                            spec["source_identity"]) if adopt_legacy else None
        missing, adoptions, versions = helper._inspect_groups(scenes_path(pilot, task, split),
                                    assigned, spec["candidates"], spec["protocol_sha256"], legacy)
        if versions and versions != [config["blender_version"]]:
            raise ValueError("Pilot Blender version differs from the configured renderer")
        for index in sorted(set(assigned) - set(missing)):
            if (scenes_path(root, task, split) / f"prob_{index}").exists():
                continue
            attempt = root / "attempts" / task / str(index) / ("import-" + uuid.uuid4().hex)
            target = attempt / "images" / f"prob_{index}"
            target.parent.mkdir(parents=True)
            shutil.copytree(scenes_path(pilot, task, split) / f"prob_{index}", target)
            if index in adoptions:
                helper._atomic_json(target / MARKER, adoptions[index])
            record = {"task": task, "index": index, "protocol_sha256": spec["protocol_sha256"],
                      "source_pilot": str(pilot), "status": "imported",
                      "historical_png_hashes_available": index not in adoptions}
            helper._atomic_json(attempt / "attempt.json", record)
            publish(root, config, task, index, attempt)


def launch(root: Path, config: dict, task: str, index: int, session: str) -> dict:
    attempt = root / "attempts" / task / str(index) / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    images = attempt / "images"
    images.mkdir()
    assets = Path(config["assets"])
    spec = config["tasks"][task]
    script = attempt / "renderer.py"
    script.write_text(helper.patch_renderer((assets / f"render_images_{task}.py").read_text(),
                      task, config.get("split", "test"), [index], 1, assets, spec["protocol_sha256"]))
    argv = [config["blender"], "--background", "--threads", "1", "--python", str(script),
            "--", "--use_gpu", "0", "--output_image_dir", str(images) + "/"]
    record = {"task": task, "index": index, "protocol_sha256": spec["protocol_sha256"],
              "session": session, "status": "running", "argv": argv,
              "patched_renderer_sha256": sha256_file(script), "started_at": time.time()}
    helper._atomic_json(attempt / "attempt.json", record)
    # These are per-worker CPU limits; neither Blender nor NumPy may multiply workers.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (attempt / "blender.log").open("wb") as log:
        process = subprocess.Popen(argv, cwd=assets, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    return {"process": process, "attempt": attempt, "record": record, "started": time.monotonic()}


def stop_owned(workers: list[dict]):
    """Signal only subprocess groups created by this invocation, never Slurm jobs."""
    for worker in workers:
        if worker["process"].poll() is None:
            try:
                os.killpg(worker["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for worker in workers:
        try:
            worker["process"].wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(worker["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker["process"].wait()


def render(root: Path, config: dict, workers: int, max_seconds: float,
           cap: int = 0, pilot: Path | None = None, adopt_legacy: bool = False) -> dict:
    allocation_guard(workers)
    started = time.monotonic()
    with root_lock(root):
        initialize(root, config)
        pending = inspect(root, config)
        recover(root, config, pending)
        if pilot:
            import_pilot(root, config, pilot, adopt_legacy)
            pending = inspect(root, config)
        initial = {task: config["tasks"][task]["count"] - len(indices) for task, indices in pending.items()}
        queues = {task: deque(indices[:cap] if cap else indices) for task, indices in pending.items()}
        session = uuid.uuid4().hex
        session_path = root / "sessions" / f"{session}.json"
        record = {"schema_version": 1, "session": session, "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "hostname": socket.gethostname(), "workers": workers, "max_seconds": max_seconds,
                  "max_new_groups_per_task": cap, "orchestrator_sha256": sha256_file(Path(__file__)),
                  "started_at": time.time(), "status": "running", "reused_groups": initial,
                  "new_groups": dict.fromkeys(TASKS, 0), "worker_seconds": dict.fromkeys(TASKS, 0.0)}
        active, errors, stopped = [], [], []
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in handlers:
            signal.signal(sig, lambda signum, frame: stopped.append(signal.Signals(signum).name))

        def save(status):
            record.update(status=status, elapsed_seconds=time.monotonic() - started,
                          completed_groups={task: initial[task] + record["new_groups"][task] for task in TASKS},
                          expected_groups={task: config["tasks"][task]["count"] for task in TASKS}, errors=errors)
            helper._atomic_json(session_path, record)
            helper._atomic_json(root / "progress.json", record)

        order = deque(TASKS)
        reason = None
        try:
            save("running")
            while active or any(queues.values()):
                if stopped or time.monotonic() - started >= max_seconds:
                    reason = stopped[0] if stopped else "budget-exhausted"
                    stop_owned(active)
                for worker in list(active):
                    code = worker["process"].poll()
                    if code is None:
                        continue
                    item = worker["record"]
                    task, index = item["task"], item["index"]
                    elapsed = time.monotonic() - worker["started"]
                    status = "interrupted" if reason else "failed"
                    # A finished receipt is reusable even if SIGTERM reached Blender during shutdown.
                    marker = worker["attempt"] / "images" / f"prob_{index}" / MARKER
                    if marker.is_file():
                        try:
                            publish(root, config, task, index, worker["attempt"])
                            record["new_groups"][task] += 1
                            record["worker_seconds"][task] += elapsed
                            status = "complete"
                            print(f"{task} problem {index}: complete ({elapsed:.1f}s)", flush=True)
                        except (OSError, ValueError) as exc:
                            errors.append(str(exc))
                    elif not reason:
                        errors.append(f"{task} problem {index}: Blender exit {code} without valid completion receipt; {worker['attempt']}")
                    helper._atomic_json(worker["attempt"] / "attempt.json",
                                        dict(item, status=status, exit_code=code, elapsed_seconds=elapsed))
                    active.remove(worker)
                    save("running")
                if reason or errors:
                    break
                while len(active) < workers and any(queues.values()):
                    task = order[0]
                    order.rotate(-1)
                    if queues[task]:
                        active.append(launch(root, config, task, queues[task].popleft(), session))
                if active:
                    time.sleep(0.25)
            complete = all(initial[task] + record["new_groups"][task] == config["tasks"][task]["count"] for task in TASKS)
            save("failed" if errors else "complete" if complete else reason or "calibration-complete")
        finally:
            stop_owned(active)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        print(json.dumps({key: record[key] for key in ("status", "completed_groups", "new_groups", "worker_seconds")}))
        if errors:
            raise RuntimeError(f"Rendering failed; completed outputs preserved. Inspect {session_path}")
        return record


def build_manifests(root: Path, config: dict):
    from vlm_bench.data import prepare_clevr_art
    if config.get("shards", 1) != 1:
        raise ValueError("Assemble all shards before building a full-dataset manifest")
    split = config.get("split", "test")
    with root_lock(root):
        if json.loads((root / "render-config.json").read_text()) != config:
            raise ValueError("Render configuration changed")
        for spec in config["tasks"].values():
            if sha256_file(root / spec["labels"]) != spec["source_identity"]["labels_sha256"]:
                raise ValueError(f"Modified copied labels: {root / spec['labels']}")
        pending = inspect(root, config)
        if any(pending.values()):
            raise ValueError("Both full splits must be complete before publishing evaluation manifests")
        manifests = root / "manifests"
        if manifests.exists():
            ready = json.loads((manifests / "READY.json").read_text())
            expected = {f"clevr-art-{task}-{split}-full.jsonl" for task in TASKS}
            if (set(ready.get("manifests", {})) != expected or
                    ready.get("render_config_sha256") != sha256_file(root / "render-config.json") or
                    ready.get("counts") != {task: config["tasks"][task]["count"] for task in TASKS}):
                raise ValueError("Published manifest readiness receipt differs from configuration")
            for name, digest in ready["manifests"].items():
                path = manifests / name
                if sha256_file(path) != digest:
                    raise ValueError(f"Modified published manifest: {path}")
                # Contact sheets are immutable inputs too, not merely existence checks.
                count = 0
                with path.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        if (len(row["images"]) != 1 or
                                sha256_file(manifests / row["images"][0]) != row["metadata"]["contact_sheet_sha256"]):
                            raise ValueError(f"Modified published contact sheet in {path}")
                        count += 1
                task = "idrules" if "-idrules-" in name else "rmts"
                if count != config["tasks"][task]["count"]:
                    raise ValueError(f"Published manifest count differs: {path}")
                provenance = json.loads(path.with_name(path.name + ".provenance.json").read_text())
                if (provenance.get("manifest_sha256") != digest or
                        provenance.get("selected_count") != count or
                        provenance.get("partial_render_pool") is not False or
                        provenance.get("official_label_file") is not True):
                    raise ValueError(f"Published manifest provenance differs: {path}")
            print(manifests / "READY.json")
            return
        # Publish the two splits as one directory. If interrupted, retain the
        # isolated attempt; a later invocation can safely make a fresh pair.
        staged = root / "manifest-attempts" / uuid.uuid4().hex
        staged.mkdir(parents=True)
        hashes = {}
        for task in TASKS:
            path = prepare_clevr_art(root, staged / f"clevr-art-{task}-{split}-full.jsonl", task,
                                     "ood", split, limit=None, seed=0)
            hashes[path.name] = sha256_file(path)
        helper._atomic_json(staged / "READY.json", {"schema_version": 1, "manifests": hashes,
                            "render_config_sha256": sha256_file(root / "render-config.json"),
                            "counts": {task: config["tasks"][task]["count"] for task in TASKS}})
        staged.rename(manifests)
        print(manifests / "READY.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Existing pinned assets; read-only")
    parser.add_argument("--root", type=Path, required=True, help="Separate full-render destination")
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, help="One CPU per worker; larger allocations require user approval")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=1050, help="Reserve at least 30 seconds before Slurm time limit")
    parser.add_argument("--max-new-groups-per-task", type=int, default=0, help="Calibration cap; 0 means all remaining")
    parser.add_argument("--pilot-root", type=Path, help="Optional read-only import of complete pilot groups")
    parser.add_argument("--adopt-legacy", action="store_true", help="Explicitly permit pilot groups lacking historical PNG hashes")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--build-manifests", action="store_true")
    args = parser.parse_args(argv)
    try:
        assets, root, blender = args.assets.resolve(), args.root.resolve(), args.blender.resolve()
        if root == assets or root in assets.parents or assets in root.parents:
            raise ValueError("Full output root and source asset root must be separate, non-nested directories")
        if args.workers < 1 or args.max_seconds <= 0 or args.max_new_groups_per_task < 0:
            raise ValueError("Workers/budget must be positive; calibration cap must be nonnegative")
        if args.adopt_legacy and not args.pilot_root:
            raise ValueError("--adopt-legacy requires --pilot-root")
        if args.run or args.build_manifests:
            allocation_guard(args.workers)
        config = settings(assets, blender, args.split, args.shard_index, args.shards)
        if args.run:
            render(root, config, args.workers, args.max_seconds, args.max_new_groups_per_task,
                   args.pilot_root.resolve() if args.pilot_root else None, args.adopt_legacy)
        elif args.build_manifests:
            build_manifests(root, config)
        else:
            print(json.dumps({"configuration": config, "output_root": str(root), "action": "planned"}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"CLEVR rendering failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
