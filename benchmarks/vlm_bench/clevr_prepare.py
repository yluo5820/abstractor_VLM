"""Resumable CPU preparation of both complete, checkpointed CLEVR-ART tasks.

Each contact sheet and JSONL row is durable before moving to the next problem.
The public manifests become usable only after READY.json certifies both tasks.
No scenes, old attempts, model outputs or renderer settings are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
from PIL import Image

from . import data
from .pgm_full import _publish_bytes
from .run import atomic_json, interruption_signals, read_records, source_hash

TASKS = {"rmts": 2, "idrules": 4}
COUNT = 10000


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def make_row(root, task, index, spec, x, y, label_digest):
    directory = root / "output" / f"{task}_images/test_ood/prob_{index}"
    names = [f"CLEVR_{i}.png" for i in range(TASKS[task])]
    marker = directory / ".benchmark-complete.json"
    if directory.is_symlink() or marker.is_symlink():
        raise ValueError(f"Symlink render group: {directory}")
    receipt = json.loads(marker.read_text())
    if (receipt.get("protocol_sha256") != spec["protocol_sha256"] or
            receipt.get("index") != index or receipt.get("blender_version") != "Blender 2.79 (sub 0)" or
            set(p.name for p in directory.iterdir()) != set(names + [marker.name])):
        raise ValueError(f"Incomplete or mismatched render receipt: {directory}")
    hashes, scenes = [], []
    for name in names:
        path = directory / name
        if path.is_symlink():
            raise ValueError(f"Symlink render image: {path}")
        digest = data.sha256_file(path)
        if receipt.get("files", {}).get(name) != digest:
            raise ValueError(f"Rendered image checksum differs: {path}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (320, 240):
                raise ValueError(f"Unexpected rendered image: {path}")
            scenes.append(data._rgb(image))
        hashes.append(digest)
    target = int(y[index])
    data._validate_clevr_answer(x[index], target, task)
    label = spec["labels"]
    row = {
        "id": f"clevr-art/{task}/ood/test/{index}", "dataset": "clevr-art", "task": task,
        "regime": "ood", "split": "test", "prompt": data.CLEVR_PROMPTS[task],
        "choices": list("ABCD"[:TASKS[task]]), "answer": chr(65 + target),
        "metadata": {"source_paths": [str((directory / n).relative_to(root)) for n in names],
                     "source_sha256": hashes, "label_source": label,
                     "label_source_sha256": label_digest, "official_label_file": True,
                     "source_image_size": [320, 240], "presentation": "complete-candidate-scenes-2col-v1"},
    }
    return row, data.render_clevr(scenes)


def validate_ready(output, render_digest):
    ready = json.loads((output / "READY.json").read_text())
    names = {f"clevr-art-{task}-test-full.jsonl" for task in TASKS}
    if (ready.get("render_config_sha256") != render_digest or
            ready.get("counts") != dict.fromkeys(TASKS, COUNT) or set(ready.get("manifests", {})) != names):
        raise ValueError("CLEVR readiness receipt does not cover both complete tasks")
    for name, digest in ready["manifests"].items():
        if data.sha256_file(output / name) != digest:
            raise ValueError(f"Published manifest changed: {name}")
    return ready


def prepare(root, output, render_digest, max_seconds=3300, *, max_new=0):
    started = time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    if data.sha256_file(root / "render-config.json") != render_digest:
        raise ValueError("Render configuration checksum changed")
    config = json.loads((root / "render-config.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    if progress.get("status") != "complete" or progress.get("completed_groups") != dict.fromkeys(TASKS, COUNT):
        raise ValueError("Rendering has not completed both full tasks")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock, interruption_signals() as stop:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {"schema_version": 1, "render_root": str(root), "render_config_sha256": render_digest,
                "source_sha256": source_hash(), "seed": 0, "count_per_task": COUNT,
                "numpy_version": np.__version__, "pillow_version": Image.__version__}
        if not (output / "preparation-config.json").exists() and set(p.name for p in output.iterdir()) - {".lock"}:
            raise ValueError("Refusing existing outputs without a preparation plan")
        _publish_bytes(output / "preparation-config.json", encoded(plan))
        if (output / "READY.json").exists():
            return dict(status="complete", **validate_ready(output, render_digest))
        completed, hashes, new_count = {}, {}, 0

        def save(status):
            value = {"status": status, "completed": dict(completed), "expected": dict.fromkeys(TASKS, COUNT),
                     "job_id": os.environ.get("SLURM_JOB_ID"), "elapsed_seconds": time.monotonic() - started,
                     "new_examples": new_count}
            atomic_json(output / "preparation-progress.json", value)
            print(json.dumps(value), flush=True)
            return value

        for task in TASKS:
            spec = config["tasks"][task]
            if spec["count"] != COUNT or spec["candidates"] != TASKS[task]:
                raise ValueError("Unexpected render task cardinality")
            label_path = root / spec["labels"]
            label_digest = data.sha256_file(label_path)
            if label_digest != data.OFFICIAL_CLEVR_SHA256[spec["labels"]] or label_digest != spec["source_identity"]["labels_sha256"]:
                raise ValueError("Official symbolic label checksum differs")
            with np.load(label_path, allow_pickle=False) as labels:
                x, y = labels["x"], labels["y"]
            data._validate_clevr_arrays(x, y, task, "test")
            if len(y) != COUNT:
                raise ValueError("Expected the complete official test labels")
            scenes = root / "output" / f"{task}_images/test_ood"
            if {p.name for p in scenes.iterdir()} != {f"prob_{i}" for i in range(COUNT)}:
                raise ValueError("Render tree has missing or foreign groups")
            selected, pool = data.select_subset([(f"clevr-art/{task}/ood/test/{i}", i) for i in range(COUNT)], None, 0)
            manifest = output / f"clevr-art-{task}-test-full.jsonl"
            assets = output / f"{manifest.stem}_images"
            assets.mkdir(exist_ok=True)
            journal = output / f".{task}-rows.jsonl"
            rows = read_records(journal)
            if len(rows) > COUNT:
                raise ValueError("Preparation journal is longer than the dataset")
            completed[task] = len(rows)
            for position, row in enumerate(rows):
                if stop.signum or time.monotonic() - started >= max_seconds:
                    return save("resumable")
                identifier, index = selected[position]
                filename = hashlib.sha256(identifier.encode()).hexdigest()[:24] + ".png"
                expected_images = [f"{assets.name}/{filename}"]
                if (row.get("id") != identifier or row.get("answer") != chr(65 + int(y[index])) or
                        row.get("dataset") != "clevr-art" or row.get("task") != task or
                        row.get("regime") != "ood" or row.get("split") != "test" or
                        row.get("prompt") != data.CLEVR_PROMPTS[task] or row.get("choices") != list("ABCD"[:TASKS[task]]) or
                        row.get("images") != expected_images or
                        data.sha256_file(output / expected_images[0]) != row["metadata"]["contact_sheet_sha256"]):
                    raise ValueError("Preparation checkpoint was modified")
            completed[task] = len(rows)
            save("running")
            with journal.open("ab") as stream:
                for identifier, index in selected[len(rows):]:
                    if stop.signum or time.monotonic() - started >= max_seconds or (max_new and new_count >= max_new):
                        return save("resumable")
                    row, image = make_row(root, task, index, spec, x, y, label_digest)
                    filename = hashlib.sha256(identifier.encode()).hexdigest()[:24] + ".png"
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    payload = buffer.getvalue()
                    _publish_bytes(assets / filename, payload)
                    row["images"] = [f"{assets.name}/{filename}"]
                    row["metadata"].update(adapter_version=data.ADAPTER_VERSION,
                        contact_sheet_sha256=hashlib.sha256(payload).hexdigest(), contact_sheet_size=list(image.size),
                        selection_seed=0, available_pool_size=COUNT)
                    stream.write(encoded(row)); stream.flush(); os.fsync(stream.fileno())
                    rows.append(row); new_count += 1; completed[task] += 1
                    if new_count % 100 == 0:
                        save("running")
            _publish_bytes(manifest, b"".join(encoded(row) for row in rows))
            hashes[manifest.name] = data.sha256_file(manifest)
            provenance = {"dataset": "clevr-art", "task": task, "regime": "ood", "split": "test",
                "seed": 0, "requested_limit": None, "source_root": str(root),
                "upstream": "https://github.com/Shanka123/OCRA/tree/f15fa625d7e5b4458d82c24108b26419037d940d",
                "symbolic_examples": COUNT, "label_source_sha256": label_digest, "official_label_file": True,
                "partial_render_pool": False, "adapter_version": data.ADAPTER_VERSION,
                "selected_count": COUNT, "available_pool_size": COUNT,
                "pool_ids_sha256": hashlib.sha256("\n".join(sorted(i for i, _ in pool)).encode()).hexdigest(),
                "selection_method": "sha256(seed + NUL + canonical_id), ascending",
                "selected_ids": [r["id"] for r in rows], "answer_counts": dict(sorted(Counter(r["answer"] for r in rows).items())),
                "manifest_sha256": hashes[manifest.name], "numpy_version": np.__version__, "pillow_version": Image.__version__}
            _publish_bytes(output / (manifest.name + ".provenance.json"), (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())
        ready = {"schema_version": 1, "manifests": hashes, "counts": dict.fromkeys(TASKS, COUNT),
                 "render_config_sha256": render_digest}
        _publish_bytes(output / "READY.json", encoded(ready))
        return save("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-config-sha256", required=True)
    parser.add_argument("--max-seconds", type=float, default=3300)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
        parser.error("Preparation requires a Slurm compute allocation")
    try:
        if args.require_ready:
            print(json.dumps(validate_ready(args.output, args.render_config_sha256)))
        else:
            if not 0 < args.max_seconds <= 3300:
                raise ValueError("Preparation segments must reserve at least five minutes in a one-hour job")
            prepare(args.root, args.output, args.render_config_sha256, args.max_seconds)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"CLEVR preparation failed: {error}\n")


if __name__ == "__main__":
    main()
