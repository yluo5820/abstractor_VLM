#!/usr/bin/env python3
"""Verify all train-render shards and publish native-image indices without copying PNGs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import clevr_full as full


def validate_shards(configs, total=10000):
    if not configs:
        raise ValueError("No shard configurations")
    expected_shards = len(configs)
    for number, config in enumerate(configs):
        if (config.get("split") != "train" or config.get("shard_index") != number
                or config.get("shards") != expected_shards):
            raise ValueError("Missing, reordered, duplicate, or non-training shard")
        for key in ("assets", "blender_version", "blender_sha256", "renderer_helper_sha256", "orchestrator_sha256"):
            if config.get(key) != configs[0].get(key):
                raise ValueError(f"Inconsistent shard provenance: {key}")
    for task in full.TASKS:
        observed = []
        protocol = configs[0]["tasks"][task]["protocol_sha256"]
        for number, config in enumerate(configs):
            spec = config["tasks"][task]
            assigned = full.assigned_indices(total, number, expected_shards)
            if (spec["indices"] != assigned or spec["count"] != len(assigned)
                    or spec["protocol_sha256"] != protocol):
                raise ValueError("Shard assignment or render protocol differs")
            observed.extend(assigned)
        if sorted(observed) != list(range(total)):
            raise ValueError("Shard union is not the complete training split")


def immutable_file(path, staged):
    if path.exists():
        if full.sha256_file(path) != full.sha256_file(staged):
            raise ValueError(f"Existing published file differs: {path}")
        staged.unlink()
    else:
        staged.rename(path)


def index(root, shards):
    roots = [root / "shards" / str(i) for i in range(shards)]
    configs = [json.loads((p / "render-config.json").read_text()) for p in roots]
    validate_shards(configs, total=full.TOTAL)
    for path, config in zip(roots, configs):
        pending = full.inspect(path, config)
        if any(pending.values()):
            raise ValueError(f"Training shard is incomplete: {path}; resume its rendering job")
    manifests = root / "native-manifests"
    manifests.mkdir(exist_ok=True)
    files = {}
    for task, choices in full.TASKS.items():
        label_name = configs[0]["tasks"][task]["labels"]
        if label_name != ("RMTS" if task == "rmts" else "identity_rules") + "_ood_train.npz":
            raise ValueError("Refusing non-training labels in training index")
        expected = full.OFFICIAL_CLEVR_SHA256[label_name]
        label_source = Path(configs[0]["assets"]) / label_name
        if full.sha256_file(label_source) != expected:
            raise ValueError("Official training labels changed")
        with tempfile.NamedTemporaryFile(dir=manifests, delete=False) as temporary:
            staged_label = Path(temporary.name)
        shutil.copyfile(label_source, staged_label)
        immutable_file(root / label_name, staged_label)
        with np.load(label_source, allow_pickle=False) as labels:
            targets = labels["y"]
        if len(targets) != full.TOTAL:
            raise ValueError("Expected 10,000 training labels")
        name = f"clevr-art-{task}-train-native.jsonl"
        with tempfile.NamedTemporaryFile(mode="w", dir=manifests, delete=False) as stream:
            staged_manifest = Path(stream.name)
            for i, target in enumerate(targets):
                directory = full.scenes_path(roots[i % shards], task, "train") / f"prob_{i}"
                marker = json.loads((directory / full.MARKER).read_text())
                target = int(target)
                if not 0 <= target < choices:
                    raise ValueError("Invalid official training target")
                source_paths = [str((directory / f"CLEVR_{c}.png").relative_to(root)) for c in range(choices)]
                row = {"id": f"clevr-art/{task}/ood/train/{i}", "dataset": "clevr-art",
                       "task": task, "regime": "ood", "split": "train",
                       "choices": list("ABCD"[:choices]), "answer": chr(65 + target),
                       "metadata": {"presentation": "original-candidate-scenes",
                           "source_paths": source_paths,
                           "source_sha256": [marker["files"][f"CLEVR_{c}.png"] for c in range(choices)],
                           "source_image_size": [320, 240], "label_source": label_name,
                           "label_source_sha256": expected, "official_label_file": True,
                           "render_protocol_sha256": marker["protocol_sha256"]}}
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        immutable_file(manifests / name, staged_manifest)
        files[name] = full.sha256_file(manifests / name)
    ready = {"schema_version": 1, "split": "train", "questions": {task: full.TOTAL for task in full.TASKS},
             "scenes": full.TOTAL * sum(full.TASKS.values()), "manifests": files,
             "source_render_configs": {str(p.relative_to(root) / "render-config.json"):
                                       full.sha256_file(p / "render-config.json") for p in roots},
             "presentation": "native candidate images; no composite VLM training images generated yet"}
    path = manifests / "READY.json"
    if path.exists() and json.loads(path.read_text()) != ready:
        raise ValueError("Existing readiness record differs")
    full.helper._atomic_json(path, ready)
    print(json.dumps(ready, indent=2), flush=True)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    full.allocation_guard(1)
    if not 1 <= args.shards <= full.TOTAL:
        parser.error("Invalid shard count")
    root = args.root.resolve()
    with (root / ".index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index(root, args.shards)


if __name__ == "__main__":
    main()
