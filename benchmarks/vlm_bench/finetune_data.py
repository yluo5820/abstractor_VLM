"""Pinned CLEVR training scenes and a fixed, test-independent development split."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLEVR_PROMPTS, OFFICIAL_CLEVR_SHA256, render_clevr, sha256_file
from .run import atomic_json


def partition_ids(task="idrules", total=10000, validation=1000, seed=20261007):
    if not 0 < validation < total:
        raise ValueError("Invalid validation size")
    ids = [f"clevr-art/{task}/ood/train/{i}" for i in range(total)]
    ids.sort(key=lambda item: hashlib.sha256(f"{seed}\0{item}".encode()).hexdigest())
    return ids[validation:], ids[:validation]


def verify_row(row, root, task="idrules"):
    root = Path(root).resolve()
    if (row["dataset"] != "clevr-art" or row["task"] != task or row["split"] != "train"
            or not row["id"].startswith(f"clevr-art/{task}/ood/train/")):
        raise ValueError("Only the official training split is permitted")
    meta = row["metadata"]
    name = ("identity_rules" if task == "idrules" else "RMTS") + "_ood_train.npz"
    choices = list("ABCD" if task == "idrules" else "AB")
    if (meta["label_source"] != name or meta["label_source_sha256"] != OFFICIAL_CLEVR_SHA256[name]
            or row["choices"] != choices or row["answer"] not in choices
            or len(meta["source_paths"]) != len(choices) or len(meta["source_sha256"]) != len(choices)):
        raise ValueError("Training provenance/target mismatch")
    for relative, checksum in zip(meta["source_paths"], meta["source_sha256"]):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != checksum:
            raise ValueError(f"Training image path/hash mismatch: {relative}")


def image_for(row, root):
    candidates = []
    for relative in row["metadata"]["source_paths"]:
        with Image.open(Path(root) / relative) as image:
            if image.size != (320, 240):
                raise ValueError("Unexpected native CLEVR dimensions")
            candidates.append(image.convert("RGB"))
    return render_clevr(candidates)


def partial_rows(root, requested, task="idrules"):
    """Small correctness pilot only; never report accuracy as a benchmark here."""
    root = Path(root)
    configs = [json.loads((root / "shards" / str(i) / "render-config.json").read_text()) for i in range(4)]
    name = "identity_rules_ood_train.npz" if task == "idrules" else "RMTS_ood_train.npz"
    labels = Path(configs[0]["assets"]) / name
    if sha256_file(labels) != OFFICIAL_CLEVR_SHA256[name]:
        raise ValueError("Modified official training labels")
    with np.load(labels, allow_pickle=False) as data:
        targets = data["y"]
    rows = []
    for identifier in requested:
        i = int(identifier.rsplit("/", 1)[1])
        config = configs[i % 4]
        if config["split"] != "train" or i not in config["tasks"][task]["indices"]:
            raise ValueError("Invalid training shard configuration")
        directory = root / "shards" / str(i % 4) / "output" / f"{task}_images/train_ood/prob_{i}"
        marker_path = directory / ".benchmark-complete.json"
        if not marker_path.exists():
            continue
        marker = json.loads(marker_path.read_text())
        if marker["protocol_sha256"] != config["tasks"][task]["protocol_sha256"] or marker["index"] != i:
            raise ValueError("Invalid completion receipt")
        count = 4 if task == "idrules" else 2
        row = {"id": identifier, "dataset": "clevr-art", "task": task, "split": "train", "regime": "ood",
               "choices": list("ABCD"[:count]), "answer": chr(65 + int(targets[i])),
               "metadata": {"label_source": name, "label_source_sha256": OFFICIAL_CLEVR_SHA256[name],
                   "source_paths": [str((directory / f"CLEVR_{c}.png").relative_to(root)) for c in range(count)],
                   "source_sha256": [marker["files"][f"CLEVR_{c}.png"] for c in range(count)]}}
        verify_row(row, root, task)
        rows.append(row)
        if len(rows) >= 64:
            break
    return rows


def prepare(root, output, pilot=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "split.json"
    train_ids, validation_ids = partition_ids()
    if path.exists():
        record = json.loads(path.read_text())
        if record["pilot"] != pilot:
            raise ValueError("Cannot change pilot/full split")
    else:
        if pilot:
            train = partial_rows(root, train_ids)[:64]
            validation = partial_rows(root, validation_ids)[:16]
            if len(train) != 64 or len(validation) != 16:
                raise ValueError("Need 64 train and 16 reserved validation renders for the pilot")
            provenance = {"source": "verified completed training-render receipts", "availability_selected_pilot_only": True}
        else:
            folder = root / "native-manifests"
            ready = json.loads((folder / "READY.json").read_text())
            name = "clevr-art-idrules-train-native.jsonl"
            if ready["split"] != "train" or sha256_file(folder / name) != ready["manifests"][name]:
                raise ValueError("Training manifest is not ready/unchanged")
            rows = [json.loads(line) for line in (folder / name).read_text().splitlines()]
            by_id = {row["id"]: row for row in rows}
            if len(rows) != 10000 or len(by_id) != 10000 or set(by_id) != set(train_ids + validation_ids):
                raise ValueError("Training manifest does not cover the full official split")
            train, validation = [by_id[i] for i in train_ids], [by_id[i] for i in validation_ids]
            provenance = {"native_manifest_sha256": ready["manifests"][name], "ready_sha256": sha256_file(folder / "READY.json")}
        record = {"schema": 1, "pilot": pilot, "seed": 20261007, "task": "idrules",
                  "prompt": CLEVR_PROMPTS["idrules"], "train": train, "validation": validation,
                  "provenance": provenance, "monitor_count": 16 if pilot else 128}
        atomic_json(path, record)
    actual = {part: [row["id"] for row in record[part]] for part in ("train", "validation")}
    if (not set(actual["train"]).issubset(train_ids) or not set(actual["validation"]).issubset(validation_ids)
            or set(actual["train"]) & set(actual["validation"])):
        raise ValueError("Train/validation overlap or changed split")
    for part in ("train", "validation"):
        if len(actual[part]) != len(set(actual[part])):
            raise ValueError("Duplicate training IDs")
        for row in record[part]:
            verify_row(row, root)
    return record
