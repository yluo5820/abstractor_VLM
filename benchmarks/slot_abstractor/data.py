"""Use original panels, never the VLM contact sheets or language prompts."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .model import digest


def preprocess(image, family):
    image = np.array(image, copy=True)
    if image.dtype != np.uint8:
        raise ValueError("Expected original uint8 image")
    if family == "pgm":
        if image.ndim != 2:
            raise ValueError("PGM must be grayscale")
        image = image[:, :, None]
        size = 80
    else:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("CLEVR must be RGB")
        size = 128
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
    tensor = (tensor - 0.5) / 0.5
    # Match torchvision 0.14.1 tensor Resize's original default, antialias=False.
    return F.interpolate(tensor[None], (size, size), mode="bilinear",
                         align_corners=False, antialias=False)[0]


def clevr_examples(root, manifest, task, start=0):
    root = Path(root).resolve()
    expected = 2 if task == "rmts" else 4
    seen = set()
    with Path(manifest).open() as stream:
        for offset, line in enumerate(stream):
            row = json.loads(line)
            if row["id"] in seen:
                raise ValueError("Duplicate CLEVR example")
            seen.add(row["id"])
            if not row["id"].startswith(f"clevr-art/{task}/ood/test/"):
                raise ValueError("Unexpected CLEVR task or split")
            if offset < start:
                continue
            meta = row["metadata"]
            paths, hashes = meta["source_paths"], meta["source_sha256"]
            if len(paths) != expected or len(hashes) != expected:
                raise ValueError("Incorrect CLEVR candidate count")
            panels = []
            for relative, checksum in zip(paths, hashes):
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or digest(path) != checksum:
                    raise ValueError(f"Source image path/hash mismatch: {relative}")
                with Image.open(path) as image:
                    panels.append(preprocess(image.convert("RGB"), "clevr"))
            target = ord(row["answer"]) - ord("A")
            if not 0 <= target < expected:
                raise ValueError("Invalid CLEVR label")
            yield row["id"], torch.stack(panels), target


def pgm_examples(paths, split="test", start=0):
    import pyarrow.parquet as pq
    # A bounded Arrow batch prevents loading gigabytes of encoded images at once.
    offset = 0
    for path in sorted(map(Path, paths)):
        if not path.name.startswith(split + "-"):
            raise ValueError(f"Unexpected split: {path}")
        parquet = pq.ParquetFile(path)
        if offset + parquet.metadata.num_rows <= start:
            offset += parquet.metadata.num_rows
            continue
        for batch in parquet.iter_batches(batch_size=16, columns=["id", "panels", "choices", "target"]):
            for row in batch.to_pylist():
                offset += 1
                if offset <= start:
                    continue
                raw = row["panels"] + row["choices"]
                if len(row["panels"]) != 8 or len(row["choices"]) != 8:
                    raise ValueError("PGM requires 8 context and 8 answer panels")
                panels = []
                for panel in raw:
                    with Image.open(io.BytesIO(panel["bytes"])) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    if rgb.shape != (160, 160, 3) or not (
                        np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and
                        np.array_equal(rgb[:, :, 0], rgb[:, :, 2])):
                        raise ValueError("Invalid PGM source panel")
                    panels.append(preprocess(rgb[:, :, 0], "pgm"))
                target = int(row["target"])
                if not 0 <= target < 8:
                    raise ValueError("Invalid PGM label")
                if type(row["id"]) is not int or row["id"] < 0:
                    raise ValueError("Invalid PGM example ID")
                yield f"pgm/neutral/{split}/{row['id']}", torch.stack(panels), target


def examples(args, start=0):
    if args.task == "pgm":
        return pgm_examples(args.parquet, start=start)
    return clevr_examples(args.data_root, args.manifest, args.task, start=start)
