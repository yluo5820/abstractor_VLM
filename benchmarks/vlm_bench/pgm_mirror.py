"""Prepare a disclosed PGM neutral/test subset from pinned HF mirror shards.

This module downloads nothing. Supply local test shards from HuggingFaceM4/PGM
at REVISION, then use the usual runner on the resulting canonical manifest.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .data import PGM_PROMPT, _write_manifest, render_pgm, select_subset, sha256_file

REPOSITORY = "HuggingFaceM4/PGM"
REVISION = "195a03a7e3bc6d6a349b55986afba5319929c3bf"
ORIGINAL_TEST_COUNT = 200000
# Hub LFS SHA256 values verified through the repository tree API. These identify
# the exact parquet bytes; they do not claim equality to original NPZ archives.
SHARD_SHA256 = {
    "test-00000-of-00009.parquet": "1270f82193c76e46872b79dfd73fc3b303d61c9777a9a8b4353fb261e0f257b6",
    "test-00001-of-00009.parquet": "b324364f95c4423c8b238cc57d3bfcecf9ba9fd121eeb56d303864ad1c57303c",
    "test-00002-of-00009.parquet": "b483f216ccb16e27f73290b544602770638a576ecc82e0b5d44e1986e52c0cf5",
    "test-00003-of-00009.parquet": "539433e82c5c0a57c921397455ea02dc3de00f0cbd73f5afdc67782009629664",
    "test-00004-of-00009.parquet": "75ba00cf11ad55850c4932821940e7da2db2ef97e3a612d180df2622a90961d7",
    "test-00005-of-00009.parquet": "b54b4e1f309c9459dbe951cc5948bf8b9495d65786ffff6f08fe1ff99bbb9046",
    "test-00006-of-00009.parquet": "a8d2174cb4c5dbf0a43e841c4db54cbdaf68724f24015441b9052983c1f73b09",
    "test-00007-of-00009.parquet": "9be6a3ab7b30ff4f3591d5067034c96609be9a1d71cc685fc811b909d699a48b",
    "test-00008-of-00009.parquet": "6ea46da021635cab7e603a5169dd3eb28e361844eaf2070915ca2b190f7c684c",
}


def decode_panels(row: dict) -> np.ndarray:
    """Decode the stored image bytes without changing order or using targets."""
    panels = []
    for key in ("panels", "choices"):
        images = row.get(key)
        if not isinstance(images, list) or len(images) != 8:
            raise ValueError(f"Mirror {key} must contain exactly eight images")
        for image in images:
            payload = image.get("bytes") if isinstance(image, dict) else None
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("Mirror image requires embedded bytes; external paths are unsupported")
            with Image.open(BytesIO(payload)) as source:
                if source.size != (160, 160):
                    raise ValueError(f"Unexpected mirror panel dimensions: {source.size}")
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
                    and np.array_equal(rgb[:, :, 0], rgb[:, :, 2])):
                raise ValueError("Expected grayscale PGM panels; refusing lossy grayscale conversion")
            panels.append(rgb[:, :, 0])
    return np.stack(panels)


def prepare_mirror(shards: list[Path], output: Path, limit: int = 16, seed: int = 0) -> Path:
    """Select from the supplied shard pool, with exact hashes and explicit IDs."""
    if limit < 1 or not shards:
        raise ValueError("Supply test shards and a positive subset limit")
    if len({path.name for path in shards}) != len(shards):
        raise ValueError("Duplicate shard filenames")
    for path in shards:
        if path.name not in SHARD_SHA256:
            raise ValueError(f"Not a pinned neutral/test shard: {path.name}")
    import pyarrow
    import pyarrow.parquet as pq

    sources, files, starts, items = [], {}, {}, []
    for path in sorted(shards):
        digest = sha256_file(path)
        if digest != SHARD_SHA256[path.name]:
            raise ValueError(f"Shard checksum differs from pinned mirror revision: {path}")
        parquet = pq.ParquetFile(path)
        files[path] = parquet
        offsets = [0]
        for group in range(parquet.num_row_groups):
            offsets.append(offsets[-1] + parquet.metadata.row_group(group).num_rows)
        starts[path] = offsets
        ids = parquet.read(columns=["id"]).column("id").to_pylist()
        if len(ids) != offsets[-1]:
            raise ValueError("Parquet metadata row count mismatch")
        for row_index, original_id in enumerate(ids):
            if type(original_id) is not int or original_id < 0:
                raise ValueError("Mirror IDs must be nonnegative integers")
            items.append((f"pgm/neutral/test/{original_id}", (path, row_index, original_id)))
        sources.append({"file": path.name, "sha256": digest, "rows": len(ids),
                        "bytes": path.stat().st_size})
    if len(items) > ORIGINAL_TEST_COUNT:
        raise ValueError("Mirror shard pool exceeds the declared test-set count")
    selected, pool = select_subset(items, limit, seed)

    selected_groups = defaultdict(list)
    for identifier, reference in selected:
        path, row_index, original_id = reference
        group = bisect_right(starts[path], row_index) - 1
        selected_groups[path, group].append((identifier, row_index - starts[path][group]))
    selected_rows = {}
    for (path, group), members in selected_groups.items():
        # Project columns before conversion: malformed rule_structure metadata
        # in this mirror is irrelevant and never materialized or given to a VLM.
        table = files[path].read_row_group(group, columns=["panels", "choices", "target", "id"])
        # Read each selected group once, then retain bytes for selected rows only.
        # take() occurs before Python materialization, avoiding a full-group cache.
        chosen = table.take([offset for _, offset in members]).to_pylist()
        selected_rows.update((identifier, raw) for (identifier, _), raw in zip(members, chosen))
        del table, chosen

    def load(identifier, reference):
        path, row_index, original_id = reference
        raw = selected_rows.pop(identifier)
        if raw["id"] != original_id or type(raw["target"]) is not int or not 0 <= raw["target"] < 8:
            raise ValueError("Mirror ID/target mismatch or target outside 0..7")
        panels = decode_panels(raw)
        row = {"id": identifier, "dataset": "pgm", "task": "matrix-completion",
               "regime": "neutral", "split": "test", "prompt": PGM_PROMPT,
               "choices": list("ABCDEFGH"), "answer": chr(65 + raw["target"]),
               "metadata": {"source_kind": "third-party-converted-mirror",
                            "mirror_repository": REPOSITORY, "mirror_revision": REVISION,
                            "mirror_shard": path.name, "mirror_row_index": row_index,
                            "mirror_sample_id": original_id, "source_image_shape": list(panels.shape),
                            "presentation": "matrix-3x3-options-2x4-v1"}}
        return row, render_pgm(panels)

    return _write_manifest(Path(output), selected, pool, load, {
        "dataset": "pgm", "regime": "neutral", "split": "test", "seed": seed,
        "requested_limit": limit, "source_kind": "third-party-converted-mirror",
        "mirror_repository": REPOSITORY, "mirror_revision": REVISION,
        "mirror_url": f"https://huggingface.co/datasets/{REPOSITORY}/tree/{REVISION}",
        "declared_full_test_count": ORIGINAL_TEST_COUNT,
        "available_shard_count": len(sources), "shards": sources,
        "selection_scope": "supplied shards only; not necessarily the full test-set population",
        "original_npz_equivalence": "not independently verified; mirror image bytes decoded as stored",
        "data_terms": "Original PGM data: non-commercial research only",
        "upstream": "https://github.com/google-deepmind/abstract-reasoning-matrices",
        "pyarrow_version": pyarrow.__version__,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        result = prepare_mirror(args.parquet, args.output, args.limit, args.seed)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
