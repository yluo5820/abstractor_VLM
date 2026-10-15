"""Bounded, resumable preparation of the complete pinned PGM neutral/test mirror.

Use a new output path, with all nine local parquet shards. The hidden sibling
.<output-name>.preparing directory is an owned checkpoint and must be retained
until preparation finishes. --resume verifies its plan and completed PNG hashes.
The CLI requires a Slurm compute allocation; this module downloads nothing.
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
import re
import socket
import sqlite3
import tempfile
import time

import numpy as np
from PIL import Image

from . import data, pgm_mirror as mirror
from .data import PGM_PROMPT, render_pgm, select_subset, sha256_file


STREAM_VERSION = 1
COLUMNS = ["panels", "choices", "target", "id"]


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _publish_bytes(path: Path, payload: bytes) -> None:
    """Create atomically, or accept exactly matching bytes; never replace a file."""
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic-link output: {path}")
    if path.exists():
        if sha256_file(path) != _hash(payload):
            raise ValueError(f"Existing output differs from this preparation: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".pgm-write-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or sha256_file(path) != _hash(payload):
                raise ValueError(f"Concurrent conflicting output: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _link_verified(source: Path, target: Path) -> None:
    if target.is_symlink():
        raise ValueError(f"Refusing symbolic-link output: {target}")
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise ValueError(f"Published output differs from checkpoint: {target}")
    else:
        os.link(source, target)


def _scan(shards: list[Path], limit: int | None, seed: int):
    import pyarrow.parquet as pq

    if len(shards) != len(mirror.SHARD_SHA256) or {p.name for p in shards} != set(mirror.SHARD_SHA256):
        raise ValueError("Full preparation requires each of the nine pinned test shards exactly once")
    sources, files, items = [], {}, []
    for path in sorted(shards, key=lambda p: p.name):
        digest = sha256_file(path)
        if digest != mirror.SHARD_SHA256[path.name]:
            raise ValueError(f"Shard checksum differs from pinned mirror revision: {path}")
        parquet = pq.ParquetFile(path)
        files[path] = parquet
        ids = parquet.read(columns=["id"]).column("id").to_pylist()
        count = sum(parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups))
        if len(ids) != count:
            raise ValueError("Parquet metadata row count mismatch")
        for row_index, original_id in enumerate(ids):
            if type(original_id) is not int or original_id < 0:
                raise ValueError("Mirror IDs must be nonnegative integers")
            items.append((f"pgm/neutral/test/{original_id}", (path, row_index, original_id)))
        sources.append({"file": path.name, "sha256": digest, "rows": len(ids), "bytes": path.stat().st_size})
    if len(items) != mirror.ORIGINAL_TEST_COUNT:
        raise ValueError(f"Expected {mirror.ORIGINAL_TEST_COUNT} rows in the complete mirror; found {len(items)}")
    selected, pool = select_subset(items, limit, seed)
    return sources, files, selected, pool


def _make_row(identifier, path, row_index, original_id, raw, seed, pool_size):
    if raw["id"] != original_id or type(raw["target"]) is not int or not 0 <= raw["target"] < 8:
        raise ValueError("Mirror ID/target mismatch or target outside 0..7")
    panels = mirror.decode_panels(raw)
    row = {"id": identifier, "dataset": "pgm", "task": "matrix-completion",
           "regime": "neutral", "split": "test", "prompt": PGM_PROMPT,
           "choices": list("ABCDEFGH"), "answer": chr(65 + raw["target"]),
           "metadata": {"source_kind": "third-party-converted-mirror",
                        "mirror_repository": mirror.REPOSITORY, "mirror_revision": mirror.REVISION,
                        "mirror_shard": path.name, "mirror_row_index": row_index,
                        "mirror_sample_id": original_id, "source_image_shape": list(panels.shape),
                        "presentation": "matrix-3x3-options-2x4-v1",
                        "adapter_version": data.ADAPTER_VERSION,
                        "selection_seed": seed, "available_pool_size": pool_size}}
    return row, render_pgm(panels)


def prepare_full(shards: list[Path], output: Path, seed: int = 0, batch_size: int = 16,
                 resume: bool = False, limit: int | None = None, progress=None) -> Path:
    """Keep only ID/reference metadata plus one projected Arrow batch in memory.

    Source hashes are rechecked on every invocation. Completed images are hashed
    on resume, then skipped without rereading their embedded source-image bytes.
    A kill between PNG publication and SQLite commit is safe: that PNG is reused
    only when it matches the freshly rendered bytes exactly.
    """
    import pyarrow

    output = Path(output).absolute()
    if output.suffix != ".jsonl" or batch_size < 1 or batch_size > 256:
        raise ValueError("Use a .jsonl output and batch size in 1..256")
    shards = [Path(p) for p in shards]
    assets = output.with_name(output.stem + "_images")
    sidecar = output.with_name(output.name + ".provenance.json")
    stage = output.with_name("." + output.name + ".preparing")
    if any(p.is_symlink() for p in (output, assets, sidecar, stage)):
        raise ValueError("Refusing symbolic-link output/checkpoint paths")
    if stage.exists() and not resume:
        raise FileExistsError(f"Checkpoint exists; pass --resume: {stage}")
    if not stage.exists() and any(p.exists() for p in (output, assets, sidecar)):
        raise FileExistsError("Refusing existing output without its preparation checkpoint")

    sources, files, selected, pool = _scan(shards, limit, seed)
    identifiers = [identifier for identifier, _ in selected]
    filenames = [hashlib.sha256(i.encode()).hexdigest()[:24] + ".png" for i in identifiers]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Contact-sheet filename collision")
    plan = {"stream_version": STREAM_VERSION, "output_name": output.name,
            "adapter_version": data.ADAPTER_VERSION, "seed": seed, "requested_limit": limit,
            "selected_count": len(selected), "available_pool_size": len(pool),
            "selected_ids_sha256": _hash("\n".join(identifiers).encode()),
            "pool_ids_sha256": _hash("\n".join(sorted(i for i, _ in pool)).encode()),
            "mirror_repository": mirror.REPOSITORY, "mirror_revision": mirror.REVISION,
            "shards": sources, "numpy_version": np.__version__, "pillow_version": Image.__version__,
            "pyarrow_version": pyarrow.__version__,
            "protocol_sources": {Path(module.__file__).name: sha256_file(Path(module.__file__))
                                 for module in (data, mirror)},
            "preparer_sha256": sha256_file(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.exists():
        stage.mkdir()
        _publish_bytes(stage / "plan.json", _json(plan))
    elif not (stage / "plan.json").is_file() or json.loads((stage / "plan.json").read_text()) != plan:
        raise ValueError("Checkpoint plan differs: sources, selection, protocol or library versions changed")

    with (stage / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is preparing this output") from error
        image_dir = stage / assets.name
        if assets.exists():
            if image_dir.exists():
                raise ValueError("Conflicting staged and published image directories")
            image_dir = assets
        else:
            image_dir.mkdir(exist_ok=True)
        database = sqlite3.connect(stage / "checkpoint.sqlite3")
        try:
            database.execute("PRAGMA synchronous=FULL")
            database.execute("CREATE TABLE IF NOT EXISTS completed (rank INTEGER PRIMARY KEY, "
                             "identifier TEXT UNIQUE, record TEXT, record_sha TEXT, image_sha TEXT)")
            database.commit()
            done = set()
            for rank, identifier, record, record_sha, image_sha in database.execute("SELECT * FROM completed"):
                if not 0 <= rank < len(selected) or identifier != identifiers[rank] or _hash(record.encode()) != record_sha:
                    raise ValueError("Checkpoint record differs from the selected plan")
                target = image_dir / filenames[rank]
                if target.is_symlink() or not target.is_file() or sha256_file(target) != image_sha:
                    raise ValueError(f"Completed image missing or modified: {target}")
                done.add(rank)
            if assets.exists() and len(done) != len(selected):
                raise ValueError("Published image directory has an incomplete checkpoint")
            by_source = {(path.name, index): (rank, identifier, original_id)
                         for rank, (identifier, (path, index, original_id)) in enumerate(selected)
                         if rank not in done}
            if progress:
                progress(len(done), len(selected))
            for path, parquet in files.items():
                group_start = 0
                for group in range(parquet.num_row_groups):
                    group_rows = parquet.metadata.row_group(group).num_rows
                    if not any((path.name, i) in by_source for i in range(group_start, group_start + group_rows)):
                        group_start += group_rows
                        continue
                    row_index = group_start
                    for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group], columns=COLUMNS,
                                                      use_threads=False):
                        raw_rows = batch.to_pylist()
                        for raw in raw_rows:
                            reference = by_source.get((path.name, row_index))
                            if reference is not None:
                                rank, identifier, original_id = reference
                                row, image = _make_row(identifier, path, row_index, original_id, raw, seed, len(pool))
                                payload = BytesIO()
                                image.save(payload, format="PNG")
                                png = payload.getvalue()
                                target = image_dir / filenames[rank]
                                _publish_bytes(target, png)
                                row["images"] = [f"{assets.name}/{filenames[rank]}"]
                                row["metadata"].update({"contact_sheet_sha256": _hash(png),
                                                        "contact_sheet_size": list(image.size)})
                                record = _json(row).decode()
                                database.execute("INSERT INTO completed VALUES (?, ?, ?, ?, ?)",
                                                 (rank, identifier, record, _hash(record.encode()), _hash(png)))
                                done.add(rank)
                            row_index += 1
                        database.commit()
                        if progress:
                            progress(len(done), len(selected))
                        del raw_rows, batch
                    if row_index != group_start + group_rows:
                        raise ValueError("Parquet batch row count mismatch")
                    group_start += group_rows
            if len(done) != len(selected):
                raise ValueError("Incomplete preparation; checkpoint retained")

            # Export in the exact pilot hash-ranking order, independent of physical
            # shard/row-group traversal. SQLite keeps full records off the Python heap.
            manifest = stage / output.name
            labels = Counter()
            fd, temporary = tempfile.mkstemp(prefix=".pgm-manifest-", dir=stage)
            temporary = Path(temporary)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    for (record,) in database.execute("SELECT record FROM completed ORDER BY rank"):
                        stream.write(record)
                        labels[json.loads(record)["answer"]] += 1
                    stream.flush()
                    os.fsync(stream.fileno())
                _link_verified(temporary, manifest)
            finally:
                temporary.unlink(missing_ok=True)
            provenance = {
                **{k: v for k, v in plan.items() if k not in ("output_name", "selected_ids_sha256")},
                "dataset": "pgm", "regime": "neutral", "split": "test",
                "source_kind": "third-party-converted-mirror",
                "mirror_url": f"https://huggingface.co/datasets/{mirror.REPOSITORY}/tree/{mirror.REVISION}",
                "declared_full_test_count": mirror.ORIGINAL_TEST_COUNT,
                "available_shard_count": len(sources),
                "selection_scope": "complete pinned neutral/test mirror population",
                "original_npz_equivalence": "not independently verified; mirror image bytes decoded as stored",
                "data_terms": "Original PGM data: non-commercial research only",
                "upstream": "https://github.com/google-deepmind/abstract-reasoning-matrices",
                "selection_method": "sha256(seed + NUL + canonical_id), ascending",
                "selected_ids": identifiers, "answer_counts": dict(sorted(labels.items())),
                "manifest_sha256": sha256_file(manifest),
                "preparation_method": "projected Arrow batches with SQLite checkpoints",
            }
            staged_sidecar = stage / sidecar.name
            _publish_bytes(staged_sidecar, (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())
            if image_dir != assets:
                if assets.exists():
                    raise FileExistsError(f"Refusing existing image output: {assets}")
                image_dir.rename(assets)
            _link_verified(staged_sidecar, sidecar)
            _link_verified(manifest, output)
            return output
        finally:
            database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Optional nested subset; all nine source shards still required")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    last_reported = [-1000]
    started = [None, None]

    def progress(done, total):
        if started[0] is None:
            started[:] = [time.monotonic(), done]
        if done == total or done - last_reported[0] >= 1000:
            elapsed = time.monotonic() - started[0]
            rate = (done - started[1]) / elapsed if elapsed else 0
            print(json.dumps({"prepared": done, "total": total,
                              "elapsed_seconds": round(elapsed, 1),
                              "examples_per_second": round(rate, 3),
                              "eta_seconds": round((total - done) / rate) if rate else None}), flush=True)
            last_reported[0] = done

    try:
        if not os.environ.get("SLURM_JOB_ID") or re.search(r"login|headnode", socket.gethostname(), re.I):
            raise RuntimeError("Full image preparation requires a Slurm compute allocation")
        result = prepare_full(args.parquet, args.output, args.seed, args.batch_size, args.resume, args.limit, progress)
    except (ValueError, OSError, ImportError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
