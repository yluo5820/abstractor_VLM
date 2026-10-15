"""Resumable native-input evaluation; requires a trained reasoning checkpoint."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import sqlite3
import time

import torch

from .data import examples
from .dataset_lock import validate_sources
from .model import NativeModel, SPECS, digest, require_compute, seed_all, unwrapped_state, UPSTREAM_COMMIT
from .pilot import atomic_json


def trained_state(path, provenance, task):
    metadata = json.loads(Path(provenance).read_text())
    family = "pgm" if task == "pgm" else "clevr"
    expected = {"task": task, "regime": "neutral" if task == "pgm" else "ood",
                "training_split": "train", "checkpoint_sha256": digest(path),
                "encoder_sha256": SPECS[family]["encoder_sha256"]}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Reasoning checkpoint provenance mismatch: {key}")
    if not metadata.get("training_description") or metadata.get("checkpoint_selection") not in (
            "validation", "final-epoch", "author-provided"):
        raise ValueError("Describe training and checkpoint selection; test-selected checkpoints are disallowed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "abstractor_model_state_dict" not in checkpoint:
        raise ValueError("This is not a reasoning checkpoint; a slot encoder alone cannot be benchmarked")
    return unwrapped_state(checkpoint["abstractor_model_state_dict"]), metadata


class Results:
    """Atomic batches make interruptions safe without truncating earlier predictions."""
    def __init__(self, path, config):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS predictions (ordinal INTEGER PRIMARY KEY, id TEXT UNIQUE, target INTEGER, prediction INTEGER, logits TEXT)")
        encoded = json.dumps(config, sort_keys=True)
        old = self.db.execute("SELECT value FROM config WHERE id=1").fetchone()
        if old and old[0] != encoded:
            self.db.close()
            raise ValueError("Resume configuration differs; use a new output directory")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (encoded,))
        count, maximum = self.db.execute("SELECT COUNT(*), MAX(ordinal) FROM predictions").fetchone()
        if count and maximum != count - 1:
            raise ValueError("Noncontiguous saved results")

    @property
    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def append(self, items, logits):
        if logits.ndim != 2 or len(items) != len(logits) or not torch.isfinite(logits).all():
            raise ValueError("Prediction batch shape/values are invalid")
        offset = self.count
        rows = []
        for i, ((example_id, _, target), scores) in enumerate(zip(items, logits.tolist())):
            prediction = max(range(len(scores)), key=scores.__getitem__)
            rows.append((offset + i, example_id, target, prediction, json.dumps(scores)))
        with self.db:
            self.db.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", rows)

    def summary(self):
        total, correct = self.db.execute("SELECT COUNT(*), SUM(target=prediction) FROM predictions").fetchone()
        return {"evaluated": total, "correct": correct or 0,
                "accuracy_percent": 100 * correct / total if total else None}

    def export(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as stream:
            for ordinal, example_id, target, prediction, logits in self.db.execute(
                    "SELECT * FROM predictions ORDER BY ordinal"):
                row = {"ordinal": ordinal, "id": example_id, "answer": chr(65 + target),
                       "prediction": chr(65 + prediction), "correct": target == prediction,
                       "logits": json.loads(logits)}
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def batch_seed(seed, ordinal):
    raw = hashlib.sha256(f"slot-abstractor-eval-v1:{seed}:{ordinal}".encode()).digest()
    return int.from_bytes(raw[:4], "big")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["pgm", "rmts", "idrules"], required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-provenance", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--parquet", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=3300)
    args = parser.parse_args()
    if args.batch_size < 1 or args.seconds < 1:
        parser.error("batch-size and seconds must be positive")
    if args.task == "pgm" and not args.parquet or args.task != "pgm" and not (args.manifest and args.data_root):
        parser.error("Provide native Parquet files for PGM, or manifest and original scene root for CLEVR")
    # Fail early on missing heads before starting any GPU inference.
    state, metadata = trained_state(args.checkpoint, args.checkpoint_provenance, args.task)
    require_compute()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = sorted(args.parquet) if args.task == "pgm" else [args.manifest]
    config = {"protocol": "slot-abstractor-native-v1", "task": args.task,
              "upstream_commit": UPSTREAM_COMMIT, "checkpoint": metadata,
              "sources": validate_sources(args.task, sources),
              "runner": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
              "batch_size": args.batch_size, "seed": args.seed, "torch": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(), "dtype": "float32", "tf32": False,
              "slot_sampling": "fresh seeded draws per fixed batch; no voting",
              "resize": "bilinear align_corners=False antialias=False"}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # A kernel-held lock is released on process death; prevents simultaneous writers.
    import fcntl
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = Results(output / "predictions.sqlite3", config)
    total_expected = 200000 if args.task == "pgm" else 10000
    if args.task == "pgm":
        import pyarrow.parquet as pq
        source_count = sum(pq.ParquetFile(p).metadata.num_rows for p in sources)
    else:
        with open(args.manifest) as stream:
            source_count = sum(1 for line in stream if line.strip())
    if records.count > source_count:
        raise ValueError("More predictions than inputs")
    model = NativeModel(args.upstream, args.encoder, args.task, "cuda")
    model.head.load_state_dict(state, strict=True)
    model.eval()
    stop = []
    for event in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(event, lambda *_: stop.append(True))
    start = time.monotonic()
    initial = records.count
    stream = examples(args, start=initial)
    with torch.no_grad():
        while records.count < source_count and not stop and time.monotonic() - start < args.seconds:
            offset = records.count
            items = list(itertools.islice(stream, args.batch_size))
            if not items:
                raise ValueError("Input iterator ended before expected count")
            seed_all(batch_seed(args.seed, offset))
            logits = model(torch.stack([row[1] for row in items]))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite model outputs")
            records.append(items, logits.cpu())
            result = records.summary()
            result.update({"expected_full_test_count": total_expected, "input_count": source_count,
                           "complete": records.count == source_count == total_expected,
                           "elapsed_seconds_this_session": time.monotonic() - start,
                           "examples_this_session": records.count - initial,
                           "job_id": os.environ["SLURM_JOB_ID"]})
            atomic_json(output / "summary.json", result)
            if records.count % (args.batch_size * 100) == 0:
                print(json.dumps(result), flush=True)
    stream.close()
    result = records.summary()
    result.update({"expected_full_test_count": total_expected, "input_count": source_count,
                   "complete": records.count == source_count == total_expected,
                   "elapsed_seconds_this_session": time.monotonic() - start,
                   "examples_this_session": records.count - initial,
                   "job_id": os.environ["SLURM_JOB_ID"]})
    atomic_json(output / "summary.json", result)
    records.export(output / "predictions.jsonl")
    print(json.dumps(result), flush=True)
    records.db.close()


if __name__ == "__main__":
    main()
