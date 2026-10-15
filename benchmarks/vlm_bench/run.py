"""Slurm-only inference with immutable inputs and resumable, strict scoring."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import sys
import time

from PIL import Image

PARSER_VERSION = "strict-letter-v1"
GROUP_FIELDS = ("dataset", "task", "regime", "split")


class AllocationInterrupted(RuntimeError):
    """A termination request honored after the current prediction is durable."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"Received {signal.Signals(signum).name}; committed predictions can be resumed")


class StopRequest:
    def __init__(self):
        self.signum = None

    def handle(self, signum, _frame):
        # Never raise between write/flush/fsync and the in-memory commit. Slurm's
        # early signal should allow the in-flight prediction to finish first.
        if self.signum is None:
            self.signum = signum

    def check(self):
        if self.signum is not None:
            raise AllocationInterrupted(self.signum)


@contextmanager
def interruption_signals():
    request = StopRequest()
    previous = {}
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, request.handle)
        yield request
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_manifest(path: Path, limit: int = 0) -> list[dict]:
    """Validate every row, then select an ordered prefix of an already seeded manifest."""
    rows, seen = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "dataset", "task", "regime", "split", "prompt"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Line {line_number}: missing/non-string {key}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate example id: {row['id']}")
        seen.add(row["id"])
        choices = row.get("choices")
        if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
            raise ValueError(f"{row['id']}: choices must contain 2–26 labels")
        if choices != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choices)]):
            raise ValueError(f"{row['id']}: expected contiguous labels A, B, ...")
        if row.get("answer") not in choices:
            raise ValueError(f"{row['id']}: answer not among choices")
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(f"{row['id']}: protocol requires exactly one contact sheet")
        if not isinstance(images[0], str) or Path(images[0]).is_absolute():
            raise ValueError(f"{row['id']}: image path must be relative to the manifest")
        image_path = (path.parent / images[0]).resolve()
        if not image_path.is_relative_to(path.parent.resolve()):
            raise ValueError(f"{row['id']}: image escapes manifest directory")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        row["_image_path"] = image_path
        rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows[:limit] if limit else rows


def parse_answer(raw: str, choices: list[str]) -> str | None:
    """Never extract an arbitrary letter from prose or a list of possible answers."""
    match = re.fullmatch(r"\s*(?:Answer\s*:\s*)?\(?([A-Z])\)?[.]?\s*", raw, re.I)
    answer = match.group(1).upper() if match else None
    return answer if answer in choices else None


def wilson(correct: int, total: int) -> list[float] | None:
    if not total:
        return None
    p, z = correct / total, 1.959963984540054
    divisor = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / divisor
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return [max(0.0, middle - half), min(1.0, middle + half)]


class SummaryAccumulator:
    """Build group totals once; each durable prediction costs constant work."""

    def __init__(self, planned: list[dict]):
        self.planned = len(planned)
        self.evaluated = 0
        self.groups = defaultdict(lambda: {"planned": 0, "evaluated": 0,
                                          "correct": 0, "invalid": 0,
                                          "latency": 0.0, "chance": 0.0})
        for row in planned:
            self.groups[tuple(row[k] for k in GROUP_FIELDS)]["planned"] += 1

    def add(self, record: dict) -> None:
        group = self.groups[tuple(record[k] for k in GROUP_FIELDS)]
        group["evaluated"] += 1
        group["correct"] += record["correct"]
        group["invalid"] += record["prediction"] is None
        group["latency"] += record["latency_seconds"]
        group["chance"] += 1 / len(record["choices"])
        self.evaluated += 1

    def snapshot(self) -> dict:
        metrics = []
        for key, group in sorted(self.groups.items()):
            n, correct = group["evaluated"], group["correct"]
            metrics.append({
                **dict(zip(GROUP_FIELDS, key)), "planned": group["planned"], "evaluated": n,
                "correct": correct, "accuracy": correct / n if n else None,
                "wilson_95_interval": wilson(correct, n), "invalid_outputs": group["invalid"],
                "invalid_rate": group["invalid"] / n if n else None,
                "chance_accuracy": group["chance"] / n if n else None,
                "mean_latency_seconds": group["latency"] / n if n else None,
            })
        return {"planned": self.planned, "evaluated": self.evaluated,
                "complete": self.evaluated == self.planned, "groups": metrics,
                "interval_note": "Wilson binomial intervals; correlated examples may violate independence."}


def summarize(records: list[dict], planned: list[dict]) -> dict:
    """Invalid outputs stay in the denominator; incomplete runs expose coverage."""
    totals = SummaryAccumulator(planned)
    for record in records:
        totals.add(record)
    return totals.snapshot()


def runtime_progress(session_start: float, evaluation_start: float, evaluated: int,
                     already_completed: int, planned: int, now: float | None = None) -> dict:
    """Operational timings for this process only; never reuse a previous session's rate."""
    now = time.monotonic() if now is None else now
    processed = evaluated - already_completed
    session_elapsed = max(0.0, now - session_start)
    evaluation_elapsed = max(0.0, now - evaluation_start)
    rate = processed / evaluation_elapsed if processed and evaluation_elapsed else None
    remaining = planned - evaluated
    return {
        "completed_this_session": processed,
        "session_elapsed_seconds": session_elapsed,
        "startup_seconds": evaluation_start - session_start,
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "end_to_end_examples_per_second": processed / session_elapsed if processed and session_elapsed else None,
        "evaluation_examples_per_second": rate,
        "remaining_examples": remaining,
        "estimated_remaining_seconds": remaining / rate if rate else (0.0 if not remaining else None),
        "estimate_basis": "Current-session evaluation-loop rate, including image reads, inference, durable writes, summaries and logging; excludes startup, future queueing and restarts. This is a projection, not a guarantee.",
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def require_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Inference requires a Slurm compute allocation; use cluster/ice.py.")
    if re.search(r"login|headnode", socket.gethostname(), re.I):
        raise RuntimeError("Refusing inference on a login node, even with SLURM_JOB_ID set.")
    if not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")
            or os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("No GPU allocation is visible in the Slurm environment.")


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    # A killed job may leave a partial final JSON line. Preserve it for inspection,
    # then retry just that example. Never silently discard complete corrupt records.
    if data and not data.endswith(b"\n"):
        offset = data.rfind(b"\n") + 1
        partial_path = path.with_name(path.name + ".partial-" + str(time.time_ns()))
        partial_path.write_bytes(data[offset:])
        with path.open("r+b") as stream:
            stream.truncate(offset)
        data = data[:offset]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def validate_records(records: list[dict], rows: list[dict]) -> None:
    """Reject edited/stale outcomes before trusting any completed prediction."""
    planned = {row["id"]: row for row in rows}
    completed = set()
    for record in records:
        example_id = record.get("id") if isinstance(record, dict) else None
        if not isinstance(example_id, str) or example_id not in planned or example_id in completed:
            raise ValueError("Duplicate or unexpected ids in saved predictions")
        completed.add(example_id)
        row = planned[example_id]
        for key in ("dataset", "task", "regime", "split", "choices", "answer"):
            if record.get(key) != row[key]:
                raise ValueError(f"Saved prediction {example_id}: {key} differs from manifest")
        raw = record.get("raw_output")
        if not isinstance(raw, str):
            raise ValueError(f"Saved prediction {example_id}: raw_output must be text")
        parsed = parse_answer(raw, row["choices"])
        if "prediction" not in record or record["prediction"] != parsed:
            raise ValueError(f"Saved prediction {example_id}: prediction differs from strict parser")
        if type(record.get("correct")) is not bool or record["correct"] != (parsed == row["answer"]):
            raise ValueError(f"Saved prediction {example_id}: correctness differs from manifest/parser")
        elapsed = record.get("latency_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"Saved prediction {example_id}: latency must be finite and nonnegative")


def run(args: argparse.Namespace) -> dict:
    with interruption_signals() as stop:
        return _run(args, stop)


def _run(args: argparse.Namespace, stop: StopRequest) -> dict:
    session_start = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    summary_every = getattr(args, "summary_every", 25)
    cpu_threads = getattr(args, "cpu_threads", 2)
    interop_threads = getattr(args, "interop_threads", 1)
    if type(summary_every) is not int or summary_every < 1:
        raise ValueError("summary-every must be a positive integer")
    if any(type(value) is not int or value < 1 for value in (cpu_threads, interop_threads)):
        raise ValueError("cpu-threads and interop-threads must be positive integers")
    manifest = Path(args.manifest).resolve()
    rows = load_manifest(manifest, args.max_samples)
    stop.check()
    if args.dry_run:
        result = {"mode": "input-validation-only", "examples": len(rows),
                  "manifest_sha256": sha256(manifest), "model": args.model,
                  "revision": args.revision, "first_id": rows[0]["id"]}
        print(json.dumps(result, indent=2))
        return result
    require_allocation()
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus and max(cpu_threads, interop_threads) > int(allocated_cpus):
        raise ValueError("Requested torch threads exceed SLURM_CPUS_PER_TASK")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision or ""):
        raise ValueError("Use an immutable 40-character model commit SHA (--revision).")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing this run directory") from error
        image_digest = hashlib.sha256()
        for row in rows:
            stop.check()
            image_digest.update(row["id"].encode())
            image_digest.update(sha256(row["_image_path"]).encode())
        config = {
            "protocol": "frozen-zero-shot-single-contact-sheet-v1",
            "model": args.model, "revision": args.revision.lower(),
            "manifest_sha256": sha256(manifest), "images_sha256": image_digest.hexdigest(),
            "source_sha256": source_hash(), "parser": PARSER_VERSION,
            "max_new_tokens": args.max_new_tokens, "dtype": args.dtype,
            "seed": args.seed, "max_samples": args.max_samples,
            "cpu_threads": cpu_threads, "interop_threads": interop_threads,
            "tokenizers_parallelism": False,
            "example_ids": [row["id"] for row in rows],
        }
        config_path = output / "config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Run inputs/config/code changed. Choose a new output directory.")
        else:
            if (output / "predictions.jsonl").exists():
                raise ValueError("Predictions exist without provenance config; use a new directory.")
            atomic_json(config_path, config)
        records = read_records(output / "predictions.jsonl")
        validate_records(records, rows)
        totals = SummaryAccumulator(rows)
        for record in records:
            totals.add(record)
        completed = {record["id"] for record in records}
        previous_path = output / "backend.json"
        if records and not previous_path.is_file():
            raise ValueError("Saved predictions have no backend provenance; use a new output directory")
        stop.check()
        if len(completed) == len(rows):
            result = totals.snapshot()
            summary_path = output / "summary.json"
            if summary_path.is_file():
                try:
                    previous_runtime = json.loads(summary_path.read_text()).get("runtime")
                    if isinstance(previous_runtime, dict):
                        result["runtime"] = previous_runtime
                except (ValueError, AttributeError):
                    pass  # This derived file can be rebuilt from durable records.
            atomic_json(summary_path, result)
            print(json.dumps(result, indent=2))
            return result
        atomic_json(output / "summary.json", totals.snapshot())
        from .models import load_backend
        random.seed(args.seed)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import numpy as np
        import torch
        torch.set_num_threads(cpu_threads)
        # PyTorch permits setting inter-op threads only before parallel work;
        # an already matching value needs no second initialization on reuse.
        if torch.get_num_interop_threads() != interop_threads:
            torch.set_num_interop_threads(interop_threads)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        backend = load_backend(args.model, config["revision"], dtype=args.dtype,
                               max_new_tokens=args.max_new_tokens)
        metadata = backend.metadata
        # Prevent mixing library/device/model resolution across resumed segments.
        if previous_path.exists() and json.loads(previous_path.read_text()) != metadata:
            raise ValueError("Backend provenance changed on resume; choose a new output directory")
        atomic_json(previous_path, metadata)
        evaluation_start = time.monotonic()
        session = {"started_utc": started_utc,
                   "evaluation_started_utc": datetime.now(timezone.utc).isoformat(),
                   "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                   "python": platform.python_version(), "already_completed": len(records),
                   "gpu": torch.cuda.get_device_name(0), "status": "running",
                   "summary_every": summary_every, "cpu_threads": cpu_threads,
                   "interop_threads": interop_threads,
                   "startup_seconds": evaluation_start - session_start}
        session_path = output / ("session-" + str(time.time_ns()) + ".json")
        atomic_json(session_path, session)
        torch.cuda.reset_peak_memory_stats()

        def save_progress():
            result = totals.snapshot()
            result["runtime"] = runtime_progress(session_start, evaluation_start, totals.evaluated,
                                                  session["already_completed"], len(rows))
            session["runtime"] = result["runtime"]
            session["completed_this_session"] = result["runtime"]["completed_this_session"]
            atomic_json(output / "summary.json", result)
            atomic_json(session_path, session)
            return result

        try:
            with (output / "predictions.jsonl").open("a") as stream:
                for row in rows:
                    stop.check()
                    if row["id"] in completed:
                        continue
                    with Image.open(row["_image_path"]) as source:
                        source_image = {"width": source.width, "height": source.height, "mode": source.mode}
                        image = source.convert("RGB")
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    raw = backend.predict(image, row["prompt"])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    prediction = parse_answer(raw, row["choices"])
                    record = {k: row[k] for k in ("id", "dataset", "task", "regime", "split",
                                                 "choices", "answer")}
                    record.update(raw_output=raw, prediction=prediction,
                                  correct=prediction == row["answer"], latency_seconds=elapsed,
                                  source_image=source_image,
                                  prediction_metadata=dict(getattr(backend, "last_prediction_metadata", {})))
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    records.append(record)
                    totals.add(record)
                    if totals.evaluated % summary_every == 0:
                        progress = save_progress()["runtime"]
                        print("Progress " + json.dumps(progress), flush=True)
                    print(f"{len(records)}/{len(rows)} {row['id']} prediction={prediction}", flush=True)
                    stop.check()
            session["status"] = "complete"
        except BaseException as error:
            session["status"] = "interrupted" if isinstance(error, AllocationInterrupted) else "interrupted-or-failed"
            if isinstance(error, AllocationInterrupted):
                session["interruption_signal"] = signal.Signals(error.signum).name
            session["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            session["ended_utc"] = datetime.now(timezone.utc).isoformat()
            session["completed_this_session"] = len(records) - session["already_completed"]
            session["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result = save_progress()
        print(json.dumps(result, indent=2))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="Unique resumable run directory")
    parser.add_argument("--model", choices=["lfm25", "smolvlm2", "internvl35", "spatialevo"], default="lfm25")
    parser.add_argument("--revision", required=True, help="40-character model commit SHA")
    parser.add_argument("--max-samples", type=int, default=16, help="0 = all manifest rows; default 16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2,
                        help="Torch intra-op CPU threads, pinned in resume config (default 2)")
    parser.add_argument("--interop-threads", type=int, default=1,
                        help="Torch inter-op threads, pinned in resume config (default 1)")
    parser.add_argument("--summary-every", type=int, default=25,
                        help="Refresh derived summary after this many durable predictions; always refresh on exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading model")
    args = parser.parse_args()
    if args.max_samples < 0 or min(args.max_new_tokens, args.summary_every, args.cpu_threads, args.interop_threads) < 1:
        parser.error("max-samples must be >=0; token, summary and thread counts must be >=1")
    try:
        run(args)
    except AllocationInterrupted as error:
        parser.exit(128 + error.signum, f"Interrupted: {error}\n")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
