#!/usr/bin/env python3
"""One bounded GPU segment: technical preflight, then durable full PGM evaluation.

Resubmit the identical command to continue. This program never submits jobs.
Exit 0 means complete OR an explicitly recorded resumable time/signal stop;
inspect pipeline-status.json to distinguish them. Technical failures exit 1.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

PREFLIGHT_COUNT = 32
FULL_COUNT = 200000
RESULT_FILES = ("config.json", "backend.json", "predictions.jsonl", "summary.json")


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: dict):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hash(source: Path) -> str:
    result = hashlib.sha256()
    files = sorted((source / "vlm_bench").glob("*.py"))
    if not files:
        raise ValueError("Source does not contain vlm_bench Python files")
    for path in files:
        result.update(path.name.encode())
        result.update(path.read_bytes())
    return result.hexdigest()


@contextmanager
def locked_output(root: Path, policy: dict):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this comparison output") from None
        try:
            path = root / "pipeline-config.json"
            if path.exists():
                if json.loads(path.read_text()) != policy:
                    raise ValueError("Pipeline inputs/code/model/settings changed; use a new output root")
            else:
                if set(p.name for p in root.iterdir()) - {".pipeline.lock"}:
                    raise ValueError("Refusing a nonempty output root without pipeline-config.json")
                atomic_json(path, policy)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def expected_config(policy: dict, phase: str) -> dict:
    return {"model": policy["model"], "revision": policy["revision"],
            "source_sha256": policy["source_sha256"],
            "manifest_sha256": policy[phase + "_manifest_sha256"],
            "max_samples": PREFLIGHT_COUNT if phase == "preflight" else 0,
            "max_new_tokens": policy["max_new_tokens"], "dtype": policy["dtype"],
            "seed": policy["seed"], "cpu_threads": policy["cpu_threads"], "interop_threads": 1,
            "tokenizers_parallelism": False, "parser": "strict-letter-v1",
            "protocol": "frozen-zero-shot-single-contact-sheet-v1"}


def verify_run_config(output: Path, policy: dict, phase: str):
    config = json.loads((output / "config.json").read_text())
    expected = expected_config(policy, phase)
    differences = [key for key, value in expected.items() if config.get(key) != value]
    count = PREFLIGHT_COUNT if phase == "preflight" else policy.get("full_count", FULL_COUNT)
    if differences or len(config.get("example_ids", [])) != count:
        raise ValueError(f"{phase} runner config differs from pipeline: {differences or ['example count']}")
    return config


def preflight_diagnostics(output: Path) -> dict:
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines() if line]
    if len(rows) != PREFLIGHT_COUNT or len({row["id"] for row in rows}) != PREFLIGHT_COUNT:
        raise ValueError("Preflight did not commit exactly 32 unique predictions")
    invalid = sum(row.get("prediction") is None for row in rows)
    budget = sum(row.get("prediction_metadata", {}).get("reached_token_budget") is True for row in rows)
    return {"examples": len(rows), "invalid_outputs": invalid, "parseable_outputs": len(rows) - invalid,
            "reached_token_budget": budget,
            "answer_distribution": dict(sorted(Counter(str(row.get("prediction")) for row in rows).items())),
            "gate": "32 complete examples; at least one parseable output; no reached_token_budget; accuracy not used"}


def seal_complete(output: Path, policy: dict, phase: str) -> dict:
    verify_run_config(output, policy, phase)
    summary = json.loads((output / "summary.json").read_text())
    count = PREFLIGHT_COUNT if phase == "preflight" else policy.get("full_count", FULL_COUNT)
    if summary.get("complete") is not True or summary.get("planned") != count or summary.get("evaluated") != count:
        raise ValueError(f"{phase} exited without a complete {count}-example summary")
    diagnostics = preflight_diagnostics(output) if phase == "preflight" else None
    if diagnostics:
        atomic_json(output / "technical-gate.json", diagnostics)
        if diagnostics["parseable_outputs"] < 1 or diagnostics["reached_token_budget"]:
            raise ValueError("Technical preflight failed; inspect technical-gate.json before running full evaluation")
    receipt = {"schema_version": 1, "phase": phase, "policy_sha256": canonical_hash(policy),
               "files": {name: sha(output / name) for name in RESULT_FILES}, "diagnostics": diagnostics}
    atomic_json(output / "COMPLETE.json", receipt)
    return receipt


def completed_receipt(output: Path, policy: dict, phase: str) -> dict | None:
    path = output / "COMPLETE.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if (receipt.get("phase") != phase or receipt.get("policy_sha256") != canonical_hash(policy)
            or set(receipt.get("files", {})) != set(RESULT_FILES)):
        raise ValueError(f"Modified/incompatible completion receipt: {path}")
    for name, expected in receipt["files"].items():
        if sha(output / name) != expected:
            raise ValueError(f"Completed run file changed: {output / name}")
    verify_run_config(output, policy, phase)
    if phase == "preflight":
        diagnostics = preflight_diagnostics(output)
        if diagnostics != receipt.get("diagnostics") or diagnostics["parseable_outputs"] < 1 or diagnostics["reached_token_budget"]:
            raise ValueError("Completed preflight no longer passes its technical gate")
    return receipt


def child_command(policy: dict, output: Path, phase: str) -> list[str]:
    return [policy["python"], "-m", "vlm_bench.run", "--manifest", policy[phase + "_manifest"],
            "--output", str(output), "--model", policy["model"], "--revision", policy["revision"],
            "--max-samples", str(PREFLIGHT_COUNT if phase == "preflight" else 0),
            "--max-new-tokens", str(policy["max_new_tokens"]), "--dtype", policy["dtype"],
            "--seed", str(policy["seed"]), "--cpu-threads", str(policy["cpu_threads"]),
            "--interop-threads", "1", "--summary-every", "25"]


def supervise(command: list[str], cwd: Path, deadline: float, stop: list[str], grace: float = 240) -> dict:
    """Budget covers child startup, all validation, model loading and inference."""
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    reason, sent, forced = None, None, False
    try:
        while process.poll() is None:
            now = time.monotonic()
            if reason is None and (stop or now >= deadline):
                reason = stop[0] if stop else "segment-deadline"
                sent = now
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if sent is not None and now - sent >= grace:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                forced = True
                break
            time.sleep(0.25)
        code = process.wait()
        return {"return_code": code, "stop_reason": reason, "forced_kill": forced}
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def pipeline(policy: dict, root: Path, deadline: float, grace: float = 240) -> dict:
    started = time.monotonic()
    stop = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    for sig in previous:
        signal.signal(sig, lambda signum, frame: stop.append(signal.Signals(signum).name))
    try:
        with locked_output(root, policy):
            segments = root / "segments"
            segments.mkdir(exist_ok=True)
            record_path = segments / (str(time.time_ns()) + ".json")
            record = {"schema_version": 1, "job_id": os.environ.get("SLURM_JOB_ID"),
                      "host": socket.gethostname(), "policy_sha256": canonical_hash(policy),
                      "status": "running", "phases": {}, "started_at": time.time()}

            def save(status):
                record.update(status=status, elapsed_seconds=time.monotonic() - started)
                atomic_json(record_path, record)
                atomic_json(root / "pipeline-status.json", record)
                print(json.dumps({"pipeline_status": status, "phases": record["phases"]}), flush=True)

            save("running")
            try:
                for phase in ("preflight", "full"):
                    output = root / phase
                    receipt = completed_receipt(output, policy, phase)
                    if receipt:
                        record["phases"][phase] = {"status": "verified-complete", "diagnostics": receipt["diagnostics"]}
                        continue
                    if stop or time.monotonic() >= deadline:
                        record["phases"][phase] = {"status": "not-started", "stop_reason": stop[0] if stop else "segment-deadline"}
                        save("resumable")
                        return record
                    result = supervise(child_command(policy, output, phase), Path(policy["source"]), deadline, stop, grace)
                    record["phases"][phase] = result
                    if result["return_code"] == 0:
                        receipt = seal_complete(output, policy, phase)
                        result.update(status="complete", diagnostics=receipt["diagnostics"])
                        save("running")
                        continue
                    # Only our deadline/forwarded signal may make an incomplete
                    # segment successful. OOM/config/model errors stop the chain.
                    expected_stop = result["return_code"] in (143, 130, -signal.SIGTERM)
                    expected_stop |= result["return_code"] == -signal.SIGKILL and result["forced_kill"]
                    if result["stop_reason"] and expected_stop:
                        result["status"] = "interrupted-resumable"
                        save("resumable")
                        return record
                    raise RuntimeError(f"{phase} runner failed with exit {result['return_code']}; no automatic full continuation")
                save("complete")
                return record
            except BaseException as exc:
                record["error"] = type(exc).__name__ + ": " + str(exc)
                save("failed")
                raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--segment-seconds", type=float, default=35400)
    parser.add_argument("--grace-seconds", type=float, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--full-count", type=int, default=FULL_COUNT,
                        help="Expected complete manifest size (CLEVR-ART: 10000 per task)")
    parser.add_argument("--expected-dataset")
    parser.add_argument("--expected-task")
    args = parser.parse_args(argv)
    deadline = time.monotonic() + args.segment_seconds
    try:
        if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname().lower():
            raise RuntimeError("Comparison requires a Slurm GPU compute allocation")
        if not (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
            raise RuntimeError("No Slurm GPU allocation is recorded")
        if args.cpu_threads < 1 or args.cpu_threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
            raise ValueError("CPU threads exceed allocation")
        if not (0 < args.segment_seconds <= 35400 and 0 < args.grace_seconds <= 240 and args.max_new_tokens > 0):
            raise ValueError("Use segment <=35400s, grace <=240s, and positive token budget")
        if len(args.revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in args.revision):
            raise ValueError("Revision must be an immutable 40-character commit SHA")
        source = args.source.resolve()
        if args.full_count < PREFLIGHT_COUNT:
            raise ValueError("Full manifest must contain at least the preflight count")
        if args.expected_dataset or args.expected_task:
            count = 0
            with args.full_manifest.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if (args.expected_dataset and row.get("dataset") != args.expected_dataset or
                            args.expected_task and row.get("task") != args.expected_task):
                        raise ValueError("Manifest dataset/task differs from requested benchmark")
                    count += 1
            if count != args.full_count:
                raise ValueError("Manifest count differs from requested complete task")
        policy = {"schema_version": 1, "source": str(source), "source_sha256": source_hash(source),
                  "coordinator_sha256": sha(Path(__file__)), "python": sys.executable,
                  "wrapper_sha256": sha(source / "cluster" / "compare_pgm.sbatch"),
                  "model": args.model, "revision": args.revision.lower(),
                  "preflight_manifest": str(args.preflight_manifest.resolve()),
                  "preflight_manifest_sha256": sha(args.preflight_manifest),
                  "full_manifest": str(args.full_manifest.resolve()), "full_manifest_sha256": sha(args.full_manifest),
                  "max_new_tokens": args.max_new_tokens, "dtype": args.dtype,
                  "seed": args.seed, "cpu_threads": args.cpu_threads,
                  "preflight_count": PREFLIGHT_COUNT, "full_count": args.full_count}
        if args.expected_dataset or args.expected_task:
            policy.update(expected_dataset=args.expected_dataset, expected_task=args.expected_task,
                          task_wrapper_sha256=sha(source / "cluster" / "compare_clevr.sbatch"))
        pipeline(policy, args.output.resolve(), deadline, args.grace_seconds)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Comparison pipeline failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
