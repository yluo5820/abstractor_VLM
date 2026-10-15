"""Bounded Slurm GPU audit of frozen LFM CPU threads and native image batches.

Measures processor/transfer/generation/decode with preloaded PIL images. It does
not measure disk I/O, loading, queueing, or rendering and does not change run.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import statistics
import sys
import threading
import time

from PIL import Image

from .models import DEFAULT_REVISIONS
from .run import atomic_json, load_manifest, parse_answer, require_allocation, sha256, source_hash


@contextmanager
def left_padding(tokenizer):
    previous = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = previous


def generated_length(tokens, eos_token_id, pad_token_id):
    eos = set(eos_token_id if isinstance(eos_token_id, (tuple, list)) else [eos_token_id])
    for index, token in enumerate(tokens):
        if token in eos:
            return index + 1
        if pad_token_id is not None and token == pad_token_id:
            return index
    return len(tokens)


def predict_batch(backend, samples, torch):
    """Batch native conversations; never pass labels or rule metadata to model."""
    conversations = [[{"role": "user", "content": [
        {"type": "image", "image": sample["image"].convert("RGB")},
        {"type": "text", "text": sample["row"]["prompt"]},
    ]}] for sample in samples]
    with left_padding(backend.processor.tokenizer):
        inputs = backend.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
        width = int(inputs["input_ids"].shape[-1])
        lengths = inputs["attention_mask"].sum(dim=-1).tolist()
        shapes = {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}
        inputs = inputs.to(device=backend.model.device, dtype=backend.model.dtype)
        with torch.inference_mode():
            outputs = backend.model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=backend.max_new_tokens,
                use_cache=True, return_dict_in_generate=False,
            )
        continuation = outputs[:, width:].detach().cpu().tolist()
        texts = backend.processor.batch_decode(continuation, skip_special_tokens=True)
    config = backend.model.generation_config
    metadata = [{"input_tokens": int(length), "padded_input_tokens": width,
                 "generated_tokens": generated_length(ids, config.eos_token_id, config.pad_token_id),
                 "generation_width": len(ids), "input_shapes": shapes}
                for length, ids in zip(lengths, continuation)]
    if len(texts) != len(samples) or len(metadata) != len(samples):
        raise RuntimeError("Native batch output count differs from input count")
    return [text.strip() for text in texts], metadata


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def current_rss_bytes():
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return peak_rss_bytes()


class RssSampler:
    """Observed RSS maximum, sampled at 20ms; not an exact per-condition peak."""
    def __enter__(self):
        self.peak = current_rss_bytes()
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.02):
                self.peak = max(self.peak, current_rss_bytes())
        self.worker = threading.Thread(target=sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.worker.join()
        self.peak = max(self.peak, current_rss_bytes())


def load_samples(path, count):
    path = Path(path).resolve()
    rows = load_manifest(path, count)
    digest = hashlib.sha256()
    samples = []
    for row in rows:
        digest.update(row["id"].encode())
        digest.update(sha256(row["_image_path"]).encode())
        with Image.open(row["_image_path"]) as image:
            source = {"width": image.width, "height": image.height, "mode": image.mode}
            samples.append({"row": row, "image": image.convert("RGB"), "source_image": source})
    return samples, {"path": str(path), "manifest_sha256": sha256(path),
                     "images_sha256": digest.hexdigest(), "sample_count": len(samples),
                     "sample_ids": [row["id"] for row in rows], "selection": "manifest prefix"}


def compare_outputs(records, baseline):
    if baseline is None:
        return {"reference": "this serial condition", "raw_mismatches": 0,
                "prediction_mismatches": 0, "input_token_mismatches": 0}
    differences = {"raw_mismatches": [], "prediction_mismatches": [], "input_token_mismatches": []}
    for record in records:
        reference = baseline[record["id"]]
        for name, changed in (
            ("raw_mismatches", record["raw_output"] != reference["raw_output"]),
            ("prediction_mismatches", record["prediction"] != reference["prediction"]),
            ("input_token_mismatches", record["prediction_metadata"].get("input_tokens") !=
             reference["prediction_metadata"].get("input_tokens")),
        ):
            if changed:
                differences[name].append(record["id"])
    return {**{key: len(ids) for key, ids in differences.items()}, "different_ids": differences}


def measure_condition(backend, samples, torch, threads, batch_size, kind, baseline,
                      warmup_batches=1, deadline=float("inf")):
    torch.set_num_threads(threads)
    gc.collect()
    torch.cuda.empty_cache()
    result = {"kind": kind, "cpu_threads": threads, "batch_size": batch_size,
              "planned": len(samples), "status": "running", "records": [], "batches": []}
    rss = None
    torch.cuda.reset_peak_memory_stats()

    def predict(batch):
        if kind == "serial":
            return [backend.predict(batch[0]["image"], batch[0]["row"]["prompt"])], [
                dict(backend.last_prediction_metadata)]
        return predict_batch(backend, batch, torch)

    def budget_check():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Audit time budget reached between batches")

    try:
        warm_start = time.perf_counter()
        for _ in range(warmup_batches):
            budget_check()
            predict(samples[:batch_size])
        torch.cuda.synchronize()
        result["warmup_seconds"] = time.perf_counter() - warm_start
        torch.cuda.reset_peak_memory_stats()
        with RssSampler() as rss:
            start, cpu_start = time.perf_counter(), time.process_time()
            try:
                for offset in range(0, len(samples), batch_size):
                    budget_check()
                    batch = samples[offset:offset + batch_size]
                    torch.cuda.synchronize()
                    batch_start = time.perf_counter()
                    texts, metadata = predict(batch)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - batch_start
                    result["batches"].append({"size": len(batch), "wall_seconds": elapsed})
                    for sample, raw, detail in zip(batch, texts, metadata):
                        row = sample["row"]
                        parsed = parse_answer(raw, row["choices"])
                        result["records"].append({"id": row["id"], "answer": row["answer"],
                            "raw_output": raw, "prediction": parsed, "correct": parsed == row["answer"],
                            "source_image": sample["source_image"], "prediction_metadata": detail})
            finally:
                result["wall_seconds"] = time.perf_counter() - start
                result["cpu_seconds"] = time.process_time() - cpu_start
        result["peak_rss_observed_bytes"] = rss.peak
        result["status"] = "complete"
    except Exception as error:
        result["status"] = ("time-budget-exhausted" if isinstance(error, TimeoutError) else
                            "out-of-memory" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed")
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if rss is not None:
            result["peak_rss_observed_bytes"] = rss.peak
        result["process_lifetime_peak_rss_bytes"] = peak_rss_bytes()
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        gc.collect()
        torch.cuda.empty_cache()
    records, wall = result["records"], result.get("wall_seconds", 0)
    result.update(evaluated=len(records), correct=sum(r["correct"] for r in records),
                  invalid_outputs=sum(r["prediction"] is None for r in records))
    result["accuracy"] = result["correct"] / len(records) if records else None
    result["samples_per_second"] = len(records) / wall if wall else None
    result["amortized_seconds_per_sample"] = wall / len(records) if records else None
    latencies = [batch["wall_seconds"] for batch in result["batches"]]
    result["median_batch_latency_seconds"] = statistics.median(latencies) if latencies else None
    result["max_batch_latency_seconds"] = max(latencies) if latencies else None
    result["comparison_to_serial"] = compare_outputs(records, baseline)
    return result


def best_condition(conditions, equivalent=False):
    valid = [row for row in conditions if row["status"] == "complete"]
    if equivalent:
        valid = [row for row in valid if all(row["comparison_to_serial"][key] == 0 for key in
                 ("raw_mismatches", "prediction_mismatches", "input_token_mismatches"))]
    return max(valid, key=lambda row: row["samples_per_second"], default=None)


def run_audit(args):
    require_allocation()
    threads = sorted(set(args.threads))
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))
                                    if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))
    if not threads or min(threads) < 1 or max(threads) > allocated:
        raise ValueError(f"Thread settings must fit the {allocated} allocated CPUs")
    if args.time_budget_seconds <= 0 or args.warmup_batches < 0:
        raise ValueError("Time budget must be positive and warmup count nonnegative")
    manifests = [("pgm", args.pgm_manifest, args.pgm_samples),
                 ("clevr", args.clevr_manifest, args.clevr_samples)]
    datasets, inputs = {}, {}
    for key, path, count in manifests:
        if path:
            if count < 1:
                raise ValueError("Sample counts must be positive")
            datasets[key], inputs[key] = load_samples(path, count)
    if not datasets:
        raise ValueError("Provide --pgm-manifest and/or --clevr-manifest")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "lfm-native-batch-performance-audit-v1", "status": "initializing",
              "started_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
              "source_sha256": source_hash(), "job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "allocated_cpus": allocated,
              "settings": {"threads": threads, "max_batch": args.max_batch, "phase": args.phase,
                           "dtype": args.dtype, "max_new_tokens": 16, "seed": args.seed,
                           "warmup_batches": args.warmup_batches, "time_budget_seconds": args.time_budget_seconds},
              "timing_scope": "preloaded PIL images; processor + device transfer + generation + decode; excludes load, disk I/O, rendering, queue",
              "rss_note": "20ms sampled resident bytes; lifetime ru_maxrss reported separately",
              "thread_note": "torch intra-op threads; tokenizer parallelism disabled; inter-op fixed to one",
              "conditions": [], "recommendations": {}}
    with output.open("x") as stream:
        json.dump(report, stream)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from .models import load_backend
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(threads[0])
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        name = torch.cuda.get_device_name(0)
        if args.require_gpu_name and args.require_gpu_name.lower() not in name.lower():
            raise ValueError(f"Expected GPU name containing {args.require_gpu_name!r}; got {name!r}")
        load_start = time.perf_counter()
        backend = load_backend("lfm25", DEFAULT_REVISIONS["lfm25"], dtype=args.dtype, max_new_tokens=16)
        torch.cuda.synchronize()
        report.update(backend=backend.metadata, model_load_seconds=time.perf_counter() - load_start,
                      status="running", gpu_name=name)
        atomic_json(output, report)
        deadline = time.perf_counter() + args.time_budget_seconds
        for dataset, samples in datasets.items():
            baseline, local = None, []
            visited = set()

            def measure(nthreads, size, kind):
                nonlocal baseline
                key = (nthreads, size, kind)
                if key in visited:
                    return None
                visited.add(key)
                condition = measure_condition(backend, samples, torch, nthreads, size, kind, baseline,
                                              args.warmup_batches, deadline)
                condition["dataset"] = dataset
                if baseline is None and condition["status"] == "complete":
                    baseline = {row["id"]: row for row in condition["records"]}
                local.append(condition)
                report["conditions"].append(condition)
                best = best_condition(local, equivalent=True)
                if best:
                    report["recommendations"][dataset] = {field: best[field] for field in
                        ("kind", "cpu_threads", "batch_size", "samples_per_second", "amortized_seconds_per_sample")}
                atomic_json(output, report)
                print(f"{dataset} {kind} threads={nthreads} batch={size} {condition['status']} "
                      f"items/s={condition['samples_per_second']}", flush=True)
                if condition["status"] == "time-budget-exhausted":
                    raise TimeoutError("Audit time budget reached; completed conditions are saved")
                if baseline is None:
                    raise RuntimeError("Serial baseline failed; inspect saved audit condition")
                return condition

            measure(threads[0], 1, "serial")
            if args.phase in ("all", "serial"):
                for nthreads in threads[1:]:
                    measure(nthreads, 1, "serial")
            serial = best_condition(local, equivalent=True)
            if args.phase in ("all", "batch"):
                for size in (1, 2, 4, 8, 16):
                    if size > args.max_batch or size > len(samples):
                        break
                    condition = measure(serial["cpu_threads"], size, "batch")
                    if condition and condition["status"] != "complete":
                        break
                fastest_batch = best_condition([row for row in local if row["kind"] == "batch"])
                if args.phase == "all" and fastest_batch:
                    for nthreads in threads:
                        measure(nthreads, fastest_batch["batch_size"], "batch")
            best = best_condition(local, equivalent=True)
            report["recommendations"][dataset] = {key: best[key] for key in
                ("kind", "cpu_threads", "batch_size", "samples_per_second", "amortized_seconds_per_sample")}
            atomic_json(output, report)
        report["status"] = "complete"
    except TimeoutError as error:
        report.update(status="time-budget-exhausted", error=str(error))
    except BaseException as error:
        report.update(status="failed-or-interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.set_num_threads(old_threads)
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgm-manifest")
    parser.add_argument("--clevr-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pgm-samples", type=int, default=32)
    parser.add_argument("--clevr-samples", type=int, default=16)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-batch", type=int, choices=[1, 2, 4, 8, 16], default=8)
    parser.add_argument("--phase", choices=["all", "serial", "batch"], default="all")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--require-gpu-name", help="Optional expected GPU-name substring")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--time-budget-seconds", type=float, default=300,
                        help="Soft audit limit after model load, checked between batches; partial audit exits 3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        report = run_audit(args)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps({"status": report["status"], "output": str(Path(args.output).resolve()),
                      "recommendations": report["recommendations"]}, indent=2))
    if report["status"] == "time-budget-exhausted":
        parser.exit(3, "Partial audit: time budget exhausted; completed and partial conditions are saved.\n")


if __name__ == "__main__":
    main()
