"""Bounded compatibility/timing check. Never reports random-head accuracy."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from .data import clevr_examples, pgm_examples, preprocess
from .model import NativeModel, require_compute, seed_all, UPSTREAM_COMMIT, CANONICAL_COMMIT


def sync():
    torch.cuda.synchronize()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--clevr-encoder", required=True)
    parser.add_argument("--pgm-encoder", required=True)
    parser.add_argument("--clevr-root", required=True)
    parser.add_argument("--pgm-shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--tasks", nargs="+", choices=["rmts", "idrules", "pgm"], default=["rmts", "idrules", "pgm"])
    parser.add_argument("--pgm-batches", nargs="+", type=int, default=[1, 4, 12, 24])
    args = parser.parse_args()
    if any(batch < 1 or batch > 96 for batch in args.pgm_batches):
        parser.error("PGM batch sizes must be between 1 and the paper's global batch size 96")
    require_compute()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    report = {"purpose": "technical pilot only; no trained reasoning checkpoint, no benchmark accuracy",
              "upstream": UPSTREAM_COMMIT, "canonical_encoder_source": CANONICAL_COMMIT,
              "job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "dtype": "float32", "tf32": False,
              "cases": [], "finished": False}
    atomic_json(args.output, report)
    for task in args.tasks:
        gc.collect()
        torch.cuda.empty_cache()
        model = NativeModel(args.upstream, args.pgm_encoder if task == "pgm" else args.clevr_encoder,
                            task, "cuda")
        seed_all(17)
        if task == "pgm":
            source = pgm_examples([args.pgm_shard])
        else:
            manifest = Path(args.clevr_root) / "manifests-v2" / f"clevr-art-{task}-test-full.jsonl"
            source = clevr_examples(args.clevr_root, manifest, task)
        # Only input pixels are used. Discard labels; random heads/updates are discarded.
        example_id, panels, _ = next(source)
        source.close()
        panels = panels[None].cuda()
        one_image = panels[:, 0]
        seed_all(41)
        with torch.no_grad():
            reference = model.encoder(one_image, model.device)
        seed_all(41)
        features, positions = model.encode(one_image)
        differences = [(features - reference[3]).abs().max().item(),
                       (positions - reference[4]).abs().max().item()]
        if max(differences) != 0.0:
            raise AssertionError(f"Encoder optimization differs from original: {differences}")
        del reference, features, positions
        case = {"task": task, "source_example": example_id,
                "encoder_equivalence_max_abs_diff": differences,
                "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                "head_parameters": sum(p.numel() for p in model.head.parameters()), "batches": []}
        report["cases"].append(case)
        optimizer = torch.optim.Adam(model.head.parameters(), lr=8e-5)
        for batch_size in ([1, 4, 16] if task != "pgm" else args.pgm_batches):
            if time.monotonic() - start > args.seconds:
                case["budget_exhausted"] = True
                break
            sample = panels.expand(batch_size, -1, -1, -1, -1).contiguous()
            # Explicit synthetic labels: no fitting against benchmark test answers.
            targets = torch.arange(batch_size, device="cuda") % (8 if task == "pgm" else sample.shape[1])
            times = []
            try:
                torch.cuda.reset_peak_memory_stats()
                for iteration in range(4):
                    sync()
                    begin = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(sample)
                    loss = F.cross_entropy(logits, targets)
                    loss.backward()
                    if not torch.isfinite(loss) or not all(
                        p.grad is None or torch.isfinite(p.grad).all() for p in model.head.parameters()):
                        raise AssertionError("Nonfinite loss or gradients")
                    optimizer.step()
                    sync()
                    if iteration:
                        times.append(time.monotonic() - begin)
                train_seconds = statistics.median(times)
                model.eval()
                with torch.no_grad():
                    sync()
                    begin = time.monotonic()
                    model(sample)
                    sync()
                    eval_seconds = time.monotonic() - begin
                case["batches"].append({"batch_size": batch_size,
                    "train_step_seconds_median": train_seconds, "train_step_samples": times,
                    "train_examples_per_second": batch_size / train_seconds,
                    "eval_examples_per_second": batch_size / eval_seconds,
                    "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "note": "Repeated resident inputs, fresh stochastic slots, synthetic targets; excludes dataset I/O and queueing"})
                model.train()
                model.encoder.eval()
            except torch.cuda.OutOfMemoryError:
                case["batches"].append({"batch_size": batch_size, "oom": True})
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
            atomic_json(args.output, report)
            print(json.dumps(case["batches"][-1]), flush=True)
            del sample, targets, logits, loss
        del optimizer, model, panels, one_image
    report["elapsed_seconds"] = time.monotonic() - start
    report["finished"] = True
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
