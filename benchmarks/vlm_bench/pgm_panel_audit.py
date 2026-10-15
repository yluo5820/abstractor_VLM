"""Paired validation diagnostic: base/LoRA x stitched/separate PGM panels."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image

from .data import sha256_file
from .pgm_panel_inputs import messages, split_composite
from .run import atomic_json, parse_answer, require_allocation, wilson


def summarize(records, planned):
    arms = {}
    comparisons = {}
    for weight in ("base", "lora20"):
        corrects = {}
        for presentation in ("composite", "separate"):
            key = f"{weight}/{presentation}"
            values = [r["results"][key] for r in records]
            corrects[presentation] = np.array([v["correct"] for v in values], dtype=float)
            n, k = len(values), int(corrects[presentation].sum())
            arms[key] = {"evaluated": n, "correct": k, "accuracy": k / n if n else None,
                         "wilson_95": wilson(k, n),
                         "invalid": sum(v["prediction"] is None for v in values),
                         "mean_seconds": float(np.mean([v["seconds"] for v in values])) if n else None,
                         "mean_input_tokens": float(np.mean([v["input_tokens"] for v in values])) if n else None}
        difference = corrects["separate"] - corrects["composite"]
        if len(difference):
            rng = np.random.default_rng(20261014)
            bootstrap = np.array([rng.choice(difference, len(difference), replace=True).mean()
                                  for _ in range(2000)])
            comparisons[weight] = {"separate_minus_composite": float(difference.mean()),
                                   "paired_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
                                   "only_separate_correct": int((difference == 1).sum()),
                                   "only_composite_correct": int((difference == -1).sum())}
    return {"split": "official validation subset", "planned": planned,
            "evaluated_matched_questions": len(records), "complete": len(records) == planned,
            "arms": arms, "paired_comparisons": comparisons,
            "caveat": "LoRA was trained on composites; separate-panel evaluation is an input distribution shift. "
                      "Native token allocation and position-label text differ between presentations."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    require_allocation()
    import torch
    from peft import PeftModel
    from .models import load_backend

    plan = json.loads(args.plan.read_text())
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # Never silently mix a repeated run with an existing set of measurements.
    stream = (output / "predictions.jsonl").open("x")
    records = []
    stopping = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    try:
        for filename, expected in plan["pinned_inputs"].items():
            if sha256_file(Path(filename)) != expected:
                raise ValueError(f"Pinned input hash mismatch: {filename}")
        root = Path(plan["data_root"])
        rows = [json.loads(line) for line in (root / "validation.jsonl").read_text().splitlines()]
        if len(rows) != 1000 or len({r["id"] for r in rows}) != 1000:
            raise ValueError("Expected the unchanged 1,000-question validation subset")
        for row in rows:
            if (row["split"] != "validation" or not row["id"].startswith("pgm/neutral/validation/")
                    or row["choices"] != list("ABCDEFGH")):
                raise ValueError("Invalid validation provenance")
        rows.sort(key=lambda r: hashlib.sha256(("panel-audit-20261014:" + r["id"]).encode()).hexdigest())
        prior = json.loads(Path(plan["expected_validation"]).read_text())
        prior_by_id = {r["id"]: r for r in prior["predictions"]}
        if set(prior_by_id) != {r["id"] for r in rows}:
            raise ValueError("Prior validation does not cover exactly the same IDs")
        atomic_json(output / "config.json", {**plan, "plan_sha256": sha256_file(args.plan),
                    "job_id": os.environ["SLURM_JOB_ID"], "ordered_ids": [r["id"] for r in rows]})
        torch.manual_seed(20261014)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        backend = load_backend("lfm25", revision=plan["revision"], device="cuda", dtype="bfloat16", max_new_tokens=16)
        backend.model = PeftModel.from_pretrained(backend.model, plan["adapter"], is_trainable=False)
        backend.model.requires_grad_(False)
        backend.model.eval()
        atomic_json(output / "backend.json", backend.metadata)
        # A small preflight proves disabling the adapter leaves base predictions
        # unchanged across intervening adapter-enabled inference.
        preflight_done = False
        for index, row in enumerate(rows):
            if stopping or time.time() >= args.deadline - 90:
                break
            image_path = (root / row["images"][0]).resolve()
            if not image_path.is_relative_to(root.resolve()):
                raise ValueError("Image path escapes data root")
            expected_hash = row["metadata"]["contact_sheet_sha256"]
            if sha256_file(image_path) != expected_hash:
                raise ValueError(f"Changed validation image: {row['id']}")
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            inputs = {}
            for mode in ("composite", "separate"):
                encoded = backend.processor.apply_chat_template(messages(image, row["prompt"], mode),
                    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", truncation=False)
                n = int(encoded["input_ids"].shape[-1])
                if n + 16 > min(32768, backend.context_limit or 32768):
                    raise ValueError("Native context length exceeded; truncation is forbidden")
                if "pixel_values" not in encoded or encoded["pixel_values"].numel() == 0:
                    raise ValueError("Images were dropped by the processor")
                if mode == "separate" and encoded["pixel_values"].shape[0] != 16:
                    raise ValueError("Expected one native image encoding per 160x160 panel")
                inputs[mode] = (encoded.to(device=backend.model.device, dtype=backend.model.dtype), n)
            if not preflight_done:
                prompts = {mode: backend.processor.apply_chat_template(messages(image, row["prompt"], mode),
                            add_generation_prompt=True, tokenize=False) for mode in inputs}
                atomic_json(output / "presentation.json", {
                    "example_id": row["id"], "rendered_prompts": prompts,
                    "input_shapes": {m: {k: list(v.shape) for k, v in t.items()} for m, (t, _) in inputs.items()},
                    "source_panel_pixel_sha256": [hashlib.sha256(p.tobytes()).hexdigest() for p in split_composite(image)]})
            results = {}
            # Alternate ordering to reduce systematic timing/warmup effects.
            weights = ("base", "lora20") if index % 2 == 0 else ("lora20", "base")
            modes = ("composite", "separate") if index % 2 == 0 else ("separate", "composite")
            for weight in weights:
                with backend.model.disable_adapter() if weight == "base" else nullcontext():
                    for mode in modes:
                        encoded, n = inputs[mode]
                        torch.cuda.synchronize()
                        started = time.monotonic()
                        with torch.inference_mode():
                            generated = backend.model.generate(**encoded, do_sample=False, num_beams=1,
                                max_new_tokens=16, use_cache=True, return_dict_in_generate=False)
                        torch.cuda.synchronize()
                        seconds = time.monotonic() - started
                        continuation = generated[0, n:]
                        raw = backend.processor.decode(continuation, skip_special_tokens=True).strip()
                        prediction = parse_answer(raw, row["choices"])
                        results[f"{weight}/{mode}"] = {"prediction": prediction, "raw_output": raw,
                            "correct": prediction == row["answer"], "seconds": seconds, "input_tokens": n,
                            "generated_tokens": int(continuation.numel())}
            if not preflight_done:
                with backend.model.disable_adapter():
                    repeated = backend.predict(image, row["prompt"])
                if repeated != results["base/composite"]["raw_output"]:
                    raise ValueError("Base disable-adapter replay or original inference-path parity failed")
                preflight_done = True
            expected = prior_by_id[row["id"]]
            if (expected["answer"] != row["answer"] or
                    expected["prediction"] != results["lora20/composite"]["prediction"]):
                raise ValueError(f"Composite checkpoint replay differs from archived validation at {row['id']}")
            record = {"id": row["id"], "answer": row["answer"], "image_sha256": expected_hash, "results": results}
            # A durable row always contains all four arms for one matched question.
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
            if len(records) % 25 == 0 or len(records) == 1:
                progress = {"matched_questions": len(records), "planned": len(rows), "time": time.time()}
                atomic_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        summary = summarize(records, len(rows))
        summary["stop_reason"] = "complete" if len(records) == len(rows) else "signal_or_deadline"
        summary["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        atomic_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                    "committed_matched_questions": len(records)})
        raise
    finally:
        stream.close()


if __name__ == "__main__":
    main()
