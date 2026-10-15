#!/usr/bin/env python3
"""Create an auditable Markdown summary from saved runs without loading models."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def report(runs: Path, output: Path):
    lines = ["# LFM2.5 visual reasoning: first accessible-data evaluation", "",
             "Generated " + datetime.now(timezone.utc).isoformat() + ".", "",
             "Frozen, zero-shot, native image processing, greedy decoding, 16 output tokens. "
             "PGM uses the pinned HuggingFaceM4 neutral/test mirror. CLEVR-ART RMTS uses "
             "official OCRA test labels and Blender-rendered scenes. These conditions differ "
             "from the task-supervised Slot-Abstractor paper experiments.", "",
             "| Run | State | Correct / evaluated | Accuracy | 95% Wilson interval | Chance | Invalid |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    details = []
    for summary_path in sorted(runs.glob("*/summary.json")):
        folder = summary_path.parent
        summary = json.loads(summary_path.read_text())
        config = json.loads((folder / "config.json").read_text())
        records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines() if line]
        for group in summary["groups"]:
            if not group["evaluated"]:
                continue
            interval = "–".join(f"{100*x:.1f}%" for x in group["wilson_95_interval"])
            state = "complete" if summary["complete"] else f"PARTIAL ({summary['evaluated']}/{summary['planned']})"
            lines.append(f"| {folder.name} | {state} | {group['correct']} / {group['evaluated']} | "
                         f"{100*group['accuracy']:.2f}% | {interval} | {100*group['chance_accuracy']:.1f}% | "
                         f"{group['invalid_outputs']} |")
        predictions = Counter(r["prediction"] or "INVALID" for r in records)
        targets = Counter(r["answer"] for r in records)
        mode = predictions.most_common(1)[0][0] if predictions else ""
        details += [f"## {folder.name}", "",
                    f"Model: `{config['model']}` at `{config['revision']}`.", "",
                    "Predictions by label: " + ", ".join(f"{k}={v}" for k,v in sorted(predictions.items())) + ".",
                    "Targets by label: " + ", ".join(f"{k}={v}" for k,v in sorted(targets.items())) + "."]
        if records:
            details += [f"The model selected its modal label ({mode}) on "
                        f"{predictions[mode]}/{len(records)} examples ({100*predictions[mode]/len(records):.1f}%). "
                        f"Always choosing that label would score "
                        f"{targets[mode]}/{len(records)} on these evaluated examples. "
                        "This diagnostic helps distinguish answer-position bias from useful predictions."]
        if folder.name == "lfm25-pgm-neutral-hf-pilot512" and summary["complete"]:
            group = summary["groups"][0]
            low, high = group["wilson_95_interval"]
            if low <= group["chance_accuracy"] <= high:
                details += ["The accuracy interval includes uniform-random chance. "
                            "This pilot provides no evidence of above-chance accuracy under the current protocol; "
                            "it does not establish a general inability of the model to reason."]
        backend = json.loads((folder / "backend.json").read_text())
        sessions = [json.loads(p.read_text()) for p in sorted(folder.glob("session-*.json"))]
        peak_bytes = max((s.get("peak_gpu_allocated_bytes", 0) or 0 for s in sessions), default=0)
        capped = sum(bool(r.get("prediction_metadata", {}).get("reached_token_budget")) for r in records)
        mean_latency = sum(r["latency_seconds"] for r in records) / len(records) if records else 0
        details += ["", f"Device: {backend.get('gpu_name')}; precision: {backend.get('dtype')}.", "",
                    f"Mean generation latency: {mean_latency:.3f} seconds/example; "
                    f"peak PyTorch allocated GPU memory: {peak_bytes / 2**30:.2f} GiB. "
                    f"Outputs reaching the token limit: {capped}/{len(records)}. "
                    "Latency excludes model loading and data preparation; allocated memory is not total device usage.", "",
                    f"Raw predictions: [{folder.name}/predictions.jsonl]({folder / 'predictions.jsonl'}).", "",
                    f"Input manifest SHA-256: `{config['manifest_sha256']}`.", ""]
    lines += ["", "The 16-example runs are implementation checks, with broad uncertainty. "
              "The 512-example run is a sampled pilot, not the full 200,000-example neutral test set. "
              "No extrapolation or other PGM generalization regime is represented. "
              "Only complete runs should be used as final measurements.", "", *details,
              "## Current scope", "",
              "Only LFM2.5 has been evaluated. SmolVLM2 and InternVL3.5 backends are prepared but unrun. "
              "CLEVR identity rules and larger render batches are deferred. The original PGM archive "
              "endpoint returned access-denied errors; original NPZ equivalence of the neutral mirror "
              "has not been independently established. Pretraining overlap is unknown.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.runs.resolve(), args.output.resolve())
