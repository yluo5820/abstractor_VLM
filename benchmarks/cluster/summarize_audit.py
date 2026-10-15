#!/usr/bin/env python3
"""Render a completed or partial performance audit without loading any model."""
import argparse
import json
from pathlib import Path


def report(source, output):
    audit = json.loads(source.read_text())
    backend = audit.get("backend", {})
    lines = ["# GPU resource audit", "",
             f"State: **{audit['status']}**. Slurm job `{audit['job_id']}` on `{audit['hostname']}`.", "",
             f"GPU: {backend.get('gpu_name', 'not initialized')}; dtype: {backend.get('dtype', 'unknown')}.", "",
             audit["timing_scope"] + ".", "",
             "| Task | Path | CPU threads | Batch | State | Examples/s | GPU peak GiB | Host RSS GiB | Changed answers vs serial |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in audit["conditions"]:
        rate = row.get("samples_per_second")
        speed = f"{rate:.3f}" if rate else "—"
        rss = row.get("peak_rss_observed_bytes", row.get("process_lifetime_peak_rss_bytes", 0)) / 2**30
        gpu = row.get("peak_gpu_allocated_bytes", 0) / 2**30
        changes = row["comparison_to_serial"]["prediction_mismatches"]
        lines.append(f"| {row['dataset']} | {row['kind']} | {row['cpu_threads']} | {row['batch_size']} | "
                     f"{row['status']} | {speed} | {gpu:.2f} | {rss:.2f} | {changes} |")
    lines += ["", "## Extrapolations", "",
              "Only complete conditions matching the serial outputs and input-token counts on the audited sample "
              "are eligible below. This small-sample agreement does not guarantee identical outputs on a full dataset. "
              "Batching is currently an audit path; the production evaluation runner remains serial.", ""]
    for task, choice in audit.get("recommendations", {}).items():
        count = 200000 if task == "pgm" else 10000
        hours = count * choice["amortized_seconds_per_sample"] / 3600
        lines.append(f"- {task}: {choice['kind']}, {choice['cpu_threads']} CPU threads, batch {choice['batch_size']}; "
                     f"{choice['samples_per_second']:.3f} examples/s; {hours:.2f} inference hours for {count:,} examples.")
    if not audit.get("recommendations"):
        lines.append("No completed task-level recommendation is available yet.")
    lines += ["", "Add queueing, model loading, data preparation and persistence overhead. CLEVR figures cover RMTS, "
              "not the unmeasured identity-rules task. A successful software test is not evidence of multi-GPU scaling. "
              "RSS is sampled resident memory, not total node usage; GPU figures are PyTorch allocated memory.", "",
              f"Raw audit: `{source.name}`. All per-example outputs, tokens, timings and errors remain in that JSON.", ""]
    if audit.get("error"):
        lines += ["Recorded error: " + audit["error"], ""]
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report(args.source, args.output)
