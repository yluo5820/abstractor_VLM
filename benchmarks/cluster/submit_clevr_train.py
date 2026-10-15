#!/usr/bin/env python3
"""Submit the authorized four-shard CPU campaign; idempotent, no GPU jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / "plan.json").read_text())
    # These are the concrete resources audited and authorized on 7 October.
    expected = {"account": "coc", "qos": "coc-ice", "partition": "ice-cpu", "shards": 4,
                "cpus_per_render_job": 16, "memory_gib_per_render_job": 32,
                "walltime_per_render_job": "16:00:00", "internal_seconds": 57000, "gpu_count": 0}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Campaign resource plan differs from the reviewed CPU request")
    for relative, digest in json.loads((campaign / "files.sha256.json").read_text()).items():
        if hashlib.sha256((campaign / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
    source = campaign / "source"
    environment = "/storage/ice1/9/8/yluo432/envs/vlm-bench"
    common = ["sbatch", "--parsable", "--account=coc", "--partition=ice-cpu", "--qos=coc-ice",
              "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--mem=32G", "--time=16:00:00"]
    def submit(name, command):
        receipt = campaign / f"{name}.json"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["command"] != command:
                raise ValueError("Saved submission command changed")
            return old["job_id"]
        print(json.dumps(command), flush=True)
        if not args.execute:
            return f"DRY_{name}"
        response = subprocess.check_output(command, text=True).strip()
        job_id = response.split(";")[0]
        if not job_id.isdigit():
            raise ValueError(f"Unrecognized scheduler receipt: {response}")
        save(receipt, {"job_id": job_id, "command": command})
        print(f"{name}: {job_id}", flush=True)
        return job_id
    # Kernel lock prevents simultaneous invocations submitting duplicate jobs.
    import fcntl
    with (campaign / ".submit.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        render_ids = []
        for shard in range(plan["shards"]):
            command = common + [f"--job-name=clevr-train-oct07-{shard}",
                f"--output={campaign}/render-{shard}-%j.out", f"--error={campaign}/render-{shard}-%j.err",
                str(source / "cluster/render_full.sbatch"), str(source), environment, plan["assets"],
                str(Path(plan["data_root"]) / "shards" / str(shard)), str(plan["internal_seconds"]),
                "all", "train", str(shard), str(plan["shards"])]
            if shard == 0 and args.execute and not (campaign / "render-0.json").exists():
                subprocess.run(command[:1] + ["--test-only"] + command[1:], check=True)
            render_ids.append(submit(f"render-{shard}", command))
        finish = ["sbatch", "--parsable", "--account=coc", "--partition=ice-cpu", "--qos=coc-ice",
                  "--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--mem=8G", "--time=01:00:00",
                  "--job-name=clevr-train-index", "--kill-on-invalid-dep=yes",
                  "--dependency=afterok:" + ":".join(render_ids),
                  f"--output={campaign}/index-%j.out", f"--error={campaign}/index-%j.err",
                  str(source / "cluster/index_clevr_train.sbatch"), str(source), environment,
                  plan["data_root"], str(plan["shards"])]
        index_id = submit("index", finish)
        result = {"render_jobs": render_ids, "index_job": index_id, "gpu_jobs": []}
        if args.execute:
            save(campaign / "submitted.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
