#!/usr/bin/env python3
"""Submit the approved LFM pilot and bounded CLEVR identity LoRA trial."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "train"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    plan = json.loads((campaign / "plan.json").read_text())
    expected = {"account": "coc", "partition": "ice-gpu", "qos": "coc-ice", "gpu": "gpu:h200:1",
                "cpus": 2, "memory_gib": 16, "pilot_walltime": "00:20:00", "train_walltime": "02:00:00"}
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Resource plan differs from the authorized bounded experiment")
    for name, checksum in json.loads((campaign / "files.sha256.json").read_text()).items():
        if hashlib.sha256((campaign / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Frozen source/dependency changed: {name}")
    command = ["sbatch", "--parsable", "--account=coc", "--partition=ice-gpu", "--qos=coc-ice",
               "--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--mem=16G", "--gres=gpu:h200:1",
               f"--time={plan[args.stage + '_walltime']}", "--signal=B:USR1@120",
               f"--job-name=lfm-lora-id-{args.stage}", f"--output={campaign}/{args.stage}-%j.out",
               f"--error={campaign}/{args.stage}-%j.err"]
    if args.stage == "train":
        pilot_id = json.loads((campaign / "pilot-job.json").read_text())["job_id"]
        command += [f"--dependency=afterok:{pilot_id}:{plan['render_index_job']}", "--kill-on-invalid-dep=yes"]
    command += [str(campaign / "source/cluster/finetune_lfm.sbatch"), str(campaign), args.stage]
    receipt = campaign / f"{args.stage}-job.json"
    with (campaign / ".submit.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record["command"] != command:
                raise ValueError("Existing submission has a different command")
        elif args.execute:
            subprocess.run(command[:1] + ["--test-only"] + command[1:], check=True)
            response = subprocess.check_output(command, text=True).strip()
            job = response.split(";")[0]
            if not job.isdigit():
                raise ValueError(f"Invalid scheduler receipt: {response}")
            record = {"job_id": job, "command": command}
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(receipt)
        else:
            record = {"command": command, "dry_run": True}
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
