#!/usr/bin/env python3
"""Local, standard-library-only launcher for one explicitly verified ICE checkout."""
import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
SOCKET = BENCH / ".local" / "ice.sock"
SIMPLE = re.compile(r"[A-Za-z0-9_.+-]+\Z")
REMOTE_PATH = re.compile(r"/[A-Za-z0-9_./+-]+\Z")


def absolute_path(value, name):
    # Restrict rsync's remote path syntax as well as ordinary shell arguments.
    if not isinstance(value, str) or not REMOTE_PATH.fullmatch(value):
        raise ValueError(f"{name} must be an absolute remote path without shell metacharacters or spaces")
    if ".." in value.split("/") or value.rstrip("/") in ("", "/home", "/scratch", "/tmp"):
        raise ValueError(f"{name} must identify a specific directory/file, not a storage root")
    return value.rstrip("/")


def load_profile(path):
    p = json.loads(Path(path).read_text())
    if p.get("verified") is not True or not str(p.get("verification_notes", "")).strip():
        raise ValueError("Profile is unverified. Inspect ICE and record evidence before allocating or syncing.")
    if (p.get("ssh_alias"), p.get("login_hostname"), p.get("login_username")) != (
        "ice-login", "login-ice.pace.gatech.edu", "yluo432"
    ):
        raise ValueError("Connection differs from the inspected SSH config; reverify before changing this launcher")
    for field in ("account", "partition"):
        if not isinstance(p.get(field), str) or not SIMPLE.fullmatch(p[field]):
            raise ValueError(f"Provide the verified {field}")
    if not isinstance(p.get("qos"), str) or (p["qos"] and not SIMPLE.fullmatch(p["qos"])):
        raise ValueError("qos must be the verified QoS, or empty when the account default is verified")
    for field in ("cpus", "memory_gib", "gpu_count"):
        if type(p.get(field)) is not int or p[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    wall = re.fullmatch(r"(\d{2,3}):([0-5]\d):([0-5]\d)", str(p.get("time", "")))
    if not wall:
        raise ValueError("time must be HH:MM:SS")
    seconds = int(wall[1]) * 3600 + int(wall[2]) * 60 + int(wall[3])
    if seconds < 1:
        raise ValueError("time must be positive")
    request = p.get("gpu_request", {})
    flag, value = request.get("flag"), request.get("value")
    if flag == "--gres":
        match = re.fullmatch(r"gpu(?::[A-Za-z0-9_.+-]+)?:(\d+)", str(value))
    elif flag in ("--gpus", "--gpus-per-node"):
        match = re.fullmatch(r"(?:[A-Za-z0-9_.+-]+:)?(\d+)", str(value))
    else:
        raise ValueError("gpu_request.flag must be --gres, --gpus, or --gpus-per-node")
    if not match or int(match[1]) != p["gpu_count"]:
        raise ValueError("GPU request must match gpu_count and contain exactly one GPU resource type")
    limits = p.get("scheduler_limits", {})
    for field in ("max_wall_seconds", "max_gpu_minutes", "max_cpu_minutes"):
        if field in limits and (type(limits[field]) is not int or limits[field] < 1):
            raise ValueError(f"scheduler_limits.{field} must be a positive integer")
    if seconds > limits.get("max_wall_seconds", seconds):
        raise ValueError("Requested wall time exceeds the verified partition limit")
    if "max_gpu_minutes" in limits and p["gpu_count"] * seconds > 60 * limits["max_gpu_minutes"]:
        raise ValueError("GPU count times wall time exceeds the verified GPU-minute limit")
    if "max_cpu_minutes" in limits and p["cpus"] * seconds > 60 * limits["max_cpu_minutes"]:
        raise ValueError("CPU count times wall time exceeds the verified CPU-minute limit")
    increased = p["cpus"] > 2 or p["memory_gib"] > 16 or p["gpu_count"] > 1 or seconds > 1200
    if increased and not str(p.get("user_resource_approval", "")).strip():
        raise ValueError("Above the initial ceiling (1 GPU, 2 CPUs, 16 GiB, 20 min). Ask the user first and record their explicit approval in user_resource_approval.")
    for field in ("project_path", "environment_path", "cache_path"):
        p[field] = absolute_path(p.get(field), field)
    if not isinstance(p.get("modules"), list) or any(not isinstance(m, str) or not re.fullmatch(r"[A-Za-z0-9_./+-]+", m) for m in p["modules"]):
        raise ValueError("modules must be a list of verified module names")
    return p


def resource_args(p):
    args = [f"--account={p['account']}", f"--partition={p['partition']}"]
    if p["qos"]:
        args.append(f"--qos={p['qos']}")
    args += [f"{p['gpu_request']['flag']}={p['gpu_request']['value']}", "--nodes=1", "--ntasks=1",
             f"--cpus-per-task={p['cpus']}", f"--mem={p['memory_gib']}G", f"--time={p['time']}"]
    return args


def ssh_args(tty=False):
    return ["ssh", "-S", str(SOCKET), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"] + (["-tt"] if tty else []) + ["ice-login"]


def remote_command(p, argv, *, tty=False, prefix=""):
    modules = ""
    if p["modules"]:
        modules = shlex.join(["module", "load", *p["modules"]]) + "; "
    body = "set -e; " + modules + prefix + "exec " + shlex.join(argv)
    return ssh_args(tty) + ["bash -lc " + shlex.quote(body)]


def verify_connection_config():
    config = subprocess.run(["ssh", "-G", "ice-login"], text=True, capture_output=True, check=True).stdout
    values = dict(line.split(" ", 1) for line in config.splitlines() if " " in line)
    if values.get("hostname") != "login-ice.pace.gatech.edu" or values.get("user") != "yluo432":
        raise ValueError("The live SSH config has changed; inspect hostname and username before proceeding")
    subprocess.run(["ssh", "-S", str(SOCKET), "-O", "check", "ice-login"], check=True)


def show_job_status(p, job_id):
    """Read the live controller, falling back only for its explicit unknown-job error."""
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("--job must be a numeric job ID")
    verify_connection_config()
    command = remote_command(p, ["env", "LC_ALL=C", "scontrol", "show", "job", job_id])
    print(shlex.join(command), flush=True)
    live = subprocess.run(command, text=True, capture_output=True)
    missing = re.fullmatch(r"(?:slurm_load_jobs? error:\s*)?Invalid job id specified\.?",
                           live.stderr.strip(), re.IGNORECASE)
    # SSH errors (255), unavailable controllers, permission errors and unexpected
    # scheduler failures must retain their original diagnostics, not become "finished".
    if live.returncode == 0 or live.returncode == 255 or not missing:
        sys.stdout.write(live.stdout)
        sys.stderr.write(live.stderr)
        live.check_returncode()
        return
    if live.stdout:
        sys.stdout.write(live.stdout)
    print(f"Job {job_id} is absent from the live controller; checking Slurm accounting.", flush=True)
    fields = "JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,AllocTRES,MaxRSS"
    # --jobs makes sacct search from the Unix epoch by default, so completed jobs
    # from prior days are included. Keep job steps to expose their memory statistics.
    command = remote_command(p, ["env", "LC_ALL=C", "sacct", f"--jobs={job_id}",
                                 f"--user={p['login_username']}", "--noheader", "--parsable2",
                                 f"--format={fields}"])
    accounting = subprocess.run(command, text=True, capture_output=True)
    sys.stderr.write(accounting.stderr)
    if accounting.returncode:
        sys.stdout.write(accounting.stdout)
        accounting.check_returncode()
    matches = []
    for line in accounting.stdout.splitlines():
        identity = line.split("|", 1)[0].strip()
        if "|" in line and (identity == job_id or identity.startswith((job_id + ".", job_id + "_", job_id + "+"))):
            matches.append(line)
    if not matches:
        if accounting.stdout:
            sys.stdout.write(accounting.stdout)
        raise ValueError(f"No accounting record found for job {job_id} owned by {p['login_username']}. "
                         "Check the job ID; accounting may be delayed or its history may no longer be retained. "
                         "Completion has not been established.")
    print("Slurm accounting history (job and steps):")
    print(fields.replace(",", "|"))
    print("\n".join(matches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(BENCH / ".local" / "ice-profile.json"))
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("validate")
    for action in ("sync", "interactive", "submit", "retrieve"):
        sub = subs.add_parser(action)
        sub.add_argument("--execute", action="store_true", help="Execute the printed command (default only previews it)")
        if action == "submit":
            sub.add_argument("--manifest", required=True, help="Absolute manifest path on ICE")
            sub.add_argument("--output", required=True, help="Absolute output path on ICE")
            sub.add_argument("--model", choices=("lfm25", "smolvlm2", "internvl35", "spatialevo"), default="lfm25")
            sub.add_argument("--revision", required=True, help="Exact 40-character Hugging Face commit hash")
            sub.add_argument("--max-samples", default="16", help="Smoke sample cap (default 16), or 'all' for full evaluation")
            sub.add_argument("--afterok", help="Wait for successful completion of this numeric preparation job ID")
    status = subs.add_parser("status")
    status.add_argument("--job", help="Optional numeric job ID for detailed status")
    args = parser.parse_args()
    p = load_profile(args.profile)
    project = p["project_path"]
    if args.action == "validate":
        print("Verified profile parses successfully. Slurm resources:")
        print(shlex.join(resource_args(p)))
        return
    if args.action == "sync":
        exclusions = [".local/", ".venv*/", "datasets/", "data/", "runs/", "logs/", "cache/", ".cache/", "__pycache__/", "*.pyc", ".git/", "*.egg-info/"]
        # Deliberately no --delete. Only the benchmark checkout is synchronized.
        cmd = ["rsync", "-av", *[f"--exclude={item}" for item in exclusions], "-e", shlex.join(ssh_args()[:-1]), str(BENCH) + "/", f"ice-login:{project}/"]
        preparation = remote_command(p, ["mkdir", "-p", project])
        print(shlex.join(preparation), flush=True)
        print(shlex.join(cmd), flush=True)
        if args.execute:
            verify_connection_config()
            subprocess.run(preparation, check=True)
            subprocess.run(cmd, check=True)
        return
    if args.action == "retrieve":
        destination = BENCH / ".local" / "remote-runs"
        cmd = ["rsync", "-av", "-e", shlex.join(ssh_args()[:-1]), f"ice-login:{project}/runs/", str(destination) + "/"]
        print(shlex.join(cmd), flush=True)
        if args.execute:
            verify_connection_config()
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)
        return
    if args.action == "interactive":
        command = ["srun", *resource_args(p), "--job-name=vlm-debug", f"--chdir={project}", "--pty", "bash", "-l", f"{project}/cluster/interactive.sh", project, p["environment_path"], p["cache_path"]]
        cmd = remote_command(p, command, tty=True)
    elif args.action == "submit":
        if p["gpu_count"] != 1:
            raise ValueError("This evaluation launcher runs one model worker. Use one GPU per independent shard/job; multiple GPUs need an explicit multi-worker launcher.")
        for field in ("manifest", "output"):
            setattr(args, field, absolute_path(getattr(args, field), field))
        if not re.fullmatch(r"[a-fA-F0-9]{40}", args.revision):
            raise ValueError("Pin the exact 40-character Hugging Face commit hash, not main or a branch")
        if args.max_samples != "all" and (not args.max_samples.isdigit() or int(args.max_samples) < 1):
            raise ValueError("--max-samples must be positive or 'all'")
        logs = f"{project}/runs/slurm"
        command = ["sbatch", "--parsable", *resource_args(p), "--signal=B:TERM@90", "--job-name=vlm-eval", f"--chdir={project}", f"--output={logs}/%j.out", f"--error={logs}/%j.err", f"{project}/cluster/evaluate.sbatch", project, p["environment_path"], p["cache_path"], args.manifest, args.output, args.model, args.revision, args.max_samples]
        if args.afterok:
            if not re.fullmatch(r"\d+", args.afterok):
                raise ValueError("--afterok must be a numeric job ID")
            command[2:2] = [f"--dependency=afterok:{args.afterok}", "--kill-on-invalid-dep=yes"]
        prefix = shlex.join(["mkdir", "-p", logs]) + "; "
        cmd = remote_command(p, command, prefix=prefix)
    else:
        if args.job:
            show_job_status(p, args.job)
            return
        command = ["squeue", "-u", p["login_username"]]
        cmd = remote_command(p, command)
    print(shlex.join(cmd), flush=True)
    if args.action == "status" or args.execute:
        verify_connection_config()
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ICE: {exc}", file=sys.stderr)
        sys.exit(2)
