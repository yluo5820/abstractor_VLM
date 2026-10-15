# ICE through SSH and Slurm

The SSH connection and a modest Slurm GPU allocation were verified on September 23, 2026 (EDT). The inspected SSH configuration identifies `ice-login` as `yluo432@login-ice.pace.gatech.edu`; the reached login host was `login-ice-gnr-2`. The verified resources are account `coc`, partition `coc-gpu`, QoS `coc-ice`, and `--gres=gpu:v100:1`. The live setup record, exact paths, and reusable direct allocation command are in [VERIFIED_ICE.md](VERIFIED_ICE.md).

An old compute-node alias, recorded hostname, or previous job ID is not evidence of a current allocation. Check the scheduler when reconnecting. The example profile remains deliberately unverified; the working configuration is saved separately in `.local/ice-profile.json`.

All commands below start in this project's `benchmarks` directory. Local commands run on the Mac unless explicitly marked as running in the compute shell. No desktop or IDE is required.

## 1. Authenticate, then inspect

Enable the Georgia Tech VPN if required, then run:

```bash
bash cluster/connect.sh
```

Enter passwords and Duo/MFA responses in that terminal. The connection uses a dedicated control socket at `benchmarks/.local/ice.sock`; it does not expose credentials to the agent. Return to Codex after authentication. If the connection later expires, run the same command again.

The following command performs read-only discovery and saves a dated report under `.local`:

```bash
bash cluster/inspect.sh
```

It reads the current user's jobs, up to ten existing job configurations, account associations, QoS definitions, partitions and GPU resource names. It also reports existing modules, environment/project directory names, storage availability, and bounded HTTPS connection checks. It installs nothing, submits no jobs, and cancels nothing. Failed or restricted queries are retained in the report; they must not be interpreted as permission to guess.

The current account, partition, QoS, and GPU syntax were established from scheduler associations and partition/GPU discovery, then exercised in a real allocation. `coc-ice` is the verified QoS; the account is `coc` and the partition is `coc-gpu`. For another account or changed cluster configuration, repeat discovery and use an existing working portal job when available, with PACE support or documentation to resolve ambiguity.

## 2. Record the verified configuration

The verified profile already exists at `.local/ice-profile.json`. Do not replace it with the example. For an unrelated future setup, start from `cluster/profile.example.json` in a separate profile file and pass its path using `ice.py --profile PATH`.

Fill in account, partition, QoS, and the exact GPU request. Common Slurm syntax forms supported by the launcher include `--gres=gpu:1`, `--gres=gpu:TYPE:1`, and `--gpus=1`; only use the form supported by ICE's discovered configuration. An empty QoS means the default was verified. Record the evidence and date in `verification_notes`, then set `verified` to `true`.

The designated remote checkout is `/storage/ice1/9/8/yluo432/vlm-benchmark`; it receives the contents of local `benchmarks/`. The new, separate environment is `/storage/ice1/9/8/yluo432/envs/vlm-bench`, and the selected cache root is `/storage/ice1/9/8/yluo432/vlm-cache`. Existing environments were inspected and left unchanged. These paths are saved as `project_path`, `environment_path`, and `cache_path`; no additional modules are currently required. Paths must be absolute and use ordinary letters, digits, dots, slashes, underscores, pluses, and hyphens.

The initial resource ceiling is one GPU, two CPUs, 16 GiB RAM, and 20 minutes. The current first pass uses these limits with readily accessible PGM neutral data and already-rendered CLEVR-ART RMTS. Larger CPU rendering work is deferred; no larger request has been approved or submitted. Do not increase resources automatically after a failure. A future larger request requires the user's explicit permission, recorded in `user_resource_approval`; this includes longer batch wall times. Set `gpu_count` consistently with the actual GPU request.

```bash
python3 cluster/ice.py validate
python3 cluster/ice.py sync
python3 cluster/ice.py sync --execute
```

The first `sync` previews the exact command. `--execute` copies only this benchmark checkout with `rsync`, without deletion. Local environments, data, results, caches, logs, and `.local` are excluded. The selected remote directory is created if necessary. Existing files with matching benchmark paths may be updated, so inspect the destination before selecting it. The live SSH hostname and username are checked before execution.

## 3. Open a modest compute shell

```bash
python3 cluster/ice.py interactive
python3 cluster/ice.py interactive --execute
```

This runs the verified equivalent of `srun RESOURCE_FLAGS --pty bash -l`, through the authenticated login connection. The startup script checks `SLURM_JOB_ID`, rejects login hostnames, requires Slurm-assigned GPU identifiers, queries GPU availability, and prints the job's allocation details before opening the interactive shell. Its final shell uses `bash --noprofile --norc -i`: the user's normal startup files otherwise overwrite `HF_HOME` and activate conda base. This preserves the selected benchmark environment and cache. No separate SSH connection to a compute node is used.

Commands in another local terminal do not inherit this shell or its allocation. Within the compute shell, check the Python path and GPU support before inference. Reuse the inspected environment when possible:

```bash
hostname -f
printf '%s\n' "$SLURM_JOB_ID"
source /storage/ice1/9/8/yluo432/envs/vlm-bench/bin/activate
python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())'
```

The capitalized paths in this guide are placeholders. Substitute the inspected remote paths; do not execute them literally.

If a new environment is needed, verify PACE's user-local installation policy first. With the chosen Python module loaded, run this **in the allocated compute shell**:

```bash
bash cluster/setup_environment.sh --policy-verified \
  /ABSOLUTE/NEW/ENVIRONMENT /ABSOLUTE/VERIFIED/CACHE \
  /ABSOLUTE/VERIFIED/PROJECT/requirements.txt
```

This creates a new virtual environment, installs the designated requirements, and records resolved versions. It refuses to overwrite an existing environment. Review the project's requirements and Python compatibility before installation. Confirm outbound access from the compute node as well; successful login-node checks do not establish compute-node access.

Codex CLI `0.156.1` is installed at `/storage/ice1/9/8/yluo432/tools/codex-0.156.1/bin/codex`, using the official native package and its verified checksum. The installation needed neither Node/npm nor `sudo` and did not edit shell startup files. `codex login status` confirmed `Logged in using ChatGPT`; an actual `codex exec` call from the compute node returned `READY`, confirming model connectivity. This was a connectivity check, not a benchmark. To start remote Codex **inside the allocated compute shell**:

```bash
cd /storage/ice1/9/8/yluo432/vlm-benchmark
/storage/ice1/9/8/yluo432/tools/codex-0.156.1/bin/codex login status
/storage/ice1/9/8/yluo432/tools/codex-0.156.1/bin/codex
```

If authentication has expired, the user completes `codex login --device-auth` directly with that binary. If remote Codex is unavailable, keep Codex local and use this one designated checkout and the SSH/Slurm launchers. A successful login-node HTTP check does not establish compute-node model connectivity; an HTTP 403 from a HEAD request to an authentication site's root is not a definitive test of the actual login flow.

Exit the compute shell when debugging is complete to release that interactive allocation. The launcher does not cancel pre-existing jobs. If a disconnection leaves a job behind, inspect it and ask the user before cancelling it.

## 4. Submit evaluation as its own Slurm job

The LFM2.5 smoke runs and PGM neutral pilot job `5919573` have completed. The example below documents the reusable procedure for a new run. Prepare its dataset and manifest in the remote storage selected after quota inspection, pin the model's exact 40-character Hugging Face revision, and begin a new model/data configuration with 16 samples:

```bash
python3 cluster/ice.py submit \
  --manifest /ABSOLUTE/REMOTE/MANIFEST.jsonl \
  --output /ABSOLUTE/VERIFIED/PROJECT/runs/lfm25-smoke \
  --model lfm25 --revision EXACT_40_CHARACTER_COMMIT_HASH
```

Add `--execute` to submit the reviewed command. This uses `sbatch` with its own resources, independent of the interactive job. Logs are written to `PROJECT/runs/slurm/JOB_ID.out` and `.err`. The batch script validates the GPU allocation before invoking `python -m vlm_bench.run` with the selected environment, cache, manifest, output path, model, and revision.

Use `--max-samples all` for a full evaluation only after a successful smoke run and a runtime estimate. The wrapper translates this to the runner's `--max-samples 0`. If the estimate needs more than the initial resources or wall time, obtain approval and update the profile before submission. Never run substantial inference on a login node.

```bash
python3 cluster/ice.py status
python3 cluster/ice.py status --job JOB_ID
python3 cluster/ice.py retrieve
python3 cluster/ice.py retrieve --execute
```

`retrieve` copies the remote checkout's `runs/` into local `.local/remote-runs/`, without deletion and without overwriting local `runs/`. Save results under the remote checkout's `runs/` to use this helper. For a finished job no longer visible in `squeue`, use `sacct` on the login node to inspect its exit state and resource usage; do not infer completion from disappearance alone.

The launcher's default is to print commands. Only `--execute` mutates the remote checkout, requests an allocation, submits a job, or retrieves results. `status` and `inspect.sh` execute read-only queries. Local launcher checks can be run with `python3 cluster/test_ice.py`; they do not contact ICE.

## Sources and remaining verification

- [PACE ICE documentation supplied for this task](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042503). The preparation environment could not read this ServiceNow page; no installation policy or Slurm values are claimed from it.
- [Official ICE orientation](https://docs.pace.gatech.edu/ice_cluster/img/ICE_orientation_spring2023.pdf). The official document's indexed software-stack section explicitly describes custom Python environments and installing one's own software; the PDF itself was unavailable to the browsing tool. This evidence supported user-local installation, without claiming special approval of a particular external service.
- [Official Codex CLI installation](https://learn.chatgpt.com/docs/codex/cli) and [headless authentication](https://learn.chatgpt.com/docs/auth).
- [Slurm srun reference](https://slurm.schedmd.com/srun.html), including allocation resource options and interactive execution.
- [Slurm sbatch reference](https://slurm.schedmd.com/sbatch.html), for jobs that allocate their own resources.
- [Slurm sacctmgr reference](https://slurm.schedmd.com/sacctmgr.html), for association and QoS discovery.

SSH authentication, the Slurm resource configuration, storage paths and quota, the separate Python environment, CUDA availability, and user-local tool installation have been exercised on ICE. `pip check`, Codex login verification, and a compute-node Codex connectivity call passed. LFM2.5 completed 16-example smoke evaluations on both the accessible PGM neutral mirror and already-rendered CLEVR-ART RMTS data, with zero invalid outputs. The saved backend dtype is `torch.bfloat16`, using emulation on V100; these runs must not be described as FP16. Results and artifact links are recorded in [VERIFIED_ICE.md](VERIFIED_ICE.md).

The initial interactive job `5919352` was released and recorded by `sacct` as `COMPLETED`, exit `0:0`, after 18:34. The separate 512-example PGM neutral pilot, job `5919573`, used the same one-V100, two-CPU, 16-GiB, 20-minute request and completed with exit `0:0` after 12:43. All 512 examples were evaluated, and no allocation remained at the final queue check. See [VERIFIED_ICE.md](VERIFIED_ICE.md) and [the report](../reports/first-pass.md) for final results. Do not submit a duplicate pilot. The first-pass scope and deferred work are recorded in [NEXT_RUNS.md](NEXT_RUNS.md).
