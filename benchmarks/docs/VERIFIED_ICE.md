# Verified ICE setup record

Recorded September 23, 2026, EDT (September 24 UTC). This is a snapshot of a working connection, allocation, and software setup. Job IDs and compute nodes below are historical observations, not reservations for future sessions.

The user subsequently selected H200 as the standard. The default profile now requests it through `ice-gpu`; see [the H200 audit](GPU_AUDIT.md) for job `5920012` and verified GPU-hour limits. The V100 commands below reproduce the initial condition.

| Setting | Verified value |
| --- | --- |
| Local SSH alias | `ice-login` |
| Configured login address | `yluo432@login-ice.pace.gatech.edu` |
| Reached login host | `login-ice-gnr-2` |
| Slurm account | `coc` |
| Partition | `coc-gpu` |
| QoS | `coc-ice` |
| GPU request | `--gres=gpu:v100:1` |
| Initial allocation | 1 GPU, 2 CPUs, 16 GiB RAM, 20 minutes |
| Historical V100 profile | `benchmarks/.local/ice-v100-profile.json` |
| Designated remote checkout | `/storage/ice1/9/8/yluo432/vlm-benchmark` |
| Python environment | `/storage/ice1/9/8/yluo432/envs/vlm-bench` |
| Cache root | `/storage/ice1/9/8/yluo432/vlm-cache` |
| Hugging Face cache | `/storage/ice1/9/8/yluo432/vlm-cache/huggingface` |
| Additional modules | None in the verified profile |

The authenticated SSH master uses the local socket `benchmarks/.local/ice.sock`. The user completed VPN/password/MFA steps directly. Reauthenticate with `bash cluster/connect.sh` when the master expires; keep using the login alias rather than an old compute-node alias.

Scheduler evidence included the `ice-slurm` association for account `coc` and user `yluo432`, with `coc-ice` allowed/default QoS. The `coc-gpu` partition permitted that QoS, and GPU discovery exposed V100 resources. A real allocation then confirmed the request worked.

## Reuse the connection and request a fresh allocation

The preferred local command, from `benchmarks/`, is:

```bash
python3 cluster/ice.py --profile .local/ice-v100-profile.json interactive --execute
```

The equivalent direct workflow starts with `ssh ice-login`. On the login node, run:

```bash
srun --account=coc --partition=coc-gpu --qos=coc-ice \
  --gres=gpu:v100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=16G --time=00:20:00 \
  --job-name=vlm-debug \
  --chdir=/storage/ice1/9/8/yluo432/vlm-benchmark \
  --pty bash -l \
  /storage/ice1/9/8/yluo432/vlm-benchmark/cluster/interactive.sh \
  /storage/ice1/9/8/yluo432/vlm-benchmark \
  /storage/ice1/9/8/yluo432/envs/vlm-bench \
  /storage/ice1/9/8/yluo432/vlm-cache
```

The outer login shell initializes the node's environment. The project startup script then validates the Slurm GPU allocation, activates the benchmark environment, selects its cache, and opens `bash --noprofile --norc -i`. This last step matters: the user's normal shell startup was observed to reset `HF_HOME` and activate conda base.

Inside the resulting compute shell, inspect and launch Codex:

```bash
hostname -f
printf '%s\n' "$SLURM_JOB_ID" "$HF_HOME"
python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())'
cd /storage/ice1/9/8/yluo432/vlm-benchmark
/storage/ice1/9/8/yluo432/tools/codex-0.156.1/bin/codex login status
/storage/ice1/9/8/yluo432/tools/codex-0.156.1/bin/codex
```

Codex authentication was independently confirmed: `login status` returned `Logged in using ChatGPT`. An actual `codex exec` call inside the compute node returned `READY`; this verifies model connectivity only. Its retrieved output is saved in [codex-connection-check.txt](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/codex-connection-check.txt). If authentication expires, the user runs `/storage/ice1/9/8/yluo432/tools/codex-0.156.1/bin/codex login --device-auth` and completes the browser step privately. No authentication token or one-time device code belongs in this record.

Exit this compute shell when finished to release its allocation. Another local terminal does not inherit it. Use `sbatch` for longer experiments, with each batch job requesting its own resources. Obtain explicit user approval before increasing resources or wall time, cancelling existing jobs, or making destructive changes.

## Allocation observed during setup

| Observation | Value |
| --- | --- |
| Interactive job | `5919352` |
| Compute hostname | `atl1-1-02-009-31-0.pace.gatech.edu` |
| GPU | NVIDIA Tesla V100-PCIE, 16 GB |
| NVIDIA driver | `575.57.08` |
| Start time | September 23, 2026, 23:22:07 EDT |
| Scheduled end | September 23, 2026, 23:42:07 EDT |
| Final accounting state | `COMPLETED`, exit `0:0` |
| Recorded elapsed time | `18:34` |

This interactive allocation was released after setup and smoke tests. Request a fresh allocation for further interactive work; do not reconnect to its recorded compute node. Before acting on any job, inspect `squeue -u "$USER"` and, when relevant, `scontrol show job JOB_ID`. The completed state above was confirmed through `sacct`; absence from the queue alone does not establish success.

## Storage and software

At initial inspection, scratch had a 300 GB quota with approximately zero usage; home used approximately 16.4 GB of its 30 GB quota. These figures preceded package/model/data downloads and will change. Scratch was selected for this checkout, environment, cache, and installed tools.

Existing environments were inspected first. A new environment was created for the benchmark; model-specific pinned requirements were installed separately without changing the old environments.

| Component | Observed version or location |
| --- | --- |
| Python | `3.11.7` |
| PyTorch | `2.7.1+cu126` |
| Transformers | `5.1` |
| CUDA availability in PyTorch | `True` |
| Dependency consistency | `pip check` passed |
| Codex CLI | `0.156.1` at `/storage/ice1/9/8/yluo432/tools/codex-0.156.1/bin/codex` |
| Blender | `2.79b` under `/storage/ice1/9/8/yluo432/tools/blender-2.79b` |

Codex and Blender were installed from official distributions with checksums checked. Codex uses the complete native package, including its auxiliary tools; it does not require Node/npm or `sudo`. The Codex installation helper does not edit shell startup files.

The [official ICE orientation](https://docs.pace.gatech.edu/ice_cluster/img/ICE_orientation_spring2023.pdf) has an indexed software-stack section supporting custom Python environments and user-installed software. The [ServiceNow ICE page supplied for this task](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042503) was inaccessible to the documentation tool; no scheduler settings were inferred from it. Actual account, partition, QoS, and GPU settings came from ICE itself.

Official Codex references: [CLI installation](https://learn.chatgpt.com/docs/codex/cli) and [authentication](https://learn.chatgpt.com/docs/auth). A HEAD request returning HTTP 403 from an authentication site's root does not by itself determine whether the supported device-login flow works.

## Experiment status at this snapshot

The user narrowed the first pass to readily accessible PGM neutral and already-rendered CLEVR-ART RMTS, using LFM2.5-VL-1.6B. The 16-example PGM neutral mirror smoke run completed with 2 correct (12.5%); the 16-example RMTS smoke run completed with 6 correct (37.5%). Both had zero invalid outputs. Simple red/blue image controls also returned the correct answers. These small runs validate the evaluation path and provide preliminary observations; they do not establish broad reasoning performance or cover all systematic-generalization regimes.

Retrieved local artifacts are linked below. Matching run directories are stored under `/storage/ice1/9/8/yluo432/vlm-benchmark/runs/` on ICE.

- PGM neutral smoke: [summary](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-pgm-neutral-hf-smoke/summary.json) and [predictions](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-pgm-neutral-hf-smoke/predictions.jsonl).
- CLEVR-ART RMTS smoke: [summary](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-clevr-rmts-smoke/summary.json) and [predictions](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-clevr-rmts-smoke/predictions.jsonl).
- PGM neutral 512-example pilot: [summary](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-pgm-neutral-hf-pilot512/summary.json), [predictions](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-pgm-neutral-hf-pilot512/predictions.jsonl), [recorded configuration](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-pgm-neutral-hf-pilot512/config.json), [Slurm stdout](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/slurm/5919573.out), and [Slurm stderr](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/slurm/5919573.err).

Pilot job `5919573` completed with Slurm state `COMPLETED`, exit `0:0`, and elapsed `00:12:43`. It requested its own one V100 GPU, two CPUs, 16 GiB RAM, and 20 minutes, independently of the released interactive allocation. The final summary has `complete: true` and exactly 512 predictions: 67 correct (13.09%), zero invalid outputs, 95% Wilson interval 10.44%–16.28%. Mean measured generation latency was 1.418 seconds/example and peak PyTorch allocated GPU memory was 4.23 GiB. Precision was `torch.bfloat16`, including emulation on V100, not native BF16 tensor-core execution.

The pilot's 483 E predictions indicate a strong answer-position bias; always choosing E also scores 67/512 on this sample. The accuracy interval contains the 12.5% chance rate. These results do not establish above-chance reasoning under this frozen, zero-shot protocol. Read the [complete report](../reports/first-pass.md) for all run details and limitations.

All input manifests, selected IDs, and contact-sheet hashes were verified after retrieval. Both jobs' completed states were confirmed with `sacct`, and `squeue -u "$USER"` was empty at the final check on September 24 around 03:54 UTC. Larger CPU rendering requests were deferred and were not submitted. No allocation remains in use for this task.
