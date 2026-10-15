# H200 resource and resumption audit

Updated September 24, 2026. H200 is the user's requested standard for subsequent experiments. The earlier V100 results remain a separate hardware condition.

## Completed results

Job `5920012` finished successfully: Slurm `COMPLETED`, exit `0:0`. It was submitted at 00:24:02 EDT, started at 00:33:16 and ended at 00:36:54. The actual queue wait was **9 minutes 14 seconds** and runtime **3 minutes 38 seconds**. The earlier test-only estimates of many hours did not predict this actual start. The live job record was later purged; accounting retains the result.

The device was an NVIDIA H200. All 18 profiling conditions completed; archive/source fingerprints, sample identities and scores were verified after retrieval. See the [raw audit report](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/gpu-audit-2026-09-24/h200.md) and [full JSON](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/gpu-audit-2026-09-24/h200.json).

| Task/condition | Observed examples/s | Sample | Full-test inference extrapolation |
| --- | ---: | ---: | ---: |
| PGM, serial, 2 CPU threads | 10.027 | 32 | 200,000 examples: **5.54 hours** |
| PGM, native batch 8, 2 threads | 6.530 | 32 | Slower than serial in this implementation |
| RMTS, serial, 2 CPU threads | 28.535 | 16 | 10,000 examples: **5.84 minutes** |
| RMTS, native batch 8, 4 threads | 64.914 | 16 | Two answers changed; not adopted as an equivalent condition |

The initial recommendation is **one H200, two CPUs, 16 GiB host RAM, serial inference**. The two-thread measurement was made inside a four-CPU allocation; validate sustained throughput with an actual two-CPU allocation before a full run. Peak Slurm-reported host RSS was about 5.11 GiB (including loading), and measured serial PGM GPU allocation peaked at 3.22 GiB. More CPUs or large RAM requests are not supported by these measurements. Model loading took 76.6 seconds.

The PGM rate is about 14.2 times the earlier V100 pilot's measured rate, but this compares a small in-memory H200 audit with the earlier configuration, not a controlled hardware-only study. It excludes file reading/persistence, loading and dataset preparation. An 8–10-hour H200 allocation is a plausible planning window for neutral/test after preparation and sustained-throughput validation, not a guaranteed completion time. Resume remains available. No full benchmark or multi-GPU run was launched.

All PGM conditions matched their H200 serial reference on these 32 examples. RMTS batches of two or more changed two of 16 predictions while leaving aggregate accuracy unchanged; they are excluded from the equivalence-based recommendation. Tiny timing differences between nearby conditions should not be overinterpreted. Identity-rules rendering/inference remains unmeasured.

The 32 shared PGM outputs also match the earlier V100 pilot, but H200 serial RMTS differs from V100 on three of 16 examples (7/16 versus 6/16 correct). Image/manifest hashes are identical. Keep those hardware conditions separate; do not combine them in a resumed accuracy run. The audit fixes PyTorch inter-op threads to one and preloads images, unlike the ordinary end-to-end runner. These short, single-pass conditions establish a practical starting point rather than a proven optimum; a sustained run is needed to confirm completion-time estimates.

`cluster/ice.py status --job JOB_ID` now checks historical `sacct` records when `scontrol` reports an invalid live job ID. Authentication/network/controller errors remain errors. Twenty cluster tests passed after this correction, and the user's exact status command was verified against this completed job.

## Verified access and limits

The user's Slurm association is account `coc`, QoS `coc-ice`. The shared `ice-gpu` partition permits that QoS and exposes H200 and H100 nodes. The earlier `coc-gpu` partition did not contain those GPU types. `coe-gpu` also contains them but does not permit this user's verified QoS, so it is not used.

Live QoS limits are `MaxTRESMinsPerJob=gres/gpu=960,cpu=30720`. The GPU partition has `MaxTime=16:00:00`. Therefore the requested GPU count multiplied by requested hours must be at most **16 GPU-hours per job**; the corresponding CPU allowance is 512 CPU-hours per job. Examples are one GPU × 16 hours, two × 8 hours, and four × 4 hours. These are limits on requested allocation, not a promise of immediate access. No aggregate per-user GPU-minute limit appeared in the inspected fields; that does not imply unlimited practical concurrency or priority.

The launcher now validates recorded partition, GPU-minute and CPU-minute limits before submission. Its regular evaluation command requires one GPU per worker so that a multi-GPU request cannot silently leave extra GPUs unused. A future parallel run should use disjoint data shards and independent model workers; the present production runner is not a multi-GPU launcher.

## Submitted audit

Job **5920012** was submitted at September 24, 00:24:02 EDT and was pending at the initial post-submission check. It has since completed as recorded above. Earlier `sbatch --test-only` projections were hours away and varied between requests; they were not reservations or firm completion times.

- Account `coc`, partition `ice-gpu`, QoS `coc-ice`.
- One H200, four CPU cores, 16 GiB host RAM, 20-minute maximum allocation.
- Explicit BF16; the device name must contain H200, with no automatic fallback to another GPU.
- Same pinned LFM2.5 checkpoint and existing PGM/CLEVR images. No new model weights or full datasets are downloaded.
- Fixed 32-example PGM sample and 16-example RMTS sample; native short-answer prompts unchanged.
- CPU thread counts 1, 2, 4; batch sizes 1, 2, 4, 8, with warm-up and measured conditions.
- Native batch outputs and input-token counts are checked against the serial reference. Changed outputs are exposed, not silently treated as equivalent.
- Reports include throughput, latency, CPU process time, sampled host RSS, GPU allocated/reserved peaks, answer validity, and accuracy on this small sample.

The audit has a 720-second soft measurement budget after model loading; it checks the deadline between batches and preserves partial conditions. Full Slurm runtime is capped at 20 minutes. Native batching errors are saved while serial results remain useful. A partial/time-budget result is not a completed audit.

The preceding immediate V100 validation request (`5919978`) could not acquire a slot within its 15-second bound. It exited without allocating resources or running inference; Slurm recorded cancellation of that unstarted request. No existing user job was cancelled. The audit subsequently exercised native batching on H200. Multi-GPU scaling remains unmeasured.

The prior V100 pilot reported peak host RSS of approximately 1.32 GiB and peak PyTorch allocated GPU memory of 4.23 GiB. Host RAM and GPU VRAM are separate resources. These measurements support starting with 16 GiB host RAM rather than requesting a large amount arbitrarily; H200 measurements will refine that choice. Four CPUs allow a bounded thread sweep and do not establish that four are needed for every future run.

## Frozen code and outputs

The queued job runs from an isolated source snapshot, so later checkout edits cannot change its code. Snapshot SHA-256:

`39342f98a3dcc8ab255cc20194f0abc3afc4323fe9622ab69e73c12c5d89f9e5`

Remote output directory:

`/storage/ice1/9/8/yluo432/vlm-benchmark/runs/gpu-audit-2026-09-24`

The job will write `h200.json` and a readable `h200.md`, plus `5920012.out`/`.err`. The JSON preserves per-example observations. It also retains the source archive, extracted source, and resolved environment. A completion report extrapolates PGM neutral and RMTS runtime only from completed measured conditions; it does not invent identity-rules timings.

From local `benchmarks/`, reuse these commands in a later session:

```bash
bash cluster/connect.sh
python3 cluster/ice.py status --job 5920012
python3 cluster/ice.py retrieve --execute
```

The connection helper reuses a live connection without prompting. If it has expired, the user completes password/Duo privately. Successful batch submission is independent of that connection, so the queued/running job survives laptop disconnection. A fresh login may be needed to inspect or retrieve the output, not to let the batch execute.

The default local profile is now H200 with two CPUs and 16 GiB host RAM at `.local/ice-profile.json`; the four-CPU audit profile remains `.local/ice-h200-audit-profile.json`. The earlier V100 profile is retained at `.local/ice-v100-profile.json`. These profiles still request only 20 minutes until a longer experiment is deliberately selected.

## Resume improvements completed

Every committed prediction is flushed and synced to disk. A repeated invocation with identical inputs, code, model, environment and hardware provenance skips those predictions. The runner rejects changed settings instead of mixing experimental conditions. Use a frozen snapshot for long multi-allocation experiments; do not mix the previous V100 outputs into a new H200 run.

The runner now handles SIGTERM/SIGINT at a safe boundary after persisting the in-flight answer. The normal batch launcher requests `--signal=B:TERM@90`; its batch script replaces itself with the Python runner, so the early signal reaches that process. A termination-driven partial run exits 143 and has an interrupted session record. A subsequent identical submission resumes it. An abrupt SIGKILL cannot run cleanup, but already-synced complete records remain recoverable and a truncated final line is preserved separately for inspection.

Incremental statistics replace the former repeated history scans, and summary snapshots default to every 25 predictions plus exit. This removes quadratic summary work for large manifests. The final summary, prediction log, and scheduler exit status must still be checked together.

Validation: **36 package tests and 14 cluster tests passed**. A real subprocess test delivered SIGTERM during an example, verified the committed prefix was byte-preserved, restarted with only unfinished examples, and checked final equivalence without duplicate work. These are local software tests; a cross-allocation H200 continuation has not yet been run. Resumption is supported, but automatic chains of long jobs have not been submitted.

## Original PGM archives

The user's signed-in screenshot shows original archives, including extrapolation at 40.4 GB and neutral at 51.1 GB. Anonymous CLI requests still fail. This is an authentication/access distinction, not evidence that the files are absent. Listing alone does not prove object-download permission. See [the bounded access audit](PGM_ACCESS_AUDIT.md) for the separate Google login and 1 KiB read check needed before fetching original data. The H200 audit already has its neutral mirror inputs and does not need that login.
