# First-pass scope and follow-up

**Current update, September 24:** The user has expanded the scope to full LFM2.5 evaluation on PGM neutral/test (200,000), CLEVR-ART RMTS/test (10,000), and CLEVR-ART identity-rules/test (10,000), using H200. See [the full-run plan](FULL_RUN.md) for preparation, scheduling and current resource approvals. Statements below describing the earlier 16-example CLEVR limit or deferred scope are historical.

The user selected readily accessible data for the first pass: LFM2.5-VL-1.6B on PGM neutral and already-rendered CLEVR-ART RMTS. Larger CPU rendering requests are deferred, with no new approval sought and no larger jobs submitted. Keep the existing one-V100, two-CPU, 16-GiB, 20-minute resource limits.

## Completed PGM neutral pilot

All nine pinned neutral/test mirror shards have been downloaded on ICE. A deterministic 512-example manifest samples across their full 200,000-example pool. Pilot job `5919573` completed successfully (`COMPLETED`, exit `0:0`, elapsed `00:12:43`), with all 512 examples evaluated. Its score was 67/512 (13.09%) with zero invalid outputs. The results have been retrieved; do not submit a duplicate pilot.

From the local `benchmarks/` directory:

```bash
python3 cluster/ice.py status --job 5919573
python3 cluster/ice.py retrieve --execute
```

The recorded remote output directory is `/storage/ice1/9/8/yluo432/vlm-benchmark/runs/lfm25-pgm-neutral-hf-pilot512`. Its logs are under the remote checkout's `runs/slurm/5919573.out` and `.err`. Scheduler accounting, the completed summary, and all 512 prediction records were checked together. See [VERIFIED_ICE.md](VERIFIED_ICE.md) for evidence and artifact links, and [the report](../reports/first-pass.md) for results.

The saved backend dtype is `torch.bfloat16`, using emulation on V100. The completed PGM pilot measured approximately 1.42 seconds/example for generation, excluding loading and preparation. Do not label these runs FP16 or assume that this throughput carries over to another precision, GPU, or model.

The pilot uses the converted PGM neutral mirror. Disclose that source and keep conclusions specific to neutral data; it does not test the unavailable official extrapolation regime.

## Completed validation and available RMTS data

SSH authentication, scheduler settings, the Python/CUDA environment, and remote Codex are working. Codex `login status` returned `Logged in using ChatGPT`, and a compute-node `codex exec` connectivity check returned `READY`. The initial interactive allocation `5919352` has been released, with `sacct` showing `COMPLETED`, exit `0:0`, and elapsed `18:34`.

The LFM2.5 smoke runs completed on 16 PGM neutral examples and 16 already-rendered RMTS examples, each with zero invalid outputs. Red/blue image controls also returned correct answers. Summaries and predictions are linked in [VERIFIED_ICE.md](VERIFIED_ICE.md). These small smoke runs establish a working inference/evaluation path; they are not a full comparison of model families or held-out appearance regimes.

Use the already-rendered RMTS material for this first-pass report. Inspect the saved answers and failures alongside the PGM pilot before deciding whether a larger dataset or another model is warranted. The other model backends are prepared but have not been benchmarked in this first pass.

## Deferred work

The earlier proposal for 512 RMTS examples and 512 identity-rule examples would require new rendering. Based on observed original-quality rendering times, the combined work was estimated at approximately 12–15 CPU-hours before overhead. The proposed two CPU-only jobs, each with up to 8 CPUs, 16 GiB RAM, and 2 hours, exceed the current first-pass resource scope. They were not submitted, and their thread scaling has not been measured.

Keep that rendering proposal deferred. Revisit it only if the user later chooses to expand the dataset scope; any larger resource request then needs explicit approval. There is no need to request that approval to finish the current accessible-data pilot and report.

The official PGM extrapolation download remains inaccessible. Neutral mirror results and the existing RMTS smoke data do not resolve the full original benchmark request across systematic-generalization regimes.
