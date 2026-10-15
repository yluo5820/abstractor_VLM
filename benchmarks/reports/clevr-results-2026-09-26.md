# Completed CLEVR-ART comparison — September 26, 2026

Verified against live Slurm accounting and saved results at **11:47 pm EDT,
September 26**. All six evaluations completed successfully. The last finished
at **4:03:37 am EDT** that day. The user queue is empty and all GPU allocations
have been released. No further jobs were submitted during this progress check.

## Full results

Each cell uses the same complete **10,000-question OOD test task** across models.
There are 60,000 model predictions in total, distinct from the 60,000 scenes
rendered to construct the 20,000 questions. Invalid outputs count as wrong;
there were **zero invalid outputs and zero generation-budget hits** in all runs.

| Model | Relational matching (RMTS), chance 50% | Identity rules, chance 25% |
| --- | --- | --- |
| LFM2.5-VL-1.6B | 4,775 / 10,000 — **47.75%** | 2,489 / 10,000 — **24.89%** |
| SmolVLM2-2.2B | 5,048 / 10,000 — **50.48%** | 2,539 / 10,000 — **25.39%** |
| SpatialEvo-3B | 5,576 / 10,000 — **55.76%** | 4,138 / 10,000 — **41.38%** |

SpatialEvo is the strongest of these three models under the fixed direct-answer
protocol. Its improvement over uniform random guessing is 5.76 percentage points
on RMTS and 16.38 points on identity rules. Its Wilson 95% intervals are
54.79–56.73% and 40.42–42.35%, respectively. These intervals assume independent
examples; correlated examples can weaken that interpretation.

The names do not imply identical parameter counts: SpatialEvo's stored model has
approximately 4.07 billion total parameters despite its 3B name. These results
do not isolate the effect of spatial training from architecture, size or native
image preprocessing, and are not directly equivalent to task-trained
Slot-Abstractor results.

## Answer distributions

The weak baseline results include severe answer-position bias, not failures to
produce a final answer. Every raw response was exactly one allowed letter after
trimming whitespace, and every generation used two tokens including termination.

| Model | RMTS predictions | Identity-rule predictions |
| --- | --- | --- |
| LFM2.5 | A: 839; B: 9,161 | A: 10,000 |
| SmolVLM2 | B: 10,000 | B: 10,000 |
| SpatialEvo | A: 4,680; B: 5,320 | A: 2,522; B: 5,915; C: 804; D: 759 |

RMTS labels are A: 4,952 and B: 5,048. Identity-rule labels are A: 2,489,
B: 2,539, C: 2,501 and D: 2,471. Thus SmolVLM2 exactly matches the most frequent
label baseline on both tasks; LFM's identity score is its constant-A baseline.
SpatialEvo exceeds the most frequent label baseline on both tasks, although it
still favors B on identity rules. Valid parsing does not establish that a model
uses the image correctly or rule out prompt/image-layout sensitivity.

## Actual time and resources

Each GPU job used **one H200, two CPUs and 16 GiB host RAM**, with a two-hour
limit. All finished within the first allocation, so no resume is needed.
Elapsed times below include the separate technical preflight, loading and full
evaluation, and exclude time waiting in the queue.

| Model | RMTS job / elapsed | Identity-rule job / elapsed | Combined GPU time |
| --- | --- | --- | --- |
| LFM2.5 | 5948210 / 10m33s | 5948211 / 11m03s | 21m36s |
| SmolVLM2 | 5948212 / 30m28s | 5948213 / 50m40s | 1h21m08s |
| SpatialEvo | 5948214 / 16m48s | 5948216 / 20m42s | 37m30s |

Total evaluation allocation time was **2h20m14s of H200 time**, rather than the
12 GPU-hours requested as upper limits. Jobs overlapped: the first evaluation
started at 3:01:27 am and the last ended at 4:03:37 am, a 1h02m10s elapsed span.
This excludes prior rendering, CPU preparation and queue waits. PGM has different
image sizes and sample counts, so these timings should not be applied to PGM.

The repaired input preparation completed in approximately 85 minutes across two
CPU segments. The remaining two segments detected completion and exited promptly.
The readiness check succeeded before any of the six GPU evaluations began.
All 60,000 scenes had already finished rendering on September 25.

## Protocol and verification

The models used the same manifest, images, answer mapping, seeded question order,
prompt and strict letter parser within each task. Generation was greedy with a
16-new-token ceiling; native image processors and chat templates were retained.
The 32-question technical preflights are separate from the 10,000 full predictions
and do not inflate the reported counts. No accuracy-based prompt tuning was done.
Model revisions and frozen code provenance are listed in the
[campaign report](clevr-three-models-2026-09-26.md).

All result directories were downloaded locally. A read-only audit verified every
completion-receipt file hash, 10,000 unique ordered IDs per run, raw-response
correctness against labels, agreement with the summaries, matching manifest and
image hashes across models, consistent source/protocol, inputs within context
limits, H200 backend identity, and successful completion without forced kills.
The audit checks saved provenance; it does not independently establish the
semantic correctness of the original dataset labels.

Local evidence:

- `benchmarks/.local/remote-runs/clevr-controls-2026-09-26/`: all six full result
  directories, raw predictions, configurations, sessions and preflights.
- `benchmarks/.local/campaigns/clevr-controls-2026-09-26/status-20260927T034737Z.json`:
  live queue, accounting, preparation and result snapshot.
- `benchmarks/.local/campaigns/clevr-controls-2026-09-26/final-results-audit.json`:
  independent rescoring, answer counts and checksum verification.
- `benchmarks/.local/campaigns/clevr-controls-2026-09-26/audit_completed_results.py`:
  reproducible local audit.

Remote originals remain under
`/storage/ice1/9/8/yluo432/vlm-benchmark/runs/clevr-controls-2026-09-26/`.

The earlier PGM comparison segments remain stopped at their saved checkpoints;
this progress check did not resume them or allocate more compute.
