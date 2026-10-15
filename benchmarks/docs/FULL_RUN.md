# Full LFM2.5 baseline: PGM and CLEVR-ART

**Verified September 24, 9:55 pm EDT:** PGM preparation and full H200 evaluation completed successfully. CLEVR rendering is 40.1% complete. See [the current status and completed PGM result](../reports/full-run-status-2026-09-24.md). Scheduling details below are retained for reproducibility.

The user authorized scheduling the full benchmark on September 24, 2026. This
includes **both datasets**, using frozen LFM2.5-VL-1.6B on H200:

| Evaluation | Questions | Preparation |
| --- | ---: | --- |
| PGM neutral/test, pinned HuggingFaceM4 mirror | 200,000 | Convert the nine already downloaded Parquet shards into contact sheets |
| CLEVR-ART relational matching, OOD test | 10,000 | Render 20,000 original-quality scenes |
| CLEVR-ART identity rules, OOD test | 10,000 | Render 40,000 original-quality scenes |

The original PGM extrapolation archive is separate and remains unavailable to
anonymous downloads. That does not prevent this neutral evaluation. CLEVR's
symbolic labels, original renderer and assets are already available; its full
scene images have to be generated. Sixteen RMTS questions were rendered for the
earlier pilot, not the full test set.

## Scheduling and resources

PGM preparation uses a CPU-only batch job with 2 CPUs, 16 GiB RAM and up to four
hours. Evaluation requests one H200, 2 CPUs, 16 GiB RAM and ten hours. An
`afterok` dependency prevents evaluation from occupying a GPU until preparation
succeeds. A failed prerequisite prevents the dependent job from running.

The first 10,000 full-preparation examples measured 16.139 contact sheets/second,
projecting roughly 3.44 CPU-job hours for conversion. This initial rate may change;
it is separate from model inference. The preparation job started after about one
second in the queue and initially used about 4.8 GiB host RAM.

The small H200 audit projected 5.54 hours for PGM inference from only 32 examples.
It did not measure a full run. The production run reports sustained throughput
and remaining-time estimates every 25 questions, including image reads and
durable prediction writes. Inspect the first 1,000–2,000 questions before
relying on the projection. Startup, queue time and preparation are additional.

CLEVR rendering is a separate CPU workload. A short 2-CPU, 16-GiB, 20-minute
calibration checked both tasks; local tests covered safe restart behavior. The full run uses
8 independent CPU workers, 16 GiB RAM, up to three sequential 16-hour jobs, one
active at a time. This caps the rendering allocation at 384 CPU-hours
and 48 elapsed compute hours before queueing; jobs stop early when complete.
The user explicitly approved this larger rendering request on September 24.
All three chunks and their dependent preparation/evaluation jobs have been submitted.

Calibration job 5921706 completed successfully (exit 0:0, elapsed 4:13) with all
4 RMTS and 4 identity-rule problems rendered, 24 scenes total. Mean worker wall
time including Blender startup was 34.40 seconds per RMTS problem and 71.09 seconds
per identity-rule problem. This projects to 293 CPU-hours, or 36.6 elapsed hours
with 8 equally fast independent workers, before queue delays and other overhead.
This is a small-sample planning estimate, not a completion guarantee. The first
full 8-worker job 5921712 is running. Prepared images are reusable for later models.

## Restart and integrity

Each campaign runs from its own source snapshot. Do not modify that snapshot or
resume with changed models, manifests, images, settings or backend versions.
PGM preparation uses bounded batches and a durable image journal. CLEVR publishes
only complete, validated problem groups; interrupted attempts remain isolated.
The full benchmark never silently treats a partially prepared dataset as complete.

Evaluation flushes and synchronizes every prediction to disk. Slurm sends an
early termination signal, allowing the in-flight prediction to be committed.
An identical invocation resumes the saved run, skipping completed questions.
Resubmission is explicit; a time limit does not automatically create unlimited
new allocations. A final score is complete only when `summary.json` records
`complete: true` and the expected number of examples.

SSH/Duo authentication is needed only if the reusable connection expires.
Submitted batch jobs continue independently of the laptop and SSH session.
Use the status helper for live queue state or accounting history after completion.

## Accepted PGM jobs

- **5921697**: CPU preparation, 2 CPUs / 16 GiB / 4 hours, submitted September 24.
- **5921699**: full LFM2.5 evaluation, 1 H200 / 2 CPUs / 16 GiB / 10 hours;
  waits for successful completion of 5921697.

Local receipt: `.local/campaigns/full-lfm25-2026-09-24/pgm-submission.json`.
Remote campaign: `/storage/ice1/9/8/yluo432/vlm-benchmark/runs/full-lfm25-2026-09-24/`.
Full predictions: `/storage/ice1/9/8/yluo432/vlm-benchmark/runs/lfm25-pgm-neutral-h200-full/`.

From the Mac, check all jobs or a particular job without a new login while the
reusable SSH connection remains alive:

```bash
python3 /Users/yluo/Downloads/Projects/VLM/benchmarks/cluster/ice.py status
python3 /Users/yluo/Downloads/Projects/VLM/benchmarks/cluster/ice.py status --job 5921699
```

An expired connection requires the existing `cluster/connect.sh` authentication
step. It does not interrupt the submitted jobs. Status lookup falls back to
Slurm accounting when a completed job has left the live queue.


## Accepted CLEVR calibration job

**5921706** completed 4 new problems per task (24 scenes total), using 2 CPUs,
16 GiB RAM and a 20-minute cap. Its completed images seed the full resumable
render root, `/storage/ice1/9/8/yluo432/vlm-benchmark/data/clevr-full-seed1`.
The user approved full 8-CPU rendering. Slurm accepted the following sequential
chain; completed images persist across chunks and an unnecessary chunk exits early.

| Stage | Accepted job | Waits for |
| --- | --- | --- |
| render chunk 1 | 5921712 | afterok:5921706 |
| render chunk 2 | 5921713 | afterok:5921712 |
| render chunk 3 | 5921714 | afterok:5921713 |
| publish manifests | 5921715 | afterok:5921714 |
| evaluate rmts | 5921716 | afterok:5921715 |
| evaluate idrules | 5921717 | afterok:5921716 |

Each CLEVR evaluation requests one H200, 2 CPUs, 16 GiB RAM and one hour.
The two evaluations run in sequence. The manifest stage fails if either
10,000-question split is incomplete; GPU
evaluations then do not run. This bounds total rendering requests without
misreporting a partial benchmark as complete. A later continuation requires a
new explicit submission from the retained checkpoint.


## Validation at submission

All 54 package tests and 21 cluster-launcher tests passed. Real calibration
produced both task types successfully, and representative contact sheets were
visually inspected. No full-dataset model score has yet completed.
