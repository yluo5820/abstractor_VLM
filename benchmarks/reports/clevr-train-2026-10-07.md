# CLEVR training renders and baseline decision — 7 October 2026

The user authorized beginning the complete CLEVR training render and increasing CPU parallelism. The user prefers spending GPU time on fine-tuning the existing VLMs if reproducing Slot Abstractor requires extensive training. This campaign therefore schedules **only CPU rendering and dataset verification**. Slot Abstractor's reported scores remain literature reference values; they are not our reproduced results.

## Submitted CPU campaign

| Job | Work | Resources | Limit |
| --- | --- | --- | --- |
| 6091746 | Training shard 0: original question indices 0, 4, 8, … | 16 CPUs, 32 GiB RAM, no GPU | 16 h |
| 6091747 | Training shard 1: original question indices 1, 5, 9, … | 16 CPUs, 32 GiB RAM, no GPU | 16 h |
| 6091748 | Training shard 2: original question indices 2, 6, 10, … | 16 CPUs, 32 GiB RAM, no GPU | 16 h |
| 6091749 | Training shard 3: original question indices 3, 7, 11, … | 16 CPUs, 32 GiB RAM, no GPU | 16 h |
| 6091750 | Verify all images and publish native training indices after all four jobs succeed | 2 CPUs, 8 GiB RAM, no GPU | 1 h |

Account `coc`, QoS `coc-ice`, partition `ice-cpu` were verified live. The CPU partition permits 18-hour jobs. The QoS permits 30,720 CPU-minutes per job (512 CPU-hours); each render request is 15,360 CPU-minutes. The four jobs can run concurrently, for a total of 64 CPUs and 128 GiB host RAM. No GPU allocation or GPU continuation is attached to these jobs.

All four render jobs started at **3:51:23 pm EDT**. The initial progress check verified active 16-CPU allocations for each, training-scene completions in every shard and no reported renderer errors. The verification/index job is pending their successful completion. Startup evidence is retained as `startup-audit.json` in the campaign archive.

Previous test rendering measured approximately 32 seconds per RMTS question and 69 seconds per identity question with one CPU worker per question. The corresponding ideal total is about 281 CPU-hours, or 4.4 wall-hours with 64 simultaneously active workers. A practical initial estimate is **4–8 hours of rendering once all jobs start**, plus queueing and final validation. Train images and allocated nodes can differ in rendering cost; this estimate is not a guarantee. The jobs have generous 16-hour limits and stop scheduling new work after 15h50m. Completed questions are durable; an unfinished shard can be resumed. The completion job refuses to announce a ready dataset until every expected question and image validates.

## Data and preparation

The official OCRA train files contain 10,000 RMTS questions (two candidate images each) and 10,000 identity-rule questions (four each): **60,000 new images**. These are separate from all existing test images and the public test archive. Train labels use their pinned official SHA256 values. The original Blender 2.79b renderer retains CPU mode, 64 samples, 320×240 pixels and seed `1 + original problem index`; parallel sharding does not renumber examples or alter their seeds.

Rendering output:

`/storage/ice1/9/8/yluo432/vlm-benchmark/data/clevr-train-seed1/shards/{0,1,2,3}/output/{rmts,idrules}_images/train_ood/prob_<original-index>/`

When all images are verified, the dependent job publishes:

- `data/clevr-train-seed1/native-manifests/clevr-art-rmts-train-native.jsonl`
- `data/clevr-train-seed1/native-manifests/clevr-art-idrules-train-native.jsonl`
- `data/clevr-train-seed1/native-manifests/READY.json`

Each row retains its canonical training ID, official answer and candidate order, image paths and SHA256 values. Indices reference the existing shard images without duplicating PNGs. These are native scene indices, not contact sheets or a finalized VLM fine-tuning format; later preparation can assemble the same presentation as our evaluations. The existing test data remain available for held-out evaluation. Before submission, storage use was about 71.3 GiB out of 300 GiB and 487,968 of 1,000,000 file entries.

The source and scheduler receipts are frozen under:

`/storage/ice1/9/8/yluo432/vlm-benchmark/runs/clevr-train-2026-10-07/`

The corresponding local archive is `benchmarks/.local/campaigns/clevr-train-2026-10-07/`. It includes the resource/authorization plan, per-job submission receipts and all source hashes. The submission helper reuses existing job receipts rather than submitting duplicates. The same saved render command can resume a shard under a fresh allocation if needed; keep its exact frozen source and data root. No continuous monitor was scheduled. Slurm jobs continue independently of the local SSH connection.

Twenty-two rendering/preparation tests passed, including existing interrupted-run recovery, new noncontiguous shard handling, train/test separation, exact coverage, native-index round-trip and unchanged image bytes. Slurm script syntax and scheduler resource validation also passed.

## What Slot Abstractor reproduction would actually train

The supplied pretrained slot/vision encoder stays frozen. The complete reasoning module and answer scorer start from random weights: approximately **7.89 million trainable parameters for each CLEVR task**, and **2.38 million for PGM**. The estimates below already reuse the pretrained encoder and exclude its pretraining. They are full supervised reasoning-network training, rather than evaluation of downloaded task-ready weights.

| Scope | Published schedule | Measured H200 compute projection |
| --- | --- | ---: |
| CLEVR RMTS, one run | 10,000 examples × 50 epochs; batch 16 | 1.34 h |
| CLEVR identity, one run | 10,000 examples × 200 epochs; batch 16 | 9.84 h |
| Both CLEVR tasks, five runs as in the reported aggregate | Five independent runs of both schedules | 55.9 h |
| PGM neutral, one run | 1.2 million examples × 73 epochs; batch 96 | 173.7 h |

These projections derive from the completed H200 audits in [the Slot Abstractor report](slot-abstractor-2026-10-07.md). They exclude dataset preparation, reading/decoding varied inputs, checkpoint writes, validation, queueing and restarts. The training schedules and five-run CLEVR reporting are documented in [the paper](https://arxiv.org/html/2403.03458v2). Training all of its other PGM regimes is outside these estimates.

The recommended next GPU experiment is fine-tuning the existing VLMs on training data, followed by the same held-out tests. This directly addresses whether the observed weakness comes from missing task training. Keep the paper's Slot Abstractor numbers explicitly labeled **published supervised results**, and our earlier VLM numbers labeled **zero-shot**. No Slot Abstractor training, VLM fine-tuning or architecture modifications were submitted in this campaign.
