# Slot Abstractor: implementation and H200 audit, 7 October 2026

The native evaluator and short hardware pilots are complete. **No reproduced Slot Abstractor accuracy is available yet.** Both public repositories supply pretrained slot encoders, but neither includes the trained task reasoning heads. The requested repository's README instructs users to train before evaluating. We therefore need author-provided heads or a supervised baseline-training reproduction before we can measure its test score.

## Published scores versus our existing results

| Model and setup | PGM neutral | CLEVR-ART RMTS | CLEVR-ART identity |
| --- | ---: | ---: | ---: |
| Slot Abstractor, paper, supervised training | 91.50% | 96.34 ± 0.50% | 91.61 ± 0.20% |
| LFM2.5-VL-1.6B, our zero-shot run | 12.37% (200,000) | 47.75% | 24.89% |
| SmolVLM2-2.2B, our zero-shot run | 12.57% (116,527; partial) | 50.48% | 25.39% |
| SpatialEvo, our zero-shot run | 12.97% (173,554; partial) | 55.76% | 41.38% |
| Chance | 12.50% | 50.00% | 25.00% |

CLEVR evaluations have 10,000 questions each. The paper reports a mean ± SEM across five independently trained CLEVR networks, and one trained model for PGM. The three VLMs share 116,527 completed PGM questions; on that common subset their scores are 12.34%, 12.57%, and 13.03%. A paper score on its complete test set is not a same-subset comparison to our partial runs. Sources: [paper Tables 3–4](https://arxiv.org/html/2403.03458v2), [local result archive](local-results-archive.md), [CLEVR result audit](clevr-results-2026-09-26.md).

Slot Abstractor is a small specialist trained with task labels and native separate panels/candidates. Our VLMs received composite images and language prompts without task training. The score gap therefore mixes training, architecture and presentation. It does not establish that an abstractor alone explains the difference. The later proposed VLM fine-tuning experiment is still needed to examine the training hypothesis.

## What was implemented and verified

The wrapper pins the requested repository at `2219512022157f8f8e5e357d8d29a44f6fbe6691`. Its PGM and CLEVR model code matches the paper repository at `100ca0bc87bca519e85f3d134084c1078caf27ef`. The latter supplies the missing PGM encoder. Both encoder files passed pinned SHA256 checks and strict restricted checkpoint loading. See [the reproduction guide](../docs/SLOT_ABSTRACTOR.md) for all provenance and commands.

Native input adapters preserve the current benchmark IDs, answer order and labels. Evaluation binds the exact test manifests/Parquet checksums to a trained reasoning checkpoint and run configuration. Completed batches are saved transactionally; repeated invocations resume without silently changing weights, data or random slot draws. JSONL exports retain per-question logits and predictions. Encoder-only checkpoints are rejected. No production training/resume launcher has been implemented yet; that follows the compute decision and preparation of separate training inputs.

Eight local tests passed, covering candidate order, label-independent inputs, pixel normalization/resize, corrupted-image rejection, checkpoint-key handling, rejection of encoder-only/unpinned inputs, atomic resumption/export and restart-stable slot seeds. Both upstream architectures also loaded and produced finite correctly shaped scores locally. Slurm script syntax passed. On H200, the decoder-skipping optimization matched the full original encoder's feature and position tensors exactly for RMTS, identity rules and PGM. All pilot training updates used synthetic targets and disposable heads; none used test answers or produced a claimed benchmark accuracy.

## Measured H200 costs

Two bounded allocations used one H200, two CPUs and 16 GiB host memory each. The existing environment was preserved; einops 0.4.1 was isolated in the campaign directory. Runtime was PyTorch 2.7.1+cu126, float32, TF32 disabled. Median training times use three measured steps after a warmup step, including frozen slot extraction, loss/backward, gradient-finiteness checks and Adam updates. Test-rate measurements use one inference step after warmup.

| Task | Batch | Training examples/s | Evaluation examples/s | Peak allocated CUDA memory | Published schedule: projected training per seed | One complete test pass: compute-only projection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RMTS | 16 | 103.65 | 295.20 | 0.53 GiB | 10,000 × 50 epochs: **1.34 h** | 0.56 min |
| Identity rules | 16 | 56.44 | 145.06 | 0.54 GiB | 10,000 × 200 epochs: **9.84 h** | 1.15 min |
| PGM neutral | 96 | 140.05 | 305.77 | 67.22 GiB | 1,200,000 × 73 epochs: **173.74 h** | 10.90 min |

These short measurements repeat already resident input pixels with fresh stochastic slots. They exclude reading/decoding varied data, preparing training images, saving checkpoints, validation, queues and restarts. They are **compute projections, not promised wall times**. Running five CLEVR seeds to reproduce the paper's aggregate would take roughly 55.9 compute-hours before overhead. Batch 96 fits in one H200 and preserves PGM's published total batch size, avoiding the much slower batch-24 projection of about 530 hours. One PGM epoch projects to 2.38 compute-hours; a 10-hour exploratory segment would cover at most roughly four epochs before overhead, far short of the published 73.

| Job | Purpose | State / exit | Allocated elapsed time | Start–end, ICE local time |
| --- | --- | --- | --- | --- |
| 6091228 | All three tasks, architecture/encoder equivalence, small-batch timing | COMPLETED / 0:0 | 3m20s | 15:26:21–15:29:41 |
| 6091343 | PGM batches 24, 48, 96 | COMPLETED / 0:0 | 15s | 15:31:51–15:32:06 |

Total H200 allocation consumption: **3m35s**. The queue was empty after completion. Full raw pilot JSON and Slurm logs are saved locally under `benchmarks/.local/campaigns/slot-abstractor-2026-10-07` and `slot-abstractor-batch96-2026-10-07`, with matching remote campaign directories. Initial cold startup accounts for much of the first job's elapsed time.

## Next decision

Current data are test-only. A CLEVR training reproduction needs 60,000 additional rendered scenes from the separate official training specifications; PGM needs its training and validation splits. Training data must remain separate from the current public test archive.

Decision update: the user authorized starting the training renders with increased CPU parallelism, and prefers using the published Slot Abstractor numbers if reproducing them would distract from VLM fine-tuning. Four 16-CPU rendering jobs plus a CPU verification job are now queued; see [the rendering campaign](clevr-train-2026-10-07.md). No Slot Abstractor GPU training, VLM fine-tuning or architecture modification has been submitted. The proposed 20-H200-hour reproduction budget was not approved or used.
