# Slot Abstractor reproduction

This is a supervised task-specific baseline, separate from the existing zero-shot VLM campaign. The evaluator is implemented and the native architectures and pretrained vision weights pass an H200 compatibility pilot. **No reproduced accuracy is available yet: the public repositories do not include the trained reasoning heads.** Training the original baseline is part of reproduction; it is separate from the proposed later VLM fine-tuning and architecture experiments.

## Pinned sources

| Asset | Source |
| --- | --- |
| User-requested repository | [slotabstractor/slotabstractor](https://github.com/slotabstractor/slotabstractor), commit `2219512022157f8f8e5e357d8d29a44f6fbe6691` |
| Paper's repository | [Shanka123/Slot-Abstractor](https://github.com/Shanka123/Slot-Abstractor), commit `100ca0bc87bca519e85f3d134084c1078caf27ef` |
| CLEVR pretrained encoder | Included in the requested repository; SHA256 `fc4d0e88f5c6425e7ed38fa79b0cf6287b19de73a7691496066b4bcb2225099d` |
| PGM pretrained encoder | Retrieved from the paper's repository; SHA256 `bec5621708498348837626f06081c52681ca133719b6afe73f730997e47b52cd` |

The two repositories' PGM and CLEVR architecture/training sources matched byte-for-byte at these revisions. Architecture hashes are enforced by `slot_abstractor/model.py`. The downloaded checkpoints load using PyTorch's restricted `weights_only=True` loader. The PGM encoder's `module.` prefix is stripped only when present. No randomly initialized head is accepted as a benchmark checkpoint.

## Evaluation protocol

| Task | Native input | Test set | Paper accuracy |
| --- | --- | ---: | ---: |
| PGM neutral | 8 context + 8 answer panels, grayscale 80×80 | 200,000 | 91.5% |
| CLEVR-ART RMTS | 2 original candidate scenes, RGB 128×128 | 10,000 | 96.34 ± 0.50% |
| CLEVR-ART identity rules | 4 original candidate scenes, RGB 128×128 | 10,000 | 91.61 ± 0.20% |

The CLEVR paper values are means ± standard error over five trained networks; PGM uses one trained network. Sources: [paper Tables 3–4 and training appendix](https://arxiv.org/html/2403.03458v2), [official reproduction instructions](https://github.com/slotabstractor/slotabstractor#clevr-art).

Use the same question identities, labels and underlying scenes/panels as our VLM runs. Slot Abstractor directly scores answer candidates; no language prompt or composite contact sheet is involved. Pixels are converted to float32 in [-1,1] and bilinearly resized with `align_corners=False, antialias=False`, matching the older torchvision tensor-resize default. Stochastic slot draws remain enabled and are seeded per fixed evaluation batch so interruption does not shift the random stream for subsequent batches.

CLEVR evaluation checks original scene hashes from `manifests-v2`, and PGM streams original PNG panels from the nine existing neutral test Parquet shards. PGM IDs use the same `pgm/neutral/test/<id>` convention as the VLM archives. Source hashes, code hashes, batch size, seed, checkpoint identity and runtime are bound to the output database. Mismatches reject a resume. SQLite commits entire batches atomically; partial runs report their actual denominator. Full-test completion requires 200,000 or 10,000 unique predictions, as appropriate. Repeat the same invocation to resume, requesting a new Slurm allocation each time. There is no automatic GPU resubmission.

Our scenes were independently rendered with seed 1, and the runtime is newer than the paper's PyTorch 1.13.1. Together with fresh random training/slot draws, this means numerical equality with the paper is not guaranteed. A single successful training run also does not reproduce a five-run mean and SEM. Comparisons to the earlier VLMs differ in both supervised training and input presentation; they cannot isolate an architectural advantage.

## Compatibility changes and checks

The original network definitions are used directly. The unused logger import is replaced with standard logging. Hard-coded CUDA position grids become nonpersistent buffers so device moves work without changing checkpoint keys. The encoder's unused reconstruction decoder is skipped for reasoning: the H200 pilot checked both returned feature and position tensors against the original full autoencoder with the same random seed, obtaining exactly zero difference on all three tasks. The frozen vision weights are preserved; all updates in the timing pilot use disposable heads and synthetic targets, never benchmark answers.

The pilot used one NVIDIA H200, two CPUs, 16 GiB host RAM, float32, TF32 disabled, PyTorch 2.7.1+cu126 and einops 0.4.1. Einops is staged in an isolated campaign directory; the existing benchmark environment was not modified. Training-step measurements include fresh slot extraction, a 24-layer reasoning head, loss, backward pass and an Adam step. Inputs are repeated resident examples, so projections exclude real data loading, checkpointing, validation, queueing and future restarts.

## Running when reasoning weights are available

The initial frozen pilot/evaluation bundle is on ICE at:

`/storage/ice1/9/8/yluo432/vlm-benchmark/runs/slot-abstractor-evaluator-2026-10-07`

It contains `source/`, `runtime/`, `upstream/`, `weights/`, and `files.sha256.json`. Live development files are in the local `benchmarks/slot_abstractor/` package. Preserve frozen campaign bundles; create a new one for code revisions. The saved vision checkpoints under `weights/` are **not** the missing reasoning checkpoints.

A trained checkpoint must contain `abstractor_model_state_dict`, with strict architecture-compatible keys. Supply a separate JSON provenance file with these fields:

```json
{
  "task": "rmts",
  "regime": "ood",
  "training_split": "train",
  "checkpoint_sha256": "SHA256_OF_THE_TRAINED_REASONING_CHECKPOINT",
  "encoder_sha256": "fc4d0e88f5c6425e7ed38fa79b0cf6287b19de73a7691496066b4bcb2225099d",
  "training_description": "Actual source, training data, seed, epoch and hyperparameters",
  "checkpoint_selection": "final-epoch"
}
```

Use `validation` for a PGM checkpoint selected by validation accuracy, or `author-provided` with an actual author-supplied artifact. Never select weights using our test results. Provenance declarations must be verified from training records; the evaluator cannot establish their truth merely by reading JSON.

Within a new explicitly allocated GPU job, the module invocation is:

```bash
python -m slot_abstractor.evaluate \
  --task rmts --upstream "$campaign/upstream" \
  --encoder "$campaign/weights/clevr.pth.tar" \
  --checkpoint /absolute/path/to/trained-rmts.pt \
  --checkpoint-provenance /absolute/path/to/trained-rmts.json \
  --data-root /storage/ice1/9/8/yluo432/vlm-benchmark/data/clevr-full-seed1 \
  --manifest /storage/ice1/9/8/yluo432/vlm-benchmark/data/clevr-full-seed1/manifests-v2/clevr-art-rmts-test-full.jsonl \
  --output /absolute/path/to/new/results --batch-size 16 --seconds 3300
```

The reusable `cluster/slot_abstractor_evaluate.sbatch` wrapper accepts the campaign path, environment path, then these evaluator arguments; its caller supplies verified account/partition/QoS/GPU/time flags. Use `--task idrules` and its manifest for identity rules. For PGM use `--task pgm`, its encoder and trained neutral head, and `--parquet` followed by all nine pinned test shards instead of CLEVR arguments. The module refuses inference on a login host.

## Remaining reproduction work

Obtain trained reasoning heads from the authors, or prepare training data and train them. The present implementation includes the evaluator and disposable timing pilot, not a production training/resume launcher. Existing rendered CLEVR data are test-only: a train split needs another 60,000 scenes. PGM needs its 1.2-million-example training split and validation split. Avoid expanding PGM into millions of small files because of the ICE file quota; use a bounded-memory, indexed packed format for training and exact global shuffling.

For a training reproduction, retain the supplied frozen encoders. Train with Adam at 8e-5: CLEVR batch 16 for 50 RMTS epochs and 200 identity epochs; PGM neutral global batch 96 for 73 epochs, selecting the head with validation data. Save head, optimizer, epoch/batch position and random states atomically so Slurm wall limits do not restart training. Budget approval and train-data preparation are required before production submission. VLM LoRA and architecture modifications remain separate later steps.
