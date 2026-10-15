# LFM2.5 visual reasoning: first accessible-data evaluation

Generated 2026-09-24T03:56:27.647530+00:00.

Frozen, zero-shot, native image processing, greedy decoding, 16 output tokens. PGM uses the pinned HuggingFaceM4 neutral/test mirror. CLEVR-ART RMTS uses official OCRA test labels and Blender-rendered scenes. These conditions differ from the task-supervised Slot-Abstractor paper experiments.

| Run | State | Correct / evaluated | Accuracy | 95% Wilson interval | Chance | Invalid |
| --- | --- | --- | --- | --- | --- | --- |
| lfm25-clevr-rmts-smoke | complete | 6 / 16 | 37.50% | 18.5%–61.4% | 50.0% | 0 |
| lfm25-pgm-neutral-hf-pilot512 | complete | 67 / 512 | 13.09% | 10.4%–16.3% | 12.5% | 0 |
| lfm25-pgm-neutral-hf-smoke | complete | 2 / 16 | 12.50% | 3.5%–36.0% | 12.5% | 0 |

The 16-example runs are implementation checks, with broad uncertainty. The 512-example run is a sampled pilot, not the full 200,000-example neutral test set. No extrapolation or other PGM generalization regime is represented. Only complete runs should be used as final measurements.

## lfm25-clevr-rmts-smoke

Model: `lfm25` at `919fde3d022e3f90a4716006f993938ee8c2eb97`.

Predictions by label: A=2, B=14.
Targets by label: A=8, B=8.
The model selected its modal label (B) on 14/16 examples (87.5%). Always choosing that label would score 8/16 on these evaluated examples. This diagnostic helps distinguish answer-position bias from useful predictions.

Device: Tesla V100-PCIE-16GB; precision: torch.bfloat16.

Mean generation latency: 0.295 seconds/example; peak PyTorch allocated GPU memory: 3.17 GiB. Outputs reaching the token limit: 0/16. Latency excludes model loading and data preparation; allocated memory is not total device usage.

Raw predictions: [lfm25-clevr-rmts-smoke/predictions.jsonl](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-clevr-rmts-smoke/predictions.jsonl).

Input manifest SHA-256: `2d81a1a9661dbd856cf8130a8c8f62ad5c96f192550feeb8001da953502fe3b8`.

## lfm25-pgm-neutral-hf-pilot512

Model: `lfm25` at `919fde3d022e3f90a4716006f993938ee8c2eb97`.

Predictions by label: A=5, B=3, C=13, D=8, E=483.
Targets by label: A=58, B=67, C=67, D=55, E=67, F=62, G=69, H=67.
The model selected its modal label (E) on 483/512 examples (94.3%). Always choosing that label would score 67/512 on these evaluated examples. This diagnostic helps distinguish answer-position bias from useful predictions.
The accuracy interval includes uniform-random chance. This pilot provides no evidence of above-chance accuracy under the current protocol; it does not establish a general inability of the model to reason.

Device: Tesla V100-PCIE-16GB; precision: torch.bfloat16.

Mean generation latency: 1.418 seconds/example; peak PyTorch allocated GPU memory: 4.23 GiB. Outputs reaching the token limit: 0/512. Latency excludes model loading and data preparation; allocated memory is not total device usage.

Raw predictions: [lfm25-pgm-neutral-hf-pilot512/predictions.jsonl](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-pgm-neutral-hf-pilot512/predictions.jsonl).

Input manifest SHA-256: `f81bc8f4340d9d176d9b04267ef07fa4caf8ff8c6099499aa74b2779d9668a5e`.

## lfm25-pgm-neutral-hf-smoke

Model: `lfm25` at `919fde3d022e3f90a4716006f993938ee8c2eb97`.

Predictions by label: C=1, E=15.
Targets by label: A=4, B=1, C=5, E=3, F=1, G=1, H=1.
The model selected its modal label (E) on 15/16 examples (93.8%). Always choosing that label would score 3/16 on these evaluated examples. This diagnostic helps distinguish answer-position bias from useful predictions.

Device: Tesla V100-PCIE-16GB; precision: torch.bfloat16.

Mean generation latency: 1.849 seconds/example; peak PyTorch allocated GPU memory: 4.23 GiB. Outputs reaching the token limit: 0/16. Latency excludes model loading and data preparation; allocated memory is not total device usage.

Raw predictions: [lfm25-pgm-neutral-hf-smoke/predictions.jsonl](/Users/yluo/Downloads/Projects/VLM/benchmarks/.local/remote-runs/lfm25-pgm-neutral-hf-smoke/predictions.jsonl).

Input manifest SHA-256: `51e5c66d86960983c40bbe397b9e097be6a0f0122d7be167f023ae3173d4b4de`.

## Current scope

Only LFM2.5 has been evaluated. SmolVLM2 and InternVL3.5 backends are prepared but unrun. CLEVR identity rules and larger render batches are deferred. The original PGM archive endpoint returned access-denied errors; original NPZ equivalence of the neutral mirror has not been independently established. Pretraining overlap is unknown.
