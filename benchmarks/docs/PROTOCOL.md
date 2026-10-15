# Evaluation protocol v1

## Primary question

How accurately can a frozen compact VLM solve the official CLEVR-ART and PGM tasks, and how does accuracy vary across appearance domains and rule regimes? This is a capability baseline. It does not by itself isolate the effect of visual compression, fine-tuning, or a training-set generalization mechanism.

## Model order

1. LiquidAI/LFM2.5-VL-1.6B.
2. HuggingFaceTB/SmolVLM2-2.2B-Instruct.
3. OpenGVLab/InternVL3_5-1B-HF, the official native Transformers conversion. Coordinate the same checkpoint, rendering, sample IDs, and scoring with teammates before comparing results from another pipeline.

Use a pinned model/processor revision, native chat template, greedy decoding, no few-shot examples, and 16 new tokens. No quantization in the initial baseline. Record actual precision and image processing configuration. Single-image contact sheets standardize the information supplied while retaining each model's native image preprocessing; resulting token budgets need not be equal.

## Data controls

- Use official data and original targets. Keep sample source, original split, regime, IDs, hashes, and generation settings.
- PGM: preserve panel order, eight context panels and eight answer choices. Use the official reshape correction, not a channel transpose. Keep each generalization regime separate. Neutral provides a reference; extrapolation is the Slot-Abstractor repository default. Held-out triple and relation-pair regimes add distinct tests.
- CLEVR-ART: keep the official RMTS and identity-rule candidate layouts and 0-based target mapping. Render the held-out object properties from the official test symbolic files. Train-domain examples may be evaluated as a separately labeled appearance control; they are not a held-out benchmark estimate or used to optimize test prompts.
- Seeds select subsets before model predictions. Use the same manifest for all models. Do not tune prompts, labels, sampling, or rendering based on test accuracy.
- Keep labels, scene attributes, PGM meta-targets, and solution metadata out of the image and prompt. Human-readable candidate letters are allowed.
- Inspect actual contact sheets and model-processed image dimensions before measuring accuracy. Poor perception at reduced resolution is a possible reason for low performance and must be examined before attributing it to reasoning.

## Stages

**Smoke:** 16 items per task/regime, only for confirming the implementation. Visually inspect at least one real example per task and check targets against source files.

**Pilot:** proposed 512 items per task/regime, fixed seed 0; report the sample count and uncertainty. Choose the full-run sample size and walltime from measured throughput and remaining storage. Obtain approval for a larger allocation before submitting it.

**Full evaluation:** evaluate all examples in the agreed manifest, with `--max-samples 0`, in resumable batch jobs. Report its relationship to the full official test set. Do not label a sampled subset as the full benchmark.

## Scoring and reporting

The model returns one candidate letter. The parser accepts a single letter (case-insensitive), optionally parenthesized or prefixed with `Answer:` and followed by a period. Responses with prose, multiple letters, or an unavailable label are invalid and count as incorrect. Raw responses are retained for audit; changing the parser is a new protocol version.

Report each dataset/task/regime/split separately: correct/evaluated/planned, accuracy, 95% Wilson interval, invalid-output rate, chance accuracy, mean generation latency, hardware/precision, and peak GPU memory. Chance is 1/8 for PGM, 1/2 for CLEVR-ART RMTS, and 1/4 for identity rules. Wilson intervals assume independent Bernoulli samples; related generated scenes may weaken that approximation. Compare models on matched IDs rather than unrelated random subsets.

Checkpoint reuse, pretraining overlap with public benchmarks, dataset generation differences, resolution, native processor differences, and precision are limitations. Frozen-VLM scores should be presented separately from models supervised on the benchmark training set. An OOD split name alone does not establish that those concepts were absent from a pretrained VLM's training data.

## Source authority

- [Slot-Abstractor README](https://github.com/Shanka123/Slot-Abstractor): original task scope and task-specific training workflows.
- [OCRA](https://github.com/Shanka123/OCRA): CLEVR-ART scene generation and task labels.
- [DeepMind abstract-reasoning-matrices](https://github.com/google-deepmind/abstract-reasoning-matrices): PGM shape correction, targets, regimes, and non-commercial research data terms. The source code license and dataset terms are distinct.

No empirical result is asserted in this protocol.
