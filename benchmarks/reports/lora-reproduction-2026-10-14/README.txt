LoRA reproduction configuration — completed runs, 14 October 2026

Read common-config.json for the complete shared optimization/input/loss recipe.
models.json pins the original model + processor commits, exact full target module
names, trainable counts and image processors. Do not use target_modules="all-linear":
that would also adapt components that we froze. Use the listed language-backbone targets.
prompts.json contains the exact instructions. Use each model's native chat template.

runs.json distinguishes each evaluated checkpoint's steps, unique questions and total
exposures from the final training run's counts. Each runs/<key>/ directory contains
unaltered saved training config/summary plus adapter settings from the publication
inventory. The latter records only inventoried fields; standard PEFT defaults are
spelled out in common-config.json. Optimizer beta/epsilon settings are the pinned
PyTorch defaults rather than explicit overrides in the original code.

Each model/task was trained separately. The first campaign allocated one H200 for
2h per model/task, including startup and validation; it was wall-time limited, not
fixed-epoch. The large PGM campaign started fresh LoRA adapters from the pretrained
models on 4 H200s for up to 4h. It did not continue the 9000-question runs. LFM then
continued from its selected 144000-example checkpoint to the fixed 240000-example
endpoint, retaining optimizer/RNG. Original LFM CLEVR training was seeded but did
not enable deterministic algorithms; repaired and expanded runs did.

To target the reported training scale on different hardware, use the selected
checkpoint's step/exposure counts; equal wall hours will NOT reproduce equal
training. Numerical equality also depends on runtime, GPU, topology and dropout
streams. The exact evaluated adapters are already shared on ICE:
/storage/ice1/9/8/yluo432/public-datasets/vlm-lora-adapters-2026-10-14-v1
That bundle includes its own README, INDEX, pinned loader and checksums. Base model
weights must be obtained separately at the revisions in models.json. It does not
contain optimizer states, so use full training checkpoints (not adapter-only weights)
when continuing the same optimizer trajectory.

The exact expanded PGM subset, validation and full test data are shared at:
/storage/ice1/9/8/yluo432/public-datasets/pgm-neutral-train20pct-seed20261007-v1
Do not independently resample another 20% subset. Use its documented manifests and
original IDs. Metadata paths in original run configs refer to our cluster account;
your teammate must map these to their local data directories.

This configuration pack contains no model weights or images. Reference input code
is supplied to specify rendering, native processing and strict scoring; it is not
a standalone training CLI. No relational-module or uncompleted second-epoch results
are included. Validate pack contents with SHA256SUMS.
