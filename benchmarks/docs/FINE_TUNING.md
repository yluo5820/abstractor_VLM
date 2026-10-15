# First LoRA adaptation experiment

This is a bounded adaptation study, not a reproduction of Slot Abstractor or evidence of convergence. The first campaign trains **LFM2.5-VL-1.6B on CLEVR-ART identity rules**. RMTS, PGM and the other VLMs are follow-up experiments, not jobs submitted by this campaign.

## Protocol

- Pinned base: `LiquidAI/LFM2.5-VL-1.6B`, revision `919fde3d022e3f90a4716006f993938ee8c2eb97`.
- Official CLEVR identity training split: 10,000 questions. Sort IDs by SHA256 of `20261007 + NUL + canonical_id`; reserve the first 1,000 for validation, train on the other 9,000. Rendering availability does not change this split.
- The throughput/correctness pilot uses 64 already rendered training questions and 16 already rendered reserved-validation questions. These availability-selected subsets are for engineering checks only. Pilot weights are never the starting point of production training.
- Assemble the same two-column, four-candidate contact sheet as the zero-shot baseline, and use the same prompt, native chat template and image preprocessing. The loss covers only the correct assistant answer and chat-ending tokens. Expanded image/instruction tokens must match the inference prefix exactly and are masked from supervision.
- Freeze the vision encoder, multimodal projector, language embeddings, output head and all other original weights. Attach LoRA to every linear layer inside `model.language_model`, including attention, MLP and convolution input/output projections. The convolution kernels remain frozen. Rank 16, alpha 32, dropout 0.05; the actual target names and trainable parameter count are recorded.
- BF16 base, FP32 adapters; no quantization. AdamW, learning rate 1e-4, linear warm-up over 20 optimizer updates, then constant rate; weight decay 0.01, gradient clipping 1.0. Microbatch 1 with 16-example gradient accumulation. A short final batch is normalized by its actual size.
- Use the allocation's internal time budget, reserving time for validation and checkpointing. **There is no default epoch cap** (`--max-epochs 0`). The initial three-epoch cutoff was removed at the user's request before training started. Record actual examples, epochs, optimizer steps and training seconds. No automatic resource increase or continuation submission.

## Validation and interpretation

Evaluate a fixed 128-example subset of the reserved validation split before training, every 200 optimizer steps when time permits, and at the end. Keep the checkpoint with the highest strict greedy answer accuracy, breaking ties by lower answer loss. The unmodified base is an eligible checkpoint, so a worse adapter is not automatically selected.

After selection, evaluate that checkpoint on the full 1,000 reserved validation questions if the remaining time permits. Incomplete validation explicitly reports its denominator. The 128 selection questions are part of that validation set: the result is a development score, not an independent test score. Baseline and adapted comparisons should first use the same 128 IDs; do not compare a 128-example baseline directly with a different 1,000-example estimate.

The benchmark test split is neither read nor scored during training or selection. A later evaluation loads `best.json`'s adapter and uses the existing 10,000-question test manifest and unchanged generation/parser settings. That evaluation requires its own approved allocation or an explicitly budgeted evaluation phase; this campaign does not silently consume additional GPU time for it.

Improvement supports task adaptation under this representation and training method. A flat short run does not prove architecture incapacity, and a falling loss alone is not a reasoning score. Report these task-supervised runs separately from the earlier zero-shot results and the paper's trained specialized models.

## Safety checks and resumption

The pilot checks fresh-adapter equivalence, finite nonzero adapter gradients, absence of frozen-weight gradients, a checkpoint round trip, and equivalence of the next optimizer update after reloading weights, optimizer moments and RNG. Production requires a successful pilot of the exact source/model revision, or an explicit compatibility record pinned to both old and new source hashes. The stopping-policy amendment uses this narrow exception: AST comparisons confirmed the optimizer loop body and all existing core functions were unchanged; eight focused tests passed. It does not claim the updated stopping policy itself received another GPU pilot.

Checkpoints are written into a staging directory and atomically published before `latest.json` is updated. They include PEFT safetensors, optimizer state, CPU/CUDA RNG, epoch, shuffled-data offset and example count. The production job checkpoints every 25 optimizer steps, after validation, and before exiting on its internal deadline or an early scheduler signal. Resume the same frozen campaign with the same arguments in a fresh allocation; existing checkpoints must match model, source, data, versions and hyperparameters. `latest.json` resumes training; `best.json` is for evaluation. Re-submitting the original submission helper returns its existing receipt rather than creating an unapproved continuation.

The pilot and trial each request one H200, two CPUs and 16 GiB host RAM. The pilot has a 20-minute scheduler cap; the trial has a two-hour cap and waits for both the pilot and CPU training-index job. The script has a shorter internal deadline and reserves time for checkpointing/validation. A failed prerequisite blocks the trial. Jobs run independently of SSH.

Dependencies are installed only in the campaign's isolated `runtime/` directory: PEFT 0.18.1, Accelerate 1.12.0 and psutil 6.1.1. Existing PyTorch/Transformers and model caches are reused unchanged. Source and wheel hashes, scheduler receipts, protocol configuration, split records, learning curves and validation outputs stay in the campaign archive.

## Entry points

- `vlm_bench/finetune_data.py`: immutable split and verified native scene loading.
- `vlm_bench/finetune.py`: bounded training, validation, checkpoint selection and resumption.
- `cluster/finetune_lfm.sbatch`: allocation guard and isolated runtime.
- `cluster/submit_finetune.py`: idempotent submission with exact resource limits.

Source support: [Liquid AI model card and LoRA guidance](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B#-fine-tuning), [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/developer_guides/lora). The chosen learning rate, batch size, rank and stopping limits are our initial experimental settings, not claimed optimal settings or the Slot Abstractor recipe.
