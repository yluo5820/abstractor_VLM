# CLEVR-ART three-model campaign — September 26, 2026

**Completed; verified at 11:47 pm EDT:** all six GPU evaluations finished,
each with 10,000 predictions and zero invalid outputs. SpatialEvo scored 55.76%
on RMTS and 41.38% on identity rules. The queue is empty and allocations are
released. See the [final results and raw-output audit](clevr-results-2026-09-26.md).
The preparation and scheduling description below records the original setup.

| Model | RMTS job (10,000 questions) | Identity-rule job (10,000 questions) |
| --- | --- | --- |
| LFM2.5-VL-1.6B | `5948210` | `5948211` |
| SmolVLM2-2.2B | `5948212` | `5948213` |
| SpatialEvo-3B | `5948214` | `5948216` |

CPU preparation segments are `5948205`, `5948206`, `5948207`, `5948208`;
readiness job `5948209` must succeed before evaluation. The frozen source
archive SHA256 is `85939c8196b9b2d7736346167389b1991075ebf3321f13f5923d6ecd2e7d0ab7`.

The user requested all three evaluated VLM families on CLEVR-ART: LFM2.5-VL-1.6B,
SmolVLM2-2.2B-Instruct, and SpatialEvo-3B (approximately 4.07B total parameters).
Each model evaluates both complete OOD test tasks: **10,000 RMTS questions** and
**10,000 identity-rule questions**, for six model/task evaluations in total.

## Rendering and preparation repair

All **60,000 rendered scenes** completed successfully at 7:46 pm EDT on September
25, with no renderer errors. The following one-hour CPU manifest job `5921715`
received SIGTERM at 58m26s. It left an unpublished partial attempt with 9,517 RMTS
contact sheets. Its two dependent LFM jobs `5921716` and `5921717` were cancelled
by Slurm before running. They produced no full CLEVR results to reuse.

The repair creates a separate `manifests-v2` directory from the existing rendered
scenes. It validates each source group against its renderer receipt and PNG
hashes while composing the contact sheet, and durably saves each completed row.
It never alters original scenes or failed staging attempts. This avoids the old
all-or-nothing publication losing the entire conversion attempt at a time limit.

Preparation uses **four bounded one-hour CPU segments**, each with **two CPUs and
16 GiB RAM** and a 55-minute internal stop. Segments resume the same output and
exit quickly once it is ready. A separate ten-minute readiness check requires
both complete 10,000-row manifests before any GPU evaluation becomes eligible.
Preparation failures stop the dependent jobs; incomplete preparation is never
silently treated as a full dataset. No more jobs are submitted automatically.

The pinned render-configuration SHA256 is
`db166511959435062ac8a1eeac2991cdf5103a9cdff0da0f9f94b5c66a8fd171`.

## Evaluation protocol and resources

All models receive the same original-renderer contact sheets, label mapping,
seeded question order and task prompt. No symbolic feature arrays are given to
the models. RMTS has A–B choices (50% chance); identity rules has A–D choices
(25% chance). Native image processors and chat templates remain model-specific.

Every evaluation uses greedy direct-answer generation, a 16-new-token ceiling,
and strict letter parsing. Invalid outputs count as wrong. A 32-question technical
preflight requires at least one parseable answer and zero token-budget hits before
the full task begins. Accuracy is not used to select models or tune prompts.
The 32 pilot outputs are stored separately and do not inflate the 10,000-question
full-run count.

Each of the six jobs requests **one H200, two CPUs, 16 GiB RAM and two hours**.
The internal 110-minute deadline leaves time to save and release the allocation.
The jobs may run independently after preparation. This is a resource limit, not
a measured promise of completion. Each has its own resumable outputs, and no
automatic GPU continuation is enabled. The previous PGM segments remain stopped.

| Model key | Repository | Immutable revision |
| --- | --- | --- |
| `lfm25` | `LiquidAI/LFM2.5-VL-1.6B` | `919fde3d022e3f90a4716006f993938ee8c2eb97` |
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `482adb537c021c86670beed01cd58990d01e72e4` |
| `spatialevo` | `lidingm/SpatialEvo-3B` | `5fd684324a3d7e759c948b42f4a51b9b016c8d5f` |

All weights are cached and GPU loading is offline. The campaign uses a fresh,
frozen code snapshot. Existing render/PGM snapshots remain unchanged.

## Validation and records

The benchmark suite passed **75 tests** and the cluster launcher passed **21**.
The six new preparation tests include exact manifest, provenance and PNG-byte
equivalence to the previous adapter; partial-row recovery; resume without
recomposing completed examples; and rejection of changed source images,
checkpoints, render plans or published manifests. The existing watchdog tests
exercise graceful interruption, forced termination of an owned stuck child, and
failure propagation. A new test verifies CLEVR's 10,000-example completeness
criterion instead of PGM's 200,000.

Submission receipts, profile, source hashes and initial failure evidence are in
`benchmarks/.local/campaigns/clevr-controls-2026-09-26/` locally. The remote campaign
root is `/storage/ice1/9/8/yluo432/vlm-benchmark/runs/clevr-controls-2026-09-26/`.
