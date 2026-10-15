# Expanded task adaptation campaign — 2026-10-07

Evaluate LFM2.5-VL-1.6B, SmolVLM2-2.2B and SpatialEvo-3B on CLEVR-ART identity rules, CLEVR-ART RMTS and PGM neutral after task-specific LoRA adaptation. SpatialEvo's repository name says 3B; its actual stored architecture is approximately 4.07B parameters. The completed LFM identity campaign is reused, not repeated.

Each of the other eight combinations uses an independent fresh base plus rank-16/alpha-32/dropout-0.05 LoRA on language-only linear layers. Original weights stay frozen; BF16 base, FP32 adapters, AdamW 1e-4, microbatch 1, accumulation 16, constant learning rate after 20-update warmup. The two-hour trial has no epoch cap and reserves validation/checkpoint time. This equal time budget is not an equal number of optimizer steps or epochs: report actual coverage for each model.

CLEVR uses the existing rendered official training splits: 9,000 training questions and 1,000 validation questions per task, deterministically partitioned by ID as in the first trial. PGM uses a deterministic 9,000-question sample across all 54 pinned mirror training shards and a separate 1,000-question sample from official validation shards. It does not claim full-PGM training. No test questions participate in adaptation, checkpoint selection, pilot selection, or sample-size decisions.

A 20-minute pilot of the exact frozen source and model revision must pass eight optimizer updates, initial adapter equivalence, finite/nonzero adapter gradients, frozen-weight checks, checkpoint roundtrip and next-update resume equivalence before training can start. Failed prerequisite jobs block/cancel only their dependent jobs through Slurm's configured dependency policy; no manual cancellation is performed.

Select by a fixed 128-question validation monitor, breaking accuracy ties by lower answer loss. Report full 1,000-question development validation separately, then score the selected checkpoint on the unchanged official test manifests: 10,000 questions for each CLEVR task and 200,000 for PGM neutral. All models use their previously pinned revisions, original composite-image presentation, native chat templates, greedy16 generation, BF16 and strict-letter parser. Evaluation checks the full manifest checksum and records adapter hashes. Predictions are durable and resumable. PGM neutral is a test split but is not an OOD regime; CLEVR uses the original appearance-shift OOD tests.

Resources per GPU job: one H200, two CPUs, 16 GiB host RAM. Pilot 20m; training 2h; CLEVR test 2h; PGM test 10h. These are scheduler caps, not runtime estimates. No automatic resource increase or unbounded continuation is authorized. A partial 10h PGM test is explicitly marked incomplete and requires an approved continuation to finish. PGM data preparation uses two CPUs, 16 GiB and a 2h cap; pinned compressed source shards occupy about 38.0 GB.

Frozen source and submission receipts: `benchmarks/.local/campaigns/adaptation-controls-2026-10-07/`. Remote counterpart: `/storage/ice1/9/8/yluo432/vlm-benchmark/runs/adaptation-controls-2026-10-07/`. Model/task outputs are stored in `<model>/<task>/{pilot,train,test}`. Existing shared environments and prior campaigns are unchanged.

Timing correction: previous LFM identity test pipeline took 661 seconds (~11 minutes), compared with 17m41s for the adapted test job. The adapted job was not faster. Earlier full LFM PGM inference took ~8h end to end for 200,000 questions. Queue delays are separate from these runtimes.

## Checkpoint replay repair

The original seven SmolVLM/SpatialEvo/PGM pilot chains failed their strict1e-6 next-update comparison; Slurm cancelled their dependent jobs. Diagnostic6096079 reloaded exact failed checkpoints and replayed the same training example twice under each execution policy. Forward losses agreed, while default backward gradients differed, producing parameter differences: SmolVLM RMTS2.6211e-5, SpatialEvo RMTS1.8818e-5, LFM PGM1.2019e-6. With deterministic algorithms enabled, all three had exactly zero gradient, loss and parameter difference. The workspace setting alone was insufficient (present in both cases).

The isolated repair campaign adds torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG=:4096:8 before process startup; the tolerance and scientific protocol are unchanged. The setting is recorded in training configuration. Seven new strict pilots6096099–6096135 gate fresh2h trials and original full tests, using the same resources. Original LFM RMTS continues with its original frozen source and passed pilot; original LFM identity is retained. The diagnostic provides numerical reproducibility evidence, not benchmark accuracy. Pilot throughput will quantify any deterministic-kernel slowdown. Retry receipts and hashes are in benchmarks/.local/campaigns/adaptation-repair-2026-10-07.

## Completed LFM RMTS test

Job6095401 completed normally in15m46s. Selected step3464 after55,376 training presentations (6.153passes) scored9,939/10,000 (99.39%) on full official appearance-shift OOD test, versus47.75%zero-shot (+51.64percentage points). No invalid answers. Validation1000/1000 is separately a development score. Downloaded full predictions/provenance, SHA256verified; independently recounted and confirmed identical baseline/test manifest and aggregate image hashes. Other models/tasks remain pending completion.

## Completed repaired RMTS tests

Both full10k appearance-shift tests completed successfully: SmolVLM8905/10000 (89.05%) versus50.48%baseline; SpatialEvo10000/10000 (100%) versus55.76%baseline. Zero invalid outputs. Independently recounted downloaded predictions, verified remote/local SHA256 and identical baseline manifest/image hashes. Equal two-hour caps produced different coverage: SmolVLM12536 presentations1.393epochs, selectedstep784; SpatialEvo14664presentations1.629epochs, selectedstep917. These are full OOD test scores, separate from development validation. Remaining identity and PGM tasks are still running.

## SpatialEvo identity test

Full official OOD identity test completed:9579/10000 (95.79%) versus41.38%baseline, zero invalid answers. Independently recounted and checksumverified downloaded results; baseline manifest and aggregate image hashes match. Training10808presentations1.201passes, selectedstep676. All PGM training also completed and was archived with hashes; PGM full tests and SmolVLM identity test remain running.

## SmolVLM identity test

Full original OOD identity test completed6854/10000 (68.54%) versus25.39%baseline; no invalid answers. Downloaded predictions independently rescored, remote/local SHA256verified and baseline manifest/image hashes identical. Training6080presentations0.676epochs, selectedstep380; final development validation was partial397/590 and must not be described as1000-example validation. All six CLEVR comparisons are complete. PGM full tests continue; initial measured rates suggest SmolVLM and SpatialEvo need continuation beyond10h caps. No additional jobs submitted.

## Full LFM PGM neutral result

Job6096127 completed8h34m22s. Full200,000-question test scored26,192correct (13.096%), versus24,747correct (12.3735%) zero-shot: +0.7225percentage points. No invalid answers. Downloaded full predictions/checksums, independently rescored, confirmed exact same200k IDs, targets, manifest and image hashes as baseline. Neutral test is not an OOD regime. Two-hour adaptation trained22,656presentations2.517passes over9k; selected earlier checkpointstep1000 after15,992presentations. Small improvement remains close to12.5%chance. Other PGM evaluations remain partial and packaging job remains active.
