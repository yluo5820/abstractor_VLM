# PGM comparison campaign — September 24, 2026

**Update:** These submissions have since run. See [September 25 progress](progress-2026-09-25.md) for partial results, runtime and CLEVR status. The submission snapshot below is historical.

Submitted at 10:13 pm EDT; verified at 10:14 pm EDT:

| Job | Work | Verified status |
| --- | --- | --- |
| `5931563` | CPU-only model download/checksums | Completed successfully in 2m44s |
| `5931708` | SmolVLM2 preflight, then full PGM | Pending: Priority; no start estimate |
| `5931709` | SpatialEvo preflight, then full PGM | Pending: Priority; no start estimate |

Both GPU requests were accepted with exactly one H200, two CPUs, 16 GiB RAM and
10 hours. The download dependency has been satisfied. No comparison GPU inference
has been verified yet. Local validation passed 68 benchmark tests and 21 cluster
launcher tests. Frozen source archive SHA256:
`942209eaf2fad5381800613e70ace61bf13ddda11a7559cdf86cc82430efa5c5`.

The user requested two additional full PGM neutral comparisons after the LFM2.5
run produced 12.3735% accuracy and selected E on 93.8345% of examples. Its Wilson
95% interval includes the 12.5% chance rate. That result does not establish a
prompt bug or a general inability to reason.

| Model | Immutable checkpoint | Purpose |
| --- | --- | --- |
| SmolVLM2-2.2B-Instruct | `482adb537c021c86670beed01cd58990d01e72e4` | Independent compact general-purpose comparison |
| SpatialEvo-3B | `5fd684324a3d7e759c948b42f4a51b9b016c8d5f` | Spatial-training comparison using Qwen2.5-VL architecture |

[SpatialEvo's authors](https://huggingface.co/lidingm/SpatialEvo-3B) describe
spatial reinforcement learning from geometric environments. This does not
demonstrate PGM performance in advance. Although the checkpoint is named “3B,”
its stored BF16 tensors contain 4,065,787,904 parameters, approximately 4.07B.
It was selected for this campaign because its official weights are public and
its native Transformers backend fits the existing environment. The alternative
[Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) currently has
an access agreement requiring the user's account information.

## Fixed comparison protocol

Both models evaluate the existing 200,000-example neutral/test mirror manifest,
in the same order as LFM, with the same RGB contact sheets, A–H answer mapping,
question text, strict answer parser, greedy decoding, and 16-new-token ceiling.
This is a **direct-answer condition**, not a reasoning-enabled evaluation or a
reproduction of the models' published evaluations. Each model uses its native
chat template and image processor; their effective image representations differ.
Invalid answers count as wrong. No prompt or answer-position tuning is performed.

- Manifest SHA256: `c413d475a0191e2a4e23d1aeda5814e61c433e7b29ec2f04ab14818e289bc340`.
- LFM reference aggregate image SHA256: `d1eb83484b325eec9a2fdb992f233f8b2c84298a3799e7331b9e43b9c1f49443`.
- Data: `data/full-lfm25-2026-09-24/pgm-neutral-full.jsonl` on ICE.
- Source mirror and limitations: [PGM mirror notes](../docs/PGM_MIRROR.md).

Each first allocation performs 32 preflight examples from the existing pilot
manifest. The gate requires complete, durable outputs, at least one parseable
answer, and no response reaching the token ceiling. Accuracy and answer diversity
are not gating criteria. A failure stops the expensive full evaluation for
inspection; it does not trigger automatic prompt tuning. Raw outputs, processing
metadata and the gate decision are saved. These examples belong to the test pool
and are implementation checks, not an independent development set.

## Resources and continuation

Weights are downloaded and checksummed in a CPU-only Slurm preparation job.
Each model receives its own one-H200, two-CPU, 16-GiB allocation with a 10-hour
wall limit. The internal segment deadline leaves time to save the current answer
and release the allocation. Both jobs can run independently when resources are
available; CLEVR rendering and queued evaluations remain separate.

The first 10-hour segment is not a promise that all 200,000 examples will finish.
The new models' end-to-end speeds have not yet been measured. The comparison
wrapper supports resubmission against the same frozen source, model revision and
output directory. It resumes durable predictions; a completed run exits without
loading the model. Additional segments should use the saved submission command,
after checking the preceding segment has stopped. Resume never mixes modified
prompts, checkpoints, images, code or backend provenance.

The campaign's local submission receipts and frozen archive are under
`benchmarks/.local/campaigns/pgm-controls-2026-09-24/`. Remote campaign outputs are
under `/storage/ice1/9/8/yluo432/vlm-benchmark/runs/pgm-controls-2026-09-24/`.
Queue acceptance, preflight validation and full completion are separate states.

Check either job from the Mac (reuse or renew the private SSH connection first):

```bash
python3 /Users/yluo/Downloads/Projects/VLM/benchmarks/cluster/ice.py status --job 5931708
python3 /Users/yluo/Downloads/Projects/VLM/benchmarks/cluster/ice.py status --job 5931709
```

For a stopped, incomplete run, exact continuation commands are saved in
`.local/campaigns/pgm-controls-2026-09-24/smolvlm2-resume-command.txt` and
`spatialevo-resume-command.txt`. Check `pipeline-status.json` and logs first:
`resumable` is a deliberate checkpoint, `failed` needs investigation, and
`complete` needs no continuation. No additional segments have been submitted.
