# Full benchmark status — September 24, 2026, 09:55 PM EDT

PGM neutral/test completed successfully. CLEVR rendering is progressing without reported errors. Only LFM2.5-VL-1.6B has been scheduled for evaluation; SmolVLM2, InternVL and optional models have not started.

| Stage | Progress | Scheduler state |
| --- | --- | --- |
| PGM image preparation, job 5921697 | 200,000 images | COMPLETED, exit 0:0; 03:06:23 |
| PGM LFM2.5 evaluation, job 5921699 | 200,000/200,000 questions | COMPLETED, exit 0:0; 07:58:32 |
| CLEVR RMTS rendering | 4,014/10,000 problems | RUNNING on 8 CPUs in job 5921712 |
| CLEVR identity-rule rendering | 4,011/10,000 problems | Same CPU job |
| CLEVR continuation jobs 5921713–5921714 | Checkpoints retained | PENDING on dependencies |
| CLEVR manifest/completeness job 5921715 | Waits for full rendering | PENDING on dependency |
| CLEVR H200 evaluations 5921716–5921717 | Not started | PENDING on dependencies |

## Completed PGM result

- Correct: **24,747/200,000 = 12.3735%**.
- Chance: **12.5%**. Wilson 95% interval: **12.230%–12.519%**.
- Invalid outputs: **0**.
- Every saved response is one uppercase letter, with no token-budget hits. The model chose **E for 187,669 questions (93.83%)**, so the strong answer bias from the pilot persists.
- Independent verification reconciled all 200,000 unique predictions, their order, parsing, scores, and configured IDs with the saved summary. Always choosing E would score 12.4675% on these labels.
- One NVIDIA H200, BF16, 2 CPU threads, 1 inter-op thread. GPU released after successful completion at 6:44 pm EDT on September 24.
- Full H200 allocation: 7h 58m 32s. Runner startup/input validation/model loading: 71.23 minutes; evaluation loop including reads and durable writes: 6.78 hours. Summed generation time: 4.87 hours. Future planning should use the measured whole-job duration rather than the small in-memory audit extrapolation.

This is the frozen zero-shot **protocol-v1 result on the converted HuggingFaceM4 neutral mirror**, not a reproduction of Slot-Abstractor's supervised training results. It provides no evidence of above-chance accuracy under this protocol. Prompt/layout sensitivity and other models remain untested.

[Saved summary](../.local/remote-runs/lfm25-pgm-neutral-h200-full/summary.json), [raw predictions](../.local/remote-runs/lfm25-pgm-neutral-h200-full/predictions.jsonl), [configuration](../.local/remote-runs/lfm25-pgm-neutral-h200-full/config.json).

## CLEVR estimate

**24,072/60,000 scenes (40.12%)** have been published in complete, verified problem groups. Current rendering consumes 8 CPUs and no GPU. Pending H200 jobs hold no GPU allocation.

At the measured per-task worker rates, approximately **21.0 compute hours remain**, projecting to around **Friday September 25, 06:56 PM EDT** before additional queue delays, manifest preparation and GPU evaluation. Slurm currently provides no start estimate for the dependent jobs. The first rendering chunk stops at its 15h 50m internal budget, checkpoints completed groups, and the queued next chunk resumes them.

No new jobs were submitted, no resource requests increased, and no running jobs cancelled during this status check. The previously supplied homework script was only inspected and has not been run.
