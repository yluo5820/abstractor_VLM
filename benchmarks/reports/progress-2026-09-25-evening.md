# Benchmark progress — September 25, 2026, 5:57 pm EDT

Both PGM comparison allocations have ended safely at their internal segment
deadlines. Neither full 200,000-example comparison has finished; both checkpoints
are resumable, and neither model currently occupies a GPU. No additional PGM
allocations have been submitted.

| Model | Questions evaluated | Coverage | Correct | Accuracy | Invalid | State |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LFM2.5-VL-1.6B | 200,000 / 200,000 | 100% | 24,747 | 12.3735% | 0 | Full evaluation complete |
| SmolVLM2-2.2B | 116,527 / 200,000 | 58.26% | 14,646 | 12.5688% | 0 | Checkpointed; segment finished |
| SpatialEvo-3B | 173,554 / 200,000 | 86.78% | 22,503 | 12.9660% | 1 | Checkpointed; segment finished |

Chance is 12.5%. SpatialEvo's final segment score is 12.9660%, a 0.466 percentage
point margin above chance under this protocol. Its reported Wilson interval is
12.809%–13.125%. These rows cover different prefix lengths; the earlier matched
116,527-example comparison remains available in the [afternoon report](progress-2026-09-25.md).
No new claims about training effects or general reasoning ability are implied.

SmolVLM2 job `5931708` stopped at 11:49 am EDT after 9h50m23s. SpatialEvo job
`5931709` stopped at 4:42 pm EDT after 9h50m25s. Both recorded a deliberate
`segment-deadline`, graceful interruption with no forced kill, and `resumable`
pipeline status. Slurm's `COMPLETED` refers to the checkpointed allocation, not
completion of the entire dataset. Smol has 83,473 questions left; SpatialEvo has
26,446 left. The new counts are read from the saved final-segment summaries.

## CLEVR-ART

**56,918 / 60,000 scenes (94.86%)** are verified complete: 9,493 / 10,000 RMTS problems and 9,483 / 10,000 identity-rule problems. Only 3,082 scenes remain.

The third and final planned rendering chunk (`5921714`) is running on eight
CPUs. The second chunk ended successfully, and the third started immediately.
Checkpoint verification has completed and new images are being produced. There
are no reported rendering errors. The current measured task rates imply roughly
1.8 hours of rendering remain, around 7:45–8:15 pm EDT if throughput holds.

Manifest preparation (`5921715`) and the two LFM H200 evaluations (`5921716`,
`5921717`) remain pending dependencies. Their queue waits and execution time are
additional to the rendering estimate. No GPU inference is currently running for
this campaign.

Raw scheduler status, summaries and log tails are retained in
`.local/campaigns/pgm-controls-2026-09-24/status-20260925T215749Z.json`.
No jobs were submitted, cancelled, resumed or modified during this check.
