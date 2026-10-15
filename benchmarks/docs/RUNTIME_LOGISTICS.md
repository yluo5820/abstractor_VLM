# Runtime and logistics estimates — September 24, 2026

**Current update, September 24:** The user has expanded the scope to full LFM2.5 evaluation on PGM neutral/test (200,000), CLEVR-ART RMTS/test (10,000), and CLEVR-ART identity-rules/test (10,000), using H200. See [the full-run plan](FULL_RUN.md) for preparation, scheduling and current resource approvals. Statements below describing the earlier 16-example CLEVR limit or deferred scope are historical.

Planning only: no new jobs, model downloads, or resource increases were requested. “Full” means test-set evaluation, not training or running all training examples.

The H200 audit has since completed successfully: approximately 10.027 PGM examples/s, projecting to **5.54 inference hours** for neutral/test from a 32-example sample. The following original tables remain V100 extrapolations. See [GPU_AUDIT.md](GPU_AUDIT.md) for the measured H200 conditions, overhead caveats and verified 16-GPU-hour-per-job constraint.

## LFM2.5, measured on one V100

Current protocol: one example at a time, native processor, greedy short answers, BF16 tensors including emulation on V100. PGM measured 1.418445 seconds/example across 512 questions. RMTS measured 0.295480 seconds/example across only 16 questions. Timings exclude model loading, preparation, file persistence and queueing; no batching or native FP16 speedup has been measured.

| Test workload | Count | Inference extrapolation |
| --- | ---: | ---: |
| PGM neutral pilot | 512 | 12.1 minutes; actual whole batch 12:43 |
| PGM subset | 2,000 | 47.3 minutes |
| PGM subset | 10,000 | 3.94 hours |
| Full PGM neutral test | 200,000 | 78.8 hours / 3.28 GPU-days |
| All eight PGM test regimes, if obtained | 1,600,000 | 630.4 hours / 26.3 GPU-days, assuming equal throughput |
| Full CLEVR-ART RMTS test | 10,000 | 49.2 minutes, from a small timing sample |
| Full CLEVR-ART identity-rules test | 10,000 | Unmeasured; four candidates differ from RMTS's two |

For PGM neutral, 90–100 GPU-hours is a provisional V100 engineering budget after validating several-thousand-example chunks, not a measured completion time or confidence interval. Queueing and preparation are additional. The runner's former repeated record scans have since been replaced by incremental statistics and periodic summaries; the GPU inference timing itself has not been remeasured by that software improvement.

Nine neutral/test mirror shards (6.23 GB compressed, 200,000 examples) are already downloaded. Complete contact-sheet conversion and expanded storage are not yet timed/sized.

## CLEVR rendering, reusable across models

OCRA supplies 10,000 test labels per task. RMTS requires two scenes per question; identity rules requires four. At approximately 14–17 seconds per observed RMTS scene on one CPU worker:

- RMTS test requires 20,000 scenes: roughly 78–94 CPU-hours.
- Both test tasks require 60,000 scenes: roughly 233–283 CPU-hours **if identity scenes cost the same**. Identity scenes contain more objects and remain unmeasured.
- Eight independent equally fast workers would ideally take 29–35 elapsed hours for both tasks, before overhead. Parallelization and scaling have not been validated; requesting eight CPUs alone does not guarantee eightfold speedup.

This preparation is done once and reused for every model. It need not occupy a GPU. Larger rendering requests remain deferred.

## Other models: conditional budgets, not measured forecasts

SmolVLM2 and InternVL3.5 have prepared backends but no measured GPU timings. Optional Gemma 4 E2B needs additional integration. Parameter counts cannot reliably predict image preprocessing/prefill/decoding time.

| Measured seconds/example on the relevant task | PGM test, 200,000 | One CLEVR test task, 10,000 |
| ---: | ---: | ---: |
| 1 | 55.6 GPU-hours | 2.8 GPU-hours |
| 2 | 111.1 GPU-hours | 5.6 GPU-hours |
| 5 | 277.8 GPU-hours | 13.9 GPU-hours |

These rows are scenarios for any candidate, not forecasts assigned to particular models. Measure each task separately. Combined inference hours = 55.56 × PGM seconds/example + 2.778 × RMTS seconds/example + 2.778 × identity-rules seconds/example. A warm-up and small real sample per model/task supplies the missing measurements.

Gemma E2B has 2.3B effective parameters and 5.1B including embeddings; it is not a 2B stored checkpoint. Its current implementation requirements differ from our verified environment. Sources: [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct), [InternVL](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF), [Gemma](https://huggingface.co/google/gemma-4-E2B-it), [Gemma implementation guide](https://ai.google.dev/gemma/docs/capabilities/thinking).

## PGM access error

The [official GitHub repository](https://github.com/google-deepmind/abstract-reasoning-matrices) works. It links to [Google Cloud Storage](https://console.cloud.google.com/storage/browser/ravens-matrices). A bounded GET of `https://storage.googleapis.com/ravens-matrices/extrapolation.tar.gz` returned HTTP 403 and XML `AccessDenied` on September 24: the anonymous caller lacks `storage.objects.get`, with the qualification that the object may not exist. That cannot distinguish private data from moved/deleted objects. Current repository issues/comments gave no replacement location.

The user can try the official bucket console while signed into Google and report whether files are visible. Signing in does not grant object permission. If denied, a current link/permission from the owner or an authorized existing copy is needed. GitHub, PACE and VPN credentials do not themselves supply Google Cloud object permissions. The [neutral-only mirror](https://huggingface.co/datasets/HuggingFaceM4/PGM) works independently. [Google authentication documentation](https://docs.cloud.google.com/storage/docs/request-endpoints#authenticated-browser-downloads).

## Reusable authentication and batch jobs

- SSH/Duo is for a new authenticated PACE connection. Our private connection can serve multiple tasks/jobs. `ControlPersist=1h` means one hour idle without clients, not a fixed expiry one hour after login. Network/VPN loss, sleep or server policy can end it sooner. [OpenSSH reference](https://man.openbsd.org/ssh_config#ControlPersist).
- Slurm itself does not require a new Duo login for every job.
- Remote Codex caches its separate login and refreshes tokens during use. A fresh allocation using the same home environment normally reuses it; invalidated or cleared credentials require sign-in. Our Python benchmark batch scripts do not use Codex authentication at all. [Official Codex authentication](https://learn.chatgpt.com/docs/auth#login-caching).
- Once a batch submission succeeds, Slurm runs it independently of the laptop/SSH connection. Reconnect later to retrieve results. Do not assume this persistence for an interactive debug shell. [Slurm documentation](https://slurm.schedmd.com/sbatch.html).
- Code, environment, weights and data stay in the verified ICE paths and are reused, subject to cluster storage retention policy.

## Actual queue experience and limits

| Job | Submitted, Sep 23 EDT | Started | Wait | Runtime |
| --- | --- | --- | ---: | ---: |
| 5919352 interactive | 23:21:52 | 23:22:07 | 15 seconds | 18:34 |
| 5919573 PGM pilot | 23:41:26 | 23:41:39 | 13 seconds | 12:43 |

Both requested one V100, two CPUs, 16 GiB RAM and 20 minutes, and completed with exit 0:0. This small request was easy to obtain that evening; larger/longer requests and busier periods may wait much longer.

The live `coc-gpu` partition reported a 16-hour maximum per job; `coc-ice` QoS showed no separate MaxWall value. The project's 20-minute request was our modest initial limit, not the cluster maximum. Full runs need resumable chunks; 10,000 PGM examples take about four inference hours. Increasing wall time or concurrency requires the user's approval under the original instructions. This document does not request or authorize that increase.
