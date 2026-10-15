# Compact VLM reasoning benchmarks on PACE ICE

**Shareable dataset, September 30:** The exact CLEVR-ART test inputs are downloaded and fully verified: a **6.61 GB archive with 20,000 images, labels, exact prompts, source notices and a README**. The archive, checksum and README are also readable by ICE users at `/storage/ice1/9/8/yluo432/public-datasets/clevr-art-ood-test-seed1-v1`. Other project files remain private. See [the archive index](reports/local-results-archive.md) for copying, verification and access-revocation instructions. No cloud upload is needed for ICE collaborators.

**Local archive, September 28:** All predictions produced so far are available offline: 200,000 LFM, 116,527 SmolVLM2 and 173,554 SpatialEvo PGM records, plus all six completed CLEVR evaluations. The final PGM segments were downloaded and verified against remote checksums; no experiments were resumed. See [local result locations and verification](reports/local-results-archive.md).

**Latest verified status, September 26, 11:47 pm EDT:** All six CLEVR evaluations completed, each with 10,000 questions and zero invalid outputs. RMTS / identity-rule accuracy: LFM2.5 **47.75% / 24.89%**, SmolVLM2 **50.48% / 25.39%**, SpatialEvo **55.76% / 41.38%** (chance: 50% / 25%). SmolVLM2 always answered B; LFM always answered A on identity rules. The queue is empty and GPUs are released. See [final results, timings and raw-output audit](reports/clevr-results-2026-09-26.md).

**Comparison campaign:** SmolVLM2-2.2B (`5931708`) and SpatialEvo-3B (`5931709`) use the same full PGM inputs and direct-answer protocol, with a technical preflight and resumable H200 segments. See [comparison protocol and original scheduling](reports/pgm-controls-2026-09-24.md).

**Current update, September 24:** The user has expanded the scope to full LFM2.5 evaluation on PGM neutral/test (200,000), CLEVR-ART RMTS/test (10,000), and CLEVR-ART identity-rules/test (10,000), using H200. See [the full-run plan](docs/FULL_RUN.md) for preparation, scheduling and current resource approvals. Statements below describing the earlier 16-example CLEVR limit or deferred scope are historical.

The initial model was **LFM2.5-VL-1.6B**, frozen and zero-shot; SmolVLM2-2.2B and SpatialEvo have now also run. The same input manifests and scoring additionally support the official Transformers conversion of InternVL3.5-1B. ICE login, Slurm allocation, isolated Python environment, original CLEVR rendering, and inference have been verified. See [verified setup](docs/VERIFIED_ICE.md). The initial pilot plans and results below are historical; the status above and linked final report supersede them.

The [completed first-pass report](reports/first-pass.md) records PGM neutral at **67/512 (13.09%)**, versus 12.5% chance, and the CLEVR-ART RMTS smoke check at **6/16 (37.5%)**, versus 50% chance. All outputs parsed successfully. PGM predictions strongly favored E (483/512); the pilot provides no evidence of above-chance accuracy under this protocol. The RMTS sample is too small for a firm conclusion. Both Slurm allocations completed successfully and released their resources.

## Current handoff

**H200 is now the verified standard.** Audit job `5920012` completed successfully after a 9:14 queue wait and 3:38 runtime. PGM reached 10.027 examples/s on 32 samples with two CPU threads, projecting to 5.54 inference hours for neutral/test before overhead. The default profile uses `ice-gpu`, account `coc`, QoS `coc-ice`, one H200, two CPUs and 16 GiB host RAM. See [GPU audit and resumption](docs/GPU_AUDIT.md) for caveats and batching differences. The original V100 profile remains available separately.

The Mac SSH configuration was inspected: `ice-login` connects to `yluo432@login-ice.pace.gatech.edu`. The user completed SSH/Duo and remote Codex authentication. The old `ice-gpu` alias is not evidence of a current allocation. For a later session, reconnect with:

Run on the Mac, completing password/Duo prompts in your terminal:

```bash
bash /Users/yluo/Downloads/Projects/VLM/benchmarks/cluster/connect.sh
```

No credentials need to be shared in chat. The private socket under `.local/` lets tasks reuse your login until the connection has been idle for one hour; interruptions can end it sooner. No `~/.ssh/config` changes were made. See [runtime and recurring logistics](docs/RUNTIME_LOGISTICS.md) for scaling estimates and authentication reuse.

The initial V100 inspection established account `coc`, partition `coc-gpu`, QoS `coc-ice`, and `--gres=gpu:v100:1`; that historical profile is `.local/ice-v100-profile.json`. The current `.local/ice-profile.json` selects H200 through `ice-gpu`. `cluster/profile.example.json` remains an unverified template. The dedicated remote checkout is `/storage/ice1/9/8/yluo432/vlm-benchmark`. See [the cluster guide](docs/CLUSTER.md).

## Scope and sequence

1. Inspect ICE and reuse suitable project storage/environment if present. Choose one remote benchmark checkout and save its exact path in the verified local profile.
2. Request one GPU, two CPUs, 16 GiB RAM, and 20 minutes using verified Slurm flags. Check actual hostname, job ID, GPU, and memory. No inference runs on the login node.
3. Locate existing data before downloading. CLEVR-ART requires official symbolic labels **and rendered scenes**. PGM accepts original NPZ samples or the separately documented, pinned neutral mirror. Acquisition and adapter details are in [DATASETS.md](docs/DATASETS.md).
4. Inspect contact sheets and run a 16-example smoke test per task. Confirm model loading, valid answer format, image visibility, runtime, and memory. A smoke test is only an implementation check.
5. Prepare deterministic evaluation manifests, initially a proposed 512 examples per task/regime. Use LFM2.5 first. Estimate the full run from measured throughput; obtain user approval before increasing resource requests.
6. Use `sbatch` for experiments. Each job requests its own resources, writes resumable predictions, and exits on completion. Evaluate the remaining models only after the first model and protocol work.

The user narrowed this first pass to readily accessible data: PGM neutral and the 16 already-rendered CLEVR-ART relational-match-to-sample examples. Identity rules, larger CLEVR render batches, and inaccessible PGM extrapolation/other regimes are deferred. ART, SVRT, and V-PROM in Slot-Abstractor remain extensions; this implementation does not claim support for them.

## Local validation and remote evaluation

Use a Python 3.10+ environment with NumPy and Pillow. Installing the lightweight package does not install model weights or PyTorch:

```bash
cd /Users/yluo/Downloads/Projects/VLM/benchmarks
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

On the allocated compute node, use the verified environment and dependency pins in [MODELS.md](docs/MODELS.md). Model revisions are recorded there and in `vlm_bench.models.DEFAULT_REVISIONS`.

Validate a prepared manifest without a GPU:

```bash
python -m vlm_bench.run --manifest data/pgm-neutral-test.jsonl \
  --output runs/lfm25-neutral-smoke --model lfm25 \
  --revision 919fde3d022e3f90a4716006f993938ee8c2eb97 --dry-run
```

Remove `--dry-run` **inside the GPU allocation** for the first 16 examples. `--max-samples 0` evaluates every row in the manifest. Use different output directories for smoke tests, pilots, full runs, changed inputs, and different models. Repeating an identical invocation resumes completed work.

Each run saves:

- `config.json`: protocol, seeds, model revision, manifest/image/code hashes, selected IDs.
- `backend.json`: model, processor, software, and precision provenance.
- `predictions.jsonl`: raw text, parsed answer, target, correctness, and latency per example.
- `summary.json`: per-task/regime/split accuracy, Wilson intervals, invalid-answer rate, chance level, latency, and completion counts.
- `session-*.json`: allocation, hardware, peak GPU memory, and run status.

Full output must not be called complete until `summary.json` says `complete: true`. Malformed answers count as incorrect. Runtime failures stop execution and remain visible in session logs; they are not silently discarded or counted as successful evaluations.

## Interpretation

These are pretrained VLM baselines. Slot-Abstractor's published pipeline trains task-specific models, so its results are a different experimental condition. In a frozen VLM, original PGM/CLEVR train/test labels describe dataset domains, not an experiment in which that VLM was trained on the official training split. Strong claims about systematic generalization require this qualification. See [PROTOCOL.md](docs/PROTOCOL.md).

Primary sources: [Slot-Abstractor](https://github.com/Shanka123/Slot-Abstractor), [OCRA](https://github.com/Shanka123/OCRA), [DeepMind PGM](https://github.com/google-deepmind/abstract-reasoning-matrices), and [PACE documentation supplied by the user](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042503). The ServiceNow article could not be retrieved in this session. Scheduler settings were verified on ICE; the accessible official orientation supports custom environments and user-installed software. Evidence and limitations are recorded in [VERIFIED_ICE.md](docs/VERIFIED_ICE.md).
