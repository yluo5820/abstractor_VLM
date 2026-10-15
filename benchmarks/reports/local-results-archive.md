# Local benchmark result archive

## Shareable CLEVR-ART image dataset — September 30

The exact rendered test inputs are now packaged and downloaded locally as
`.local/exports/clevr-art-ood-test-seed1-v1/clevr-art-ood-test-seed1-v1.tar.gz`.
The archive is **6,612,104,418 bytes (6.61 GB)** and contains 20,000 unchanged
composite PNGs, both complete manifests with labels and exact prompts, official
symbolic test files, generation provenance, upstream notices, a README and a
standalone verification script. It covers 10,000 RMTS and 10,000 identity-rule
OOD test questions. It does not contain the training sets.

The adjacent `.tar.gz.sha256` checks the downloaded archive. Its SHA256 is
`54d41a8a14a5d77702ce7a9ffc80e2b1aff71310389102db2f73b237b59b5c87`.
`VERIFIED.json` records successful local verification of all 20,017 archive
members, every image hash and both original pinned manifest hashes. A copy of
the included `README.md` is alongside the archive for convenient preview.

The CPU-only packaging job `6001594` completed in 13m13s using two CPUs and
16 GiB RAM; no GPU was allocated. The archive was verified locally at
2026-09-30 06:34 UTC. At the user's request, the archive, checksum and README
were made readable by other ICE users at 07:14 UTC, without a cloud upload:

```text
/storage/ice1/9/8/yluo432/public-datasets/clevr-art-ood-test-seed1-v1
```

A teammate logged into ICE can copy the three files from their own scratch or
project directory:

```bash
cp -n /storage/ice1/9/8/yluo432/public-datasets/clevr-art-ood-test-seed1-v1/* .
```

Use a CPU allocation for the full checksum verification and extraction; no GPU
is needed. The included README explains the dataset layout and verification.
The folder is read-only to other users. This is filesystem sharing within ICE,
not an internet download link. The archive and checksum are hard links to the
already-verified files, so publication uses no additional archive storage.

The four pre-existing top-level folders (`vlm-cache`, `envs`, `vlm-benchmark`,
and `tools`) remain accessible only to the owner. Their permission bits changed
from 0755 to 0700 before storage-root traversal was enabled (0700 to 0711);
this preserves their prior effective privacy. Other users cannot list the
storage root. A default ACL makes newly created root entries private unless
the owner deliberately changes their permissions. POSIX permissions and ACLs
were audited, hard-link identity was checked, and inherited privacy was tested
with temporary empty entries. No second-account login was performed.

The publication receipt and before/after ACL records are saved under
`.local/campaigns/clevr-share-2026-09-30/`. To revoke this share while preserving
the archive, run on ICE as the owner:

```bash
chmod 700 /storage/ice1/9/8/yluo432/public-datasets
```

Packaging source, checks, submission and scheduler receipts are saved under
`.local/campaigns/clevr-share-2026-09-30/`. The remote archive remains in
`/storage/ice1/9/8/yluo432/vlm-benchmark/exports/clevr-art-ood-test-seed1-v1-20260930/`.

## Model prediction archives

Updated September 28, 2026, 11:28 pm EDT. All raw predictions produced so far
for the three PGM runs and six CLEVR-ART evaluations are now available locally
for offline analysis. The archive contains saved outputs and run metadata;
it is not a local mirror of the model weights or source image datasets.

## PGM neutral/test

| Model | Local predictions | Correct | Accuracy | Evaluation status |
| --- | ---: | ---: | ---: | --- |
| LFM2.5-VL-1.6B | 200,000 | 24,747 | 12.3735% | Full 200,000 complete |
| SmolVLM2-2.2B | 116,527 | 14,646 | 12.5688% | Stopped segment; 83,473 remain |
| SpatialEvo-3B | 173,554 | 22,503 | 12.9660% | Stopped segment; 26,446 remain |

The local archive is complete for the work already performed. The SmolVLM2 and
SpatialEvo evaluations themselves are still partial. No new jobs were submitted
or resumed while retrieving these records. Rows cover different prefix lengths;
uniform chance is 12.5%. On the shared first 116,527 examples, accuracy is
12.3371%, 12.5688% and 13.0296%, respectively.

Paths relative to `benchmarks/`:

- LFM: `.local/remote-runs/lfm25-pgm-neutral-h200-full/`.
- SmolVLM2: `.local/remote-runs/pgm-controls-2026-09-24-final-segments/smolvlm2/`.
- SpatialEvo: `.local/remote-runs/pgm-controls-2026-09-24-final-segments/spatialevo/`.

For the comparison models, `full/` contains raw predictions, summaries,
configurations, backend metadata and session records. The parent directories
also include pipeline status/configuration, segment records and separate
technical-preflight results. The original shorter snapshots under
`.local/remote-runs/pgm-controls-2026-09-24/` were preserved; use the
`-final-segments` directories for the latest available records.

The September 28 retrieval downloaded and verified **36 remote files totaling
186,593,772 bytes**, including SpatialEvo's previously missing 52,229 predictions.
Every downloaded file's SHA256 and size matches the remote source. A local audit
independently rescored all 490,081 available PGM predictions, matched summary
counts, verified unique ordered IDs and common input/protocol provenance, and
confirmed the older snapshots are unchanged prefixes. SpatialEvo has one invalid
response (`S`), counted as wrong; the other two have none. No run hit the
generation budget.

Checksums and audit:

- `.local/remote-runs/pgm-controls-2026-09-24-final-segments/remote-checksums.json`
- `.local/remote-runs/pgm-controls-2026-09-24-final-segments/local-verification.json`
- Reproducible audit: `.local/campaigns/pgm-controls-2026-09-24/audit_final_archive.py`

## CLEVR-ART

All six full evaluations, each with 10,000 predictions, remain available under
`.local/remote-runs/clevr-controls-2026-09-26/{model}/{task}/full/`.
Model keys are `lfm25`, `smolvlm2` and `spatialevo`; task keys are `rmts` and
`idrules`. See [the verified results](clevr-results-2026-09-26.md).

SSH is unnecessary for analyzing these saved results. Running additional
experiments or retrieving source images that are only on ICE still requires
cluster access.
