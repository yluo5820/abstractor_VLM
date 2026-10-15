# PGM neutral test mirror

The [HuggingFaceM4/PGM mirror](https://huggingface.co/datasets/HuggingFaceM4/PGM)
contains the neutral regime as converted Parquet files. Its original-NPZ pixel
equivalence has not been independently verified. Keep this distinction in result
tables, and follow the [original PGM terms](https://github.com/google-deepmind/abstract-reasoning-matrices)
for non-commercial research. This mirror does not provide the extrapolation or
other systematic-generalization regimes.

The pinned revision is `195a03a7e3bc6d6a349b55986afba5319929c3bf`.
There are nine test shards, totaling 6,233,095,978 bytes (about 5.81 GiB), and
200,000 declared test examples. Each shard is roughly 690–696 MB. Downloading all
nine permits selection from the entire test population. Selecting from one shard
is suitable for a smoke test, but is recorded as a subset of that available pool.

After choosing an approved PACE project/scratch location, activate the dedicated
environment and install the optional converter dependency:

```bash
python -m pip install -r benchmarks/requirements-data.txt
```

Set `PGM_MIRROR_DIR` to that approved location. The following acquisition snippet
fetches only the test split from the exact revision using the existing
`huggingface-hub` dependency:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceM4/PGM",
    repo_type="dataset",
    revision="195a03a7e3bc6d6a349b55986afba5319929c3bf",
    allow_patterns=["data/test-*.parquet"],
    local_dir=os.environ["PGM_MIRROR_DIR"],
    max_workers=4,
)
PY
```

The direct URL shape is
`https://huggingface.co/datasets/HuggingFaceM4/PGM/resolve/195a03a7e3bc6d6a349b55986afba5319929c3bf/data/test-00000-of-00009.parquet`,
with the shard index ranging from `00000` through `00008`. The converter verifies
every supplied shard's complete SHA256 against its pinned Hub LFS checksum before
reading examples; incorrect bytes or an unrecognized filename are rejected.

Create a new manifest for the pilot, with `PGM_PILOT_MANIFEST` set to a new `.jsonl`
path under the approved project/scratch location:

```bash
PYTHONPATH=benchmarks python -m vlm_bench.pgm_mirror \
  --parquet "$PGM_MIRROR_DIR"/data/test-*.parquet \
  --output "$PGM_PILOT_MANIFEST" --limit 512 --seed 0
```

The converter performs no network requests. It ranks canonical sample IDs by a
seeded hash, reads each selected Parquet row group once, and retains only selected
image bytes. It projects only `panels`, `choices`, `target`, and `id`; malformed
rule-structure metadata is never read into Python or included in model prompts.
Context and choice order are preserved, and target indices `0`–`7` map to `A`–`H`.

The manifest sidecar records the mirror revision, complete shard hashes and sizes,
available and selected counts, selection seed, and exact selected IDs. Existing
manifests are never overwritten. A larger pilot uses a new run directory because
its manifest and executable-source hashes differ from an earlier smoke run.
