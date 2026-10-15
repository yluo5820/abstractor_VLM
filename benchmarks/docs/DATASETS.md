# Dataset preparation and evaluation protocol

**Current update, September 24:** The user has expanded the scope to full LFM2.5 evaluation on PGM neutral/test (200,000), CLEVR-ART RMTS/test (10,000), and CLEVR-ART identity-rules/test (10,000), using H200. See [the full-run plan](FULL_RUN.md) for preparation, scheduling and current resource approvals. Statements below describing the earlier 16-example CLEVR limit or deferred scope are historical.

Start with LFM2.5-VL-1.6B on a small, fixed manifest. Use exactly the same manifest,
contact sheets, prompts and decoding settings when adding SmolVLM2 or InternVL.
This package implements the two requested datasets. The broader Slot-Abstractor
repository also contains ART, SVRT and V-PROM; they need separate adapters and
protocols, rather than treating every repository folder as interchangeable data.
[Slot-Abstractor overview](https://github.com/Shanka123/Slot-Abstractor)

The current scope is the readily accessible **PGM neutral/test mirror** and the
**16 already-rendered CLEVR-ART RMTS test examples**. Those RMTS scenes were
rendered on ICE with selection seed `0` and render seed `1`; a real LFM2.5 GPU
smoke check also completed on ICE. Larger CPU rendering proposals, including
512-example CLEVR runs, are deferred. Identity-rules rendering has not been
tested on the cluster. Original PGM extrapolation remains inaccessible.

## Supported inputs

| Dataset | Supported task and regime | Inputs | Choices |
|---|---|---|---|
| PGM | `matrix-completion`; eight named regimes | One `PGM_REGIME_SPLIT_ID.npz` per example | A–H |
| PGM mirror | `matrix-completion`; `neutral` only | Pinned HuggingFaceM4/PGM test Parquet shards; separate mirror converter | A–H |
| CLEVR-ART | `rmts`, `ood` | Official symbolic NPZ plus two rendered candidate scenes per problem | A–B |
| CLEVR-ART | `idrules`, `ood` | Official symbolic NPZ plus four rendered candidate scenes per problem | A–D |

CLEVR-ART's two tasks test equality patterns over object appearance. Its training
objects are small/medium rubber cubes with cyan/brown/green/gray colors; its test
objects are large metal spheres/cylinders with yellow/purple/blue/red colors.
The original tasks present complete candidate scenes: top-row context repeats
for every candidate. RMTS compares pairs, and identity rules compares two rows
of three objects. [OCRA paper, Sections 4.1 and S2](https://arxiv.org/pdf/2306.02500)

The adapters deliberately accept only grounded upstream formats. CLEVR-ART's
NPZ files contain symbolic features and labels, **not rendered images**. Generic
CLEVR question-answer data, symbolic feature drawings, or a different scene
generator are not substitutes for these images.
[Original renderers](https://github.com/Shanka123/OCRA/tree/f15fa625d7e5b4458d82c24108b26419037d940d)

## PGM acquisition

The official source is [DeepMind's dataset repository](https://github.com/google-deepmind/abstract-reasoning-matrices),
which points to the `ravens-matrices` Google Cloud Storage bucket. It names:

```text
neutral              interpolation       extrapolation
attr.rel.pairs       attr.rels            attrs.pairs
attrs.shape.color    attrs.line.type
```

Each regime has 1.2 million training, 20,000 validation, and 200,000 test questions.
Do not download or unpack all regimes for a smoke test.
[PGM paper, Appendix A.1](https://arxiv.org/pdf/1807.04225)

At the 2026-09-24 UTC check, anonymous bucket listing returned HTTP 401, and direct
HEAD requests to both of these historically named objects returned HTTP 403:

```text
https://storage.googleapis.com/ravens-matrices/neutral.tar.gz
https://storage.googleapis.com/ravens-matrices/extrapolation.tar.gz
```

The original extrapolation archive remains inaccessible. An accessible
[HuggingFaceM4/PGM neutral mirror](https://huggingface.co/datasets/HuggingFaceM4/PGM)
is now staged on ICE: nine test shards at revision
`195a03a7e3bc6d6a349b55986afba5319929c3bf`, totaling 6,233,095,978 bytes and
covering 200,000 declared examples. The mirror converter verifies full shard
hashes against that pinned revision. Pixel equivalence to the unavailable
original NPZ files has not been independently established; identify results as
**PGM neutral, converted HuggingFaceM4 mirror**. See [PGM_MIRROR.md](PGM_MIRROR.md)
for acquisition, conversion and provenance details. This mirror does not supply
extrapolation or the other held-out generalization regimes.

For original regimes, use an authorized lab/teammate cache or obtain access from
the maintainers. No original PGM archive has been downloaded. Current original
compressed archive sizes are unverified. The 200,000-example test set has
81.92 GB of uncompressed panel pixels at one byte per pixel, before container
overhead; compressed size is different. Plan scratch space before expanding a
whole archive.

For a cache containing original NPZ files, run from `benchmarks/` after installing
the package:

```bash
python -m vlm_bench.data pgm \
  --root "$PGM_ROOT" \
  --output data/pgm-neutral-original-test-100.jsonl \
  --regime neutral --split test --limit 100 --seed 0
```

That command is for an original NPZ cache, not the staged Parquet mirror; use
the separate mirror converter for the current run. The active regime is
`neutral`. The Slot-Abstractor default is `extrapolation`, which remains deferred
until its data is accessible. Each regime and source remains a separate result.
[Slot-Abstractor PGM instructions](https://github.com/Shanka123/Slot-Abstractor#pgm)

The adapter uses the documented correction
`image.reshape(16, 160, 160)` even if the stored shape is `(160,160,16)`.
Transposing that array scrambles the official panels. Panels 0–7 are context;
panels 8–15 are ordered choices; target 0 maps to A, target 7 to H.
Only exact regime/split filenames are selected. Symbolic `meta_matrix` and
`meta_target` are never included in the model input.
[Official format and correction](https://github.com/google-deepmind/abstract-reasoning-matrices#update----fix-re-array-shape-and-readability-of-the-data)

**Data terms:** DeepMind specifies non-commercial research use for this data.
The repository's Apache-2.0 code license does not replace those dataset terms.
[Upstream terms](https://github.com/google-deepmind/abstract-reasoning-matrices)

## CLEVR-ART assets and renderer

The staging helper downloads only 18 required, checksum-pinned upstream files
(20,112,951 bytes total), including labels, renderers, Blender assets, utility
code, and licenses. It does not download model checkpoints. Existing matching
files are reused; different files are never replaced.

```bash
python cluster/prepare_data.py clevr-assets --destination "$CLEVR_ROOT"
```

The source revisions are:

- OCRA: `f15fa625d7e5b4458d82c24108b26419037d940d`.
- CLEVR generator: `f0ce2c81750bfae09b5bf94d009f42e055f2cb3a`.

OCRA's renderer imports `utils.py` from the original CLEVR generator; that file is
not the OCRA `util.py`. The staging helper supplies the correct pinned dependency
and records its checksum. [CLEVR utility source](https://github.com/facebookresearch/clevr-dataset-gen/blob/f0ce2c81750bfae09b5bf94d009f42e055f2cb3a/image_generation/utils.py)

The four released label files were inspected directly: each contains 10,000
examples; RMTS `x` has shape `(10000,6,4)`, identity rules `x` has shape
`(10000,9,4)`, and each `y` has shape `(10000,)`. Feature columns are color, shape,
size and material. Each RMTS label file is 2,000,490 bytes; each identity-rules
label file is 2,960,490 bytes. This is only symbolic data; rendered PNG storage
depends on the chosen number of examples. Source SHA-256 values are embedded in
`vlm_bench/data.py`, and staging verifies Git blob hashes too.
[Pinned data and code](https://github.com/Shanka123/OCRA/tree/f15fa625d7e5b4458d82c24108b26419037d940d)

### Blender runtime

Use a working Blender 2.78/2.79 installation. The original renderer uses the old
Blender API; a modern Blender executable is not a drop-in replacement.
[Original CLEVR setup instructions](https://github.com/facebookresearch/clevr-dataset-gen#step-1-generating-images)

If cluster policy permits a user-local executable, the verified official Linux
x86_64 archive is:

- [Blender 2.79b portable archive](https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2), 143,200,446 bytes.
- SHA-256: `43824a4e0b0c6de6fa34ff224eec44c1cc9f26a95f6f3c8c2558d1c05704183c`, checked against the [official checksum file](https://download.blender.org/release/Blender2.79/release279b.sha256).

Download and verify the checksum before extraction into a project tools
directory. Check `ldd /path/to/blender` for missing libraries and
`/path/to/blender --version`. No sudo is needed. Blender bundles its own Python;
the benchmark Python environment is separate. The generated wrapper adds the
utility import path, so installing a `.pth` file into Blender is unnecessary.

### Rendering and resuming

The supplied helper modifies a copy of the exact upstream script to select the
NPZ split and bounded example indices, seed random choices, add the utility
import path, and checkpoint completed problem groups. It preserves object
mappings, scene assets, camera/light settings,
320×240 resolution, and 64 Cycles samples. A source copy and JSON record make
the change inspectable. Upstream's `--num_images` argument alone is insufficient:
its task loop is hardcoded to 10,000 examples.
[RMTS source](https://github.com/Shanka123/OCRA/blob/f15fa625d7e5b4458d82c24108b26419037d940d/render_images_rmts.py),
[identity-rules source](https://github.com/Shanka123/OCRA/blob/f15fa625d7e5b4458d82c24108b26419037d940d/render_images_idrules.py)

The current 16-example RMTS set is already rendered and can be converted or
evaluated without more rendering. The following fresh-root command is retained
for future setup/debugging; it is not a request to expand the current run.
Omit `--run` when preparing scripts on a login node:

```bash
python cluster/prepare_data.py clevr-render \
  --root "$CLEVR_ROOT" --task rmts --split test \
  --limit 1 --seed 0 --render-seed 1 \
  --blender "$BLENDER_BIN" --threads 1
```

In an allocated compute shell, add `--run` to execute. The helper refuses to run
without a Slurm job ID, on a hostname containing `login`, with a different
Blender major/minor version, or with more renderer threads than allocated CPUs.
CPU rendering avoids depending on the old Blender CUDA interface. This still
belongs on a compute node. Record runtime for the first two PNGs before sizing a
larger rendering job. It does not write scene-answer JSONs; those writes are
commented out in the upstream renderer.

Completed groups can be reused in the same staged root with `--resume`. Before
reuse, the helper validates the full existing output against the source assets,
label hashes, selection seed, render seed, Blender version, PNG decoding and
resolution, and each PNG's recorded checksum. Increasing `--limit` with the same
seeds extends the deterministic sample and renders only new groups. Such an
increase is deferred for the current task; the larger CPU proposals are not
authorized by this guide.

The initial ICE RMTS run used the earlier helper, before per-group checkpoints
existed. Its completed run can be imported once with `--resume --adopt-legacy`:

```bash
python cluster/prepare_data.py clevr-render \
  --root "$CLEVR_ROOT" --task rmts --split test \
  --limit 16 --seed 0 --render-seed 1 \
  --blender "$BLENDER_BIN" --threads 1 \
  --resume --adopt-legacy
```

Without `--run`, this validates the import and saves an inspectable plan. With
`--run` in a compute allocation, it seals the old images with new checksums;
if all 16 groups are present, no new scenes are rendered. Legacy import requires
a successful completed run record, the exact generated script, matching source
and label hashes, and valid PNGs. It explicitly records that historical PNG
hashes were unavailable. Subsequent reuse needs only `--resume`.

New runs write an atomic `.benchmark-complete.json` receipt after every complete
problem group, so completed work remains reusable after interruption. A lock
prevents concurrent helper processes from writing the same task/split, and each
attempt receives a separate plan record. Incomplete, foreign, or modified groups
are refused; **no partial files are removed or silently re-rendered**. Inspect
such a group before choosing a recovery action. A different rendering protocol
needs a separate staged root rather than mixing results.

For future identity-rules work, change `--task rmts` to `--task idrules`; this
path has not undergone a real cluster render. For a future appearance reference
use `--split train`; the label filename is switched deliberately.
The OCRA README's identity-rules test-generation sentence names a training file;
the helper uses `identity_rules_ood_test.npz` for test data and validates held-out
feature ranges. [OCRA instructions](https://github.com/Shanka123/OCRA#clevr-art)

### Manifest conversion

Expected directory layout:

```text
CLEVR_ROOT/
  RMTS_ood_train.npz
  RMTS_ood_test.npz
  identity_rules_ood_train.npz
  identity_rules_ood_test.npz
  output/
    rmts_images/test_ood/prob_314/CLEVR_0.png
    rmts_images/test_ood/prob_314/CLEVR_1.png
    idrules_images/test_ood/prob_271/CLEVR_0.png
    idrules_images/test_ood/prob_271/CLEVR_1.png
    idrules_images/test_ood/prob_271/CLEVR_2.png
    idrules_images/test_ood/prob_271/CLEVR_3.png
```

`train_ood` uses the same structure. Problem numbers index the corresponding NPZ
label array directly; directory enumeration order never determines labels.

```bash
python -m vlm_bench.data clevr-art \
  --root "$CLEVR_ROOT" \
  --output data/clevr-rmts-ood-test-16.jsonl \
  --task rmts --regime ood --split test --limit 16 --seed 0
```

Only `ood` with `train`/`test` is supported for this release. A validation split
is not invented. The converter checks symbolic shapes, held-out feature ranges,
target indices, consistency with the task's equality rule, and complete PNG
candidate sets. It refuses inconsistent data instead of silently discarding
examples. OCRA's MIT license and the inherited CLEVR BSD license/patent file are
preserved by staging; retain them with copied assets.
[OCRA license](https://github.com/Shanka123/OCRA/blob/f15fa625d7e5b4458d82c24108b26419037d940d/LICENSE.md),
[CLEVR license](https://github.com/facebookresearch/clevr-dataset-gen/blob/f0ce2c81750bfae09b5bf94d009f42e055f2cb3a/LICENSE)

## Manifest contract and controls

Each JSONL record contains `id`, `dataset`, `task`, `regime`, `split`, `images`,
`prompt`, `choices`, `answer`, and `metadata`. `images` has exactly one relative
path, resolved relative to the manifest's directory. Labels are strings, never
model-specific token IDs. The evaluator must send only the prompt and PNG to
the model; `answer` and provenance are scoring data.

One lossless RGB PNG presents the full problem. PGM displays a 3×3 matrix with a
fixed missing cell and a 2×4 labeled choice grid. CLEVR-ART displays all complete
candidate scenes in two columns. Candidate order stays unchanged, and originals
are not resized or cropped. The model's own image processor may subsequently
resize images; record its actual configuration. Inspect the first real contact
sheet before treating accuracy as a model result.

Examples are ranked by SHA-256 of `seed + NUL + canonical_id`. Smaller limits
are prefixes of larger ones, independent of filesystem order and Python RNG
versions. A prepared manifest is fixed across models. If the available source
pool is only a partial cache, this is a subset of that cache, not automatically
a random sample of the entire dataset. The CLEVR renderer selects from all
10,000 official IDs before rendering; keep the same seed for conversion.

The sidecar `FILE.jsonl.provenance.json` stores source location, upstream source,
selection seed/method, available pool size, selected IDs, class counts, library
versions, and manifest hash. Each record includes source and PNG hashes. CLEVR
also records whether its label NPZ matches the official checksum and whether
the render pool is partial. Existing manifests/assets are never overwritten.

Treat 1–16 examples as integration checks, not evidence of benchmark performance.
A 100-example pilot estimates cost and common failures. Report per-task/regime
accuracy, uncertainty, invalid-answer rate, sample count, and uniform-choice
chance levels (PGM 12.5%, RMTS 50%, identity rules 25%) before larger runs.

For pretrained VLMs, call this zero-shot evaluation on the released appearance
splits. Without controlling pretraining exposure or training the model on the
official training split, an OOD score does not reproduce the original papers'
controlled systematic-generalization experiment. Keep any prompt development
on a separate development manifest; freeze the prompt before held-out scoring.

## Verification performed

- Nineteen data/resume tests passed. They cover panel-byte order, target-to-letter mapping, split
  isolation, fixed candidate order, pixel/color preservation, deterministic
  nested subsets, overwrite prevention, rejection of invalid/incomplete data,
  checkpoint integrity, concurrent-writer rejection, and legacy-run extension
  without overwriting existing PNGs or provenance.
- All 40,000 released CLEVR-ART symbolic examples were checked against the task
  equality rules and their training/test feature ranges.
- All 18 staged asset checksums and both generated test-renderer scripts were
  verified; the generated scripts parse under Python 3.5 syntax.
- Real original-quality rendering of 16 RMTS test problems (32 candidate PNGs)
  completed on ICE with selection seed `0` and render seed `1`. A real LFM2.5 GPU
  smoke check completed on ICE. These small checks establish working execution;
  use their saved run outputs for measured results rather than treating synthetic
  tests as benchmark evidence.
- Identity-rules cluster rendering and larger CLEVR datasets have not been
  verified. Current work uses the accessible neutral PGM mirror and existing
  RMTS images; the larger CPU rendering proposals are deferred.

```bash
python -m unittest discover -s tests -p test_data.py -v
```
