# CLEVR-ART OOD test: RMTS and identity rules

This is the exact rendered evaluation input used for our September 2026 VLM
comparison: **10,000 relational-match-to-sample (RMTS) questions and 10,000
identity-rule questions**. It contains finished PNG images, answer labels and
the exact task prompts. Blender is not required to use these files.

This is a derived evaluation package built from the official OCRA symbolic test
files and original renderers. It covers these two OOD test sets, not training
data or the full collection of datasets in Slot-Abstractor.

## Open and verify

Extract `clevr-art-ood-test-seed1-v1.tar.gz` using your archive utility, or:

```bash
tar -xzf clevr-art-ood-test-seed1-v1.tar.gz
cd clevr-art-ood-test-seed1-v1
python3 verify_bundle.py
```

Verification reads every file and checks its SHA256. It also checks the question
counts, IDs, choices, labels, image paths and image hashes. This takes a few
minutes and needs only Python 3.10 or later. The adjacent `.tar.gz.sha256` file
checks the archive before extraction (`shasum -a 256 -c` on macOS, or
`sha256sum -c` on Linux).

## Files and tasks

| Task | Manifest | Questions | Choices | Uniform chance |
| --- | --- | ---: | --- | ---: |
| RMTS | `clevr-art-rmts-test-full.jsonl` | 10,000 | A, B | 50% |
| Identity rules | `clevr-art-idrules-test-full.jsonl` | 10,000 | A, B, C, D | 25% |

Each JSONL line contains `id`, `images`, `prompt`, `choices`, `answer`, and source
metadata. `images[0]` is relative to the manifest's directory. Keep the two image
directories alongside their manifests when moving the package. The manifests
and image bytes have not been changed from the completed evaluations.

RMTS images are 688 x 342 pixels and show two labeled candidate scenes. Each
candidate repeats the source pair above a candidate pair. Identity-rule images
are 688 x 630 pixels and show four labeled candidate scenes; each scene contains
two rows of three objects. Select the candidate preserving the demonstrated
identity relation. Identity depends on color, shape, size and material.

`prompts.json` is a convenient copy of the exact prompts in the manifests.
Use those prompts when comparing against our scores, rather than replacing
them with a generic instruction to choose an answer. The optional `.npz` files
contain official symbolic features and targets for auditing, not model inputs.
`provenance/` records source versions and generation settings. Original
cluster paths and original scene references in provenance/metadata are historical
records; inference uses only the relative `images` paths in this package.

## Load an example

```python
from pathlib import Path
import json

root = Path("clevr-art-ood-test-seed1-v1")
with (root / "clevr-art-rmts-test-full.jsonl").open() as stream:
    for line in stream:
        example = json.loads(line)
        image_path = root / example["images"][0]
        prompt = example["prompt"]
        # Pass only this image and prompt to your model.
        # Use example["answer"] afterward to score the model's answer.
        break
```

The example assumes you run Python from the parent of the extracted folder.
If you are already inside it, use `root = Path(".")`.

## Comparison protocol

Our reported results use frozen models, zero-shot prompts, one composite image
per question, native model image processors/chat templates, BF16 and greedy
decoding with at most 16 new tokens. Evaluate both manifests in full. Labels and
symbolic attributes must not be included in model input. No task-specific
fine-tuning or demonstrations were used in our experiments.

Score each task separately. The parser accepts one allowed letter,
case-insensitively, optionally prefixed by `Answer:`, parenthesized, and/or
followed by a period. Prose, multiple letters and letters outside the task's
choices are invalid and count as wrong. Preserve raw responses. The exact
parser expression is `\s*(?:Answer\s*:\s*)?\(?([A-Z])\)?[.]?\s*`, matched
against the entire response with case-insensitive matching.

The `ood` designation refers to the original task's appearance split. It does
not prove that a pretrained VLM has never encountered related data. These
frozen-VLM evaluations differ from task-trained Slot-Abstractor experiments.

## Generation and provenance

The 60,000 original candidate scenes were rendered at 320 x 240 with Blender
2.79b, 64 Cycles samples, and base render seed 1 using the original OCRA scene
construction. Candidate scenes were then assembled into the 20,000 PNG inputs
included here. Selection seed 0 determines the recorded example order.
Source pins and original manifest hashes are in `provenance/bundle-spec.json`.

- OCRA: https://github.com/Shanka123/OCRA
  (commit `f15fa625d7e5b4458d82c24108b26419037d940d`).
- CLEVR generator: https://github.com/facebookresearch/clevr-dataset-gen
  (commit `f0ce2c81750bfae09b5bf94d009f42e055f2cb3a`).
- Slot-Abstractor: https://github.com/Shanka123/Slot-Abstractor
- OCRA paper: *Systematic Visual Reasoning through Object-Centric Relational
  Abstraction* (NeurIPS 2023), https://arxiv.org/abs/2306.02500.
- Slot-Abstractor paper: *Slot Abstractors: Toward Scalable Abstract Visual
  Reasoning* (ICML 2024), linked from its repository.

Please retain the included upstream notices in `licenses/` and cite the source
work when using the dataset. This package is our rendered copy, not an official
new release by the original authors.
