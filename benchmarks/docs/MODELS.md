# Frozen VLM model backends

Start with LFM2.5, validate the complete data-to-answer path on a small sample,
then add SmolVLM2 and coordinate the InternVL run with the existing team pipeline.
LFM2.5 has completed the full 200,000-example PGM neutral mirror run on an
allocated PACE H200 with BF16 tensors, following the earlier V100 checks.
SmolVLM2, InternVL and SpatialEvo native APIs/configurations have been checked;
their GPU integration remains subject to the recorded smoke runs. No model
weights were downloaded to the local Mac.

| Priority/key | Repository | Immutable default revision |
| --- | --- | --- |
| 1 / `lfm25` | [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) | `919fde3d022e3f90a4716006f993938ee8c2eb97` |
| 2 / `smolvlm2` | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | `482adb537c021c86670beed01cd58990d01e72e4` |
| 3 / `internvl35` | [OpenGVLab/InternVL3_5-1B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF) | `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| Spatial specialist / `spatialevo` | [lidingm/SpatialEvo-3B](https://huggingface.co/lidingm/SpatialEvo-3B) | `5fd684324a3d7e759c948b42f4a51b9b016c8d5f` |

The first three public Hub configurations were checked on 2026-09-23;
SpatialEvo was checked on 2026-09-24. These
are ordinary full-precision checkpoints, with the runtime precision chosen
explicitly. InternVL uses the publisher's HF-format conversion of InternVL3.5-1B;
record this repository when comparing to teammates' original custom-code format.
Its current config declares native `InternVLForConditionalGeneration` with a
Qwen3 text backbone. None of the four configs declares an `auto_map`; the
backend always disables remote repository code.

SpatialEvo was selected because its publisher's checkpoint is publicly accessible
and declares native `Qwen2_5_VLForConditionalGeneration`, supported by the existing
Transformers 5.1.0 environment. The [pinned checkpoint index](https://huggingface.co/lidingm/SpatialEvo-3B/blob/5fd684324a3d7e759c948b42f4a51b9b016c8d5f/model.safetensors.index.json)
reports **4,065,787,904 stored parameters**, despite the model's "3B" name, and
8,131,575,808 bytes of BF16 tensors across two shards. Its config declares FP32;
use explicit `--dtype bfloat16` on H200 to control runtime precision. No dependency
changes or remote repository code are required.

The [Cosmos-Reason2-2B publisher page](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
currently requires authenticated acceptance and contact-information sharing.
It was not selected for this run, and no alternative weight mirror is used.

## Environment

Use a dedicated user environment with Python 3.10–3.12. Inspect existing server
environments and installation policy first. The pinned requirements are based
on these primary sources:

- The [LFM2.5 model card](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)
  requires Transformers 5.1 or later and documents the native auto model API.
- [Transformers 5.1.0 dependency declarations](https://github.com/huggingface/transformers/blob/v5.1.0/setup.py)
  permit PyTorch 2.4+, Hugging Face Hub 1.3–<2, tokenizers 0.22–0.23, and
  Pillow 10.0.1–15. The selected versions satisfy those constraints; package
  metadata was also checked through the public PyPI JSON API.
- [PyTorch's official version matrix](https://pytorch.org/get-started/previous-versions/)
  pairs torch 2.7.1 with torchvision 0.22.1 and publishes CUDA 11.8, 12.6,
  and 12.8 wheel sets. Choose the wheel set after checking the GPU driver.

The verified ICE environment uses Python `3.11.7`, PyTorch `2.7.1+cu126`, and
Transformers `5.1.0`. CUDA availability was `True`, the benchmark package was
installed in editable mode, and `pip check` passed. Saved backend provenance confirms LFM inference used `torch.bfloat16` on
the V100; this is a completed integration check for LFM, not evidence that the
other model backends have been exercised.

The designated remote checkout `/storage/ice1/9/8/yluo432/vlm-benchmark` contains
the files from the local project's `benchmarks/` directory directly. To reproduce
the verified installation, activate the intended dedicated environment and run
from that checkout. These CUDA 12.6 wheels were verified with the current ICE
driver; reassess compatibility if the machine or driver changes.

```bash
cd /storage/ice1/9/8/yluo432/vlm-benchmark
source /storage/ice1/9/8/yluo432/envs/vlm-bench/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-models.txt
python -m pip install -e .
python -m pip check
mkdir -p /storage/ice1/9/8/yluo432/vlm-benchmark/results
python -m pip freeze > /storage/ice1/9/8/yluo432/vlm-benchmark/results/environment.freeze.txt
```

For another project checkout, change the explicit checkout and environment paths
together. Do not install into an existing shared environment. The requirements pin the major runtime
components; the freeze file captures the complete resolved transitive dependency
set. The command above writes that capture to the displayed absolute path; run
outputs also retain their own backend/package provenance. Torchvision is required for native image
preprocessing; `num2words` supports SmolVLM's processor. The backends request
PyTorch SDPA, so a separately compiled `flash-attn` package is unnecessary.

Use a verified project/scratch cache location with sufficient quota for weights,
then export `HF_HOME` there. Model fetches must follow PACE's verified download
policy. If compute nodes have no outbound access, prefetch each pinned snapshot
from an allowed host into that cache and run the job with `HF_HUB_OFFLINE=1`.
No API tokens are stored in the project.

## API and measurement protocol

```python
from vlm_bench.models import DEFAULT_REVISIONS, load_backend

# Call only after the runner has verified the compute allocation.
backend = load_backend("lfm25", DEFAULT_REVISIONS["lfm25"],
                       device="cuda", dtype="auto", max_new_tokens=16)
answer = backend.predict(composite_pil_image, question_prompt)
metadata = backend.metadata
sample_metadata = backend.last_prediction_metadata
```

`MODEL_IDS` and `DEFAULT_REVISIONS` are importable without torch. An omitted
revision uses the immutable default; an explicit branch/tag is resolved once
through a config-only Hub fetch, after which both model and processor are loaded
from the resolved SHA. The metadata includes that SHA, package versions,
processor configuration, chat-template hash, device, actual dtype, and generation
settings, finite context limit, and explicit `truncation=False`.
`trust_remote_code=True` is rejected: use the native HF-format model.

All models receive the same benchmark composite image and task prompt through
their own native chat template. The image is converted to RGB, and each model's
default processor controls resizing and tiling. This measures the model-plus-
processor system; it does not equalize visual token counts. Per-example tensor
shapes and input-token counts are available in `last_prediction_metadata` for
auditing that distinction.

The loader rejects missing native chat templates or image processors before
loading weights. Prediction rejects missing/empty image tensors and an expanded
input-token count plus output budget exceeding the finite native context limit.
It never truncates the question to fit. Qwen's actual `image_grid_thw` values are
recorded alongside tensor shapes; no model-specific resolution override is used.
These checks add small bookkeeping before generation, so new timing measurements
include that overhead.

For SpatialEvo, the pinned native template supplies its generic helpful-assistant
system message and leaves generation at the assistant prefix; it does not force
reasoning tags. The benchmark supplies the **unchanged PGM prompt and composite
image**, the same 16-token greedy budget, and the same strict final-letter parser
used for LFM. This is a controlled direct-answer comparison, not a reproduction
of an author's potentially longer reasoning evaluation. Inspect the GPU smoke
for answer-format failures and budget hits before the full run. Preserve those
failures in scoring; do not silently extract letters from explanations.

The models are frozen, run in evaluation/inference mode, and use greedy decoding
with one beam and a shared output budget. Only newly generated tokens are
decoded, so the answer parser cannot accidentally read the question's labels.
The checkpoint's other generation defaults are retained and logged. No custom
thinking system prompt is injected. A response that reaches the 16-token budget
is flagged; inspect these during the smoke run and, if changing the budget,
apply the same preregistered change to all models before the main benchmark.

`auto` uses PyTorch's `is_bf16_supported()` probe to select BF16, otherwise FP16. In PyTorch 2.7.1 the probe includes emulation by default: these V100 runs actually selected BF16 despite the hardware lacking native BF16 tensor cores. The saved `backend.json` is authoritative; do not label these results FP16 or infer native BF16 throughput. Use explicit `--dtype float16` for a future native FP16 condition, in a separate run directory. See [the exact PyTorch 2.7.1 implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/__init__.py).
CPU selects FP32. Explicit `bfloat16`, `float16`, and `float32` are supported.
The backend never silently moves a failed CUDA request to CPU. Record precision
and GPU model across comparisons, because greedy decoding alone does not promise
bitwise identical results across hardware or kernels.

Native API references: [LFM2-VL 5.1 documentation](https://huggingface.co/docs/transformers/v5.1.0/en/model_doc/lfm2_vl),
[SmolVLM2 model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct),
[InternVL 5.1 documentation](https://huggingface.co/docs/transformers/v5.1.0/en/model_doc/internvl),
and [Qwen2.5-VL 5.1 documentation](https://huggingface.co/docs/transformers/v5.1.0/en/model_doc/qwen2_5_vl).
