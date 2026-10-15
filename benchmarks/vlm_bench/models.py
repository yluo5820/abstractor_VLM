"""Native Transformers backends for the compact, frozen VLM benchmark.

Importing this module does not import PyTorch or download any files. Callers must
enforce their compute-allocation policy before calling :func:`load_backend`.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


MODEL_IDS = {
    "lfm25": "LiquidAI/LFM2.5-VL-1.6B",
    "smolvlm2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "internvl35": "OpenGVLab/InternVL3_5-1B-HF",
    "spatialevo": "lidingm/SpatialEvo-3B",
}

# Public Hub API/config metadata verified 2026-09-23; SpatialEvo added 2026-09-24.
# No model weights were downloaded during these metadata checks.
DEFAULT_REVISIONS = {
    "lfm25": "919fde3d022e3f90a4716006f993938ee8c2eb97",
    "smolvlm2": "482adb537c021c86670beed01cd58990d01e72e4",
    "internvl35": "9191dbccf312b537016f041b25d61c72e7c5c9f3",
    "spatialevo": "5fd684324a3d7e759c948b42f4a51b9b016c8d5f",
}

_MODEL_TYPES = {"lfm25": "lfm2_vl", "smolvlm2": "smolvlm", "internvl35": "internvl",
                "spatialevo": "qwen2_5_vl"}
_COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")


def _versions() -> dict[str, str | None]:
    result = {}
    for package in (
        "torch", "torchvision", "transformers", "huggingface-hub",
        "tokenizers", "safetensors", "Pillow", "numpy", "num2words",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def resolve_revision(model_key: str, revision: str | None = None) -> str:
    """Resolve a Hub reference to an immutable SHA, fetching only config.json.

    The default and full commit hashes need no network lookup. Branches/tags are
    resolved once via the normal Hub cache, which also supports HF_HUB_OFFLINE.
    Every subsequent processor/weight request uses the resulting commit.
    """
    if model_key not in MODEL_IDS:
        raise ValueError(f"Unknown model {model_key!r}; choose from {', '.join(MODEL_IDS)}")
    requested = revision or DEFAULT_REVISIONS[model_key]
    if _COMMIT.fullmatch(requested):
        return requested.lower()
    from huggingface_hub import hf_hub_download

    config_path = Path(hf_hub_download(MODEL_IDS[model_key], "config.json", revision=requested))
    # Do not resolve symlinks: snapshot files usually point to a blobs/ directory.
    resolved = config_path.parent.name
    if config_path.parent.parent.name != "snapshots" or not _COMMIT.fullmatch(resolved):
        raise RuntimeError(f"Could not recover the immutable Hub revision from {config_path}")
    return resolved.lower()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _context_limit(config: Any, tokenizer: Any) -> int:
    text_config = getattr(config, "text_config", None) or config
    candidates = [getattr(text_config, "max_position_embeddings", None),
                  getattr(config, "max_position_embeddings", None),
                  getattr(tokenizer, "model_max_length", None)]
    # HF uses extremely large sentinel integers for an unspecified tokenizer
    # limit. They are not evidence that the model supports that context length.
    finite = [value for value in candidates if type(value) is int and 0 < value < 1_000_000_000]
    if not finite:
        raise RuntimeError("No finite native context limit found; refusing an unchecked token budget")
    return min(finite)


class TransformersBackend:
    """A single-image, continuation-only, greedy generation backend."""

    def __init__(self, model: Any, processor: Any, torch_module: Any,
                 max_new_tokens: int, metadata: dict[str, Any]):
        self.model = model
        self.processor = processor
        self._torch = torch_module
        self.max_new_tokens = max_new_tokens
        self.metadata = metadata
        self.context_limit = metadata.get("context_limit")
        self.last_prediction_metadata: dict[str, Any] = {}

    def predict(self, image: Image.Image, prompt: str) -> str:
        """Generate an answer without returning echoed prompt or special tokens."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", truncation=False,
        )
        # Fail if a mismatched processor silently drops the image. All registered
        # native architectures require pixel_values for this image-only protocol.
        for name in ("input_ids", "pixel_values"):
            if name not in inputs or inputs[name].numel() == 0:
                raise RuntimeError(f"Native processor returned missing/empty {name} for an image question")
        input_length = int(inputs["input_ids"].shape[-1])
        if self.context_limit is not None and input_length + self.max_new_tokens > self.context_limit:
            raise ValueError(f"Input ({input_length}) plus output budget ({self.max_new_tokens}) "
                             f"exceeds native context limit ({self.context_limit}); no truncation is applied")
        grid = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None
        # BatchFeature.to preserves integer token IDs and masks while casting
        # floating image tensors to the model's actual inference precision.
        inputs = inputs.to(device=self.model.device, dtype=self.model.dtype)
        input_shapes = {
            key: list(value.shape) for key, value in inputs.items()
            if hasattr(value, "shape")
        }
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=False,
            )
        continuation = generated[0, input_length:]
        generated_length = int(continuation.shape[-1])
        self.last_prediction_metadata = {
            "input_tokens": input_length,
            "generated_tokens": generated_length,
            "input_shapes": input_shapes,
            # Hitting the budget may also coincide with EOS; retain the literal
            # observation instead of falsely claiming the answer was truncated.
            "reached_token_budget": generated_length >= self.max_new_tokens,
        }
        if grid is not None:
            self.last_prediction_metadata["image_grid_thw"] = grid
        return self.processor.decode(continuation, skip_special_tokens=True).strip()


def load_backend(
    model_key: str,
    revision: str | None = None,
    device: str = "cuda",
    dtype: str = "auto",
    max_new_tokens: int = 16,
    trust_remote_code: bool = False,
) -> TransformersBackend:
    """Load native, safetensors-only model weights on one selected device.

    ``auto`` means BF16 on supporting CUDA devices, otherwise FP16 on CUDA and
    FP32 on CPU. This avoids SmolVLM's checkpoint FP32 storage default silently
    doubling GPU memory. The resolved precision is recorded in ``metadata``.

    All registered repositories use native Transformers implementations;
    remote repository code is deliberately unsupported by this backend.
    """
    if model_key not in MODEL_IDS:
        raise ValueError(f"Unknown model {model_key!r}; choose from {', '.join(MODEL_IDS)}")
    if dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be auto, float32, float16, or bfloat16")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if trust_remote_code:
        raise ValueError("These backends use native Transformers code; trust_remote_code must be False")

    import torch
    from packaging.version import Version
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    if Version(version("transformers")) < Version("5.1.0"):
        raise RuntimeError("This benchmark requires Transformers >=5.1.0; see requirements-models.txt")
    selected_device = torch.device(device)
    if selected_device.type not in {"cuda", "cpu"}:
        raise ValueError("This benchmark supports explicit cuda[:index] or cpu devices")
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; enter an allocated GPU job and check its environment")
        if selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(selected_device)
    if dtype == "auto":
        if selected_device.type == "cuda":
            selected_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            selected_dtype = torch.float32
    else:
        selected_dtype = getattr(torch, dtype)
    if selected_device.type == "cuda" and selected_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 is unsupported by the selected GPU; choose float16 explicitly")

    resolved = resolve_revision(model_key, revision)
    model_id = MODEL_IDS[model_key]
    load_options = {"revision": resolved, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(model_id, **load_options)
    if config.model_type != _MODEL_TYPES[model_key]:
        raise RuntimeError(f"Unexpected model_type {config.model_type!r} for {model_id} at {resolved}")
    processor = AutoProcessor.from_pretrained(model_id, **load_options)
    tokenizer = getattr(processor, "tokenizer", None)
    image_processor = getattr(processor, "image_processor", None)
    chat_template = getattr(processor, "chat_template", None)
    if chat_template is None and tokenizer is not None:
        chat_template = getattr(tokenizer, "chat_template", None)
    if tokenizer is None or image_processor is None:
        raise RuntimeError(f"Native multimodal processor is incomplete for {model_id} at {resolved}")
    if not isinstance(chat_template, (str, dict)) or not chat_template:
        raise RuntimeError(f"No native chat template for {model_id} at {resolved}; no template is substituted")
    context_limit = _context_limit(config, tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, config=config, dtype=selected_dtype,
        attn_implementation="sdpa", use_safetensors=True, **load_options,
    ).to(selected_device)
    model.eval()
    model.requires_grad_(False)

    template_bytes = json.dumps(chat_template, sort_keys=True, ensure_ascii=False).encode("utf-8")
    metadata = {
        "model_key": model_key,
        "model_id": model_id,
        "requested_revision": revision or DEFAULT_REVISIONS[model_key],
        "resolved_revision": resolved,
        "processor_revision": resolved,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "image_processor_class": type(image_processor).__name__,
        "image_processor_config": _json_safe(image_processor.to_dict()) if image_processor else None,
        "context_limit": context_limit,
        "processor_overrides": {"truncation": False},
        "chat_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "packages": _versions(),
        "device": str(model.device),
        "dtype_requested": dtype,
        "dtype": str(model.dtype),
        "attention_implementation": "sdpa",
        "trust_remote_code": False,
        "quantization": None,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
        "generation_config": _json_safe(model.generation_config.to_dict()),
        "generation_overrides": {
            "do_sample": False, "num_beams": 1, "max_new_tokens": max_new_tokens,
            "use_cache": True, "return_dict_in_generate": False,
        },
        "input_protocol": "single RGB composite image, user prompt, native chat template",
    }
    return TransformersBackend(model, processor, torch, max_new_tokens, metadata)
