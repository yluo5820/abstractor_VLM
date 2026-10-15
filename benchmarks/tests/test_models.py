from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from vlm_bench import models


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def numel(self):
        return self.values.size

    def tolist(self):
        return self.values.tolist()

    def __getitem__(self, item):
        return Tensor(self.values[item])


class Inputs(dict):
    def to(self, **kwargs):
        self.move = kwargs
        return self


class Device:
    def __init__(self, name, index=None):
        self.type = name.split(":")[0]
        self.index = index if index is not None else int(name.split(":")[1]) if ":" in name else None

    def __str__(self):
        return self.type + (f":{self.index}" if self.index is not None else "")


class ModelTests(unittest.TestCase):
    def fixtures(self, model_type="qwen2_5_vl"):
        torch = SimpleNamespace(device=Device, bfloat16="torch.bfloat16", float16="torch.float16",
            float32="torch.float32", inference_mode=nullcontext, version=SimpleNamespace(cuda="12.6"),
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                current_device=lambda: 0, set_device=Mock(), get_device_name=lambda device: "NVIDIA H200"))
        config = SimpleNamespace(model_type=model_type, text_config=SimpleNamespace(max_position_embeddings=128000))
        processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None, model_max_length=131072),
            image_processor=SimpleNamespace(to_dict=lambda: {"min_pixels": 3136, "max_pixels": 12845056}),
            chat_template="native-image-and-user-text-template")
        model = SimpleNamespace(device=Device("cuda", 0), dtype=torch.bfloat16,
            eval=Mock(), requires_grad_=Mock(), generation_config=SimpleNamespace(to_dict=lambda: {"eos_token_id": 151645}))
        model.to = Mock(return_value=model)
        transformers = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
            AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=Mock(return_value=model)))
        return torch, transformers, processor, model

    def test_specialist_factory_pins_native_config_processor_weights_and_precision(self):
        torch, transformers, processor, model = self.fixtures()
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
             patch.object(models, "version", return_value="5.1.0"), \
             patch.object(models, "_versions", return_value={"transformers": "5.1.0"}):
            backend = models.load_backend("spatialevo", dtype="bfloat16")
        repo = "lidingm/SpatialEvo-3B"
        sha = "5fd684324a3d7e759c948b42f4a51b9b016c8d5f"
        options = {"revision": sha, "trust_remote_code": False}
        transformers.AutoConfig.from_pretrained.assert_called_once_with(repo, **options)
        transformers.AutoProcessor.from_pretrained.assert_called_once_with(repo, **options)
        loaded = transformers.AutoModelForImageTextToText.from_pretrained.call_args
        self.assertEqual(loaded.args, (repo,))
        self.assertEqual(loaded.kwargs["revision"], sha)
        self.assertFalse(loaded.kwargs["trust_remote_code"])
        self.assertEqual(loaded.kwargs["dtype"], "torch.bfloat16")
        self.assertEqual(loaded.kwargs["attn_implementation"], "sdpa")
        self.assertTrue(loaded.kwargs["use_safetensors"])
        self.assertEqual(backend.metadata["resolved_revision"], sha)
        self.assertEqual(backend.metadata["processor_revision"], sha)
        self.assertEqual(backend.metadata["context_limit"], 128000)
        self.assertEqual(backend.metadata["processor_overrides"], {"truncation": False})
        self.assertEqual(backend.metadata["image_processor_config"], processor.image_processor.to_dict())
        model.requires_grad_.assert_called_once_with(False)
        model.eval.assert_called_once()

    def test_rejects_wrong_architecture_or_missing_processor_before_weight_load(self):
        for failure in ["wrong-type", "no-template", "no-image-processor"]:
            with self.subTest(failure=failure):
                torch, transformers, processor, model = self.fixtures("qwen3_vl" if failure == "wrong-type" else "qwen2_5_vl")
                if failure == "no-template":
                    processor.chat_template = None
                if failure == "no-image-processor":
                    processor.image_processor = None
                with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
                     patch.object(models, "version", return_value="5.1.0"), self.assertRaises(RuntimeError):
                    models.load_backend("spatialevo", dtype="bfloat16")
                transformers.AutoModelForImageTextToText.from_pretrained.assert_not_called()

    def test_prediction_preserves_native_image_prompt_grid_and_only_decodes_continuation(self):
        torch, _, processor, model = self.fixtures()
        inputs = Inputs(input_ids=Tensor([[10, 11, 12]]), attention_mask=Tensor([[1, 1, 1]]),
                        pixel_values=Tensor([[.2, .3]]), image_grid_thw=Tensor([[1, 28, 28]]))
        processor.apply_chat_template = Mock(return_value=inputs)
        processor.decode = Mock(return_value=" C ")
        model.generate = Mock(return_value=Tensor([[10, 11, 12, 34, 151645]]))
        backend = models.TransformersBackend(model, processor, torch, 16, {})
        image = Image.new("RGB", (700, 1016), "white")
        prompt = "Original PGM prompt; only A through H."
        self.assertEqual(backend.predict(image, prompt), "C")
        call = processor.apply_chat_template.call_args
        self.assertEqual(call.kwargs, dict(add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt", truncation=False))
        messages = call.args[0]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[1], {"type": "text", "text": prompt})
        self.assertEqual(content[0]["image"].size, image.size)
        self.assertEqual(content[0]["image"].tobytes(), image.tobytes())
        self.assertEqual(processor.decode.call_args.args[0].values.tolist(), [34, 151645])
        self.assertEqual(processor.decode.call_args.kwargs, {"skip_special_tokens": True})
        generated = model.generate.call_args.kwargs
        self.assertIs(generated["image_grid_thw"], inputs["image_grid_thw"])
        self.assertEqual(generated["max_new_tokens"], 16)
        self.assertFalse(generated["do_sample"])
        self.assertEqual(generated["num_beams"], 1)
        self.assertEqual(backend.last_prediction_metadata["input_tokens"], 3)
        self.assertEqual(backend.last_prediction_metadata["generated_tokens"], 2)
        self.assertFalse(backend.last_prediction_metadata["reached_token_budget"])
        self.assertEqual(backend.last_prediction_metadata["input_shapes"]["image_grid_thw"], [1, 3])
        self.assertEqual(backend.last_prediction_metadata["image_grid_thw"], [[1, 28, 28]])

    def test_context_guard_counts_visual_tokens_and_output_without_truncation(self):
        torch, _, processor, model = self.fixtures()
        processor.apply_chat_template = Mock(return_value=Inputs(
            input_ids=Tensor([[1, 2, 3]]), pixel_values=Tensor([[.2, .3]])))
        model.generate = Mock()
        backend = models.TransformersBackend(model, processor, torch, 16, {"context_limit": 18})
        with self.assertRaisesRegex(ValueError, "exceeds native context limit"):
            backend.predict(Image.new("RGB", (8, 8)), "Choose A or B.")
        model.generate.assert_not_called()
        self.assertFalse(processor.apply_chat_template.call_args.kwargs["truncation"])
        limit = models._context_limit(SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=8192)),
                                      SimpleNamespace(model_max_length=10**30))
        self.assertEqual(limit, 8192)
        self.assertEqual(models._context_limit(SimpleNamespace(max_position_embeddings=8192),
                                              SimpleNamespace(model_max_length=4096)), 4096)
        with self.assertRaisesRegex(RuntimeError, "No finite"):
            models._context_limit(SimpleNamespace(), SimpleNamespace(model_max_length=10**30))

    def test_missing_or_empty_image_tensor_never_generates_text_only_result(self):
        torch, _, processor, model = self.fixtures()
        model.generate = Mock()
        for pixel in [None, Tensor([])]:
            inputs = Inputs(input_ids=Tensor([[1, 2]]))
            if pixel is not None:
                inputs["pixel_values"] = pixel
            processor.apply_chat_template = Mock(return_value=inputs)
            backend = models.TransformersBackend(model, processor, torch, 16, {})
            with self.assertRaisesRegex(RuntimeError, "pixel_values"):
                backend.predict(Image.new("RGB", (8, 8)), "Choose A or B.")
        model.generate.assert_not_called()

    def test_full_revision_normalization_and_remote_code_rejection_need_no_download(self):
        self.assertEqual(models.resolve_revision("spatialevo", "A" * 40), "a" * 40)
        self.assertEqual(models.resolve_revision("spatialevo"), models.DEFAULT_REVISIONS["spatialevo"])
        with self.assertRaisesRegex(ValueError, "trust_remote_code"):
            models.load_backend("spatialevo", trust_remote_code=True)


if __name__ == "__main__":
    unittest.main()
