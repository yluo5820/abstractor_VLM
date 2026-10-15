from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from vlm_bench import perf_audit as audit


class Tensor:
    """Small CPU stand-in: the tests never load torch or model weights."""
    def __init__(self, data):
        self.array = np.asarray(data)
        self.shape = self.array.shape

    def sum(self, dim):
        return Tensor(self.array.sum(axis=dim))

    def tolist(self):
        return self.array.tolist()

    def __getitem__(self, item):
        return Tensor(self.array[item])

    def detach(self):
        return self

    def cpu(self):
        return self


class Inputs(dict):
    def to(self, **kwargs):
        self.move = kwargs
        return self


class FakeOOM(RuntimeError):
    pass


def fake_torch():
    cuda = SimpleNamespace(
        empty_cache=Mock(), synchronize=Mock(), reset_peak_memory_stats=Mock(),
        max_memory_allocated=Mock(return_value=1024), max_memory_reserved=Mock(return_value=2048),
        OutOfMemoryError=FakeOOM, manual_seed_all=Mock(), get_device_name=Mock(return_value="NVIDIA H200"),
    )
    return SimpleNamespace(
        cuda=cuda, inference_mode=nullcontext, get_num_threads=Mock(return_value=2),
        set_num_threads=Mock(), set_num_interop_threads=Mock(), manual_seed=Mock(),
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True)),
    )


def samples():
    return [{"row": {"id": str(index), "prompt": f"Choose {index}.",
                     "answer": answer, "choices": ["A", "B"], "secret_rule": "must never be sent"},
             "image": Image.new("RGB", (13 + index, 17)),
             "source_image": {"width": 13 + index, "height": 17, "mode": "RGB"}}
            for index, answer in enumerate(["A", "B"])]


class AuditTests(unittest.TestCase):
    def test_native_batch_uses_only_pixels_prompt_and_left_padding(self):
        tokenizer = SimpleNamespace(padding_side="right")
        inputs = Inputs(input_ids=Tensor([[0, 11, 12], [21, 22, 23]]),
                        attention_mask=Tensor([[0, 1, 1], [1, 1, 1]]))
        captured = []

        def template(conversations, **kwargs):
            self.assertEqual(tokenizer.padding_side, "left")
            self.assertEqual(kwargs, dict(add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors="pt", padding=True))
            captured.extend(conversations)
            return inputs

        processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=Mock(side_effect=template),
                                    batch_decode=Mock(return_value=[" A ", " B "]))
        model = SimpleNamespace(device="cuda:0", dtype="bfloat16",
            generation_config=SimpleNamespace(eos_token_id=2, pad_token_id=0),
            generate=Mock(return_value=Tensor([[0, 11, 12, 31, 2, 0], [21, 22, 23, 32, 33, 2]])))
        backend = SimpleNamespace(model=model, processor=processor, max_new_tokens=16)
        rows = samples()
        texts, metadata = audit.predict_batch(backend, rows, fake_torch())
        self.assertEqual(texts, ["A", "B"])
        self.assertEqual([r["input_tokens"] for r in metadata], [2, 3])
        self.assertEqual([r["generated_tokens"] for r in metadata], [2, 3])
        self.assertEqual([r["padded_input_tokens"] for r in metadata], [3, 3])
        processor.batch_decode.assert_called_once_with([[31, 2, 0], [32, 33, 2]], skip_special_tokens=True)
        self.assertEqual(inputs.move, {"device": "cuda:0", "dtype": "bfloat16"})
        self.assertEqual(tokenizer.padding_side, "right")
        for index, conversation in enumerate(captured):
            self.assertEqual(set(conversation[0]), {"role", "content"})
            content = conversation[0]["content"]
            self.assertEqual(content[1], {"type": "text", "text": rows[index]["row"]["prompt"]})
            self.assertEqual(content[0]["image"].size, rows[index]["image"].size)
            self.assertEqual(content[0]["image"].tobytes(), rows[index]["image"].tobytes())
        options = model.generate.call_args.kwargs
        self.assertFalse(options["do_sample"])
        self.assertEqual(options["num_beams"], 1)
        self.assertEqual(options["max_new_tokens"], 16)
        model.generate.side_effect = FakeOOM("no space")
        with self.assertRaises(FakeOOM):
            audit.predict_batch(backend, rows, fake_torch())
        self.assertEqual(tokenizer.padding_side, "right")

    def test_generation_length_counts_eos_but_excludes_batch_padding(self):
        self.assertEqual(audit.generated_length([4, 2, 0, 0], [2, 3], 0), 2)
        self.assertEqual(audit.generated_length([4, 0, 0], None, 0), 1)
        self.assertEqual(audit.generated_length([4, 5], 2, None), 2)

    def test_measurement_scores_records_and_preserves_failure(self):
        backend = SimpleNamespace(last_prediction_metadata={"input_tokens": 20})
        backend.predict = Mock(side_effect=lambda image, prompt: "A" if prompt == "Choose 0." else "B")
        torch = fake_torch()
        result = audit.measure_condition(backend, samples(), torch, 1, 1, "serial", None)
        self.assertEqual(result["status"], "complete")
        self.assertEqual((result["evaluated"], result["correct"], result["invalid_outputs"]), (2, 2, 0))
        self.assertEqual(backend.predict.call_count, 3)  # One warmup, two timed examples.
        self.assertEqual(result["peak_gpu_allocated_bytes"], 1024)
        self.assertGreater(result["wall_seconds"], 0)
        self.assertGreater(result["peak_rss_observed_bytes"], 0)
        self.assertEqual(result["records"][1]["source_image"]["width"], 14)
        baseline = {row["id"]: row for row in result["records"]}
        backend.predict = Mock(side_effect=["A", FakeOOM("no space")])
        failed = audit.measure_condition(backend, samples(), torch, 1, 1, "serial", baseline, warmup_batches=0)
        self.assertEqual(failed["status"], "out-of-memory")
        self.assertEqual(failed["evaluated"], 1)
        self.assertGreater(failed["peak_rss_observed_bytes"], 0)
        self.assertIn("no space", failed["error"])

    def test_selection_excludes_fast_but_changed_outputs(self):
        reference = {"id": "0", "raw_output": "A", "prediction": "A", "prediction_metadata": {"input_tokens": 4}}
        changed = dict(reference, raw_output="B", prediction="B", prediction_metadata={"input_tokens": 5})
        mismatch = audit.compare_outputs([changed], {"0": reference})
        self.assertEqual([mismatch[key] for key in ("raw_mismatches", "prediction_mismatches", "input_token_mismatches")], [1, 1, 1])
        stable = dict(status="complete", samples_per_second=2, comparison_to_serial=audit.compare_outputs([reference], None))
        fast = dict(status="complete", samples_per_second=5, comparison_to_serial=mismatch)
        self.assertIs(audit.best_condition([stable, fast], equivalent=True), stable)
        self.assertIs(audit.best_condition([stable, fast]), fast)

    def test_time_budget_is_explicit_partial_with_distinct_exit(self):
        backend = SimpleNamespace(predict=Mock(), last_prediction_metadata={})
        result = audit.measure_condition(backend, samples(), fake_torch(), 1, 1, "serial", None,
                                         warmup_batches=0, deadline=-1)
        self.assertEqual(result["status"], "time-budget-exhausted")
        self.assertEqual(result["evaluated"], 0)
        backend.predict.assert_not_called()
        with patch("sys.argv", ["perf_audit", "--output", "/tmp/unused.json"]), \
             patch.object(audit, "run_audit", return_value={"status": "time-budget-exhausted", "recommendations": {}}), \
             patch("sys.stderr"), patch("sys.stdout"):
            with self.assertRaises(SystemExit) as error:
                audit.main()
        self.assertEqual(error.exception.code, 3)

    def test_audit_loads_once_and_checkpoints_sweep_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (13, 17)).save(root / "image.png")
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(dict(id=str(index), dataset="pgm", task="matrix",
                regime="neutral", split="test", images=["image.png"], choices=["A", "B"],
                answer="A", prompt="Choose.")) + "\n" for index in range(2)))
            args = SimpleNamespace(threads=[1, 2], time_budget_seconds=300, warmup_batches=1,
                pgm_manifest=str(manifest), clevr_manifest=str(manifest), pgm_samples=2, clevr_samples=2,
                output=str(root / "audit.json"), max_batch=2, phase="all", dtype="bfloat16",
                seed=0, require_gpu_name="H200")
            backend = SimpleNamespace(metadata={"revision": "a" * 40})

            def measurement(_backend, batch, _torch, threads, size, kind, baseline, *rest):
                records = [{"id": row["row"]["id"], "raw_output": "A", "prediction": "A",
                            "prediction_metadata": {"input_tokens": 8}} for row in batch]
                return dict(kind=kind, cpu_threads=threads, batch_size=size, status="complete",
                    records=records, samples_per_second=threads * size, amortized_seconds_per_sample=1/(threads*size),
                    comparison_to_serial=audit.compare_outputs(records, baseline))

            with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "2"}), \
                 patch.dict("sys.modules", {"torch": fake_torch()}), \
                 patch.object(audit, "require_allocation"), \
                 patch("vlm_bench.models.load_backend", return_value=backend) as load, \
                 patch.object(audit, "measure_condition", side_effect=measurement):
                report = audit.run_audit(args)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(len(report["conditions"]), 10)
                self.assertEqual(report["recommendations"]["pgm"]["batch_size"], 2)
                self.assertEqual(report["recommendations"]["pgm"]["cpu_threads"], 2)
                load.assert_called_once()
                self.assertEqual(json.loads(Path(args.output).read_text()), report)
                with self.assertRaises(FileExistsError):
                    audit.run_audit(args)
                self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
