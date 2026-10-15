"""Integrity tests for native inputs, trained-head gating and durable results."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from slot_abstractor.data import clevr_examples, preprocess
from slot_abstractor.evaluate import Results, batch_seed, trained_state
from slot_abstractor.dataset_lock import validate_sources
from slot_abstractor.model import digest, SPECS, unwrapped_state


class SlotAbstractorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self):
        paths = []
        for i, value in enumerate([0, 255]):
            path = self.root / f"candidate-{i}.png"
            Image.fromarray(np.full((240, 320, 3), value, np.uint8)).save(path)
            paths.append(path)
        row = {"id": "clevr-art/rmts/ood/test/15", "answer": "B", "metadata": {
            "source_paths": [p.name for p in paths], "source_sha256": [digest(p) for p in paths]}}
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        return manifest, row, paths

    def test_native_candidate_order_normalization_and_label_independence(self):
        manifest, row, _ = self.fixture()
        _, pixels, target = next(clevr_examples(self.root, manifest, "rmts"))
        self.assertEqual(pixels.shape, (2, 3, 128, 128))
        self.assertEqual(target, 1)
        self.assertTrue((pixels[0] == -1).all())
        self.assertTrue((pixels[1] == 1).all())
        row["answer"] = "A"
        manifest.write_text(json.dumps(row) + "\n")
        _, changed, label = next(clevr_examples(self.root, manifest, "rmts"))
        self.assertTrue(torch.equal(pixels, changed))
        self.assertEqual(label, 0)

    def test_corrupted_source_rejected_and_resume_skips_completed(self):
        manifest, _, paths = self.fixture()
        paths[0].write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            next(clevr_examples(self.root, manifest, "rmts"))
        self.assertEqual(list(clevr_examples(self.root, manifest, "rmts", start=1)), [])

    def test_bilinear_pgm_resize(self):
        image = np.arange(160, dtype=np.uint8)[None, :].repeat(160, axis=0)
        tensor = preprocess(image, "pgm")
        self.assertEqual(tensor.shape, (1, 80, 80))
        expected = (torch.arange(80).float() * 2 + 0.5) / 255 * 2 - 1
        self.assertTrue(torch.allclose(tensor[0, 0], expected, atol=1e-7))

    def test_checkpoint_prefix_does_not_truncate_plain_names(self):
        value = torch.tensor(1)
        self.assertEqual(list(unwrapped_state({"module.layer.weight": value})), ["layer.weight"])
        self.assertEqual(list(unwrapped_state({"layer.weight": value})), ["layer.weight"])
        with self.assertRaisesRegex(ValueError, "Mixed"):
            unwrapped_state({"module.a": value, "b": value})

    def test_encoder_only_checkpoint_rejected(self):
        checkpoint = self.root / "encoder.pt"
        torch.save({"slot_model_state_dict": {}}, checkpoint)
        meta = self.root / "meta.json"
        meta.write_text(json.dumps({"task": "rmts", "regime": "ood", "training_split": "train",
            "checkpoint_sha256": digest(checkpoint), "encoder_sha256": SPECS["clevr"]["encoder_sha256"],
            "training_description": "test fixture", "checkpoint_selection": "final-epoch"}))
        with self.assertRaisesRegex(ValueError, "not a reasoning checkpoint"):
            trained_state(checkpoint, meta, "rmts")

    def test_resume_is_atomic_and_sample_weighted(self):
        path = self.root / "results.db"
        results = Results(path, {"seed": 1})
        results.append([("one", None, 1), ("two", None, 0)], torch.tensor([[0., 1.], [0., 1.]]))
        results.db.close()
        resumed = Results(path, {"seed": 1})
        self.assertEqual(resumed.count, 2)
        # Duplicate IDs roll back the ENTIRE batch, including preceding new rows.
        with self.assertRaises(Exception):
            resumed.append([("three", None, 0), ("one", None, 0)], torch.tensor([[1., 0.], [1., 0.]]))
        self.assertEqual(resumed.count, 2)
        resumed.append([("three", None, 0)], torch.tensor([[1., 0.]]))
        self.assertAlmostEqual(resumed.summary()["accuracy_percent"], 200 / 3)
        exported = self.root / "predictions.jsonl"
        resumed.export(exported)
        rows = [json.loads(line) for line in exported.read_text().splitlines()]
        self.assertEqual([row["id"] for row in rows], ["one", "two", "three"])
        self.assertEqual(sum(row["correct"] for row in rows), 2)
        resumed.db.close()
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            Results(path, {"seed": 2})

    def test_unpinned_dataset_rejected(self):
        path = self.root / "test-00000-of-00009.parquet"
        path.write_bytes(b"Not the original neutral test shard")
        with self.assertRaisesRegex(ValueError, "pinned test campaign"):
            validate_sources("pgm", [path])

    def test_slot_draw_seed_is_restart_stable(self):
        torch.manual_seed(batch_seed(7, 32))
        first = torch.randn(2, 7, 64)
        torch.randn(100)
        torch.manual_seed(batch_seed(7, 32))
        self.assertTrue(torch.equal(first, torch.randn(2, 7, 64)))
        self.assertNotEqual(batch_seed(7, 32), batch_seed(7, 48))


if __name__ == "__main__":
    unittest.main()
