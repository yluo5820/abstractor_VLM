"""Train/test separation, disjoint rendering shards and publication gates."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
import clevr_full as full
import index_clevr_train as indexer


class TrainRenderTests(unittest.TestCase):
    def config(self, number, shards=4, total=10):
        config = {"split": "train", "shard_index": number, "shards": shards, "tasks": {}}
        for task, candidates in full.TASKS.items():
            indices = full.assigned_indices(total, number, shards)
            config["tasks"][task] = {"indices": indices, "count": len(indices),
                "candidates": candidates, "protocol_sha256": task + "-train-protocol"}
        return config

    def test_partition_full_training_set_exactly_once(self):
        pieces = [full.assigned_indices(10000, i, 4) for i in range(4)]
        self.assertEqual(sorted(sum(pieces, [])), list(range(10000)))
        self.assertTrue(all(len(part) == 2500 for part in pieces))
        for bad in [(-1, 4), (4, 4), (0, 0), (0, 10001)]:
            with self.assertRaises(ValueError):
                full.assigned_indices(10000, *bad)

    def test_native_index_refuses_test_shards_and_overlap(self):
        configs = [self.config(i) for i in range(4)]
        indexer.validate_shards(configs, total=10)
        configs[1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "non-training"):
            indexer.validate_shards(configs, total=10)
        configs[1]["split"] = "train"
        configs[1]["tasks"]["rmts"]["indices"] = configs[0]["tasks"]["rmts"]["indices"]
        with self.assertRaisesRegex(ValueError, "assignment"):
            indexer.validate_shards(configs, total=10)

    def test_launch_uses_training_labels_and_original_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            (assets / "render_images_rmts.py").write_text("fixture renderer")
            config = self.config(1)
            config.update(assets=str(assets), blender="/fake/blender")
            with mock.patch.object(full.helper, "patch_renderer", return_value="fixture") as patcher, \
                    mock.patch.object(full.subprocess, "Popen"):
                full.launch(root / "render", config, "rmts", 9, "fixture-session")
            args = patcher.call_args.args
            self.assertEqual(args[1:4], ("rmts", "train", [9]))
            self.assertEqual(full.scenes_path(root, "rmts", "train"), root / "output/rmts_images/train_ood")
            self.assertEqual(full.scenes_path(root, "rmts"), root / "output/rmts_images/test_ood")

    def test_noncontiguous_shard_receipts_resume_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(1)
            config["blender_version"] = "Blender 2.79 (sub 0)"
            directory = full.scenes_path(root, "rmts", "train") / "prob_5"
            directory.mkdir(parents=True)
            files = {}
            for c in range(2):
                path = directory / f"CLEVR_{c}.png"
                Image.new("RGB", (320, 240)).save(path)
                files[path.name] = full.sha256_file(path)
            full.helper._atomic_json(directory / full.MARKER, {
                "index": 5, "files": files, "protocol_sha256": "rmts-train-protocol",
                "blender_version": config["blender_version"]})
            pending = full.inspect(root, config)
            self.assertEqual(pending, {"rmts": [1, 9], "idrules": [1, 5, 9]})

    def test_training_index_roundtrip_preserves_original_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "assets"
            assets.mkdir()
            label_hashes = {}
            for task, choices in full.TASKS.items():
                name = ("RMTS" if task == "rmts" else "identity_rules") + "_ood_train.npz"
                np.savez(assets / name, y=np.arange(6) % choices)
                label_hashes[name] = full.sha256_file(assets / name)
            for shard in range(2):
                config = self.config(shard, shards=2, total=6)
                config.update(assets=str(assets), blender_version="Blender 2.79 (sub 0)")
                for task, choices in full.TASKS.items():
                    config["tasks"][task]["labels"] = ("RMTS" if task == "rmts" else "identity_rules") + "_ood_train.npz"
                    for i in config["tasks"][task]["indices"]:
                        directory = full.scenes_path(root / "shards" / str(shard), task, "train") / f"prob_{i}"
                        directory.mkdir(parents=True)
                        hashes = {}
                        for c in range(choices):
                            path = directory / f"CLEVR_{c}.png"
                            Image.new("RGB", (320, 240), (i, c, 0)).save(path)
                            hashes[path.name] = full.sha256_file(path)
                        full.helper._atomic_json(directory / full.MARKER, {
                            "index": i, "files": hashes, "protocol_sha256": task + "-train-protocol",
                            "blender_version": config["blender_version"]})
                (root / "shards" / str(shard) / "render-config.json").write_text(json.dumps(config))
            before = {str(p): full.sha256_file(p) for p in root.rglob("*.png")}
            with mock.patch.object(full, "TOTAL", 6), mock.patch.dict(full.OFFICIAL_CLEVR_SHA256, label_hashes):
                first = indexer.index(root, 2)
                second = indexer.index(root, 2)
            self.assertEqual(first, second)
            self.assertEqual(first["scenes"], 36)
            self.assertEqual(before, {str(p): full.sha256_file(p) for p in root.rglob("*.png")})
            rows = [json.loads(line) for line in (root / "native-manifests/clevr-art-rmts-train-native.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], [f"clevr-art/rmts/ood/train/{i}" for i in range(6)])
            self.assertEqual([row["answer"] for row in rows], list("ABABAB"))
            self.assertTrue(all((root / p).is_file() for row in rows for p in row["metadata"]["source_paths"]))


if __name__ == "__main__":
    unittest.main()
