import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from vlm_bench import clevr_prepare as prepare, data


class ClevrPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "render"
        self.root.mkdir()
        self.output = self.base / "prepared"
        self.count = 3
        patcher = patch.object(prepare, "COUNT", self.count)
        patcher.start(); self.addCleanup(patcher.stop)
        config, label_hashes = {"tasks": {}}, {}
        a, b, c, d = ([color, 1, 2, 1] for color in range(4, 8))
        for task, candidates in prepare.TASKS.items():
            example = [a, a, b, c, d, d] if task == "rmts" else [a, b, a, c, d, d, b, c, a]
            target = 1 if task == "rmts" else 2
            label = ("RMTS" if task == "rmts" else "identity_rules") + "_ood_test.npz"
            np.savez(self.root / label, x=np.repeat(np.array(example)[None], self.count, axis=0),
                     y=np.full(self.count, target, dtype=np.int64))
            digest = data.sha256_file(self.root / label)
            label_hashes[label] = digest
            protocol = hashlib.sha256(task.encode()).hexdigest()
            config["tasks"][task] = {"labels": label, "count": self.count, "candidates": candidates,
                                     "protocol_sha256": protocol, "source_identity": {"labels_sha256": digest}}
            for index in range(self.count):
                directory = self.root / "output" / f"{task}_images/test_ood/prob_{index}"
                directory.mkdir(parents=True)
                hashes = {}
                for choice in range(candidates):
                    path = directory / f"CLEVR_{choice}.png"
                    Image.new("RGB", (320, 240), (20 + choice * 30, 40 + index, 60)).save(path)
                    hashes[path.name] = data.sha256_file(path)
                (directory / ".benchmark-complete.json").write_text(json.dumps({
                    "index": index, "protocol_sha256": protocol, "files": hashes,
                    "blender_version": "Blender 2.79 (sub 0)"}))
        (self.root / "render-config.json").write_text(json.dumps(config))
        self.digest = data.sha256_file(self.root / "render-config.json")
        (self.root / "progress.json").write_text(json.dumps({"status": "complete",
            "completed_groups": dict.fromkeys(prepare.TASKS, self.count)}))
        patcher = patch.dict(data.OFFICIAL_CLEVR_SHA256, label_hashes)
        patcher.start(); self.addCleanup(patcher.stop)

    def run_prepare(self, **kwargs):
        return prepare.prepare(self.root, self.output, self.digest, **kwargs)

    def test_matches_original_adapter_bytes_labels_and_provenance(self):
        self.assertEqual(self.run_prepare()["status"], "complete")
        for task in prepare.TASKS:
            name = f"clevr-art-{task}-test-full.jsonl"
            original = self.base / "original" / name
            data.prepare_clevr_art(self.root, original, task, "ood", "test")
            current = self.output / name
            self.assertEqual(current.read_bytes(), original.read_bytes())
            self.assertEqual(current.with_name(name + ".provenance.json").read_bytes(),
                             original.with_name(name + ".provenance.json").read_bytes())
            for row in map(json.loads, current.read_text().splitlines()):
                self.assertEqual((self.output / row["images"][0]).read_bytes(),
                                 (original.parent / row["images"][0]).read_bytes())
        self.assertEqual(self.run_prepare()["status"], "complete")

    def test_resumes_completed_rows_and_recovers_partial_journal_tail(self):
        self.assertEqual(self.run_prepare(max_new=2)["status"], "resumable")
        self.assertFalse((self.output / "READY.json").exists())
        with (self.output / ".rmts-rows.jsonl").open("ab") as stream:
            stream.write(b'{"id":')
        with patch.object(prepare, "make_row", wraps=prepare.make_row) as make:
            self.assertEqual(self.run_prepare()["status"], "complete")
            self.assertEqual(make.call_count, 4)
        self.assertEqual(len(list(self.output.glob("*.partial-*"))), 1)

    def test_changed_completed_image_is_rejected(self):
        self.run_prepare(max_new=1)
        row = json.loads((self.output / ".rmts-rows.jsonl").read_text())
        (self.output / row["images"][0]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "checkpoint was modified"):
            self.run_prepare()
        self.assertFalse((self.output / "READY.json").exists())

    def test_changed_render_is_rejected(self):
        path = self.root / "output/rmts_images/test_ood/prob_0/CLEVR_0.png"
        Image.new("RGB", (320, 240), "white").save(path)
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.run_prepare()
        self.assertFalse((self.output / "READY.json").exists())

    def test_changed_render_plan_or_missing_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "configuration checksum"):
            prepare.prepare(self.root, self.output, "0" * 64)
        path = self.root / "output/rmts_images/test_ood/prob_1"
        path.rename(path.with_name("foreign"))
        with self.assertRaisesRegex(ValueError, "missing or foreign"):
            self.run_prepare()

    def test_ready_receipt_cannot_hide_modified_manifest(self):
        self.run_prepare()
        manifest = self.output / "clevr-art-rmts-test-full.jsonl"
        manifest.write_text(manifest.read_text() + "{}\n")
        with self.assertRaisesRegex(ValueError, "Published manifest changed"):
            prepare.validate_ready(self.output, self.digest)


if __name__ == "__main__":
    unittest.main()
