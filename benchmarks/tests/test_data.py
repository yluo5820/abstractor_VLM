"""Dataset semantics and leakage checks using tiny, valid synthetic fixtures."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from vlm_bench.data import prepare_clevr_art, prepare_pgm, select_subset


_helper_spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parents[1] / "cluster" / "prepare_data.py"
)
render_helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(render_helper)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def pgm(self, index=0, target=3, split="test", regime="extrapolation", malformed_shape=False):
        panels = np.stack([np.full((160, 160), i * 10, dtype=np.uint8) for i in range(16)])
        # Upstream files can advertise H,W,N while bytes remain panel-major.
        raw = panels.reshape(160, 160, 16)
        if malformed_shape:
            raw = raw.reshape(-1)
        path = self.root / f"PGM_{regime}_{split}_{index}.npz"
        np.savez(path, image=raw, target=np.array(target, dtype=np.int64),
                 meta_target=np.ones(12), meta_matrix=np.ones((4, 12)))
        return path

    def rows(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_pgm_panel_order_target_and_split(self):
        self.pgm(target=7)
        self.pgm(index=1, split="train")
        self.pgm(index=2, regime="neutral")
        output = self.root / "out.jsonl"
        prepare_pgm(self.root, output, "extrapolation", "test")
        row, = self.rows(output)
        self.assertEqual(row["answer"], "H")
        self.assertEqual(row["choices"], list("ABCDEFGH"))
        self.assertEqual(row["id"], "pgm/extrapolation/test/0")
        self.assertNotIn("meta_target", row["metadata"])
        self.assertNotIn("meta_matrix", row["metadata"])
        image = Image.open(output.parent / row["images"][0])
        # Matrix begins x=98,y=42, gap=12; bottom-right remains blank.
        for i in range(8):
            self.assertEqual(image.getpixel((98 + i % 3 * 172 + 40, 42 + i // 3 * 172 + 40)), (i * 10,) * 3)
        for i in range(8):
            self.assertEqual(image.getpixel((12 + i % 4 * 172 + 40, 600 + i // 4 * 202 + 30 + 40)), ((i + 8) * 10,) * 3)
        self.assertEqual(image.getpixel((98 + 2 * 172 + 40, 42 + 2 * 172 + 40)), (238,) * 3)

    def test_pgm_target_never_changes_model_input(self):
        path = self.pgm(target=0)
        first = self.root / "first.jsonl"
        prepare_pgm(self.root, first, "extrapolation", "test")
        path.unlink()
        self.pgm(target=7)
        second = self.root / "second.jsonl"
        prepare_pgm(self.root, second, "extrapolation", "test")
        one, two = self.rows(first)[0], self.rows(second)[0]
        self.assertNotEqual(one["answer"], two["answer"])
        self.assertEqual(one["prompt"], two["prompt"])
        self.assertEqual(one["metadata"]["contact_sheet_sha256"], two["metadata"]["contact_sheet_sha256"])
        self.assertNotEqual(one["metadata"]["source_sha256"], two["metadata"]["source_sha256"])

    def test_stable_nested_subset_and_duplicate_rejection(self):
        items = [(str(i), self.root / str(i)) for i in range(20)]
        small, _ = select_subset(items, 3, 42)
        large, _ = select_subset(reversed(items), 8, 42)
        self.assertEqual(small, large[:3])
        self.assertNotEqual(small, select_subset(items, 3, 7)[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_subset(items + [items[0]], None, 42)

    def test_bad_pgm_rejected_without_published_output(self):
        self.pgm(target=8)
        output = self.root / "out.jsonl"
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_pgm(self.root, output, "extrapolation", "test")
        self.assertFalse(output.exists())
        self.assertFalse((self.root / "out_images").exists())

    def test_no_overwrite(self):
        self.pgm()
        output = self.root / "out.jsonl"
        prepare_pgm(self.root, output, "extrapolation", "test")
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_pgm(self.root, output, "extrapolation", "test")
        self.assertEqual(before, output.read_bytes())

    def clevr(self, task="rmts", index=0, bad_target=False):
        a, b, c, d = ([color, 1, 2, 1] for color in range(4, 8))
        if task == "rmts":
            example, target, count = [a, a, b, c, d, d], 1, 2
        else:
            example, target, count = [a, b, a, c, d, d, b, c, a], 2, 4
        # Index into y must come from prob_N, never directory enumeration order.
        x = np.repeat(np.array(example)[None], index + 1, axis=0)
        y = np.full(index + 1, target if not bad_target else 0, dtype=np.int64)
        name = "RMTS" if task == "rmts" else "identity_rules"
        np.savez(self.root / f"{name}_ood_test.npz", x=x, y=y)
        directory = self.root / "output" / f"{task}_images" / "test_ood" / f"prob_{index}"
        directory.mkdir(parents=True)
        for choice in range(count):
            Image.new("RGB", (320, 240), (20 + choice * 30, 40, 60)).save(directory / f"CLEVR_{choice}.png")
        return directory

    def test_clevr_rmts_index_mapping_color_and_candidate_order(self):
        self.clevr(index=7)
        output = self.root / "rmts.jsonl"
        prepare_clevr_art(self.root, output, "rmts", "ood", "test")
        row, = self.rows(output)
        self.assertEqual(row["answer"], "B")
        self.assertTrue(row["id"].endswith("/7"))
        self.assertNotIn("x", row["metadata"])
        self.assertFalse(row["metadata"]["official_label_file"])
        image = Image.open(output.parent / row["images"][0])
        self.assertEqual(image.getpixel((16 + 40, 70 + 40)), (20, 40, 60))
        self.assertEqual(image.getpixel((352 + 40, 70 + 40)), (50, 40, 60))
        provenance = json.loads(Path(str(output) + ".provenance.json").read_text())
        self.assertTrue(provenance["partial_render_pool"])
        self.assertEqual(provenance["symbolic_examples"], 8)
        self.assertEqual(provenance["manifest_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_clevr_identity_uses_all_four_choices(self):
        self.clevr(task="idrules")
        output = self.root / "idrules.jsonl"
        prepare_clevr_art(self.root, output, "idrules", "ood", "test")
        row, = self.rows(output)
        self.assertEqual(row["answer"], "C")
        self.assertEqual(row["choices"], list("ABCD"))
        image = Image.open(output.parent / row["images"][0])
        for i in range(4):
            self.assertEqual(image.getpixel((16 + (i % 2) * 336 + 40, 70 + (i // 2) * 288 + 40)), (20 + 30 * i, 40, 60))

    def test_clevr_mislabel_rejected(self):
        self.clevr(bad_target=True)
        with self.assertRaisesRegex(ValueError, "rule and target disagree"):
            prepare_clevr_art(self.root, self.root / "out.jsonl", "rmts", "ood", "test")

    def test_clevr_incomplete_render_not_silently_skipped(self):
        directory = self.clevr()
        (directory / "CLEVR_1.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete render"):
            prepare_clevr_art(self.root, self.root / "out.jsonl", "rmts", "ood", "test", limit=1)

    def test_clevr_copied_train_labels_cannot_claim_test(self):
        self.clevr()
        path = self.root / "RMTS_ood_test.npz"
        with np.load(path) as arrays:
            x, y = arrays["x"].copy(), arrays["y"].copy()
        x[:, :, 0] = 0
        np.savez(path, x=x, y=y)
        with self.assertRaisesRegex(ValueError, "out-of-domain"):
            prepare_clevr_art(self.root, self.root / "out.jsonl", "rmts", "ood", "test")


class RenderResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.scenes = self.root / "output" / "rmts_images" / "test_ood"
        self.scenes.mkdir(parents=True)
        self.protocol = "a" * 64
        self.blender = {"bpy": SimpleNamespace(app=SimpleNamespace(version_string="2.79 (sub 0)"))}
        exec(render_helper._checkpoint_source(self.protocol, 2), self.blender)

    def tearDown(self):
        self.tmp.cleanup()

    def group(self, index=7, checkpoint=True):
        directory = self.scenes / f"prob_{index}"
        directory.mkdir()
        for choice in range(2):
            Image.new("RGB", (320, 240), (40 + choice * 30, 20, 60)).save(directory / f"CLEVR_{choice}.png")
        if checkpoint:
            self.blender["_benchmark_finish_group"](str(self.scenes), index)
        return directory

    def test_resume_extension_skips_verified_complete_groups(self):
        self.group(7)
        self.group(42)
        pending, adoptions, versions = render_helper._inspect_groups(self.scenes, [7, 42, 9, 12], 2, self.protocol)
        self.assertEqual(pending, [9, 12])
        self.assertEqual(adoptions, {})
        self.assertEqual(versions, ["Blender 2.79 (sub 0)"])
        self.assertFalse(self.blender["_benchmark_start_group"](str(self.scenes), 7))
        self.assertTrue(self.blender["_benchmark_start_group"](str(self.scenes), 9))

    def test_changed_pixels_rejected_even_if_valid_png(self):
        directory = self.group()
        Image.new("RGB", (320, 240), "red").save(directory / "CLEVR_0.png")
        with self.assertRaisesRegex(ValueError, "Modified render"):
            render_helper._inspect_groups(self.scenes, [7, 9], 2, self.protocol)
        with self.assertRaisesRegex(RuntimeError, "Modified render"):
            self.blender["_benchmark_start_group"](str(self.scenes), 7)

    def test_unfinished_group_errors_without_deleting_anything(self):
        directory = self.group(checkpoint=False)
        (directory / "CLEVR_1.png").unlink()
        before = (directory / "CLEVR_0.png").read_bytes()
        with self.assertRaisesRegex(ValueError, "Incomplete or untracked"):
            render_helper._inspect_groups(self.scenes, [7], 2, self.protocol)
        self.assertEqual(before, (directory / "CLEVR_0.png").read_bytes())

    def test_seed_or_source_protocol_change_rejected(self):
        self.group()
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            render_helper._inspect_groups(self.scenes, [7], 2, "b" * 64)

    def test_unrequested_and_foreign_files_rejected(self):
        directory = self.group()
        with self.assertRaisesRegex(ValueError, "outside requested"):
            render_helper._inspect_groups(self.scenes, [9], 2, self.protocol)
        (directory / "unexpected.txt").write_text("keep me")
        with self.assertRaisesRegex(ValueError, "foreign files"):
            render_helper._inspect_groups(self.scenes, [7], 2, self.protocol)
        self.assertTrue((directory / "unexpected.txt").exists())

    def test_legacy_adoption_requires_explicit_attestation_and_valid_pngs(self):
        directory = self.group(checkpoint=False)
        with self.assertRaisesRegex(ValueError, "Incomplete or untracked"):
            render_helper._inspect_groups(self.scenes, [7], 2, self.protocol)
        legacy = {7: {"index": 7, "protocol_sha256": self.protocol,
                      "blender_version": "Blender 2.79 (sub 0)",
                      "legacy_adoption": {"historical_png_hashes_available": False}}}
        pending, adoptions, _ = render_helper._inspect_groups(self.scenes, [7, 9], 2, self.protocol, legacy)
        self.assertEqual(pending, [9])
        self.assertEqual(set(adoptions[7]["files"]), {"CLEVR_0.png", "CLEVR_1.png"})
        self.assertFalse((directory / ".benchmark-complete.json").exists())
        (directory / "CLEVR_0.png").write_bytes(b"not a PNG")
        with self.assertRaises(OSError):
            render_helper._inspect_groups(self.scenes, [7], 2, self.protocol, legacy)

    def test_generated_renderer_keeps_group_checkpoint_inside_outer_loop(self):
        source = '''from __future__ import print_function
import numpy as np
def main(args):
  prob_answer_arr = np.load("RMTS_ood_train.npz")
  for num_ex in range(10000):
    for i in range(2):
      render_scene(args, 0, [])

def render_scene(args,angle,all_feats,
                 num_objects=5):
  pass
'''
        patched = render_helper.patch_renderer(source, "rmts", "test", [9, 12], 42, self.root, self.protocol)
        import ast
        tree = ast.parse(patched, feature_version=(3, 5))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        loop = next(node for node in main.body if isinstance(node, ast.For))
        self.assertEqual(loop.body[0].test.operand.func.id, "_benchmark_start_group")
        self.assertEqual(loop.body[-1].value.func.id, "_benchmark_finish_group")
        self.assertIn('np.load("RMTS_ood_test.npz", allow_pickle=False)', patched)

    def test_active_run_lock_rejects_concurrent_writer(self):
        with render_helper._render_lock(self.root, "rmts", "test", True):
            with self.assertRaisesRegex(RuntimeError, "Another helper"):
                with render_helper._render_lock(self.root, "rmts", "test", True):
                    self.fail("Second writer acquired the lock")

    def test_legacy_complete_run_extends_without_overwriting_original_images(self):
        from vlm_bench import data as data_module
        source = '''from __future__ import print_function
import numpy as np
def main(args):
  prob_answer_arr = np.load("RMTS_ood_train.npz")
  for num_ex in range(10000):
    for i in range(2):
      render_scene(args, 0, [])

def render_scene(args,angle,all_feats,
                 num_objects=5):
  pass
'''
        source_path = self.root / "render_images_rmts.py"
        source_path.write_text(source)
        label_path = self.root / "RMTS_ood_test.npz"
        np.savez(label_path, x=np.zeros((3, 6, 4), dtype=np.int64), y=np.zeros(3, dtype=np.int64))
        label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
        selected, _ = select_subset([(f"clevr-art/rmts/ood/test/{i}", Path(str(i))) for i in range(3)], 2, 0)
        old_index, new_index = [int(identifier.rsplit("/", 1)[1]) for identifier, _ in selected]
        directory = self.group(old_index, checkpoint=False)
        original_png = (directory / "CLEVR_0.png").read_bytes()
        plan_dir = self.root / "benchmark-render-plans"
        plan_dir.mkdir()
        legacy_script = plan_dir / "legacy.py"
        legacy_script.write_text(render_helper.patch_renderer(source, "rmts", "test", [old_index], 0, self.root))
        legacy = {
            "task": "rmts", "split": "test", "selection_seed": 0, "render_seed": 0,
            "status": "complete", "exit_code": 0, "missing_files": [], "indices": [old_index],
            "ocra_commit": render_helper.OCRA_COMMIT, "clevr_commit": render_helper.CLEVR_COMMIT,
            "source_renderer_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "labels_sha256": label_hash, "render_mode": "CPU", "resolution": [320, 240],
            "patched_renderer_sha256": hashlib.sha256(legacy_script.read_bytes()).hexdigest(),
            "blender_version": "Blender 2.79 (sub 0)",
        }
        (plan_dir / "legacy.json").write_text(json.dumps(legacy))
        pinned = {path.name: render_helper.git_blob(path.read_bytes()) for path in (source_path, label_path)}

        def fake_blender(argv, cwd):
            record = json.loads(Path(argv[argv.index("--python") + 1]).with_suffix(".json").read_text())
            self.assertEqual(record["pending_indices"], [new_index])
            self.assertEqual(record["legacy_adopted_indices"], [old_index])
            namespace = {"bpy": SimpleNamespace(app=SimpleNamespace(version_string="2.79 (sub 0)"))}
            exec(render_helper._checkpoint_source(record["protocol_sha256"], 2), namespace)
            self.group(new_index, checkpoint=False)
            namespace["_benchmark_finish_group"](str(self.scenes), new_index)
            return render_helper.subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(render_helper, "OCRA_FILES", pinned), \
             mock.patch.object(render_helper, "CLEVR_FILES", {}), \
             mock.patch.dict(data_module.OFFICIAL_CLEVR_SHA256, {label_path.name: label_hash}), \
             mock.patch.dict(render_helper.os.environ, {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "2"}), \
             mock.patch.object(render_helper.socket, "gethostname", return_value="compute-1"), \
             mock.patch.object(render_helper.subprocess, "check_output", return_value="Blender 2.79 (sub 0)\n"), \
             mock.patch.object(render_helper.subprocess, "run", side_effect=fake_blender):
            render_helper.plan_render(self.root, "rmts", "test", 2, 0, 0, "blender", 1, True,
                                      resume=True, adopt_legacy=True)
        self.assertEqual((directory / "CLEVR_0.png").read_bytes(), original_png)
        self.assertEqual(json.loads((plan_dir / "legacy.json").read_text()), legacy)
        attempts = [p for p in plan_dir.glob("*.json") if p.name != "legacy.json"]
        record = json.loads(attempts[0].read_text())
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["reused_count"], 1)
        self.assertEqual(render_helper._inspect_groups(self.scenes, [old_index, new_index], 2, record["protocol_sha256"])[0], [])


if __name__ == "__main__":
    unittest.main()
