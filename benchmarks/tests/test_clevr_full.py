"""Full-render coordination checks use real PNG receipts and bounded subprocesses."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest import mock

from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "clevr_full", Path(__file__).resolve().parents[1] / "cluster" / "clevr_full.py")
full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(full)


class FullRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.assets = self.base / "assets"
        self.assets.mkdir()
        self.root = self.base / "full"
        self.config = {"schema_version": 1, "assets": str(self.assets),
                       "blender_version": "Blender 2.79 (sub 0)", "tasks": {}}
        for task, choices in full.TASKS.items():
            label = f"{task}.npz"
            (self.assets / label).write_bytes(b"fixture labels " + task.encode())
            self.config["tasks"][task] = {"labels": label, "count": 3,
                "candidates": choices, "protocol_sha256": f"protocol-{task}",
                "source_identity": {"labels_sha256": full.sha256_file(self.assets / label)}}
        self.env = mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "2"})
        self.host = mock.patch.object(full.socket, "gethostname", return_value="compute-fixture")
        self.env.start()
        self.host.start()
        self.launched = []

    def tearDown(self):
        self.env.stop()
        self.host.stop()
        self.tmp.cleanup()

    def initialize(self):
        with full.root_lock(self.root):
            full.initialize(self.root, self.config)

    def attempt(self, task, index, complete=True):
        attempt = self.root / "attempts" / task / str(index) / uuid.uuid4().hex
        directory = attempt / "images" / f"prob_{index}"
        directory.mkdir(parents=True)
        count = full.TASKS[task] if complete else 1
        hashes = {}
        for i in range(count):
            path = directory / f"CLEVR_{i}.png"
            Image.new("RGB", (320, 240), (index, i, 22)).save(path)
            hashes[path.name] = full.sha256_file(path)
        record = {"task": task, "index": index, "status": "running",
                  "protocol_sha256": self.config["tasks"][task]["protocol_sha256"]}
        full.helper._atomic_json(attempt / "attempt.json", record)
        if complete:
            full.helper._atomic_json(directory / full.MARKER, dict(record, files=hashes,
                                      blender_version=self.config["blender_version"], schema_version=2))
        return attempt, record

    def success(self, root, config, task, index, session):
        attempt, record = self.attempt(task, index)
        self.launched.append((task, index))
        process = mock.Mock()
        process.poll.return_value = 0
        return {"process": process, "attempt": attempt, "record": record, "started": time.monotonic()}

    def test_calibration_both_tasks_then_resume_only_missing(self):
        with mock.patch.object(full, "launch", side_effect=self.success), mock.patch.object(full.time, "sleep"):
            first = full.render(self.root, self.config, 2, 10, cap=1)
            self.assertEqual(first["status"], "calibration-complete")
            self.assertEqual(self.launched, [("rmts", 0), ("idrules", 0)])
            before = full.sha256_file(full.scenes_path(self.root, "rmts") / "prob_0" / "CLEVR_0.png")
            second = full.render(self.root, self.config, 2, 10)
        self.assertEqual(second["status"], "complete")
        self.assertEqual(second["completed_groups"], {"rmts": 3, "idrules": 3})
        self.assertEqual(len(self.launched), 6)
        self.assertEqual(len(set(self.launched)), 6)
        self.assertEqual(full.sha256_file(full.scenes_path(self.root, "rmts") / "prob_0" / "CLEVR_0.png"), before)
        self.assertEqual(full.inspect(self.root, self.config), {"rmts": [], "idrules": []})

    def test_budget_kills_only_owned_process_and_partial_attempt_resumes(self):
        children, partials = [], []

        def incomplete(root, config, task, index, session):
            attempt, record = self.attempt(task, index, complete=False)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            children.append(child)
            partials.append(attempt / "images" / f"prob_{index}" / "CLEVR_0.png")
            return {"process": child, "attempt": attempt, "record": record, "started": time.monotonic()}

        with mock.patch.object(full, "launch", side_effect=incomplete):
            result = full.render(self.root, self.config, 2, 0.15, cap=1)
        self.assertEqual(result["status"], "budget-exhausted")
        self.assertTrue(all(child.poll() is not None for child in children))
        self.assertEqual(result["completed_groups"], {"rmts": 0, "idrules": 0})
        self.assertTrue(all(path.exists() for path in partials))
        before = {str(path): full.sha256_file(path) for path in partials}
        with mock.patch.object(full, "launch", side_effect=self.success), mock.patch.object(full.time, "sleep"):
            result = full.render(self.root, self.config, 2, 10)
        self.assertEqual(result["status"], "complete")
        self.assertEqual({str(path): full.sha256_file(path) for path in partials}, before)

    def test_finished_unpublished_attempt_is_recovered(self):
        self.initialize()
        attempt, _ = self.attempt("rmts", 0)
        with mock.patch.object(full, "launch", side_effect=self.success), mock.patch.object(full.time, "sleep"):
            result = full.render(self.root, self.config, 2, 10)
        self.assertNotIn(("rmts", 0), self.launched)
        self.assertEqual(result["reused_groups"]["rmts"], 1)
        self.assertEqual(json.loads((attempt / "attempt.json").read_text())["status"], "recovered")

    def test_corrupt_checkpoint_never_replaced(self):
        self.initialize()
        attempt, _ = self.attempt("rmts", 0)
        image = attempt / "images" / "prob_0" / "CLEVR_0.png"
        Image.new("RGB", (320, 240), "white").save(image)
        before = image.read_bytes()
        with self.assertRaisesRegex(ValueError, "Modified render"):
            full.render(self.root, self.config, 2, 10)
        self.assertEqual(image.read_bytes(), before)
        self.assertFalse((full.scenes_path(self.root, "rmts") / "prob_0").exists())

    def test_configuration_lock_and_allocation_guard(self):
        self.initialize()
        changed = copy.deepcopy(self.config)
        changed["tasks"]["rmts"]["protocol_sha256"] = "different-seed"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            full.render(self.root, changed, 2, 10)
        with full.root_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another renderer"):
                with full.root_lock(self.root):
                    pass
        with self.assertRaisesRegex(ValueError, "allocation has 2"):
            full.allocation_guard(8)
        with mock.patch.object(full.socket, "gethostname", return_value="login-ice"):
            with self.assertRaisesRegex(RuntimeError, "login node"):
                full.allocation_guard(2)

    def test_manifest_gate_requires_all_both_tasks(self):
        self.initialize()
        attempt, _ = self.attempt("rmts", 0)
        full.publish(self.root, self.config, "rmts", 0, attempt)
        with self.assertRaisesRegex(ValueError, "Both full splits"):
            full.build_manifests(self.root, self.config)
        self.assertFalse((self.root / "manifests").exists())

    def test_launch_keeps_original_seed_candidate_loop_and_one_cpu(self):
        self.initialize()
        source = '''from __future__ import print_function
import numpy as np
import random
def main(args):
  prob_answer_arr = np.load("identity_rules_ood_train.npz")
  for num_ex in range(10000):
    for i in range(4):
      render_scene(args, 0, [])

def render_scene(args,angle,all_feats,
                 num_objects=6):
  pass
'''
        (self.assets / "render_images_idrules.py").write_text(source)
        self.config["blender"] = "/verified/blender"
        with mock.patch.object(full.subprocess, "Popen") as popen:
            worker = full.launch(self.root, self.config, "idrules", 2, "session")
        script = (worker["attempt"] / "renderer.py").read_text()
        self.assertIn("random.seed(1 + num_ex)", script)
        self.assertIn("np.random.seed((1 + num_ex) % (2 ** 32))", script)
        self.assertIn("for num_ex in [2]:", script)
        self.assertIn("for i in range(4):", script)
        self.assertIn('np.load("identity_rules_ood_test.npz", allow_pickle=False)', script)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[argv.index("--threads") + 1], "1")
        self.assertEqual(argv[argv.index("--use_gpu") + 1], "0")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(popen.call_args.kwargs["cwd"], self.assets)

    def test_sigterm_preserves_partial_and_returns_successful_checkpoint(self):
        children = []
        def incomplete(root, config, task, index, session):
            attempt, record = self.attempt(task, index, complete=False)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            children.append(child)
            return {"process": child, "attempt": attempt, "record": record, "started": time.monotonic()}
        previous = signal.getsignal(signal.SIGTERM)
        with mock.patch.object(full, "launch", side_effect=incomplete), \
                mock.patch.object(full.time, "sleep", side_effect=lambda _: os.kill(os.getpid(), signal.SIGTERM)):
            result = full.render(self.root, self.config, 2, 60, cap=1)
        self.assertEqual(result["status"], "SIGTERM")
        self.assertTrue(all(child.poll() is not None for child in children))
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(len(list((self.root / "attempts").glob("*/*/*/images/prob_*/CLEVR_0.png"))), 2)

    def test_pilot_import_preserves_original_bytes(self):
        self.initialize()
        attempt, _ = self.attempt("rmts", 0)
        pilot = self.base / "pilot"
        original = full.scenes_path(pilot, "rmts") / "prob_0"
        original.parent.mkdir(parents=True)
        shutil.copytree(attempt / "images" / "prob_0", original)
        before = {path.name: path.read_bytes() for path in original.iterdir()}
        full.import_pilot(self.root, self.config, pilot, False)
        self.assertEqual({path.name: path.read_bytes() for path in original.iterdir()}, before)
        copied = full.scenes_path(self.root, "rmts") / "prob_0"
        self.assertEqual({path.name: path.read_bytes() for path in copied.iterdir()}, before)
        self.assertEqual(full.inspect(self.root, self.config)["rmts"], [1, 2])

    def test_manifest_pair_is_atomic_idempotent_and_revalidates_labels(self):
        with mock.patch.object(full, "launch", side_effect=self.success), mock.patch.object(full.time, "sleep"):
            full.render(self.root, self.config, 2, 10)

        def make_manifest(root, output, task, regime, split, limit, seed):
            image = output.parent / (task + ".png")
            Image.new("RGB", (320, 240), "white").save(image)
            row = {"images": [image.name], "metadata": {"contact_sheet_sha256": full.sha256_file(image)}}
            output.write_text((json.dumps(row) + "\n") * 3)
            provenance = {"manifest_sha256": full.sha256_file(output), "selected_count": 3,
                          "partial_render_pool": False, "official_label_file": True}
            output.with_name(output.name + ".provenance.json").write_text(json.dumps(provenance))
            return output

        def fail_second(*args, **kwargs):
            if args[2] == "idrules":
                raise RuntimeError("simulated interruption between splits")
            return make_manifest(*args, **kwargs)

        with mock.patch("vlm_bench.data.prepare_clevr_art", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                full.build_manifests(self.root, self.config)
        self.assertFalse((self.root / "manifests").exists())
        retained = list((self.root / "manifest-attempts").iterdir())
        self.assertEqual(len(retained), 1)
        with mock.patch("vlm_bench.data.prepare_clevr_art", side_effect=make_manifest) as build:
            full.build_manifests(self.root, self.config)
            full.build_manifests(self.root, self.config)
        self.assertEqual(build.call_count, 2)
        self.assertTrue(retained[0].exists())
        self.assertTrue((self.root / "manifests" / "READY.json").exists())
        (self.root / self.config["tasks"]["rmts"]["labels"]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Modified copied labels"):
            full.build_manifests(self.root, self.config)

    def test_failure_retains_partial_and_no_false_completeness(self):
        def fail(root, config, task, index, session):
            attempt, record = self.attempt(task, index, complete=False)
            process = mock.Mock()
            process.poll.return_value = 1
            return {"process": process, "attempt": attempt, "record": record, "started": time.monotonic()}
        with mock.patch.object(full, "launch", side_effect=fail), mock.patch.object(full.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "Rendering failed"):
                full.render(self.root, self.config, 2, 10)
        progress = json.loads((self.root / "progress.json").read_text())
        self.assertEqual(progress["status"], "failed")
        self.assertEqual(progress["completed_groups"], {"rmts": 0, "idrules": 0})
        self.assertEqual(len(list((self.root / "attempts").glob("*/*/*/images/prob_*/CLEVR_0.png"))), 2)


if __name__ == "__main__":
    unittest.main()
