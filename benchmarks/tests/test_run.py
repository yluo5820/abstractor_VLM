import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy  # Import before sys.modules mocks so separate tests do not reload NumPy.
from PIL import Image
from vlm_bench.run import atomic_json, load_manifest, parse_answer, read_records, require_allocation, run, runtime_progress, summarize, validate_records


class RunnerTests(unittest.TestCase):
    def test_strict_answer_parser(self):
        for response in ("A", "a", "(A)", "Answer: A."):
            self.assertEqual(parse_answer(response, ["A", "B"]), "A")
        for response in ("I think A or B", "Answer: C", "Because A is right", "", "A B"):
            self.assertIsNone(parse_answer(response, ["A", "B"]))

    def test_invalid_answers_count_as_wrong_and_expose_incomplete(self):
        base = dict(dataset="pgm", task="matrix", regime="neutral", split="test", choices=list("ABCDEFGH"))
        records = [dict(base, correct=True, prediction="A", latency_seconds=1),
                   dict(base, correct=False, prediction=None, latency_seconds=3)]
        summary = summarize(records, [base, base, base])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["groups"][0]["accuracy"], .5)
        self.assertEqual(summary["groups"][0]["invalid_rate"], .5)
        self.assertEqual(summary["groups"][0]["chance_accuracy"], .125)

    def test_runtime_projection_uses_only_new_durable_records_and_monotonic_time(self):
        # This session resumed at 500 examples, spent 20s starting, then
        # durably processed another 1,000 examples over 100s of evaluation.
        progress = runtime_progress(10, 30, 1500, 500, 200000, now=130)
        self.assertEqual(progress["completed_this_session"], 1000)
        self.assertEqual(progress["startup_seconds"], 20)
        self.assertEqual(progress["session_elapsed_seconds"], 120)
        self.assertEqual(progress["evaluation_elapsed_seconds"], 100)
        self.assertEqual(progress["evaluation_examples_per_second"], 10)
        self.assertAlmostEqual(progress["end_to_end_examples_per_second"], 1000 / 120)
        self.assertEqual(progress["estimated_remaining_seconds"], 19850)
        empty = runtime_progress(10, 30, 500, 500, 200000, now=30)
        self.assertIsNone(empty["evaluation_examples_per_second"])
        self.assertIsNone(empty["estimated_remaining_seconds"])

    def test_login_guard(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                require_allocation()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            with patch("socket.gethostname", return_value="login-ice"):
                with self.assertRaises(RuntimeError):
                    require_allocation()
            with patch("socket.gethostname", return_value="compute-123"):
                require_allocation()

    def test_resume_recovers_only_unterminated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_bytes(b'{"id":"one"}\n{"id":')
            self.assertEqual(read_records(path), [{"id": "one"}])
            self.assertEqual(path.read_bytes(), b'{"id":"one"}\n')
            self.assertEqual(len(list(Path(directory).glob("*.partial-*"))), 1)
            path.write_text("not-json\n")
            with self.assertRaises(json.JSONDecodeError):
                read_records(path)

    def test_manifest_rejects_duplicate_ids_and_wrong_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16)).save(root / "image.png")
            row = dict(id="one", dataset="pgm", task="matrix", regime="neutral", split="test",
                       images=["image.png"], choices=["A", "B"], answer="A", prompt="Choose.")
            path = root / "test.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(load_manifest(path)[0]["answer"], "A")
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                load_manifest(path)
            row["answer"] = "C"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_resume_rejects_modified_labels_scores_and_outcomes(self):
        row = dict(id="one", dataset="pgm", task="matrix", regime="neutral", split="test",
                   choices=["A", "B"], answer="A")
        record = dict(row, raw_output="Answer: A.", prediction="A", correct=True, latency_seconds=1)
        validate_records([record], [row])
        for change in ({"answer": "B"}, {"prediction": "B"}, {"correct": False},
                       {"correct": 1}, {"raw_output": "B"}, {"latency_seconds": float("nan")},
                       {"latency_seconds": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_records([dict(record, **change)], [row])
        with self.assertRaises(ValueError):
            validate_records([record, record], [row])

    def test_interrupted_run_resumes_and_rejects_changed_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16)).save(root / "image.png")
            rows = [dict(id=str(i), dataset="pgm", task="matrix", regime="neutral", split="test",
                         images=["image.png"], choices=["A", "B"], answer="A", prompt="Choose.")
                    for i in range(2)]
            path = root / "test.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = SimpleNamespace(manifest=str(path), output=str(root / "run"), model="lfm25",
                                   revision="A" * 40, max_samples=0, max_new_tokens=16,
                                   seed=0, dtype="auto", dry_run=False)
            backend = SimpleNamespace(metadata={"revision": "a" * 40},
                                      predict=MagicMock(side_effect=["A", RuntimeError("interrupted")]))
            fake_torch = MagicMock()
            fake_torch.get_num_interop_threads.return_value = 8
            fake_torch.cuda.get_device_name.return_value = "TEST DEVICE"
            fake_torch.cuda.max_memory_allocated.return_value = 0
            model_module = SimpleNamespace(load_backend=MagicMock(return_value=backend))
            with patch.dict("sys.modules", {"torch": fake_torch, "vlm_bench.models": model_module}), \
                    patch.dict(os.environ, {"SLURM_JOB_ID": "test"}), \
                    patch("vlm_bench.run.require_allocation"), patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run(args)
                model_module.load_backend.assert_called_with("lfm25", "a" * 40,
                                                             dtype="auto", max_new_tokens=16)
                for call in backend.predict.call_args_list:
                    self.assertEqual(len(call.args), 2)
                    self.assertIsInstance(call.args[0], Image.Image)
                    self.assertEqual(call.args[1], "Choose.")
                incomplete = json.loads((root / "run/summary.json").read_text())
                self.assertEqual(incomplete["evaluated"], 1)
                self.assertFalse(incomplete["complete"])
                self.assertEqual(incomplete["runtime"]["completed_this_session"], 1)
                self.assertEqual(incomplete["runtime"]["remaining_examples"], 1)
                fake_torch.set_num_threads.assert_called_with(2)
                fake_torch.set_num_interop_threads.assert_called_with(1)
                config = json.loads((root / "run/config.json").read_text())
                self.assertEqual((config["cpu_threads"], config["interop_threads"]), (2, 1))
                self.assertFalse(config["tokenizers_parallelism"])
                first = read_records(root / "run/predictions.jsonl")[0]
                self.assertEqual(first["source_image"], {"width": 16, "height": 16, "mode": "RGB"})
                self.assertEqual(first["prediction_metadata"], {})
                backend_path = root / "run/backend.json"
                provenance = backend_path.read_text()
                backend_path.unlink()
                with self.assertRaisesRegex(ValueError, "backend provenance"):
                    run(args)
                backend_path.write_text(provenance)
                backend.metadata = {"revision": "a" * 40, "dtype": "changed"}
                with self.assertRaisesRegex(ValueError, "Backend provenance changed"):
                    run(args)
                backend.metadata = json.loads(provenance)
                backend.predict = MagicMock(return_value="B")
                backend.last_prediction_metadata = {"input_tokens": 24, "generated_tokens": 1,
                                                    "input_shapes": {"pixel_values": [1, 3, 16, 16]}}
                result = run(args)
                self.assertTrue(result["complete"])
                self.assertEqual(result["groups"][0]["accuracy"], .5)
                self.assertEqual(result["runtime"]["completed_this_session"], 1)
                self.assertEqual(result["runtime"]["estimated_remaining_seconds"], 0)
                final = read_records(root / "run/predictions.jsonl")[-1]
                self.assertEqual(final["prediction_metadata"], backend.last_prediction_metadata)
                backend.predict.assert_called_once()
                cached = run(args)
                self.assertEqual(cached["runtime"], result["runtime"])
                backend.predict.assert_called_once()
                for change in ({"revision": "b" * 40}, {"seed": 1}, {"dtype": "float16"},
                               {"max_new_tokens": 32}, {"max_samples": 1},
                               {"cpu_threads": 1}, {"interop_threads": 2}):
                    with self.subTest(change=change), self.assertRaisesRegex(ValueError, "changed"):
                        run(SimpleNamespace(**{**vars(args), **change}))
                with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1"}):
                    with self.assertRaisesRegex(ValueError, "exceed"):
                        run(args)
                with patch("vlm_bench.run.source_hash", return_value="changed-source"):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        run(args)
                Image.new("RGB", (16, 16), "white").save(root / "image.png")
                with self.assertRaisesRegex(ValueError, "changed"):
                    run(args)

    def test_large_run_writes_periodic_summaries_without_full_rescans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4)).save(root / "image.png")
            rows = [dict(id=str(i), dataset="pgm", task="matrix", regime="neutral", split="test",
                         images=["image.png"], choices=["A", "B"], answer="A", prompt="Choose.")
                    for i in range(61)]
            manifest = root / "test.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = SimpleNamespace(manifest=str(manifest), output=str(root / "run"), model="lfm25",
                                   revision="a" * 40, max_samples=0, max_new_tokens=16,
                                   seed=0, dtype="auto", dry_run=False, summary_every=25)
            backend = SimpleNamespace(metadata={}, predict=MagicMock(return_value="A"))
            fake_torch = MagicMock()
            fake_torch.get_num_interop_threads.return_value = 1
            fake_torch.cuda.get_device_name.return_value = "TEST DEVICE"
            fake_torch.cuda.max_memory_allocated.return_value = 0
            model_module = SimpleNamespace(load_backend=MagicMock(return_value=backend))
            summary_counts = []
            progress_snapshots = []

            def capture_summary(path, value):
                if path.name == "summary.json":
                    summary_counts.append(value["evaluated"])
                    progress_snapshots.append(value.get("runtime"))
                atomic_json(path, value)

            with patch.dict("sys.modules", {"torch": fake_torch, "vlm_bench.models": model_module}), \
                    patch.dict(os.environ, {"SLURM_JOB_ID": "test"}), \
                    patch("vlm_bench.run.require_allocation"), patch("builtins.print"), \
                    patch("vlm_bench.run.atomic_json", side_effect=capture_summary), \
                    patch("vlm_bench.run.summarize", side_effect=AssertionError("Full history rescan")):
                result = run(args)
            self.assertEqual(summary_counts, [0, 25, 50, 61])
            self.assertEqual([row["completed_this_session"] for row in progress_snapshots[1:]], [25, 50, 61])
            self.assertTrue(all(row["evaluation_examples_per_second"] > 0 for row in progress_snapshots[1:]))
            fake_torch.set_num_threads.assert_called_once_with(2)
            fake_torch.set_num_interop_threads.assert_not_called()
            self.assertTrue(result["complete"])
            self.assertEqual(result["groups"][0]["correct"], 61)
            self.assertEqual(len(read_records(root / "run/predictions.jsonl")), 61)

    def test_real_sigterm_commits_inflight_example_and_resumes_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 4)).save(root / "image.png")
            rows = [dict(id=str(i), dataset="pgm", task="matrix", regime="neutral", split="test",
                         images=["image.png"], choices=["A", "B"], answer="A", prompt=str(i))
                    for i in range(7)]
            manifest = root / "test.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            script = root / "fake_worker.py"
            script.write_text(textwrap.dedent('''
                import os
                from pathlib import Path
                import sys
                import time
                from types import SimpleNamespace
                from unittest.mock import MagicMock
                from vlm_bench import run as runner

                root = Path(sys.argv[1])
                mode = sys.argv[2]
                sys.argv = [sys.argv[0], *sys.argv[3:]]
                os.environ["SLURM_JOB_ID"] = "test-" + mode
                runner.require_allocation = lambda: None
                runner.source_hash = lambda: "frozen-test-source"
                original_handler = runner.StopRequest.handle
                def acknowledge(self, signum, frame):
                    original_handler(self, signum, frame)
                    (root / "acknowledged").touch()
                runner.StopRequest.handle = acknowledge
                def predict(image, prompt):
                    with (root / "calls.txt").open("a") as calls:
                        calls.write(prompt + "\\n")
                    if mode == "interrupt" and prompt == "2":
                        (root / "inflight").touch()
                        while not (root / "release").exists():
                            time.sleep(.01)
                    return "A" if int(prompt) % 2 == 0 else "B"
                backend = SimpleNamespace(metadata={"revision": "a" * 40}, predict=predict)
                fake_torch = MagicMock()
                fake_torch.cuda.get_device_name.return_value = "TEST DEVICE"
                fake_torch.cuda.max_memory_allocated.return_value = 0
                sys.modules["torch"] = fake_torch
                sys.modules["vlm_bench.models"] = SimpleNamespace(load_backend=lambda *a, **kw: backend)
                runner.main()
            '''))
            command = [sys.executable, str(script), str(root), "interrupt", "--manifest", str(manifest),
                       "--output", str(root / "run"), "--revision", "a" * 40, "--max-samples", "0",
                       "--summary-every", "25"]
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
            worker = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            def wait_for(filename):
                deadline = time.monotonic() + 10
                while not (root / filename).exists():
                    if worker.poll() is not None:
                        self.fail(f"Worker exited before {filename}: {worker.communicate()}")
                    if time.monotonic() > deadline:
                        self.fail(f"Worker did not reach {filename}")
                    time.sleep(.01)

            try:
                wait_for("inflight")
                worker.send_signal(signal.SIGTERM)
                wait_for("acknowledged")
                (root / "release").touch()
                stdout, stderr = worker.communicate(timeout=10)
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.communicate()
            self.assertEqual(worker.returncode, 128 + signal.SIGTERM, (stdout, stderr))
            predictions = root / "run/predictions.jsonl"
            committed_prefix = predictions.read_bytes()
            self.assertEqual([row["id"] for row in read_records(predictions)], ["0", "1", "2"])
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertEqual(summary["evaluated"], 3)
            self.assertFalse(summary["complete"])
            interrupted = json.loads(next((root / "run").glob("session-*.json")).read_text())
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["interruption_signal"], "SIGTERM")
            self.assertEqual(interrupted["completed_this_session"], 3)
            config_before = (root / "run/config.json").read_bytes()
            backend_before = (root / "run/backend.json").read_bytes()
            command[3] = "resume"
            resumed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertTrue(predictions.read_bytes().startswith(committed_prefix))
            self.assertEqual((root / "run/config.json").read_bytes(), config_before)
            self.assertEqual((root / "run/backend.json").read_bytes(), backend_before)
            self.assertEqual((root / "calls.txt").read_text().splitlines(), list(map(str, range(7))))
            finished = read_records(predictions)
            self.assertEqual([row["id"] for row in finished], list(map(str, range(7))))
            self.assertEqual([row["correct"] for row in finished], [True, False, True, False, True, False, True])
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["groups"][0]["correct"], 4)
            sessions = [json.loads(path.read_text()) for path in (root / "run").glob("session-*.json")]
            self.assertEqual(sorted(item["completed_this_session"] for item in sessions), [3, 4])
            self.assertEqual(sorted(item["status"] for item in sessions), ["complete", "interrupted"])


if __name__ == "__main__":
    unittest.main()
