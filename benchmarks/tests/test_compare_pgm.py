"""No GPU/model needed: test real watchdogs and immutable pipeline transitions."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

_spec = importlib.util.spec_from_file_location(
    "compare_pgm", Path(__file__).resolve().parents[1] / "cluster" / "compare_pgm.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "run"
        self.policy = {"model": "spatialevo", "revision": "a" * 40,
                       "source": str(self.base), "source_sha256": "source-hash",
                       "coordinator_sha256": "coordinator-hash", "python": sys.executable,
                       "preflight_manifest": str(self.base / "pilot.jsonl"),
                       "preflight_manifest_sha256": "pilot-sha",
                       "full_manifest": str(self.base / "full.jsonl"), "full_manifest_sha256": "full-sha",
                       "max_new_tokens": 16, "dtype": "auto", "seed": 0, "cpu_threads": 2}
        self.full_count = mock.patch.object(compare, "FULL_COUNT", 64)
        self.full_count.start()

    def tearDown(self):
        self.full_count.stop()
        self.tmp.cleanup()

    def results(self, phase, *, parseable=True, budget=False, complete=True):
        output = self.root / phase
        output.mkdir(parents=True, exist_ok=True)
        count = compare.PREFLIGHT_COUNT if phase == "preflight" else compare.FULL_COUNT
        config = dict(compare.expected_config(self.policy, phase), example_ids=[str(i) for i in range(count)])
        compare.atomic_json(output / "config.json", config)
        compare.atomic_json(output / "backend.json", {"model": "pinned-fixture"})
        with (output / "predictions.jsonl").open("w") as stream:
            for i in range(count):
                stream.write(json.dumps({"id": str(i), "prediction": "A" if parseable else None,
                    "correct": False, "answer": "B", "raw_output": "A" if parseable else "bad",
                    "prediction_metadata": {"reached_token_budget": budget}}) + "\n")
        compare.atomic_json(output / "summary.json", {"complete": complete, "planned": count,
                                                       "evaluated": count, "accuracy": 0.0})
        return output

    def success(self, command, *_):
        phase = Path(command[command.index("--output") + 1]).name
        self.results(phase)
        return {"return_code": 0, "stop_reason": None, "forced_kill": False}

    def test_preflight_then_full_zero_accuracy_allowed_and_repeat_skips_child(self):
        with mock.patch.object(compare, "supervise", side_effect=self.success) as child:
            first = compare.pipeline(self.policy, self.root, time.monotonic() + 100)
            second = compare.pipeline(self.policy, self.root, time.monotonic() + 100)
        self.assertEqual(child.call_count, 2)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(second["status"], "complete")
        self.assertEqual(second["phases"]["full"]["status"], "verified-complete")
        gate = json.loads((self.root / "preflight" / "technical-gate.json").read_text())
        self.assertEqual(gate["parseable_outputs"], 32)
        self.assertNotIn("accuracy", gate)
        self.assertEqual(gate["answer_distribution"], {"A": 32})

    def test_invalid_or_token_limited_preflight_prevents_full(self):
        for kind in ("invalid", "token-budget"):
            with self.subTest(kind=kind):
                self.root = self.base / kind
                def failed_gate(command, *_):
                    self.results("preflight", parseable=kind != "invalid", budget=kind == "token-budget")
                    return {"return_code": 0, "stop_reason": None, "forced_kill": False}
                with mock.patch.object(compare, "supervise", side_effect=failed_gate) as child:
                    with self.assertRaisesRegex(ValueError, "Technical preflight failed"):
                        compare.pipeline(self.policy, self.root, time.monotonic() + 100)
                self.assertEqual(child.call_count, 1)
                self.assertFalse((self.root / "full").exists())
                self.assertEqual(json.loads((self.root / "pipeline-status.json").read_text())["status"], "failed")
                self.assertTrue((self.root / "preflight" / "technical-gate.json").exists())

    def test_modified_model_or_predictions_cannot_skip_verification(self):
        with mock.patch.object(compare, "supervise", side_effect=self.success):
            compare.pipeline(self.policy, self.root, time.monotonic() + 100)
        changed = copy.deepcopy(self.policy)
        changed["model"] = "smolvlm2"
        with self.assertRaisesRegex(ValueError, "inputs/code/model/settings changed"):
            compare.pipeline(changed, self.root, time.monotonic() + 100)
        with (self.root / "full" / "predictions.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "Completed run file changed"):
            compare.pipeline(self.policy, self.root, time.monotonic() + 100)

    def test_deadline_is_explicit_resumable_not_false_complete(self):
        interrupted = {"return_code": 143, "stop_reason": "segment-deadline", "forced_kill": False}
        with mock.patch.object(compare, "supervise", return_value=interrupted) as child:
            result = compare.pipeline(self.policy, self.root, time.monotonic() + 100)
        self.assertEqual(result["status"], "resumable")
        self.assertEqual(result["phases"]["preflight"]["status"], "interrupted-resumable")
        self.assertEqual(child.call_count, 1)
        with mock.patch.object(compare, "supervise", side_effect=self.success):
            resumed = compare.pipeline(self.policy, self.root, time.monotonic() + 100)
        self.assertEqual(resumed["status"], "complete")

    def test_unrequested_failure_and_partial_zero_exit_fail_chain(self):
        with mock.patch.object(compare, "supervise", return_value={"return_code": -9, "stop_reason": None, "forced_kill": False}):
            with self.assertRaisesRegex(RuntimeError, "runner failed"):
                compare.pipeline(self.policy, self.root, time.monotonic() + 100)
        self.root = self.base / "unexpected-kill"
        with mock.patch.object(compare, "supervise", return_value={"return_code": -9, "stop_reason": "segment-deadline", "forced_kill": False}):
            with self.assertRaisesRegex(RuntimeError, "runner failed"):
                compare.pipeline(self.policy, self.root, time.monotonic() + 100)
        self.root = self.base / "incomplete"
        def partial(*_):
            self.results("preflight", complete=False)
            return {"return_code": 0, "stop_reason": None, "forced_kill": False}
        with mock.patch.object(compare, "supervise", side_effect=partial):
            with self.assertRaisesRegex(ValueError, "without a complete"):
                compare.pipeline(self.policy, self.root, time.monotonic() + 100)

    def test_watchdog_terminates_process_during_startup(self):
        script = "import signal,sys,time; signal.signal(signal.SIGTERM,lambda *args: sys.exit(143)); time.sleep(60)"
        started = time.monotonic()
        result = compare.supervise([sys.executable, "-c", script], self.base, started + 0.3, [], grace=0.5)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(result["stop_reason"], "segment-deadline")
        self.assertIn(result["return_code"], (143, -signal.SIGTERM))
        self.assertFalse(result["forced_kill"])

    def test_watchdog_escalates_hung_owned_process(self):
        script = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
        result = compare.supervise([sys.executable, "-c", script], self.base, time.monotonic() + 0.3, [], grace=0.3)
        self.assertEqual(result["return_code"], -signal.SIGKILL)
        self.assertTrue(result["forced_kill"])

    def test_child_has_exact_shared_inference_settings_and_all_full_rows(self):
        for phase in ("preflight", "full"):
            command = compare.child_command(self.policy, self.root / phase, phase)
            self.assertEqual(command[command.index("--max-samples") + 1], "32" if phase == "preflight" else "0")
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "16")
            self.assertEqual(command[command.index("--cpu-threads") + 1], "2")
            self.assertEqual(command[command.index("--model") + 1], "spatialevo")

    def test_clevr_expected_count_does_not_use_pgm_count(self):
        policy = dict(self.policy, full_count=10000)
        output = self.root / "full"
        output.mkdir(parents=True)
        config = dict(compare.expected_config(policy, "full"), example_ids=[str(i) for i in range(10000)])
        compare.atomic_json(output / "config.json", config)
        compare.verify_run_config(output, policy, "full")
        config["example_ids"].pop()
        compare.atomic_json(output / "config.json", config)
        with self.assertRaisesRegex(ValueError, "example count"):
            compare.verify_run_config(output, policy, "full")


if __name__ == "__main__":
    unittest.main()
