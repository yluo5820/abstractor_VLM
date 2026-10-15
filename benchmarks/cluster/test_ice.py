"""Tests for request limits and shell quoting; never connect or allocate."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import ice


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((Path(__file__).parent / "profile.example.json").read_text())
        self.profile.update(verified=True, verification_notes="Test-only scheduler evidence", account="test-account", partition="test-partition", qos="test-qos", project_path="/scratch/test/benchmarks", environment_path="/scratch/test/venv", cache_path="/scratch/test/cache")

    def load(self, **updates):
        profile = copy.deepcopy(self.profile)
        profile.update(updates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile))
            return ice.load_profile(path)

    def test_unverified_profile_refused(self):
        with self.assertRaisesRegex(ValueError, "unverified"):
            self.load(verified=False)

    def test_initial_resources(self):
        args = ice.resource_args(self.load())
        for expected in ("--gres=gpu:1", "--mem=16G", "--cpus-per-task=2", "--time=00:20:00"):
            self.assertIn(expected, args)

    def test_raised_resources_require_approval(self):
        for changed in ({"memory_gib": 17}, {"cpus": 3}, {"time": "00:20:01"}, {"gpu_count": 2, "gpu_request": {"flag": "--gres", "value": "gpu:2"}}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Ask the user"):
                self.load(**changed)

    def test_recorded_explicit_approval_permits_change(self):
        self.assertEqual(self.load(time="02:00:00", user_resource_approval="User authorized a two-hour run on DATE")['time'], "02:00:00")

    def test_gpu_count_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            self.load(gpu_request={"flag": "--gres", "value": "gpu:a100:2"})

    def test_verified_gpu_hour_budget(self):
        limits = dict(max_wall_seconds=57600, max_gpu_minutes=960, max_cpu_minutes=30720)
        fields = dict(scheduler_limits=limits, user_resource_approval="User authorized resource audit",
                      gpu_count=2, gpu_request={"flag": "--gres", "value": "gpu:h200:2"})
        self.assertEqual(self.load(time="08:00:00", **fields)["gpu_count"], 2)
        with self.assertRaisesRegex(ValueError, "GPU count times"):
            self.load(time="08:00:01", **fields)
        with self.assertRaisesRegex(ValueError, "partition limit"):
            self.load(time="16:00:01", scheduler_limits=limits, user_resource_approval="audit")

    def test_verified_cpu_hour_budget(self):
        with self.assertRaisesRegex(ValueError, "CPU count times"):
            self.load(cpus=65, time="08:00:00", user_resource_approval="audit",
                      scheduler_limits={"max_cpu_minutes": 30720})

    def test_typed_gpu_syntax(self):
        self.assertIn("--gres=gpu:a100:1", ice.resource_args(self.load(gpu_request={"flag": "--gres", "value": "gpu:a100:1"})))
        self.assertIn("--gpus=a100:1", ice.resource_args(self.load(gpu_request={"flag": "--gpus", "value": "a100:1"})))

    def test_connection_changes_refused(self):
        with self.assertRaisesRegex(ValueError, "Connection differs"):
            self.load(ssh_alias="old-compute-node")

    def test_remote_path_and_module_injection_refused(self):
        for changed in ({"project_path": "/scratch/user/$(touch-bad)"}, {"project_path": "/scratch/user/../other"}, {"modules": ["python;rm"]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.load(**changed)

    def test_empty_verified_qos_uses_default(self):
        self.assertFalse(any(arg.startswith("--qos") for arg in ice.resource_args(self.load(qos=""))))

    def test_remote_shell_receives_quoted_argv(self):
        command = ice.remote_command(self.load(), ["printf", "%s", "contains spaces; and 'quotes'"])
        shell = shlex.split(command[-1])
        self.assertEqual(shell[:2], ["bash", "-lc"])
        self.assertIn("exec printf %s 'contains spaces; and", shell[2])
        self.assertEqual(shlex.split(shell[2].split("exec ", 1)[1]), ["printf", "%s", "contains spaces; and 'quotes'"])

    def test_guard_refuses_no_allocation(self):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
        result = subprocess.run(["bash", str(Path(__file__).parent / "guard.sh")], env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SLURM_JOB_ID is absent", result.stderr)

    def test_batch_all_maps_to_explicit_zero(self):
        # Mock only the allocation probes and interpreter; no scheduler or model runs.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            binaries = base / "bin"
            binaries.mkdir()
            for name, body in {"hostname": "printf 'compute-test\\n'", "nvidia-smi": "printf 'mock GPU\\n'", "python": "printf '%s\\n' \"$@\""}.items():
                executable = binaries / name
                executable.write_text("#!/bin/sh\n" + body + "\n")
                executable.chmod(0o700)
            environment = dict(os.environ, SLURM_JOB_ID="123", SLURM_JOB_GPUS="0", PATH=str(binaries) + os.pathsep + os.environ["PATH"])
            command = ["bash", str(Path(__file__).parent / "evaluate.sbatch"), str(ice.BENCH), str(base), str(base / "cache"), str(base / "manifest.jsonl"), str(base / "output"), "lfm25", "a" * 40, "all"]
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--max-samples\n0\n", result.stdout)

    def test_live_job_status_uses_controller_without_accounting(self):
        live = subprocess.CompletedProcess([], 0, "JobId=5920012 JobState=RUNNING\n", "")
        output = io.StringIO()
        with patch("ice.verify_connection_config") as verify, \
                patch("ice.subprocess.run", return_value=live) as invoke, redirect_stdout(output):
            ice.show_job_status(self.load(), "5920012")
        verify.assert_called_once()
        invoke.assert_called_once()
        self.assertIn("JobState=RUNNING", output.getvalue())
        self.assertNotIn("accounting history", output.getvalue())
        self.assertIn("scontrol show job 5920012", invoke.call_args.args[0][-1])

    def test_purged_live_job_returns_accounting_with_job_steps(self):
        missing = subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified\n")
        records = ("5920012|vlm-audit|coc-gpu|COMPLETED|0:0|00:04:17|2026-09-23T10:00:00|2026-09-23T10:04:17|cpu=2,gres/gpu=1|\n"
                   "5920012.batch|batch||COMPLETED|0:0|00:04:17|||cpu=2|1024M\n")
        accounting = subprocess.CompletedProcess([], 0, records, "")
        output, errors = io.StringIO(), io.StringIO()
        with patch("ice.verify_connection_config"), \
                patch("ice.subprocess.run", side_effect=[missing, accounting]) as invoke, \
                redirect_stdout(output), redirect_stderr(errors):
            ice.show_job_status(self.load(), "5920012")
        self.assertEqual(invoke.call_count, 2)
        self.assertIn("Slurm accounting history", output.getvalue())
        self.assertIn("5920012|vlm-audit|coc-gpu|COMPLETED|0:0", output.getvalue())
        self.assertIn("5920012.batch", output.getvalue())
        self.assertEqual(errors.getvalue(), "")
        remote = invoke.call_args.args[0][-1]
        self.assertIn("sacct --jobs=5920012 --user=yluo432", remote)
        self.assertIn("--noheader --parsable2", remote)

    def test_unknown_job_without_accounting_is_not_reported_complete(self):
        missing = subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified\n")
        for data in ("", "9999999|another-job|coc-gpu|COMPLETED|0:0\n"):
            output = io.StringIO()
            with self.subTest(data=data), patch("ice.verify_connection_config"), \
                    patch("ice.subprocess.run", side_effect=[missing, subprocess.CompletedProcess([], 0, data, "")]), \
                    redirect_stdout(output), self.assertRaisesRegex(ValueError, "No accounting record found for job 5920012.*Completion has not been established"):
                ice.show_job_status(self.load(), "5920012")
            self.assertNotIn("Slurm accounting history", output.getvalue())

    def test_status_does_not_hide_auth_network_or_controller_errors(self):
        cases = [(255, "Permission denied (publickey).\n"),
                 (255, "ssh: connect to host login-ice port 22: Connection timed out\n"),
                 (255, "slurm_load_jobs error: Invalid job id specified\n"),
                 (1, "slurm_load_jobs error: Unable to contact slurm controller (connect failure)\n"),
                 (1, "scontrol: error: Access/permission denied\n")]
        for code, error in cases:
            output, errors = io.StringIO(), io.StringIO()
            with self.subTest(error=error), patch("ice.verify_connection_config"), \
                    patch("ice.subprocess.run", return_value=subprocess.CompletedProcess([], code, "", error)) as invoke, \
                    redirect_stdout(output), redirect_stderr(errors), self.assertRaises(subprocess.CalledProcessError):
                ice.show_job_status(self.load(), "5920012")
            invoke.assert_called_once()
            self.assertEqual(errors.getvalue(), error)
            self.assertNotIn("checking Slurm accounting", output.getvalue())

    def test_accounting_failure_preserves_diagnostic(self):
        missing = subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified\n")
        failed = subprocess.CompletedProcess([], 1, "", "sacct: error: Slurm accounting storage is disabled\n")
        output, errors = io.StringIO(), io.StringIO()
        with patch("ice.verify_connection_config"), patch("ice.subprocess.run", side_effect=[missing, failed]), \
                redirect_stdout(output), redirect_stderr(errors), self.assertRaises(subprocess.CalledProcessError):
            ice.show_job_status(self.load(), "5920012")
        self.assertEqual(errors.getvalue(), failed.stderr)
        self.assertNotIn("No accounting record", output.getvalue())

    def test_invalid_job_id_refused_before_connecting(self):
        with patch("ice.verify_connection_config") as verify, patch("ice.subprocess.run") as invoke:
            with self.assertRaisesRegex(ValueError, "numeric job ID"):
                ice.show_job_status(self.load(), "5920012;whoami")
        verify.assert_not_called()
        invoke.assert_not_called()

    def test_submission_waits_for_successful_preparation(self):
        argv = ["ice.py", "submit", "--manifest", "/scratch/test/data.jsonl",
                "--output", "/scratch/test/result", "--revision", "a" * 40,
                "--max-samples", "all", "--afterok", "12345"]
        output = io.StringIO()
        with patch("sys.argv", argv), patch("ice.load_profile", return_value=self.load()), \
                patch("ice.subprocess.run") as invoke, redirect_stdout(output):
            ice.main()
        self.assertIn("--dependency=afterok:12345", output.getvalue())
        self.assertIn("--kill-on-invalid-dep=yes", output.getvalue())
        invoke.assert_not_called()
        argv[-1] = "12345;bad"
        with patch("sys.argv", argv), patch("ice.load_profile", return_value=self.load()), \
                self.assertRaisesRegex(ValueError, "numeric job ID"):
            ice.main()


if __name__ == "__main__":
    unittest.main()
