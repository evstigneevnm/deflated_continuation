"""Coverage and failure-policy regressions for the first directory CI suite."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SUITE = "source/nonlinear_operators/cahn_hilliard/tests/run_all.sh"
sys.path.insert(0, str(ROOT / Path(SUITE).parent))
import ch_ci as suite
import run_local_ci


class DirectoryCITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = run_local_ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        cls.steps = [(job, step) for job, spec in cls.jobs.items() for step in spec.get("steps", [])
                     if SUITE in step.get("run", "") or "run_pde_validation.sh" in step.get("run", "")]

    def test_all_suite_calls_are_required_by_the_reference_gate(self):
        required = set()

        def visit(job):
            if job not in required:
                required.add(job)
                needs = self.jobs[job].get("needs", [])
                for parent in ([needs] if isinstance(needs, str) else needs):
                    visit(parent)

        visit("all-tests-passed")
        for job, step in self.steps:
            self.assertIn(job, required)
            self.assertFalse(step.get("continue-on-error"))
            self.assertFalse(self.jobs[job].get("continue-on-error"))
        self.assertEqual({j for j, s in self.steps if "run_pde_validation.sh" in s["run"]},
                         {"timestepper-pde-"+p for p in ("cpu", "cuda", "hip")})
        job, figures = next((j, s) for j, s in self.steps if "--group benchmarks" in s["run"])
        self.assertIn("SERIAL OMP", figures["run"])
        self.assertNotIn("if", figures)
        self.assertNotIn("strategy", self.jobs[job])  # Run figures once with the configured GCC.
        self.assertFalse(any("projection_sanitizers" in s.get("run", "") for s in self.jobs[job]["steps"]))

    def test_case_inventory(self):
        self.assertEqual(set(suite.CONTRACTS), {"cahn_hilliard_imex", "cahn_hilliard_energy",
                                               "cahn_hilliard_imex_lifecycle", "cahn_hilliard_unsplit"})
        for platform, name in [("SERIAL", "fourier_methods"), ("OMP", "fourier_long_euler")]:
            batches = dict(suite.benchmarks(platform))
            self.assertEqual(set(batches), {"fd_methods", "fd_adaptive", name})
            args = suite.bench.arguments(batches["fd_methods"])
            self.assertEqual(set(args.methods), {"IMEX_EULER", "IE", "CN", "SDIRK2(1)2", "BDF2", "ROS3",
                                                  "ARK324L2SA", "ROS34PW2", "BDF", "IMEX_ARK324L2SA"})
            args = suite.bench.arguments(batches["fd_adaptive"])
            self.assertEqual(set(args.adaptive_methods), {"ARK324L2SA", "ROS34PW2", "IMEX_ARK324L2SA"})
            self.assertEqual(set(args.methods), {*args.adaptive_methods, "BDF"})
            args = suite.bench.arguments(batches[name])
            self.assertEqual(args.spatial, "fourier")
            if platform == "OMP":
                self.assertEqual((args.grid, args.end, args.step, args.threads), (32, 60, .1, 1))
                self.assertEqual(set(args.methods), {"IE", "IMEX_EULER"})
            else:
                self.assertEqual(args.grid, 12)
                self.assertEqual(set(args.methods), {"IMEX_EULER", "CN", "BDF", "ROS34PW2"})

    def test_missing_toolchain_only_skips_on_optional_hosted_job(self):
        for platform, compiler in [("cuda", "NVCC"), ("hip", "HIPCC")]:
            step = next(s for job, s in self.steps if job == f"timestepper-pde-{platform}")
            for runner, outcome in [("github-hosted", "skipped"), ("self-hosted", "failed"), ("", "failed")]:
                with self.subTest(platform=platform, runner=runner), tempfile.TemporaryDirectory() as tmp:
                    env = {**os.environ, **step["env"], compiler: tmp+"/absent", "BUILD_DIR": tmp,
                           "RUNNER_ENVIRONMENT": runner, "GITHUB_ACTIONS": "true" if runner else ""}
                    result = subprocess.run(["bash", "-ec", step["run"]], cwd=ROOT, env=env, capture_output=True)
                    self.assertEqual(result.returncode == 0, outcome == "skipped")
                    report = json.loads((Path(tmp)/f"pde_{platform}/backend.json").read_text())
                    self.assertEqual((report["result"], report["executed"]), (outcome, False))

    def test_self_hosted_workflow_runs_only_when_dispatched_and_requires_selected_backend(self):
        workflow = run_local_ci.load_workflow(ROOT / ".github/workflows/ch-self-hosted.yml")
        # PyYAML's YAML 1.1 loader treats an unquoted "on" key as True.
        self.assertEqual(set(workflow.get("on", workflow.get(True))), {"workflow_dispatch"})
        job = workflow["jobs"]["ch"]
        self.assertEqual(job["runs-on"], ["self-hosted", "linux", "${{ inputs.platform }}"])
        self.assertEqual(job["env"]["PLATFORM"], "${{ inputs.platform }}")
        cmd = next(s["run"] for s in job["steps"] if "run" in s)
        self.assertIn(SUITE + " --profile ci", cmd)
        self.assertNotIn("gpu_unavailable", cmd)

    def test_compiler_success_but_kernel_cannot_execute_is_a_failure(self):
        for platform, compiler in [("CUDA", "NVCC"), ("HIP", "HIPCC")]:
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fake = root/"compiler"
                fake.write_text("#!/bin/sh\nwhile [ \"$1\" != -o ]; do shift || exit 2; done\n"
                                "cp /bin/false \"$2\"\n")
                fake.chmod(0o755)
                with patch.dict(os.environ, {"PLATFORM": platform}), contextlib.redirect_stderr(io.StringIO()):
                    rc = suite.main(["--group", "contracts", "--build-dir", tmp, "--make-option", f"{compiler}={fake}"])
                self.assertEqual(rc, 1)
                folder = root/platform/"contracts"
                self.assertTrue((folder/"probe/device_probe.bin").is_file())
                report = json.loads((folder/"summary.json").read_text())
                self.assertFalse(report["passed"])
                self.assertFalse(report["results"])
                self.assertIn("runtime probe failed", report["error"])

    def test_platform_conflict_never_falls_back(self):
        with patch.dict(os.environ, {"PLATFORM": "CUDA"}):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                suite.platforms.selection("cpu")
        with patch.dict(os.environ, {"PLATFORM": "CUAD"}):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                suite.platforms.selection()

    def test_parent_make_passes_selected_compiler_to_ch_builds(self):
        # GNU Make can reinterpret a recursively forwarded G++=... as an append to G+.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/"site.inc"
            config.write_text(f"GCC_ROOT_PATH={root}/no-fallback-compiler\n")
            compiler = root/"selected-cxx"
            compiler.write_text("#!/bin/sh\nwhile [ \"$1\" != -o ]; do shift || exit 2; done\ncp /bin/true \"$2\"\n")
            compiler.chmod(0o755)
            for group in ("contracts", "sanitizers"):
                subprocess.run(["make", "test-ch", "PLATFORM=SERIAL", f"G++ = {compiler}", f"CONFIG_FILE={config}",
                                f"BUILD_DIR={root}", f"CH_TEST_OPTIONS=--group {group}"], cwd=ROOT, check=True,
                               capture_output=True, env=dict(os.environ, PYTHON=sys.executable), timeout=30)
                for name in suite.CONTRACTS:
                    lines = (root/f"ch_ci/SERIAL/{group}/{name}.log").read_text().splitlines()
                    cmd, = [line for line in lines if line.startswith(str(compiler)+" ")]
                    self.assertEqual("-fsanitize=address,undefined" in cmd, group == "sanitizers")

    def test_case_failure_is_reported_after_running_the_other_cases(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PLATFORM": "OMP"}):
            def run(cmd, **kwargs):
                if cmd[0] != "make" and "imex_lifecycle" in cmd[0]:
                    raise subprocess.CalledProcessError(3, cmd)
            with patch.object(subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(suite.main(["--group", "contracts", "--build-dir", tmp]), 1)
            folder = Path(tmp)/"OMP/contracts"
            report = json.loads((folder/"summary.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(list(report["selected"]), list(suite.CONTRACTS))
            self.assertEqual(report["results"]["cahn_hilliard_imex_lifecycle"], "failed")
            self.assertEqual(list(report["results"].values()).count("passed"), 3)
            self.assertEqual(len(list(folder.glob("*.log"))), 4)

    def test_list_and_build_only_do_not_probe_or_run(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PLATFORM": "CUDA"}), \
                patch.object(subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            args = ["--group", "contracts", "--build-dir", tmp]
            self.assertEqual(suite.main([*args, "--list"]), 0)
            run.assert_not_called()
            self.assertEqual(suite.main([*args, "-B"]), 0)
            for call in run.call_args_list:
                self.assertEqual(call.args[0][:2], ["make", "-B"])
            self.assertEqual(run.call_count, len(suite.CONTRACTS))


if __name__ == "__main__":
    unittest.main()
