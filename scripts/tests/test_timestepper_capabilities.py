"""Keep capability evidence and the parent validation gate connected."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_timestepper_capabilities as capabilities
import run_local_ci as ci
SUITE = ROOT/"source/nonlinear_operators/pde_validation/tests"
sys.path.insert(0, str(SUITE))
import pde_ci


class CapabilityContractTest(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / capabilities.MATRIX).read_text()

    def test_matrix_has_linked_evidence(self):
        self.assertEqual(capabilities.validate()["methods"], 28)

    def test_stale_test_link_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing capability evidence"):
            capabilities.validate(text=self.text.replace("test_explicit_rk.cpp", "missing_test.cpp"))

    def test_missing_evidence_and_duplicate_method_are_rejected(self):
        row = next(line for line in self.text.splitlines() if line.startswith("| EE |"))
        for text in (self.text.replace(row, "| EE | 1 | 0 | 1 | F | none |"),
                     self.text.replace(row, row + "\n" + row)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                capabilities.validate(text=text)

    def test_compiled_metadata_must_agree(self):
        rows = capabilities.method_rows(self.text)
        metadata = "\n".join("|".join((name, *values)) for name, values in rows.items())
        self.assertTrue(capabilities.validate(metadata=metadata)["compiled_metadata_checked"])
        with self.assertRaisesRegex(ValueError, "differ from compiled"):
            capabilities.validate(metadata=metadata.replace("RK4|4|0|3", "RK4|3|0|3"))

    def test_pde_jobs_and_failure_artifacts_are_present(self):
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        gate = jobs["all-tests-passed"]["needs"]
        for name in ("timestepper-capability-contract", "timestepper-pde-cpu",
                     "timestepper-pde-cuda", "timestepper-pde-hip"):
            with self.subTest(job=name):
                self.assertIn(name, jobs)
                if name != "timestepper-capability-contract":
                    uploads = [step for step in jobs[name]["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")]
                    self.assertTrue(uploads)
                    self.assertEqual(uploads[0]["if"], "always()")
                    tag = name.rsplit("-", 1)[1]
                    artifact = "timestepper-pde-"+("${{ matrix.compiler }}" if tag == "cpu" else tag)
                    self.assertEqual(uploads[0]["with"]["name"], artifact)
                    for pattern in ("backend.json", "*.txt", "**/*.log", "**/summary.json"):
                        self.assertIn(pattern, uploads[0]["with"]["path"])
        for name in ("timestepper-pde-cpu", "timestepper-pde-cuda", "timestepper-pde-hip"):
            self.assertIn(name, gate)

    def test_job_shells_parse(self):
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        for name in ("timestepper-capability-contract", "timestepper-pde-cpu", "timestepper-pde-cuda", "timestepper-pde-hip"):
            commands = "\n".join(step.get("run", "") for step in jobs[name]["steps"])
            with self.subTest(job=name):
                subprocess.run(["bash", "-n"], input=commands, text=True, check=True, capture_output=True)

    def test_convex_split_ch_headers_and_capability(self):
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        commands = "\n".join(step.get("run", "") for step in jobs["timestepper-capability-contract"]["steps"])
        self.assertIn("source/nonlinear_operators/cahn_hilliard/*.h", commands)
        self.assertIn("source/nonlinear_operators/cahn_hilliard/detail/*.h", commands)
        self.assertIn("| General singular-mass IMEX | U:", self.text)
        self.assertIn("| Reduced convex-split CH IMEX | V", self.text)


class PDEProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = ci.load_workflow(ROOT/".github/workflows/operations.yml")["jobs"]

    def step(self, platform):
        job = self.jobs["timestepper-pde-"+platform.lower()]
        self.assertEqual(job["timeout-minutes"], 15)
        return next(s for s in job["steps"] if "run_pde_validation.sh" in s.get("run", ""))

    def test_inventory_and_list_on_each_platform(self):
        for platform, suffix in (("SERIAL", "cpu"), ("OMP", "cpu_omp"), ("CUDA", "cuda"), ("HIP", "hip")):
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)/"unused"
                listing = subprocess.check_output(["bash", str(SUITE/"run_all.sh"), "--profile", "ci", "--build-dir", str(directory),
                                                   "--list"], env=dict(os.environ, PLATFORM=platform), text=True)
                names = ("diffusion", "reaction_diffusion", "cahn_hilliard")
                expected = [f"test_pde_{n}_{suffix}.bin" for n in names]
                self.assertEqual([line.split()[1] for line in listing.splitlines()], expected)
                self.assertFalse(directory.exists())
                timeout = 180 if platform in ("SERIAL", "OMP") else 300
                self.assertEqual([(c.target, c.threads, c.timeout) for c in pde_ci.cases(platform, "pde")],
                                 [(target, 2, timeout) for target in expected])

    def test_failed_case_keeps_later_results_and_build_only_is_not_a_pass(self):
        for build_only in (False, True):
            seen = []
            def run(cmd, **kwargs):
                seen.append(cmd)
                rc = int(cmd[0].endswith("/test_pde_reaction_diffusion_cpu.bin"))
                return subprocess.CompletedProcess(cmd, rc)
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, PLATFORM="SERIAL"), \
                    patch.object(subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                args = ["--build-dir", tmp, "--build-jobs", "2", *(["-B"] if build_only else [])]
                rc = pde_ci.suite.run_profile(args, ("pde",), pde_ci.cases, ROOT, "ci-device-probe")
                d = json.loads((Path(tmp)/"SERIAL/pde/summary.json").read_text())
                self.assertEqual(rc, 0 if build_only else 1)
                self.assertFalse(d["passed"])
                self.assertEqual([r["status"] for r in d["results"].values()],
                                 ["built"]*3 if build_only else ["passed", "failed", "passed"])
                if build_only:
                    self.assertTrue(all(cmd[0] == "make" and "-B" in cmd for cmd in seen))

    def test_compiler_lanes_run_both_suites_and_propagate_failure(self):
        matrix = self.jobs["timestepper-pde-cpu"]["strategy"]["matrix"]["compiler"]
        self.assertEqual(matrix, ["g++", "clang++"])
        for platform, compiler in [("CPU", c) for c in matrix] + [("CUDA", "nvcc"), ("HIP", "hipcc")]:
            step = self.step(platform)
            # A numerical CUDA failure after preflight must still permit the CH batch.
            # Other lanes exercise propagation of a failure in the dedicated CH suite.
            setup = """
                function g++ { echo GCC; }; function clang++ { echo Clang; }
                nvcc() { echo nvcc; }; hipcc() { echo 'HIP clang'; }
                make() {
                    printf 'CALL'; printf '\\t%s' "$@"; printf '\\n'
                    if [[ $2 == PLATFORM=CUDA ]]; then
                        if [[ $1 == test-pde ]]; then
                            printf '{"results":{"diffusion":{"status":"failed"}}}' > "$BUILD_DIR/pde_cuda/CUDA/pde/summary.json"
                            return 1
                        fi
                        return 0
                    fi
                    [[ $1 != test-ch ]]
                }
                export -f g++ clang++ nvcc hipcc make
            """
            with tempfile.TemporaryDirectory() as tmp:
                env = dict(os.environ, **step.get("env", {}), BUILD_DIR=tmp, CONFIG_FILE="site.inc",
                           GITHUB_ACTIONS="true", RUNNER_ENVIRONMENT="self-hosted", NVCC="nvcc", HIPCC="hipcc",
                           PDE_TEST_OPTIONS="-B")
                result = subprocess.run(["bash", "-c", setup+step["run"].replace("${{ matrix.compiler }}", compiler)],
                                        cwd=ROOT, env=env, capture_output=True, text=True)
                calls = [line.split("\t")[1:] for line in result.stdout.splitlines() if line.startswith("CALL\t")]
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(all("PDE_TEST_OPTIONS=--group pde" in c for c in calls if c[0] == "test-pde"))
                if platform == "CPU":
                    self.assertEqual([c[:2] for c in calls], [[t, "PLATFORM="+p] for t in ("test-pde", "test-ch")
                                                              for p in ("SERIAL", "OMP", "SERIAL")])
                    self.assertTrue(all(f"G++ = {compiler}" in c for c in calls))
                    self.assertTrue(any("-fsanitize=address,undefined" in arg for arg in calls[2]))
                    self.assertEqual([c[-1] for c in calls[3:]], ["CH_TEST_OPTIONS=--group "+g
                                     for g in ("contracts", "contracts", "sanitizers")])
                    directory = Path(tmp)/"pde"/compiler
                else:
                    self.assertEqual([c[:2] for c in calls], [[t, "PLATFORM="+platform] for t in ("test-pde","test-ch")])
                    self.assertIn("CH_TEST_OPTIONS=--group contracts", calls[-1])
                    self.assertTrue(all("CONFIG_FILE=site.inc" in c for c in calls))
                    directory = Path(tmp)/("pde_"+platform.lower())
                self.assertTrue((directory/"compiler.txt").read_text().strip())
                self.assertEqual(json.loads((directory/"backend.json").read_text())["result"], "failed")

    def test_broken_required_probe_stops_before_any_numerical_suite(self):
        for platform in ("CUDA", "HIP"):
            step = self.step(platform)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fake = root/"fake-compiler"
                fake.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then echo nvcc; exit 0; fi\n"
                                "while [ \"$1\" != -o ]; do shift || exit 2; done\ncp /bin/false \"$2\"\n")
                fake.chmod(0o755)
                env = dict(os.environ, **step["env"], BUILD_DIR=tmp, CONFIG_FILE="build_configs/config_github_ubuntu_cpu.inc",
                           NVCC=str(fake), HIPCC=str(fake), GITHUB_ACTIONS="true", RUNNER_ENVIRONMENT="self-hosted")
                rc = subprocess.run(["bash", "-ec", step["run"]], cwd=ROOT, env=env, capture_output=True).returncode
                directory = root/("pde_"+platform.lower())
                summary = json.loads((directory/platform/"pde/summary.json").read_text())
                self.assertNotEqual(rc, 0)
                self.assertFalse(summary["results"])
                self.assertIn("probe failed", summary["error"])
                self.assertFalse(json.loads((directory/"backend.json").read_text())["executed"])
                self.assertFalse((directory/"ch").exists())


if __name__ == "__main__":
    unittest.main()
