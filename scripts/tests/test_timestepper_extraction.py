"""Keep extracted explicit timesteppers independent of parent/solver headers."""

import configparser
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "source/contrib/nmfd-timesteppers"
sys.path.insert(0, str(ROOT))
import run_local_ci as ci


class TimestepperExtractionTest(unittest.TestCase):
    def make(self, directory, *arguments):
        env = os.environ.copy()
        for name in ("MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "CPPFLAGS", "CPATH",
                     "NMFD_OPERATIONS_ROOT", "SCFD_ROOT", "NMFD_TIMESTEPPERS_ROOT",
                     "NMFD_NEWTON_ROOT", "NMFD_LINSOLVERS_ROOT"):
            env.pop(name, None)
        return subprocess.run(
            ["make", "--no-print-directory", "-n", *arguments], cwd=directory,
            env=env, check=True, capture_output=True, text=True, timeout=30,
        ).stdout

    def test_registration_and_single_source_owner(self):
        config = configparser.ConfigParser()
        config.read(ROOT / ".gitmodules")
        section = 'submodule "source/contrib/nmfd-timesteppers"'
        self.assertEqual(config[section]["path"], "source/contrib/nmfd-timesteppers")
        self.assertEqual(config[section]["url"],
                         "https://github.com/evstigneevnm/NMFD-timesteppers.git")
        self.assertTrue((MODULE / "include/nmfd/time_steppers/runge_kutta/explicit_time_step.h").is_file())
        for directory in ("detail", "integration", "runge_kutta", "tests"):
            self.assertFalse(list((ROOT / "source/time_stepper" / directory).glob("*.h")))
            self.assertFalse(list((ROOT / "source/time_stepper" / directory).glob("*.cpp")))
            self.assertFalse(list((ROOT / "source/time_stepper" / directory).glob("*.py")))
        self.assertEqual((MODULE / ".clang-format").read_bytes(), (ROOT / ".clang-format").read_bytes())
        self.assertNotIn("Makefile", (MODULE / ".gitignore").read_text().splitlines())

    def test_sources_use_only_public_external_dependencies(self):
        public = MODULE / "include"
        sources = [*public.rglob("*.h"), *MODULE.glob("tests/*.h"), *MODULE.glob("tests/*.cpp")]
        for path in sources:
            for header in re.findall(r'^\s*#\s*include\s*[<"]([^>"]+)', path.read_text(), re.MULTILINE):
                with self.subTest(path=path.relative_to(MODULE), header=header):
                    self.assertFalse(header.startswith(("time_stepper/", "common/", "contrib/",
                                                       "numerical_algos/")))
                    if public in path.parents or not path.name.startswith(("implicit_", "test_implicit_")):
                        self.assertFalse(header.startswith("nmfd/solvers/"))
                    if header.startswith("nmfd/time_steppers/"):
                        self.assertTrue((public / header).is_file())

    def test_parent_explicit_builds_do_not_require_newton(self):
        for target in ("test_rk_tables_explicit.bin", "test_rk_tables_implicit.bin",
                       "test_explicit_rk_cpu.bin", "test_explicit_rk_cpu_omp.bin",
                       "test_explicit_lorenz_simulation_cpu.bin"):
            output = self.make(ROOT, target, "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc",
                               "BUILD_DIR=build/timestepper_extraction_probe")
            with self.subTest(target=target):
                self.assertIn("nmfd-timesteppers/tests/", output)
                self.assertNotIn("nmfd-newton/include", output)
                self.assertNotIn("nmfd-linsolvers/include", output)
                self.assertNotRegex(output, r"(?:^|\s)-I\s*source/?(?:\s|$)")

    def test_module_defaults_are_flat_and_cuda_links_objects(self):
        output = self.make(MODULE, "test_explicit_rk_cpu_omp.bin", "BUILD_DIR=build/extraction_probe")
        self.assertIn("-I../nmfd-operations/include", output)
        self.assertIn("-I../scfd/include", output)
        self.assertNotIn("newton/include", output)
        self.assertNotIn("linsolvers/include", output)
        output = self.make(MODULE, "test_explicit_rk_cuda.bin", "BUILD_DIR=build/extraction_probe")
        self.assertIn("-x cu -c tests/test_explicit_rk.cpp", output)
        link = next(line for line in output.splitlines() if "-lcublas" in line)
        self.assertNotIn(".cpp", link)
        self.assertIn("cuda_reductions.o", link)

    def test_parent_ci_initializes_module_and_uses_migrated_scripts(self):
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        self.assertEqual(ci.expand_expressions(workflow["env"]["NMFD_TIMESTEPPERS_ROOT"], {}, ROOT), str(MODULE))
        for name, job in workflow["jobs"].items():
            if not name.startswith("timestepper-"):
                continue
            for step in job["steps"]:
                with self.subTest(job=name):
                    if step.get("uses", "").startswith("actions/checkout@"):
                        self.assertIs(step.get("with", {}).get("submodules"), True)
                    self.assertNotIn("source/time_stepper/tests/", step.get("run", ""))
        cpu = workflow["jobs"]["timestepper-explicit-rk-cpu"]
        gpu = workflow["jobs"]["timestepper-explicit-cuda"]
        self.assertIn("check_timestepper_headers_cpu", "\n".join(step.get("run", "") for step in cpu["steps"]))
        self.assertIn("check_timestepper_headers_cuda", "\n".join(step.get("run", "") for step in gpu["steps"]))

    def test_implicit_solver_dependencies_are_test_only(self):
        output = self.make(MODULE, "test_implicit_dirk_cpu.bin", "BUILD_DIR=build/implicit_probe")
        self.assertIn("-I../nmfd-newton/include", output)
        self.assertIn("-I../nmfd-linsolvers/include", output)
        header = (MODULE / "include/nmfd/time_steppers/runge_kutta/implicit_time_step.h").read_text()
        self.assertNotIn("#include <nmfd/solvers/", header)
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        for test in ("dirk", "mass-dae", "lifecycle", "lorenz", "rossler", "vdp", "nonautonomous"):
            for backend in ("cpu", "cuda"):
                self.assertIn(f"timestepper-implicit-{test}-{backend}", workflow["jobs"])

    def test_nonautonomous_matrix_preserves_explicit_independence(self):
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        for mode in ("explicit", "implicit"):
            output = self.make(ROOT, f"test_{mode}_nonautonomous_cpu.bin",
                               "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc",
                               "BUILD_DIR=build/nonautonomous_probe")
            self.assertIn(f"tests/test_{mode}_nonautonomous.cpp", output)
            if mode == "explicit":
                self.assertNotIn("nmfd-newton/include", output)
                self.assertNotIn("nmfd-linsolvers/include", output)
            for backend in ("cpu", "cuda"):
                self.assertIn(f"timestepper-{mode}-nonautonomous-{backend}", workflow["jobs"])


if __name__ == "__main__":
    unittest.main()
