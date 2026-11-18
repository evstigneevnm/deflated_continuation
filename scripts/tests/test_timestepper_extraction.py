"""Keep extracted explicit timesteppers independent of parent/solver headers."""

import configparser
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "source/contrib/nmfd-timesteppers"
sys.path.insert(0, str(ROOT))
import run_local_ci as ci


class TimestepperExtractionTest(unittest.TestCase):
    def sanitizer_cases(self, job):
        cases = []
        for family in job["strategy"]["matrix"]["family"]:
            steps = [s for s in job["steps"] if s.get("if") == f"matrix.family == '{family}'"]
            self.assertEqual(len(steps), 1, family)
            command = shlex.split(steps[0]["run"])
            self.assertEqual(command[:2], ["bash", "scripts/ci/run_timestepper_sanitizers.sh"])
            self.assertTrue(command[2:], family)
            cases.extend(command[2:])
        self.assertEqual(len(cases), len(set(cases)), "Duplicate sanitizer case")
        return set(cases)

    def test_sanitizer_batch_builds_then_runs_and_propagates_failures(self):
        harness = """
            make() { printf 'make %s\n' "$*"; return "$BUILD_RC"; }
            timeout() { printf 'run %s %s %s\n' "$OMP_NUM_THREADS" "$1" "$2"; return "$TEST_RC"; }
            export -f make timeout
            bash scripts/ci/run_timestepper_sanitizers.sh implicit_dirk explicit_nonautonomous
        """
        for build_rc, test_rc, runs in [(0, 0, 2), (7, 0, 0), (0, 9, 1)]:
            with self.subTest(build=build_rc, test=test_rc):
                result = subprocess.run(["bash", "-c", harness], cwd=ROOT, capture_output=True, text=True,
                                        env=dict(os.environ, BUILD_DIR="build/mock", BUILD_RC=str(build_rc),
                                                 TEST_RC=str(test_rc)))
                self.assertEqual(result.returncode, build_rc or test_rc)
                build, *actual = result.stdout.splitlines()
                self.assertIn("make -j2 test_implicit_dirk_cpu.bin test_explicit_nonautonomous_cpu.bin", build)
                self.assertIn("-fsanitize=address,undefined", build)
                self.assertEqual(actual, [f"run 2 90s build/mock/implicit_sanitizers/test_{case}_cpu.bin"
                                          for case in ("implicit_dirk", "explicit_nonautonomous")[:runs]])

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
        for directory in ("detail", "integration", "runge_kutta", "bdf", "tests"):
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

    def test_missing_gpu_compiler_fails_except_on_optional_hosted_runner(self):
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        for job, directory in (("timestepper-explicit-cuda", "explicit_cuda"),
                               ("timestepper-implicit-dirk-cuda", "dirk_ci"),
                               ("timestepper-imex-cuda", "imex_cuda"), ("timestepper-imex-hip", "imex_hip"),
                               ("timestepper-bdf-cuda", "bdf_cuda"), ("timestepper-bdf-hip", "bdf_hip")):
            step = next(s for s in jobs[job]["steps"] if "run" in s)
            with self.subTest(job=job):
                self.check_missing_compiler(step, directory)

    def check_missing_compiler(self, step, directory):
        for runner, actions, expected in (("github-hosted", "true", "skipped"),
                                          ("self-hosted", "true", "failed"), ("", "", "failed")):
            with self.subTest(runner=runner), tempfile.TemporaryDirectory() as tmp:
                env = dict(os.environ, **step["env"], RUNNER_ENVIRONMENT=runner, GITHUB_ACTIONS=actions,
                           NVCC=str(Path(tmp)/"missing-nvcc"), HIPCC=str(Path(tmp)/"missing-hipcc"), BUILD_DIR=tmp)
                result = subprocess.run(["bash", "-eu", "-c", step["run"]], cwd=ROOT, env=env,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0 if expected == "skipped" else 1, result.stderr)
                report = json.loads((Path(tmp)/directory/"backend.json").read_text())
                self.assertEqual(report["result"], expected)
                self.assertFalse(report["executed"])

    def test_hip_ci_toolchain_selection(self):
        script = """
            compiler() { echo "$VERSION"; return "$COMPILER_RC"; }
            source scripts/ci/hosted_gpu_policy.sh
            select_hip_compiler "$BUILD_DIR"
            printf '%s %s %s %s' "$HIPCC" "$HIP_ROOT_PATH" "$HIP_PATH" "$HIP_ARCH_LIST"
        """
        for version, arch, expected in (("nvcc", "", "75"), ("AMD clang", "", "native"),
                                        ("AMD clang", "gfx90a", "gfx90a")):
            with self.subTest(version=version, arch=arch), tempfile.TemporaryDirectory() as tmp:
                env = dict(os.environ, GPU_POLICY="required", HIPCC="compiler", ROCM_PATH="/rocm",
                           HIP_ARCH_LIST=arch, VERSION=version, COMPILER_RC="0", BUILD_DIR=tmp)
                result = subprocess.run(["bash", "-eu", "-c", script], cwd=ROOT, env=env, check=True,
                                        text=True, capture_output=True)
                self.assertEqual(result.stdout, f"compiler /rocm /rocm {expected}")
                env.update(COMPILER_RC="1", GPU_POLICY="optional-on-hosted",
                           RUNNER_ENVIRONMENT="github-hosted", GITHUB_ACTIONS="true")
                self.assertNotEqual(subprocess.run(["bash", "-eu", "-c", script], cwd=ROOT, env=env).returncode, 0)

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
        self.assertEqual(cpu["strategy"]["matrix"]["platform"], ["SERIAL", "OMP"])
        self.assertIn("test_transform_reduce_sum_cuda.bin", "\n".join(step.get("run", "") for step in gpu["steps"]))
        for job in (cpu, gpu):
            self.assertTrue(any("make test-explicit-rk" in step.get("run", "") for step in job["steps"]))
        for group in ("lorenz", "rk4", "dop853"):
            references = [step for job in workflow["jobs"].values() for step in job["steps"]
                          if f'EXPLICIT_RK_OPTIONS="--group {group}-reference"' in step.get("run", "")]
            self.assertEqual(len(references), 1)
            self.assertIn("for platform in SERIAL OMP", references[0]["run"])

    def test_implicit_solver_dependencies_are_test_only(self):
        output = self.make(MODULE, "test_implicit_dirk_cpu.bin", "BUILD_DIR=build/implicit_probe")
        self.assertIn("-I../nmfd-newton/include", output)
        self.assertIn("-I../nmfd-linsolvers/include", output)
        header = (MODULE / "include/nmfd/time_steppers/runge_kutta/implicit_time_step.h").read_text()
        self.assertNotIn("#include <nmfd/solvers/", header)

    def test_rosenbrock_engine_is_standalone_and_required_by_ci(self):
        tests = ("rosenbrock", "rosenbrock_lifecycle", "rosenbrock_backend",
                 "rosenbrock_adaptive", "rosenbrock_nonautonomous", "rosenbrock_ros3",
                 "rosenbrock_ros3_adaptive", "rosenbrock_ros3_lifecycle",
                 "rosenbrock_dense", "rosenbrock_dense_lifecycle", "rosenbrock_w",
                 "rosenbrock_w_adaptive", "rosenbrock_w_lifecycle", "rosenbrock_w_dense",
                 "rosenbrock_w_dense_lifecycle", "rosenbrock_mass",
                 "rosenbrock_mass_adaptive", "rosenbrock_mass_lifecycle",
                 "rosenbrock_mass_nonautonomous", "rosenbrock_mass_time_lifecycle",
                 "rosenbrock_dae", "rosenbrock_dae_lifecycle", "rosenbrock_variable_mass",
                 "rosenbrock_variable_mass_adaptive", "rosenbrock_variable_mass_lifecycle",
                 "rosenbrock_variable_mass_benchmarks", "rosenbrock_index2",
                 "rosenbrock_index2_moving", "rosenbrock_index2_dense", "rosenbrock_index2_lifecycle")
        for directory in (ROOT, MODULE):
            for test in tests:
                output = self.make(directory, f"test_implicit_{test}_cpu.bin", "BUILD_DIR=build/rosenbrock_probe")
                self.assertIn(f"tests/test_implicit_{test}.cpp", output)
        output = self.make(MODULE, "test_rosenbrock_cpu", "BUILD_DIR=build/rosenbrock_probe")
        for test in tests:
            self.assertIn(f"test_implicit_{test}_cpu_omp.bin", output)
        for path in (MODULE / "include/nmfd/time_steppers/rosenbrock").glob("*.h"):
            self.assertNotIn("#include <nmfd/solvers/", path.read_text())
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        self.assertEqual(jobs["timestepper-rosenbrock-cpu"]["strategy"]["matrix"]["compiler"], ["g++", "clang++"])
        loop = "for test in " + " ".join(tests) + "; do"
        for backend in ("cpu", "cuda", "hip"):
            name = f"timestepper-rosenbrock-{backend}"
            self.assertIn(name, jobs["all-tests-passed"]["needs"])
            commands = "\n".join(step.get("run", "") for step in jobs[name]["steps"])
            self.assertIn(loop, commands)
            self.assertIn("test_implicit_${test}_" + backend + ".bin", commands)
        sanitizer = jobs["timestepper-implicit-sanitizers-cpu"]
        self.assertIn("timestepper-rosenbrock-cpu", sanitizer["needs"])
        self.assertTrue({"implicit_" + test for test in tests} <= self.sanitizer_cases(sanitizer))
        headers = self.make(ROOT, "test_implicit_rosenbrock_cpu.bin", "BUILD_DIR=build/rosenbrock_probe", "-p")
        dependencies = next(line for line in headers.splitlines() if line.startswith("test_implicit_rosenbrock_cpu.bin:"))
        self.assertIn("include/nmfd/time_steppers/rosenbrock/time_step.h", dependencies)
        self.assertIn("tests/implicit_rosenbrock_test.h", dependencies)

    def test_bdf_history_engine_is_standalone_and_required_by_ci(self):
        tests = ("bdf", "bdf_lifecycle", "bdf_mass_dae", "bdf_backend", "bdf_adaptive",
                 "bdf_adaptive_lifecycle", "bdf_adaptive_mass_dae", "bdf_orders", "bdf_variable_order", "bdf_stability",
                 "bdf_dense", "bdf_dense_lifecycle", "bdf_dense_mass_dae")
        for directory in (ROOT, MODULE):
            for test in tests:
                output = self.make(directory, f"test_implicit_{test}_cpu.bin", "BUILD_DIR=build/bdf_probe")
                self.assertIn(f"tests/test_implicit_{test}.cpp", output)
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        self.check_family_workflow(jobs, "bdf", tests)
        sanitizer = jobs["timestepper-implicit-sanitizers-cpu"]
        self.assertIn("timestepper-bdf-cpu", sanitizer["needs"])
        self.assertTrue({"implicit_" + test for test in tests} <= self.sanitizer_cases(sanitizer))
        visual = "\n".join(step.get("run", "") for step in jobs["timestepper-lorenz-simulation-python-cpu"]["steps"])
        self.assertIn("--methods BDF1 BDF2 --controllers constant", visual)
        self.assertIn("--methods BDF1 BDF2 --controllers matlab scipy PID", visual)
        self.assertIn("--methods BDF --bdf-max-order 5", visual)
        self.assertIn("--methods BDF1 BDF2 BDF --controllers scipy --sample-step", visual)
        for directory in (ROOT, MODULE):
            output = self.make(directory, "test_implicit_bdf_cpu.bin", "BUILD_DIR=build/bdf_probe",
                               "-p")
            target = "test_implicit_bdf_cpu.bin" if directory == ROOT else "build/bdf_probe/test_implicit_bdf_cpu.bin"
            dependencies = next(line for line in output.splitlines() if line.startswith(target + ":"))
            self.assertIn("include/nmfd/time_steppers/bdf/implicit_time_step.h", dependencies)

    def test_grouped_dirk_inventory_and_dependencies(self):
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        cpu, cuda = "timestepper-implicit-dirk-cpu", "timestepper-implicit-dirk-cuda"
        groups = ["dirk-core", "dirk-extensions"]
        self.assertEqual(jobs[cpu]["strategy"]["matrix"], {"group": groups, "platform": ["SERIAL", "OMP"]})
        self.assertEqual(jobs[cuda]["strategy"]["matrix"], {"group": groups})
        self.assertIn(cpu, jobs[cuda]["needs"])
        self.assertIn("scfd-vector-operations-cuda", jobs[cuda]["needs"])
        self.assertIn(cuda, jobs["all-tests-passed"]["needs"])
        for dependent in ("timestepper-implicit-sanitizers-cpu", "periodic-orbit-sections-cpu",
                          "timestepper-pde-cpu", "timestepper-bdf-cpu", "timestepper-imex-cpu"):
            self.assertIn(cpu, jobs[dependent]["needs"])
        for job in (cpu, cuda):
            self.assertFalse(jobs[job]["strategy"]["fail-fast"])
            commands = "\n".join(step.get("run", "") for step in jobs[job]["steps"])
            self.assertIn("make test-rk PLATFORM=", commands)
            self.assertIn("RK_TEST_OPTIONS=\"--group ", commands)
            self.assertTrue(any(step.get("if") == "always()" and "summary.json" in step["with"]["path"]
                                for step in jobs[job]["steps"] if step.get("uses", "").startswith("actions/upload-")))

        expected = set(("dirk adaptive mass_dae lifecycle nonautonomous lorenz rossler vdp "
                        "esdirk esdirk_adaptive esdirk_mass_dae dense_output dense_lifecycle dense_mass_dae "
                        "globalization nonnormal_steady_state lorenz_steady_state rossler_steady_state "
                        "vdp_steady_state").split())
        for platform, suffix in (("SERIAL", "cpu"), ("OMP", "cpu_omp"), ("CUDA", "cuda")):
            targets = []
            for group in groups:
                listing = subprocess.check_output(
                    ["bash", str(MODULE/"tests/run_all.sh"), "--profile", "ci", "--group", group, "--list"],
                    env=dict(os.environ, PLATFORM=platform, PYTHON=sys.executable), text=True)
                targets.extend(line.split()[1] for line in listing.splitlines())
            baseline = [f"test_implicit_{name}_{suffix}.bin" for name in expected]
            if platform == "SERIAL":
                baseline.append("test_rk_tables_implicit.bin")
            self.assertCountEqual(targets, baseline)
            for directory in (ROOT, MODULE):
                output = self.make(directory, *targets, "BUILD_DIR=build/dirk_inventory_probe")
                for name in expected:
                    self.assertIn(f"tests/test_implicit_{name}.cpp", output)
        sanitizer = self.sanitizer_cases(jobs["timestepper-implicit-sanitizers-cpu"])
        self.assertTrue({f"implicit_{name}" for name in expected} <= sanitizer)
        self.assertIn("explicit_nonautonomous", sanitizer)

    def test_imex_execution_is_built_standalone_and_required_by_ci(self):
        tests = ("imex", "imex_adaptive", "imex_mass", "imex_lifecycle", "imex_backend", "imex_benchmarks",
                 "imex_dense_output", "imex_dense_lifecycle")
        for directory in (ROOT, MODULE):
            for test in tests:
                output = self.make(directory, f"test_implicit_{test}_cpu_omp.bin",
                                   "BUILD_DIR=build/imex_probe")
                self.assertIn(f"tests/test_implicit_{test}.cpp", output)
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        self.check_family_workflow(jobs, "imex", tests)
        sanitizer = jobs["timestepper-implicit-sanitizers-cpu"]
        self.assertIn("timestepper-imex-cpu", sanitizer["needs"])
        self.assertTrue({"implicit_" + test for test in tests} <= self.sanitizer_cases(sanitizer))

    def check_family_workflow(self, jobs, family, tests):
        cpu_job = f"timestepper-{family}-cpu"
        self.assertEqual(jobs[cpu_job]["strategy"]["matrix"]["compiler"], ["g++", "clang++"])
        for backend in ("cpu", "cuda", "hip"):
            name = f"timestepper-{family}-{backend}"
            job = jobs[name]
            command = f"bash scripts/ci/run_timestepper_family.sh {family} {backend.upper()}"
            if backend == "cpu":
                command += " '${{ matrix.compiler }}'"
            self.assertIn(command, [s.get("run") for s in job["steps"]])
            self.assertTrue(any(s.get("if") == "always()" and "summary.json" in s["with"]["path"]
                                for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact")))
            if backend != "cpu":
                self.assertIn(name, jobs["all-tests-passed"]["needs"])
                self.assertIn(cpu_job, job["needs"])
        for platform, backend in (("SERIAL", "cpu"), ("OMP", "cpu_omp"), ("CUDA", "cuda"), ("HIP", "hip")):
            listing = subprocess.check_output(
                ["bash", str(MODULE/"tests/run_all.sh"), "--profile", "ci", "--group", family, "--list"],
                env=dict(os.environ, PLATFORM=platform, PYTHON=sys.executable), text=True)
            self.assertCountEqual([line.split()[1] for line in listing.splitlines()],
                                  [f"test_implicit_{test}_{backend}.bin" for test in tests])

    def test_cpu_family_checks_headers_and_both_platforms_even_if_serial_fails(self):
        harness = """
            make() {
                printf '<%s>' "$@"; printf '\\n'
                [[ " $* " != *' PLATFORM=SERIAL '* ]]
            }
            export -f make
            bash scripts/ci/run_timestepper_family.sh bdf CPU "$COMPILER"
        """
        for compiler in ("g++", "clang++"):
            env = dict(os.environ, BUILD_DIR="build/mock", COMPILER=compiler)
            result = subprocess.run(["bash", "-c", harness], env=env, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            header, serial, omp = result.stdout.splitlines()
            self.assertTrue(header.startswith("<check_timestepper_headers_cpu>"))
            for line in (header, serial, omp):
                self.assertIn(f"<G++ = {compiler}>", line)
                self.assertIn(f"<BUILD_DIR=build/mock/bdf/{compiler}>", line)
            for platform, line in (("SERIAL", serial), ("OMP", omp)):
                self.assertIn(f"<test-timesteppers><PLATFORM={platform}>", line)
                self.assertIn("<TIMESTEPPER_TEST_OPTIONS=--group bdf>", line)
            invocation = self.make(ROOT, "test-timesteppers", f"G++ = {compiler}", "PLATFORM=OMP",
                                   "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc")
            self.assertIn(f'--make-option "G++ = {compiler}"', invocation)

    def test_gpu_family_passes_site_settings_and_propagates_suite_failure(self):
        harness = """
            local_nvcc() { echo nvcc; }
            local_hipcc() { echo AMD; }
            make() { printf '<%s>' "$@"; return 9; }
            export -f make local_nvcc local_hipcc
            bash scripts/ci/run_timestepper_family.sh bdf "$PLATFORM"
        """
        for platform in ("CUDA", "HIP"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                env = dict(os.environ, BUILD_DIR=directory, CONFIG_FILE="site.inc", PLATFORM=platform,
                           GPU_POLICY="required", NVCC="local_nvcc", HIPCC="local_hipcc",
                           ROCM_PATH="/opt/site-rocm", HIP_ARCH_LIST="gfx90a")
                result = subprocess.run(["bash", "-c", harness], cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 9, result.stderr)
                self.assertIn(f"<test-timesteppers><PLATFORM={platform}><CONFIG_FILE=site.inc>", result.stdout)
                self.assertIn("<TIMESTEPPER_TEST_OPTIONS=--group bdf>", result.stdout)
                if platform == "CUDA":
                    self.assertIn("<NVCC=local_nvcc>", result.stdout)
                else:
                    for value in ("HIPCC=local_hipcc", "HIP_ROOT_PATH=/opt/site-rocm", "HIP_PATH=/opt/site-rocm",
                                  "HIP_ARCH_LIST=gfx90a"):
                        self.assertIn(f"<{value}>", result.stdout)

    def test_filter_adaptation_is_standalone_and_gates_implicit_tests(self):
        for directory in (ROOT, MODULE):
            output = self.make(directory, "test_explicit_filter_adaptation_cpu_omp.bin",
                               "BUILD_DIR=build/filter_probe")
            self.assertIn("tests/test_explicit_filter_adaptation.cpp", output)
            self.assertNotIn("nmfd-newton/include", output)
            self.assertNotIn("nmfd-linsolvers/include", output)
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        self.assertIn("timestepper-explicit-rk-cpu", jobs["timestepper-implicit-dirk-cpu"]["needs"])
        self.assertIn("timestepper-explicit-cuda", jobs["all-tests-passed"]["needs"])
        self.assertIn("explicit_filter_adaptation", self.sanitizer_cases(jobs["timestepper-implicit-sanitizers-cpu"]))

    def test_nonautonomous_matrix_preserves_explicit_independence(self):
        for mode in ("explicit", "implicit"):
            output = self.make(ROOT, f"test_{mode}_nonautonomous_cpu.bin",
                               "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc",
                               "BUILD_DIR=build/nonautonomous_probe")
            self.assertIn(f"tests/test_{mode}_nonautonomous.cpp", output)
            if mode == "explicit":
                self.assertNotIn("nmfd-newton/include", output)
                self.assertNotIn("nmfd-linsolvers/include", output)
            else:
                self.assertIn("nmfd-newton/include", output)

    def test_visual_benchmark_runner_is_standalone_and_tested_in_ci(self):
        for backend in ("cpu", "cpu_omp", "cuda"):
            output = self.make(MODULE, f"ode_benchmarks_{backend}.bin", "BUILD_DIR=build/benchmark_probe")
            self.assertIn("tests/ode_benchmarks.cpp", output)
            self.assertIn("-MMD -MP -MF", output)
            self.assertIn("nmfd-newton/include", output)
            self.assertIn("nmfd-linsolvers/include", output)
        workflow = (ROOT / ".github/workflows/operations.yml").read_text()
        self.assertIn("python3 source/contrib/nmfd-timesteppers/tests/test_ode_benchmarks.py", workflow)
        self.assertIn('"$PYTHON" source/contrib/nmfd-timesteppers/tests/test_ode_benchmarks.py', workflow)
        self.assertIn("--methods ROS2 ROS3 ROS34PW2 --controllers scipy", workflow)
        self.assertIn("--systems lorenz rossler vdp vdp_stiff time_coefficient stiff_tracking state_mass time_mass", workflow)


if __name__ == "__main__":
    unittest.main()
