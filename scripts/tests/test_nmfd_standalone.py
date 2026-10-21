import importlib.util
from graphlib import TopologicalSorter
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
import sys


spec = importlib.util.spec_from_file_location(
    "standalone", Path(__file__).resolve().parents[1] / "validate_nmfd_standalone.py"
)
standalone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standalone)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run_local_ci as ci


class NmfdStandaloneTest(unittest.TestCase):
    def test_environment_cannot_select_parent_dependencies(self):
        environment = dict.fromkeys(standalone.BUILD_OVERRIDES, "/wrong/parent/tree")
        environment.update(PATH="/bin", BOOST_ROOT="/opt/boost", MPICXX="/opt/mpi/bin/mpicxx")
        self.assertEqual(standalone.clean_environment(environment), {
            "PATH": "/bin", "BOOST_ROOT": "/opt/boost", "MPICXX": "/opt/mpi/bin/mpicxx",
        })

    def test_commands_keep_dependency_defaults_and_cover_the_selected_suite(self):
        for compiler in ("gcc", "clang"):
            for suite, libraries in standalone.TARGETS.items():
                for library, targets in libraries.items():
                    with self.subTest(library=library, compiler=compiler, suite=suite):
                        command = standalone.make_command(library, compiler, suite, 2, 4)
                        for target in targets:
                            self.assertIn(target, command)
                        self.assertIn("MPI_NP=4", command)
                        self.assertIn("CXX=" + ("g++" if compiler == "gcc" else "clang++"), command)
                        self.assertFalse(any(word.split("=", 1)[0].endswith("_ROOT") for word in command))
        with self.assertRaises(ValueError):
            standalone.make_command("nmfd-linsolvers", "gcc", "mpi", 2, 1)

    def test_ci_has_separate_standalone_gates_and_required_summary(self):
        workflow = ci.load_workflow(Path(__file__).resolve().parents[2] / ".github/workflows/operations.yml")
        jobs = workflow["jobs"]
        graph = {}
        for name, job in jobs.items():
            dependencies = job.get("needs", [])
            graph[name] = [dependencies] if isinstance(dependencies, str) else dependencies
        self.assertEqual(set(TopologicalSorter(graph).static_order()), set(jobs))
        self.assertEqual(jobs["nmfd-standalone-cpu"]["strategy"]["matrix"], {
            "library": list(standalone.LIBRARIES), "compiler": ["gcc", "clang"],
        })
        self.assertEqual(jobs["nmfd-standalone-mpi"]["strategy"]["matrix"], {
            "library": ["nmfd-operations", "nmfd-newton"], "ranks": [1, 2, 4],
        })
        required = set()
        pending = ["nmfd-migration-validation-cpu"]
        while pending:
            name = pending.pop()
            self.assertIn(name, jobs)
            if name in required:
                continue
            required.add(name)
            dependencies = jobs[name].get("needs", [])
            pending.extend([dependencies] if isinstance(dependencies, str) else dependencies)
        for name in ("nmfd-standalone-cpu", "nmfd-standalone-mpi", "operations-dependency-boundaries",
                     "ks1d-sym-cpu-omp", "ks1d-full-cpu-omp", "ks2d-cpu-omp",
                     "nmfd-multiprecision-cpu", "workflow-ownership-sanitizers-cpu"):
            self.assertIn(name, required)
        for name in required:
            self.assertIn(name, jobs)

    def test_hip_link_commands_do_not_parse_objects_as_sources(self):
        root = Path(__file__).resolve().parents[2]
        for target in ("circle_bd_hip", "circle_stability_hip", "star_shaped_bd_hip"):
            with self.subTest(target=target):
                result = subprocess.run(
                    ["make", "--no-print-directory", "-n", target,
                     "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc", "BUILD_DIR=build/dependency_probe",
                     "HIPCC=hipcc", "HIP_EXTENDED_LAMBDA="],
                    cwd=root, env=standalone.clean_environment(os.environ),
                    check=True, text=True, capture_output=True, timeout=30,
                )
                commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("hipcc ")]
                main_object = "build/dependency_probe/" + target + "_main.o"
                compile_command = next(command for command in commands if command[-2] == main_object)
                link_command = next(command for command in commands if command[-2].endswith(target + ".bin"))
                self.assertIn("-c", compile_command)
                self.assertEqual([word for word in compile_command if word.endswith(".o")], [main_object])
                self.assertIn(main_object, link_command)
                self.assertIn("build/dependency_probe/gpu_reduction_ogita_kernels_hip.o", link_command)
                self.assertIn("-lhipblas", link_command)
                self.assertIn("-lhipsolver", link_command)
                self.assertFalse(any(word.endswith(".cpp") for word in link_command))

    def test_summary_fails_for_failed_skipped_or_cancelled_required_groups(self):
        root = Path(__file__).resolve().parents[2]
        workflow = ci.load_workflow(root / ".github/workflows/operations.yml")
        job = workflow["jobs"]["nmfd-migration-validation-cpu"]
        self.assertEqual(job["if"], "always()")
        for foundation in ("success", "failure", "skipped", "cancelled"):
            for application in ("success", "failure", "skipped", "cancelled"):
                with self.subTest(foundation=foundation, application=application):
                    env = dict(os.environ, FOUNDATION_RESULT=foundation, APPLICATION_RESULT=application)
                    result = subprocess.run(["bash", "-e", "-c", job["steps"][0]["run"]],
                                            env=env, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode == 0, foundation == application == "success")

    def test_hip_circle_gate_covers_closed_curve_stability_and_restart(self):
        root = Path(__file__).resolve().parents[2]
        workflow = ci.load_workflow(root / ".github/workflows/operations.yml")
        commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["circle-bd-hip"]["steps"])
        for expected in ("circle_stability_hip", "--require-closed-circle", "--expected-bifurcations 2",
                         'cmp "$CIRCLE_BUILD/stability_diagram.before_restart.dat"',
                         'cmp "$CIRCLE_BUILD/stability_curve.before_restart.dat"'):
            self.assertIn(expected, commands)

    def test_old_committed_pins_win_over_dirty_newer_working_trees(self):
        with tempfile.TemporaryDirectory(prefix="nmfd-standalone-test-") as directory:
            root = Path(directory) / "project"
            root.mkdir()
            env = standalone.clean_environment(os.environ)
            env["GIT_ALLOW_PROTOCOL"] = "file"

            def git(repository, *arguments):
                return standalone.git(
                    repository, "-c", "user.name=NMFD CI Test", "-c", "user.email=ci@example.invalid",
                    "-c", "commit.gpgsign=false", *arguments, env=env,
                )

            git(root, "init", "-q")
            contrib = root / "source/contrib"
            scfd = contrib / "scfd"
            scfd.mkdir(parents=True)
            git(scfd, "init", "-q")
            (scfd / "value").write_text("pinned dependency\n")
            git(scfd, "add", "value")
            git(scfd, "commit", "-qm", "Pinned dependency")
            dependency_pin = git(scfd, "rev-parse", "HEAD")
            (scfd / "value").write_text("newer dependency\n")
            git(scfd, "commit", "-qam", "Newer dependency")
            dependency_tip = git(scfd, "rev-parse", "HEAD")

            library = contrib / "nmfd-operations"
            library.mkdir()
            git(library, "init", "-q")
            (library / "value").write_text("pinned library\n")
            (library / ".gitmodules").write_text(
                '[submodule "contrib/scfd"]\npath = contrib/scfd\nurl = ' + scfd.as_uri() + "\n"
            )
            git(library, "add", "value", ".gitmodules")
            git(library, "update-index", "--add", "--cacheinfo", "160000," + dependency_pin + ",contrib/scfd")
            git(library, "commit", "-qm", "Pinned library")
            library_pin = git(library, "rev-parse", "HEAD")
            git(root, "update-index", "--add", "--cacheinfo",
                "160000," + library_pin + ",source/contrib/nmfd-operations")
            git(root, "commit", "-qm", "Pinned integration")

            (library / "value").write_text("newer library\n")
            git(library, "commit", "-qam", "Newer library")
            (library / "value").write_text("uncommitted library\n")
            (scfd / "value").write_text("uncommitted dependency\n")
            original_status = git(root, "status", "--porcelain")
            checkout = Path(directory) / "isolated"
            selected = standalone.prepare_checkout(root, "nmfd-operations", checkout, env)
            self.assertEqual(selected, {"revision": library_pin, "dependencies": {"contrib/scfd": dependency_pin}})
            self.assertEqual((checkout / "value").read_text(), "pinned library\n")
            self.assertEqual((checkout / "contrib/scfd/value").read_text(), "pinned dependency\n")
            self.assertEqual(git(scfd, "rev-parse", "HEAD"), dependency_tip)
            self.assertEqual((library / "value").read_text(), "uncommitted library\n")
            self.assertEqual((scfd / "value").read_text(), "uncommitted dependency\n")
            self.assertEqual(git(root, "status", "--porcelain"), original_status)
            standalone.verify_checkout(checkout, selected, env)

            shallow = Path(directory) / "shallow"
            subprocess.run(["git", "clone", "--quiet", "--depth=1", scfd.as_uri(), str(shallow)],
                           env=env, check=True)
            with self.assertRaises(subprocess.CalledProcessError):
                git(shallow, "cat-file", "-e", dependency_pin + "^{commit}")
            fetched = Path(directory) / "fetched-pin"
            standalone.clone_revision(shallow, dependency_pin, fetched, env, scfd.as_uri())
            self.assertEqual(git(fetched, "rev-parse", "HEAD"), dependency_pin)

            (checkout / "value").write_text("unexpected build modification\n")
            with self.assertRaises(subprocess.CalledProcessError):
                standalone.verify_checkout(checkout, selected, env)
            git(checkout, "restore", "value")
            git(checkout / "contrib/scfd", "checkout", "--detach", "-q", dependency_tip)
            with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                standalone.verify_checkout(checkout, selected, env)


if __name__ == "__main__":
    unittest.main()
