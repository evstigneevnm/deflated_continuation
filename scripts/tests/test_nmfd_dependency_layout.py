import configparser
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_local_ci as ci


CONTRIB = ROOT / "source/contrib"
ROOTS = {
    "SCFD_ROOT": str(CONTRIB / "scfd"),
    "NMFD_OPERATIONS_ROOT": str(CONTRIB / "nmfd-operations"),
    "NMFD_LINSOLVERS_ROOT": str(CONTRIB / "nmfd-linsolvers"),
}
LIBRARIES = {
    "nmfd-operations": ("operations.yml", "scfd"),
    "nmfd-linsolvers": ("ci.yml", "scfd"),
    "nmfd-newton": ("newton.yml", "SCFD"),
}


class NmfdDependencyLayoutTest(unittest.TestCase):
    def make_roots(self, library, overrides=None):
        env = os.environ.copy()
        for name in (*ROOTS, "CONFIG_FILE", "BUILD_DIR", "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES"):
            env.pop(name, None)
        env.update(overrides or {})
        values = " ".join(f"'$(abspath $({name}))'" for name in ROOTS)
        result = subprocess.run(
            ["make", "--no-print-directory", "-s", "-C", str(CONTRIB / library),
             f"--eval=print_dependency_roots:;@printf '%s\\n' {values}", "print_dependency_roots"],
            env=env, check=True, text=True, capture_output=True,
        )
        return dict(zip(ROOTS, result.stdout.splitlines()))

    def test_standalone_defaults_use_direct_dependencies(self):
        for library, (_, scfd) in LIBRARIES.items():
            with self.subTest(library=library):
                paths = self.make_roots(library)
                local = CONTRIB / library / "contrib"
                self.assertEqual(paths["SCFD_ROOT"], str(local / scfd))
                if library != "nmfd-operations":
                    self.assertEqual(paths["NMFD_OPERATIONS_ROOT"], str(local / "nmfd-operations"))
                if library == "nmfd-newton":
                    self.assertEqual(paths["NMFD_LINSOLVERS_ROOT"], str(local / "nmfd-linsolvers"))

    def test_parent_overrides_select_canonical_checkouts(self):
        for library in LIBRARIES:
            with self.subTest(library=library):
                self.assertEqual(self.make_roots(library, ROOTS), ROOTS)

    def test_parent_workflow_initializes_only_direct_dependencies(self):
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        expanded = {
            key: ci.expand_expressions(workflow["env"][key], {}, ROOT) for key in ROOTS
        }
        self.assertEqual(expanded, ROOTS)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    setting = step.get("with", {}).get("submodules")
                    self.assertIn(setting, (None, True))

    def test_dense_solver_jobs_select_module_local_configs(self):
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        for name in ("nmfd-dense-solve-cpu", "nmfd-direct-solver-cuda"):
            for step in workflow["jobs"][name]["steps"]:
                commands = step.get("run", "").replace("\\\n", " ")
                for command in commands.splitlines():
                    words = shlex.split(command)
                    if words[:2] != ["make", "-C"]:
                        continue
                    with self.subTest(job=name, command=command):
                        configs = [word.split("=", 1)[1] for word in words
                                   if word.startswith("CONFIG_FILE=")]
                        self.assertEqual(len(configs), 1, "module builds must override the parent config")
                        self.assertTrue((ROOT / words[2] / configs[0]).is_file())

    def test_bratu_jobs_bound_nested_parallelism(self):
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        for name in ("bratu-bd-cpu-omp", "bratu-bd-fd3-cpu-omp"):
            with self.subTest(job=name):
                job = workflow["jobs"][name]
                env = ci.merge_env(workflow.get("env"), job.get("env"))
                self.assertEqual(env["OMP_NUM_THREADS"], "2")
                self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
                for step in job["steps"]:
                    self.assertNotRegex(step.get("run", ""),
                                        r"(?m)^\s*(?:export\s+)?(?:OMP_NUM_THREADS|OPENBLAS_NUM_THREADS)\s*=")

    def test_linear_solver_headers_have_one_canonical_owner(self):
        header = Path("include/nmfd/solvers/dense1_extended_solver.h")
        self.assertTrue((CONTRIB / "nmfd-linsolvers" / header).is_file())
        self.assertFalse((CONTRIB / "nmfd-newton" / header).exists())
        newton_headers = {path.relative_to(CONTRIB / "nmfd-newton/include")
                          for path in (CONTRIB / "nmfd-newton/include").rglob("*.h")}
        linear_headers = {path.relative_to(CONTRIB / "nmfd-linsolvers/include")
                          for path in (CONTRIB / "nmfd-linsolvers/include").rglob("*.h")}
        self.assertFalse(newton_headers & linear_headers)

    def test_migrated_regressions_do_not_bind_legacy_solvers(self):
        for filename in ("source/models/KS_1D/test_KS1D_full_operator.cpp",
                         "source/stability/tests/test_stability_continuation_interface.cpp",
                         "source/main/deflation_continuation.hpp"):
            with self.subTest(filename=filename):
                self.assertNotIn("numerical_algos/lin_solvers/", (ROOT / filename).read_text())
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        commands = "\n".join(step.get("run", "") for step in
                             workflow["jobs"]["nmfd-dense1-extended-cpu"]["steps"])
        self.assertIn("check_dense1_extended_solver_cpu", commands)

    def test_multiprecision_workflows_use_canonical_solvers(self):
        for filename in ("source/models/circle/circle_bd.cpp",
                         "source/models/bratu/bratu_linear_solvers.h",
                         "source/models/star_shaped/star_shaped_linear_solvers.h"):
            with self.subTest(filename=filename):
                self.assertNotIn("numerical_algos/lin_solvers/", (ROOT / filename).read_text())
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        for job in ("circle-bd-var-prec", "bratu-bd-var-prec", "star-shaped-bd-var-prec"):
            self.assertIn("nmfd-multiprecision-cpu", workflow["jobs"][job]["needs"])

    def test_local_runner_expands_workflow_and_step_environment(self):
        workflow = {"env": {name: "${{ github.workspace }}/source/contrib/" + Path(path).name
                            for name, path in ROOTS.items()}}
        job = {"steps": [{"run": "true", "env": {"SELECTED_SCFD": "${{ env.SCFD_ROOT }}"}}]}
        with mock.patch.object(ci.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            result = ci.run_job(ROOT, workflow, "roots", job, {}, None, False)
        self.assertEqual(result.status, "PASS")
        environment = run.call_args.kwargs["env"]
        for name, path in ROOTS.items():
            self.assertEqual(environment[name], path)
        self.assertEqual(environment["SELECTED_SCFD"], ROOTS["SCFD_ROOT"])

    def test_local_runner_preserves_explicit_overrides(self):
        workflow = {"env": {"SCFD_ROOT": "${{ github.workspace }}/source/contrib/scfd"}}
        job = {"steps": [{"run": "true"}]}
        with mock.patch.object(ci.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            ci.run_job(ROOT, workflow, "override", job, {"SCFD_ROOT": "/custom/scfd"}, None, False)
        self.assertEqual(run.call_args.kwargs["env"]["SCFD_ROOT"], "/custom/scfd")

    def test_standalone_ci_separates_pinned_and_latest(self):
        for library, (filename, scfd) in LIBRARIES.items():
            with self.subTest(library=library):
                workflow = ci.load_workflow(CONTRIB / library / ".github/workflows" / filename)
                local = CONTRIB / library
                self.assertEqual(
                    ci.expand_expressions(workflow["env"]["SCFD_ROOT"], {}, local),
                    str(local / "contrib" / scfd),
                )
                events = workflow.get("on", workflow.get(True))
                self.assertIn("schedule", events)
                latest = workflow["jobs"]["latest-dependencies"]
                commands = "\n".join(step.get("run", "") for step in latest["steps"])
                self.assertIn("git submodule update --init --remote --no-fetch --checkout", commands)
                self.assertIn("git submodule status", commands)
                for forbidden in ("--recursive", "git add", "git commit", "git push"):
                    self.assertNotIn(forbidden, commands)
                for name, job in workflow["jobs"].items():
                    for step in job["steps"]:
                        if step.get("uses", "").startswith("actions/checkout@"):
                            self.assertIs(step["with"]["submodules"], True)
                        if name != "latest-dependencies":
                            self.assertNotIn("--remote", step.get("run", ""))

    def test_latest_dependency_selection_from_shallow_clones(self):
        env = dict(os.environ, GIT_ALLOW_PROTOCOL="file", GIT_TERMINAL_PROMPT="0")
        with tempfile.TemporaryDirectory(prefix="nmfd-latest-") as directory:
            root = Path(directory)

            def git(cwd, *args):
                return subprocess.run(
                    ["git", "-c", "user.name=NMFD CI Test", "-c", "user.email=ci@example.invalid",
                     "-c", "commit.gpgsign=false", *args],
                    cwd=cwd, env=env, check=True, text=True, capture_output=True, timeout=30,
                ).stdout.strip()

            dependencies = {}
            for i, branch in enumerate(("master", "nevstigneev", "nevstigneev/sync_with_bifurcator")):
                remote = root / f"remote-{i}"
                git(root, "init", "-q", "-b", "master", str(remote))
                git(remote, "commit", "-qm", "Pinned dependency", "--allow-empty")
                pinned = git(remote, "rev-parse", "HEAD")
                if branch != "master":
                    git(remote, "checkout", "-qb", branch, pinned)
                git(remote, "commit", "-qm", "Latest " + branch, "--allow-empty")
                dependencies[branch] = (remote, pinned, git(remote, "rev-parse", "HEAD"))
                git(remote, "checkout", "-q", "master")

            for library, (filename, _) in LIBRARIES.items():
                with self.subTest(library=library):
                    parent = root / library
                    git(root, "init", "-q", "-b", "master", str(parent))
                    config = configparser.ConfigParser(interpolation=None)
                    config.read(CONTRIB / library / ".gitmodules")
                    for section in config.sections():
                        name = shlex.split(section)[1]
                        path = config[section]["path"]
                        remote, pinned, _ = dependencies[config[section]["branch"]]
                        git(parent, "submodule", "add", "-q", "--name", name, remote.as_uri(), path)
                        git(parent / path, "checkout", "--detach", "-q", pinned)
                        git(parent, "config", "-f", ".gitmodules", f"submodule.{name}.branch",
                            config[section]["branch"])
                    git(parent, "add", ".")
                    git(parent, "commit", "-qm", "Pinned direct dependencies")
                    checkout = root / (library + "-shallow")
                    git(root, "clone", "-q", "--depth=1", parent.as_uri(), str(checkout))
                    git(checkout, "submodule", "update", "--init", "--depth=1")
                    for section in config.sections():
                        path = checkout / config[section]["path"]
                        self.assertEqual(git(path, "rev-parse", "--is-shallow-repository"), "true")
                        self.assertEqual(
                            git(path, "for-each-ref", "--format=%(refname)", "refs/remotes/origin").splitlines(),
                            ["refs/remotes/origin/HEAD", "refs/remotes/origin/master"],
                        )
                    head = git(checkout, "rev-parse", "HEAD")
                    index = git(checkout, "ls-files", "--stage")
                    workflow = ci.load_workflow(CONTRIB / library / ".github/workflows" / filename)
                    steps = workflow["jobs"]["latest-dependencies"]["steps"]
                    command = next(step["run"] for step in steps
                                   if step.get("name") == "Select designated dependency branch tips")
                    result = subprocess.run(
                        ["bash", "-eo", "pipefail", "-c", command], cwd=checkout, env=env,
                        check=True, text=True, capture_output=True, timeout=30,
                    )
                    for section in config.sections():
                        expected = dependencies[config[section]["branch"]][2]
                        self.assertEqual(git(checkout / config[section]["path"], "rev-parse", "HEAD"), expected)
                        self.assertIn(expected, result.stdout)
                    self.assertEqual(git(checkout, "rev-parse", "HEAD"), head)
                    self.assertEqual(git(checkout, "ls-files", "--stage"), index)

    def test_all_direct_dependencies_have_explicit_tracking_branches(self):
        branches = {"scfd": "nevstigneev", "SCFD": "nevstigneev", "nmfd-operations": "master",
                    "nmfd-linsolvers": "nevstigneev/sync_with_bifurcator"}
        for library in LIBRARIES:
            config = configparser.ConfigParser(interpolation=None)
            config.read(CONTRIB / library / ".gitmodules")
            for section in config.sections():
                with self.subTest(library=library, dependency=section):
                    dependency = Path(config[section]["path"]).name
                    self.assertEqual(config[section]["branch"], branches[dependency])


if __name__ == "__main__":
    unittest.main()
