import configparser
import os
from pathlib import Path
import shlex
import subprocess
import sys
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
                self.assertIn("git submodule update --init --remote --checkout", commands)
                self.assertIn("git submodule status", commands)
                for forbidden in ("--recursive", "git add", "git commit", "git push"):
                    self.assertNotIn(forbidden, commands)
                for name, job in workflow["jobs"].items():
                    for step in job["steps"]:
                        if step.get("uses", "").startswith("actions/checkout@"):
                            self.assertIs(step["with"]["submodules"], True)
                        if name != "latest-dependencies":
                            self.assertNotIn("--remote", step.get("run", ""))

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
