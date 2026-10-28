"""Protect the expensive reference gate and its sampling-independent comparison."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_local_ci as ci
import validate_ks1d_reference as reference


class ReferenceWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]

    def test_final_gate_covers_every_other_job_without_cycles(self):
        visited, active = set(), set()

        def visit(name):
            self.assertIn(name, self.jobs)
            self.assertNotIn(name, active, f"Cyclic CI dependency: {name}")
            if name in visited:
                return
            active.add(name)
            needs = self.jobs[name].get("needs", [])
            for dependency in ([needs] if isinstance(needs, str) else needs):
                visit(dependency)
            active.remove(name)
            visited.add(name)

        visit("all-tests-passed")
        self.assertEqual(visited, set(self.jobs) - {"ks1d-n64-reference-cpu-omp"})
        heavy = self.jobs["ks1d-n64-reference-cpu-omp"]
        self.assertEqual(heavy["needs"], "all-tests-passed")
        self.assertNotIn("if", heavy)
        self.assertNotIn("if", self.jobs["all-tests-passed"])

    def test_pinned_asset_threads_and_failure_artifacts(self):
        job = self.jobs["ks1d-n64-reference-cpu-omp"]
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            self.assertEqual(str(job["env"][name]), "1")
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn("download_reference_data.py --version 20260730", commands)
        self.assertIn('--destination "$BUILD_DIR/ks1d_n64_reference/asset"', commands)
        self.assertIn("test_KS1D_reference_cpu_omp.bin", commands)
        self.assertIn("REAL_TYPE=double", commands)
        self.assertIn("scripts/validate_ks1d_reference.py", commands)
        upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
        self.assertEqual(upload["if"], "always()")
        self.assertIn("*.log", upload["with"]["path"])
        self.assertIn("*.json", upload["with"]["path"])
        early = "\n".join(step.get("run", "") for step in self.jobs["operations-dependency-boundaries"]["steps"])
        self.assertIn("-p test_ks1d_reference.py", early)


class ReferenceComparisonTest(unittest.TestCase):
    def test_production_run_requires_augmented_selection(self):
        newton = "Newton solver: NMFD\n"
        augmented = "Continuation linear solver: augmented BiCGStab(L)\n"
        for output, accepted in ((newton, False), (newton + augmented, True),
                                 (newton + augmented + "ERROR: failed\n", False)):
            with self.subTest(output=output), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)

                def simulate(command, stdout, **kwargs):
                    stdout.write(output)
                    return reference.subprocess.CompletedProcess(command, 0)

                with patch.object(reference.subprocess, "run", side_effect=simulate):
                    if accepted:
                        reference.run(["test"], "initial", work, 1, {})
                    else:
                        with self.assertRaises(RuntimeError):
                            reference.run(["test"], "initial", work, 1, {})

    def test_assignment_ignores_curve_order_but_requires_distinct_branches(self):
        self.assertEqual(reference.assign_branches([[1., 1e-7], [1e-6, 1.]]), (1, 0))
        for costs in ([[1e-7, 1.], [1e-6, 1.]], [[math.inf]], [[math.nan]], []):
            with self.subTest(costs=costs), self.assertRaises(RuntimeError):
                reference.assign_branches(costs)

    def test_interpolation_does_not_require_equal_point_counts(self):
        target = reference.State(2., (1, 1), Path("reference"), [2., 0.])
        curve = reference.Curve("arbitrary_id", [], [
            reference.State(1., (1, 1), Path("a"), [1., 0.]),
            reference.State(3., (1, 1), Path("b"), [3., 0.]),
        ], [])
        error, predictor = reference.predictor(curve, target)
        self.assertEqual(error, 0.)
        self.assertEqual(predictor, target.values)
        curve.states[1].segment = (2, 2)
        self.assertEqual(reference.predictor(curve, target)[0], math.inf)

    def test_subperiodic_discrete_translation_retains_odd_layout(self):
        target = reference.State(2., (1, 1), Path("reference"), [0., 2.])
        curve = reference.Curve("other", [], [reference.State(2., (1, 1), Path("a"), [0., -2.])], [])
        self.assertLess(reference.predictor(curve, target)[0], 1e-14)
        curve.states[0].values = [1., -2.]
        self.assertGreater(reference.predictor(curve, target)[0], .1)

    def test_coverage_detects_missing_sections_without_requiring_same_sampling(self):
        archived = reference.Curve("a", [[1., 0.], [2., 1.], [3., 0.]], [], [])
        fine = reference.Curve("b", [[1., 0.], [1.5, .5], [2., 1.], [2.5, .5], [3., 0.]], [], [])
        self.assertLess(reference.diagram_coverage(archived, fine, (2., 1.)), 1e-14)
        missing = reference.Curve("b", [[1., 0.], [3., 0.]], [], [])
        self.assertEqual(reference.diagram_coverage(archived, missing, (2., 1.)), 1.)
        # A jump between semicurves must not masquerade as a covered segment.
        jump = reference.Curve("b", [[1., 0.], [2., 1.], [3., 0.]], [], [], [(1, 1), (2, 2), (2, 2)])
        self.assertGreater(reference.diagram_coverage(archived, jump, (2., 1.)), .5)

    def make_curve(self, directory):
        path = directory / "7"
        path.mkdir()
        (path / "metadata_curve.dat").write_text(
            "# index lambda saved id segment semicurve forced reason\n"
            "0 1 1 1 1 1 1 boundary_min\n1 3 1 2 2 2 1 boundary_max\n")
        (path / "debug_curve_all.dat").write_text("1 0.7071067811865475 1 0 1\n3 1.414213562373095 2 0 2\n")
        (path / "1").write_text("1\n0\n")
        (path / "2").write_text("2\n0\n")
        return path

    def test_missing_nonfinite_or_wrong_size_state_is_rejected(self):
        for contents in (None, "nan\n0\n", "1\n", "0\n0\n"):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as directory:
                path = self.make_curve(Path(directory))
                if contents is None:
                    (path / "1").unlink()
                else:
                    (path / "1").write_text(contents)
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    reference.read_curve(path, 2, [1., 3.])

    def test_invalid_endpoints_and_indices_are_rejected(self):
        for old, new in (("boundary_max", "max_steps"), ("1 3 1 2", "5 3 1 2"),
                         ("1 3 1 2", "1 2 1 2"), ("1 3 1 2", "1 3 0 0")):
            with self.subTest(new=new), tempfile.TemporaryDirectory() as directory:
                path = self.make_curve(Path(directory))
                metadata = path / "metadata_curve.dat"
                metadata.write_text(metadata.read_text().replace(old, new))
                with self.assertRaises(RuntimeError):
                    reference.read_curve(path, 2, [1., 3.])

    def test_snapshot_detects_all_project_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            path = self.make_curve(project)
            (project / "bifurcation_diagram.dat").write_text("archive")
            before = reference.snapshot(project)
            self.assertEqual(reference.snapshot(project), before)
            (path / "1").write_text("changed")
            self.assertNotEqual(reference.snapshot(project), before)
            (project / "extra").write_text("new")
            self.assertIn("extra", set(reference.snapshot(project)) - set(before))

    def test_config_changes_only_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            original = {"path_to_project": "old/", "nonlinear_operator": {"tolerance": 1e-10},
                        "deflation_continuation": {"maximum_continuation_steps": 5000}}
            manifest = {"metadata_points": 0, "saved_vectors": 0}
            args = type("Args", (), dict(work_dir=work, reference_dir=work,
                                       bd_executable=Path("unused"), timeout=1))()
            with patch.object(reference, "verify_reference", return_value=(manifest, original.copy(), work)), \
                 patch.object(reference, "inventory", return_value=[]), \
                 patch.object(reference, "run", side_effect=RuntimeError("stop before simulation")):
                with self.assertRaisesRegex(RuntimeError, "stop before simulation"):
                    reference.validate(args, {})
            actual = json.loads((work / "config.json").read_text())
            self.assertEqual(actual.pop("path_to_project"), str(work / "project") + "/")
            self.assertEqual(actual, {key: value for key, value in original.items() if key != "path_to_project"})

    def test_failure_still_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "replay"
            arguments = ["test", "--bd-executable", "unused", "--checker-executable", "unused",
                         "--work-dir", str(work)]
            with patch.object(sys, "argv", arguments), \
                 patch.object(reference, "validate", side_effect=RuntimeError("injected failure")), \
                 patch("builtins.print"):
                self.assertEqual(reference.main(), 1)
            report = json.loads((work / "comparison.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["error"], "injected failure")

    def test_reference_checksum_failure_precedes_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("changed")
            (path / "SHA256SUMS").write_text("0" * 64 + "  config.json\n")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                reference.verify_reference(path, path / "extracted")
            self.assertFalse((path / "extracted").exists())


if __name__ == "__main__":
    unittest.main()
