#!/usr/bin/env python3
"""Protect independent analytical sampling and bounded KS1D smoke fixtures."""

import json
import math
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_local_ci as ci
import write_ks1d_ci_config as writer


class KS1DCIConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "config.json"
        self.project = self.root / "project"

    def arguments(self, base="KS1D_test_sym.json", *extra):
        command = [
            "write_ks1d_ci_config.py", str(self.output), str(self.project),
            "--base-config", str(ROOT / "json_project_files" / base), *extra,
        ]
        with patch.object(sys, "argv", command):
            return writer.parse_args()

    def test_both_fixtures_override_production_bounds_without_changing_solvers(self):
        for base in ("KS1D_test_sym.json", "KS1D_test_full.json"):
            with self.subTest(base=base):
                args = self.arguments(base, "--steps", "32")
                original = json.loads(Path(args.base_config).read_text())
                writer.write_config(args)
                config = json.loads(self.output.read_text())
                continuation = config["deflation_continuation"]
                self.assertEqual(continuation["maximum_continuation_steps"], 32)
                self.assertEqual(continuation["analytical_max_parameter_step"], 0.1)
                self.assertEqual(continuation["continuation_parameter_bounds"], {
                    "enabled": True, "minimum": 2.0, "maximum": 3.5,
                    "resolve_with_knot_registry": True,
                })
                for section, names in (
                    ("deflation_continuation", ("linear_solver_extended", "newton_continuation", "newton_deflation")),
                    ("nonlinear_operator", ("linear_solver", "newton")),
                    ("stability_continuation", ("linear_solver", "newton")),
                ):
                    for name in names:
                        numerical = {key: value for key, value in original[section][name].items()
                                     if key not in ("verbose", "save_convergence_history", "save_norms_history")}
                        self.assertEqual(numerical, {key: config[section][name][key] for key in numerical})

    def test_parameter_spacing_is_independent_of_arclength_and_uses_custom_knots(self):
        args = self.arguments(
            "KS1D_test_full.json", "--step-size", "0.04", "--max-step-size", "0.06",
            "--analytical-max-parameter-step", "0.2", "--knots", "3.8", "2.4", "3.1",
        )
        writer.write_config(args)
        continuation = json.loads(self.output.read_text())["deflation_continuation"]
        self.assertEqual(continuation["step_size"], 0.04)
        self.assertEqual(continuation["max_step_size"], 0.06)
        self.assertEqual(continuation["analytical_max_parameter_step"], 0.2)
        self.assertEqual(continuation["continuation_parameter_bounds"]["minimum"], 2.4)
        self.assertEqual(continuation["continuation_parameter_bounds"]["maximum"], 3.8)
        args.analytical_max_parameter_step = 0.0
        writer.write_config(args)
        self.assertEqual(json.loads(self.output.read_text())["deflation_continuation"]["analytical_max_parameter_step"], 0.0)

    def test_invalid_sampling_does_not_clean_an_existing_project(self):
        self.project.mkdir()
        marker = self.project / "preserve.dat"
        marker.write_text("preserve\n")
        for value in (-0.1, math.nan, math.inf):
            with self.subTest(gap=value):
                args = self.arguments("KS1D_test_sym.json", "--clean-project")
                args.analytical_max_parameter_step = value
                with self.assertRaises(ValueError):
                    writer.write_config(args)
                self.assertTrue(marker.exists())
        for knots in ([], [2.0], [2.0, 2.0], [2.0, math.nan], [2.0, math.inf]):
            with self.subTest(knots=knots):
                args = self.arguments("KS1D_test_sym.json", "--clean-project")
                args.knots = knots
                with self.assertRaises(ValueError):
                    writer.write_config(args)
                self.assertTrue(marker.exists())
        self.assertFalse(self.output.exists())

    def test_all_cpu_and_cuda_workflow_fixtures_have_an_explicit_sampling_budget(self):
        workflow = ci.load_workflow(ROOT / ".github/workflows/operations.yml")
        count = 0
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                for line in step.get("run", "").splitlines():
                    if not line.strip().startswith("python3 scripts/write_ks1d_ci_config.py "):
                        continue
                    command = shlex.split(line)
                    with self.subTest(command=command), patch.object(sys, "argv", command[1:]):
                        args = writer.parse_args()
                        self.assertIn("--analytical-max-parameter-step", command)
                        self.assertGreater(args.analytical_max_parameter_step, 0.0)
                        required = math.ceil((max(args.knots) - min(args.knots)) / args.analytical_max_parameter_step)
                        self.assertGreaterEqual(args.steps, required + len(set(args.knots)))
                        count += 1
        self.assertEqual(count, 4)


if __name__ == "__main__":
    unittest.main()
