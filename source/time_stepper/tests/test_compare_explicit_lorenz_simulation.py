#!/usr/bin/env python3
"""Check trajectory validation and strict/diagnostic comparison behavior."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import compare_explicit_lorenz_simulation as comparison


class LorenzSimulationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trajectory.dat"
        self.metadata = {
            "format": "nmfd.lorenz_trajectory.v1", "problem": "lorenz", "method": "DOPRI54",
            "backend": "serial_cpu", "start_time": 0, "end_time": 0.01,
            "initial_state": [2.2, 30.5, 2.5],
            "parameters": {"sigma": 10, "rho": 28, "beta": 8 / 3, "epsilon": 0.0055, "delta": 0},
            "adaptation": {"relative_tolerance": 1e-10, "absolute_tolerance": 1e-12,
                           "initial_step": 0.01, "minimum_step": 1e-14, "maximum_step": 1}
        }
        self.data = np.array([[0, 2.2, 30.5, 2.5], [0.01, 2.2, 30.5, 2.5]])
        self.footer = {"status": "completed", "rows": 2}

    def write(self, complete=True):
        with self.path.open("w", encoding="utf-8") as stream:
            stream.write("# " + json.dumps(self.metadata) + "\n")
            np.savetxt(stream, self.data, fmt="%.17g")
            if complete:
                stream.write("# " + json.dumps(self.footer) + "\n")

    def test_valid_input_preserves_times_and_parameters(self):
        self.write()
        metadata, data = comparison.read_trajectory(self.path)
        self.assertEqual(metadata, self.metadata)
        np.testing.assert_array_equal(data, self.data)

    def test_bs32_metadata_is_accepted(self):
        self.metadata["method"] = "BS32"
        self.write()
        metadata, data = comparison.read_trajectory(self.path)
        self.assertEqual(metadata["method"], "BS32")
        np.testing.assert_array_equal(data, self.data)

    def test_bs32_uses_scipy_rk23(self):
        self.metadata["method"] = "BS32"
        self.write()
        args = SimpleNamespace(trajectory=self.path, reference_rtol=1e-12, reference_atol=1e-14,
                               check_rtol=1e-7, check_atol=1e-8, no_plots=True, diagnostic_only=True)
        with mock.patch.object(comparison, "solve_ivp", wraps=comparison.solve_ivp) as solver:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(comparison.compare(args), 0)
        self.assertEqual(solver.call_args.kwargs["method"], "RK23")
        summary = json.loads(self.path.with_name("trajectory_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["reference_method"], "SciPy RK23")

    def test_incomplete_trajectory_is_rejected(self):
        self.write(complete=False)
        with self.assertRaisesRegex(ValueError, "completion footer"):
            comparison.read_trajectory(self.path)

    def test_wrong_row_count_is_rejected(self):
        self.footer["rows"] = 3
        self.write()
        with self.assertRaisesRegex(ValueError, "row count"):
            comparison.read_trajectory(self.path)

    def test_nonfinite_states_are_rejected(self):
        self.data[1, 2] = np.nan
        self.write()
        with self.assertRaisesRegex(ValueError, "finite"):
            comparison.read_trajectory(self.path)

    def test_time_order_is_checked(self):
        self.data = np.vstack((self.data[0], self.data[0], self.data[1]))
        self.footer["rows"] = 3
        self.write()
        with self.assertRaisesRegex(ValueError, "time ordering"):
            comparison.read_trajectory(self.path)

    def test_interval_and_initial_state_are_checked(self):
        for column, value in ((0, 0.001), (1, 3.2)):
            with self.subTest(column=column):
                self.data[0] = [0, 2.2, 30.5, 2.5]
                self.data[0, column] = value
                self.write()
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    comparison.read_trajectory(self.path)

    def test_unsupported_method_is_rejected(self):
        self.metadata["method"] = "RK33SSP"
        self.write()
        with self.assertRaisesRegex(ValueError, "DOPRI54"):
            comparison.read_trajectory(self.path)

    def test_adaptation_settings_are_checked(self):
        self.metadata["adaptation"]["minimum_step"] = 2
        self.write()
        with self.assertRaisesRegex(ValueError, "adaptation"):
            comparison.read_trajectory(self.path)

    def test_rhs_includes_cubic_damping_and_asymmetry(self):
        values = comparison.lorenz_rhs(0, [2, 3, 4], 10, 28, 8 / 3, 0.0055, 0.2)
        np.testing.assert_allclose(values, [10 - 0.0055 * 8, 45.2, -8 / 3 * 4 + 6 - 0.2])

    def test_strict_failure_and_diagnostic_report_are_distinct(self):
        self.write()
        args = SimpleNamespace(trajectory=self.path, reference_rtol=1e-12, reference_atol=1e-14,
                               check_rtol=1e-7, check_atol=1e-8, no_plots=True, diagnostic_only=False)
        for diagnostic in (False, True):
            with self.subTest(diagnostic=diagnostic):
                args.diagnostic_only = diagnostic
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    status = comparison.compare(args)
                self.assertEqual(status, 0 if diagnostic else 1)
                self.assertIn("DIAGNOSTIC" if diagnostic else "FAIL", output.getvalue())
                summary = json.loads(self.path.with_name("trajectory_summary.json").read_text(encoding="utf-8"))
                self.assertFalse(summary["within_tolerance"])
                self.assertEqual(summary["diagnostic_only"], diagnostic)


if __name__ == "__main__":
    unittest.main()
