import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("stokes_benchmarks", ROOT / "source/nonlinear_operators/stokes/tests/stokes_benchmarks.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(order=2):
    config = runner.configurations(runner.arguments(["--systems", "stokes", "--methods", "CN",
                                                  "--base-steps", "4", "--levels", "3"]))[0]
    rows = []
    for level in range(config["levels"]):
        steps = config["base_steps"] * 2**level
        h = (config["end"]-config["start"])/steps
        rows.append(dict(steps=steps, step=h, error_rms=.01*h**order,
                         maximum_divergence_rms=0, backend_allocations_after_warmup=0))
    history = [dict(time=config["start"]+(j+1)*rows[-1]["step"],
                    error_rms=rows[-1]["error_rms"]*(j+1)/rows[-1]["steps"], divergence_rms=0)
               for j in range(rows[-1]["steps"])]
    frames = []
    for j in range(3):
        t = config["start"]+j*(config["end"]-config["start"])/2
        error = rows[-1]["error_rms"]*j/2
        coefficients = []
        for kx, ky in sorted(runner.MODE_KEYS):
            u, v = runner.exact_coefficient(kx, ky, t, config["system"])
            if (kx, ky) == (1, 1):
                u += error/2
                v -= error/2
            coefficients.append(dict(kx=kx, ky=ky, velocity=[u.real, u.imag, v.real, v.imag]))
        frames.append(dict(time=t, omitted_rms=0, coefficients=coefficients))
    report = dict(format="stokes.timestepper.validation.v1", configuration=config, passed=True,
                  expected_order=2, refinements=rows, history=history, snapshots=frames,
                  norm="physical_velocity_rms", fit_error_floor=1e-11, bulk_host_transfer_bytes=0,
                  exported_scalar_values=60, pressure_reconstructed=False)
    return config, report


class StokesBenchmarksTest(unittest.TestCase):
    def test_default_cases_and_supported_combinations(self):
        configs = runner.configurations(runner.arguments([]))
        self.assertEqual(len(configs), 17)
        self.assertEqual(len({runner.case_name(c) for c in configs}), 17)
        self.assertTrue(all(c["system"] == "stokes" or c["method"].startswith("IMEX_") for c in configs))
        configs = runner.configurations(runner.arguments(["--methods", "CN", "CN", "--systems", "stokes", "stokes"]))
        self.assertEqual(len(configs), 1)

    def test_invalid_arguments(self):
        for options in (["--start", "nan"], ["--end", "0"], ["--grid", "7"], ["--grid", "10", "--levels", "2"],
                        ["--base-steps", "5"], ["--threads", "0"], ["--timeout", "inf"], ["--plot-grid", "4"],
                        ["--make-option", "-n"], ["--make-option", "BUILD_DIR=x"], ["--systems", "oseen", "--methods", "CN"],
                        ["-B", "--plot-only", "x"], ["--start", "1e20", "--end", "1e20"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.arguments(options)

    def test_reference_lines_and_measured_order(self):
        values = runner.reference_line([.1, .05, .025], .1, .01, 3)
        self.assertAlmostEqual(values[0]/values[1], 8)
        self.assertAlmostEqual(values[1]/values[2], 8)
        config, report = fixture()
        self.assertAlmostEqual(runner.validate_report(report, config)["observed_order"], 2)

    def test_degraded_order_and_floor_are_not_passed(self):
        config, report = fixture(order=1)
        with self.assertRaisesRegex(ValueError, "Observed order"):
            runner.validate_report(report, config)
        config, report = fixture()
        for row in report["refinements"]:
            row["error_rms"] = 1e-13
        with self.assertRaisesRegex(ValueError, "Insufficient refinement"):
            runner.validate_report(report, config)

    def test_report_rejects_wrong_semantics_and_numerical_failures(self):
        config, valid = fixture()
        for key, value in (("passed", False), ("norm", "spectral_storage_rms"), ("expected_order", 3),
                           ("bulk_host_transfer_bytes", 8), ("pressure_reconstructed", True),
                           ("exported_scalar_values", 64), ("fit_error_floor", 1), ("history", [])):
            report = copy.deepcopy(valid)
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_report(report, config)
        for key, value in (("steps", 5), ("backend_allocations_after_warmup", 1),
                           ("maximum_divergence_rms", 1e-5), ("error_rms", float("nan"))):
            report = copy.deepcopy(valid)
            report["refinements"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_report(report, config)

    def test_exported_fields_independently_verify_errors_and_constraints(self):
        config, valid = fixture()
        for mutate in (lambda r: r["snapshots"][2]["coefficients"].pop(),
                       lambda r: r["snapshots"][2].update(omitted_rms=1e-4),
                       lambda r: r["snapshots"][2].update(time=5),
                       lambda r: r["snapshots"][2]["coefficients"][0]["velocity"].__setitem__(0, 2),
                       lambda r: r["history"][-1].update(error_rms=1e-7)):
            report = copy.deepcopy(valid)
            mutate(report)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                runner.validate_report(report, config)

    def test_build_stamp_tracks_options_without_timestamp_churn(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.subprocess, "run") as make:
            args = runner.arguments(["--build-dir", directory, "--methods", "CN"])
            runner.build(args, runner.configurations(args))
            stamp = Path(directory) / "bin/cpu_build_options.json"
            timestamp = stamp.stat().st_mtime_ns
            runner.build(args, runner.configurations(args))
            self.assertEqual(stamp.stat().st_mtime_ns, timestamp)
            args.make_option = ["G++=clang++"]
            args.build_only = True
            runner.build(args, runner.configurations(args))
            self.assertIn("-B", make.call_args.args[0])
            self.assertIn("G++ = clang++", make.call_args.args[0])
            self.assertTrue((Path(directory) / "cases/stokes_CN/config.json").exists())

    def test_build_only_does_not_integrate_or_import_plotting(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "build") as build:
            with patch.object(runner, "ensure_plotting", side_effect=AssertionError("Unexpected plotting import")):
                with patch.object(runner, "summarize", side_effect=AssertionError("Unexpected plotting")):
                    with patch.object(runner.subprocess, "run", side_effect=AssertionError("Unexpected execution")):
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(runner.main(["-B", "--build-dir", directory]), 0)
            self.assertTrue(build.call_args.args[0].build_only)
            self.assertFalse((Path(directory) / "runs").exists())

    def test_plot_only_never_builds(self):
        with patch.object(runner, "ensure_plotting"), patch.object(runner, "build", side_effect=AssertionError("Unexpected build")):
            with patch.object(runner, "summarize", return_value=0) as plot:
                self.assertEqual(runner.main(["--plot-only", "/tmp/stokes-run"]), 0)
                plot.assert_called_once_with(Path("/tmp/stokes-run"), 64)

    def test_timeout_is_recorded_and_runs_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "ensure_plotting"):
            with patch.object(runner, "build", return_value=Path("/tmp/test.bin")), patch.object(runner, "summarize", return_value=1):
                with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("test", 1)):
                    with contextlib.redirect_stdout(io.StringIO()):
                        for _ in range(2):
                            self.assertEqual(runner.main(["--methods", "CN", "--build-dir", directory]), 1)
            runs = list((Path(directory) / "runs").iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                manifest = json.loads((run / "manifest.json").read_text())
                self.assertEqual(manifest["cases"][0]["returncode"], 124)

    def test_failed_executable_is_not_plotted_or_reported_as_pass(self):
        config, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps(dict(format="stokes.benchmark.manifest.v1",
                cases=[dict(configuration=config, returncode=2)])))
            with patch.object(runner, "plot_solutions", side_effect=AssertionError("Unexpected plot")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(runner.summarize(path), 1)
            self.assertFalse(json.loads((path / "validation.json").read_text())[0]["passed"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Field validation requires NumPy")
    def test_fourier_reconstruction_agrees_with_physical_manufactured_velocity(self):
        import numpy as np
        for system in ("stokes", "oseen"):
            t = .7
            coefficients = []
            for kx, ky in sorted(runner.MODE_KEYS):
                u, v = runner.exact_coefficient(kx, ky, t, system)
                coefficients.append(dict(kx=kx, ky=ky, velocity=[u.real, u.imag, v.real, v.imag]))
            _, _, reference, numerical = runner.velocity_fields(dict(time=t, coefficients=coefficients), system, 32)
            self.assertLess(np.max(np.abs(np.array(reference)-numerical)), 2e-15)


if __name__ == "__main__":
    unittest.main()
