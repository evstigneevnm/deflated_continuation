import contextlib
import csv
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
SPEC = importlib.util.spec_from_file_location("orbit_benchmarks", ROOT / "source/periodic_orbit/tests/orbit_benchmarks.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class OrbitBenchmarkRunnerTest(unittest.TestCase):
    def test_configuration_is_explicit_and_unique(self):
        args = runner.arguments(["--sections", "1", "4", "4", "--methods", "RK45", "DOP853", "RK45"])
        configs = runner.configurations(args)
        self.assertEqual(len(configs), 4)
        self.assertEqual({c["sections"] for c in configs}, {1, 4})
        self.assertTrue(all(c["system"] == "analytic_cycle" and c["start_time"] == .37 for c in configs))

    def test_reject_invalid_options(self):
        for options in (["--sections", "0"], ["--sections", "65"], ["--start", "nan"],
                        ["--perturbation", "1"], ["--threads", "0"], ["-B", "--plot-only", "x"],
                        ["--make-option", "BUILD_DIR=x"], ["--make-option", "-n"], ["--newton-max-iterations", "0"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.arguments(options)

    def test_build_stamp_is_stable_and_tracks_options(self):
        with tempfile.TemporaryDirectory() as directory:
            args = runner.arguments(["--build-dir", directory, "--sections", "1"])
            with patch.object(runner.subprocess, "run") as make:
                runner.build(args, runner.configurations(args))
                stamp = Path(directory) / "bin/cpu_build_options.json"
                original = stamp.stat().st_mtime_ns
                runner.build(args, runner.configurations(args))
                self.assertEqual(stamp.stat().st_mtime_ns, original)
                args.make_option = ["G++=clang++"]
                args.build_only = True
                runner.build(args, runner.configurations(args))
                command = make.call_args.args[0]
                self.assertIn("-B", command)
                self.assertIn("G++ = clang++", command)
                self.assertEqual(json.loads(stamp.read_text())["options"], ["G++=clang++"])
                self.assertTrue((Path(directory) / "cases/analytic_cycle_DOP853_M1/config.json").exists())

    def test_build_only_never_runs_or_plots(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "build") as build:
            with patch.object(runner, "summarize", side_effect=AssertionError("Unexpected plot")):
                with patch.object(runner.subprocess, "run", side_effect=AssertionError("Unexpected integration")):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(runner.main(["-B", "--build-dir", directory]), 0)
            self.assertTrue(build.call_args.args[0].build_only)
            self.assertFalse((Path(directory) / "runs").exists())

    def test_plot_only_does_not_build(self):
        with patch.object(runner, "build", side_effect=AssertionError("Unexpected build")):
            with patch.object(runner, "summarize", return_value=0) as summarize:
                self.assertEqual(runner.main(["--plot-only", "/tmp/orbit-report"]), 0)
                summarize.assert_called_once_with(Path("/tmp/orbit-report"))

    def test_runs_are_preserved_and_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "build", return_value={"analytic_cycle": Path("/tmp/test.bin")}):
            with patch.object(runner, "summarize", return_value=1), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("test", 1)):
                    for _ in range(2):
                        self.assertEqual(runner.main(["--build-dir", directory, "--sections", "1"]), 1)
            runs = list((Path(directory) / "runs").iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                manifest = json.loads((run / "manifest.json").read_text())
                self.assertEqual(manifest["cases"][0]["returncode"], 124)
                self.assertIn("timeout", (run / "analytic_cycle_DOP853_M1/run.log").read_text())

    def test_failure_report_is_not_presented_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = runner.configurations(runner.arguments(["--sections", "1"]))[0]
            (path / "manifest.json").write_text(json.dumps({"cases": [{"configuration": config, "returncode": 2}]}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.summarize(path), 1)
            report = json.loads((path / "summary.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["stage"], "matching_only")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Numerical export validation requires NumPy")
    def test_export_validation_rejects_corrupt_reports_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = runner.configurations(runner.arguments(["--sections", "1"]))[0]
            segment = dict(flight_time=2 * math.pi, steps=4, crossing_speed=1, section_value=0, gap_norm=0)
            part = dict(succeeded=True, period=2 * math.pi, segments=[segment])
            report = dict(format="periodic_orbit.matching.v1", stage="matching_only", orbit_corrected=False,
                          configuration=config, passed=True, baseline=part, refined=part,
                          perturbed=dict(part, segments=[dict(segment, gap_norm=.02)]),
                          baseline_norm=0, refined_norm=0, perturbed_norm=.02,
                          matching_host_bytes=0, closure_error=0, trajectory_error=0, trajectory_rows=5)
            with (path / "trajectory.csv").open("w") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time", "x", "y"])
                for i in range(5):
                    t = config["start_time"] + i * math.pi / 2
                    writer.writerow([t, math.cos(t - config["start_time"]), math.sin(t - config["start_time"])])
            columns = "section,time,cx,cy,nx,ny,x,y,arrival_x,arrival_y,target_x,target_y,flight_time,steps,gap".split(",")
            def write_sections(perturbation, filename):
                with (path / filename).open("w") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerow([0, config["start_time"], 1, 0, 0, 1, 1 + perturbation, 0,
                                     1, 0, 1 + perturbation, 0, 2 * math.pi, 4, perturbation])
            write_sections(0, "sections.csv")
            write_sections(.02, "perturbed_sections.csv")
            (path / "report.json").write_text(json.dumps(report))
            self.assertTrue(runner.validate_case(path, config)["passed"])
            new_config = dict(config, system="analytic_cycle_newton", newton_tolerance=1e-10, newton_max_iterations=12)
            new_report = dict(format="periodic_orbit.newton.v1", stage="periodic_newton", orbit_corrected=True,
                              configuration=new_config, passed=True, result_status=0, corrected=part, refined=part,
                              initial=report["perturbed"], initial_norm=.02, corrected_norm=0, refined_norm=0,
                              matching_host_bytes=0, closure_errors=[0, 0], history=[.02, 0])
            for source, target in (("sections.csv", "corrected_sections.csv"),
                                   ("perturbed_sections.csv", "initial_sections.csv"),
                                   ("trajectory.csv", "refined_trajectory.csv")):
                (path / target).write_text((path / source).read_text())
            (path / "newton_history.csv").write_text("iteration,residual\n0,0.02\n1,0\n")
            (path / "report.json").write_text(json.dumps(new_report))
            self.assertTrue(runner.validate_case(path, new_config)["passed"])
            for key, value in (("orbit_corrected", False), ("matching_host_bytes", 1024), ("result_status", 9),
                               ("corrected_norm", 1e-8), ("history", [.02, float("nan")]), ("closure_errors", [0])):
                with self.subTest(newton_key=key):
                    (path / "report.json").write_text(json.dumps(dict(new_report, **{key: value})))
                    with self.assertRaises(ValueError):
                        runner.validate_case(path, new_config)
            for key, value in (("orbit_corrected", True), ("matching_host_bytes", 1024), ("baseline_norm", 1)):
                with self.subTest(key=key):
                    (path / "report.json").write_text(json.dumps(dict(report, **{key: value})))
                    with self.assertRaises(ValueError):
                        runner.validate_case(path, config)
            (path / "report.json").write_text(json.dumps(report))
            with (path / "sections.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["arrival_y"] = "0.1"
            rows[0]["gap"] = "0.1"
            with (path / "sections.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                runner.validate_case(path, config)


    def test_correction_configuration_and_independent_build_targets(self):
        args = runner.arguments(["--systems", "analytic_cycle", "analytic_cycle_newton", "--sections", "1", "4"])
        configs = runner.configurations(args)
        self.assertEqual(len(configs), 4)
        self.assertNotIn("newton_tolerance", configs[0])
        self.assertEqual(configs[2]["newton_tolerance"], 1e-10)
        self.assertEqual(runner.stage(configs), "mixed")
        self.assertEqual(runner.stage(configs[2:]), "periodic_newton")
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.subprocess, "run") as make:
            args.build_dir = Path(directory)
            executables = runner.build(args, configs)
            self.assertEqual(set(executables), set(args.systems))
            self.assertIn("periodic_orbit_benchmark_analytic_cycle_newton_cpu.bin", make.call_args.args[0])

    def test_newton_report_cannot_be_confused_with_matching_only(self):
        config = runner.configurations(runner.arguments(["--system", "analytic_cycle_newton", "--sections", "1"]))[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for report in ({"format": "periodic_orbit.matching.v1"},
                           dict(format="periodic_orbit.newton.v1", stage="periodic_newton", orbit_corrected=False)):
                (path / "report.json").write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    runner.validate_case(path, config)


if __name__ == "__main__":
    unittest.main()
