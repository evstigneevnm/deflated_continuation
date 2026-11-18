"""CH plot/report validation and build/run lifecycle, without plotting dependencies."""

import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ch_benchmarks", ROOT / "source/nonlinear_operators/cahn_hilliard/tests/ch_benchmarks.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(method="IMEX_EULER", legacy=True, grows_energy=False):
    config = runner.configuration(runner.arguments(["--grid", "8", "--end", ".4", "--step", ".1",
                                                  "--levels", "3", "--base-steps", "4", "--frames", "3"]))
    config["method"] = method
    n, count, h = config["grid"], 4, .1
    shape = [math.cos(i*2*math.pi/n) + .4*math.sin(j*2*math.pi/n) +
             .25*math.cos(2*i*2*math.pi/n)*math.cos(j*2*math.pi/n) for i in range(n) for j in range(n)]
    history, snapshots = [], []
    for k in range(count+1):
        values = [config["mean"]+config["amplitude"]*(1+(-.1 if grows_energy else .1)*k/count)*v for v in shape]
        energy, mass, _ = runner.field_diagnostics(values, config)
        row = dict(time=k*h, energy=energy, mass=mass)
        if k:
            previous = history[-1]["energy"]
            dissipation = (previous-energy)/(2*h)
            row.update(step=h, dissipation=dissipation, residual_work=0, step_defect_rms=1e-14,
                       bound_defect=energy-previous+h*dissipation,
                       roundoff_budget=256*sys.float_info.epsilon*(1+abs(previous)+abs(energy)+abs(h*dissipation)))
        history.append(row)
        if k % 2 == 0:
            snapshots.append(dict(step_index=k, time=k*h, concentration=values))
        if not legacy:
            mu = runner.field_diagnostics(values, config)[2]
            row["physical_dissipation"] = -config["mobility"]*(2*math.pi/n)**2*sum(
                a*b for a, b in zip(mu, runner.periodic_laplacian(mu, n)))
            if method != "IMEX_EULER":
                for key in ("dissipation", "residual_work", "step_defect_rms", "bound_defect", "roundoff_budget"):
                    row.pop(key, None)
    def mesh(steps):
        return dict(steps=steps, step=config["order_end"]/steps, backend_allocations_after_warmup=0)
    order = runner.METHODS[method]
    rows = [dict(mesh(4*2**level), error_rms=.01*(config["order_end"]/(4*2**level))**order) for level in range(3)]
    report = dict(format="ch.energy.visualization.v1", configuration=config, method="IMEX_EULER",
                  expected_order=order, norm="physical_concentration_rms", fit_error_floor=1e-11,
                  trajectory=dict(steps=count, step=h, backend_allocations_after_warmup=0),
                  history=history, snapshots=snapshots, refinements=rows,
                  reference=dict(coarse=mesh(128), fine=mesh(256), difference_rms=1e-6),
                  integration_bulk_host_transfer_bytes=0, export_bulk_host_transfer_bytes=3*n*n*8)
    report["method"] = method
    if not legacy:
        report["format"] = "ch.energy.visualization.v2"
        report["energy_policy"] = "convex_split_euler" if method == "IMEX_EULER" else "observe_only"
        report["reference"]["method"] = "ROS3"
        report["reference"]["difference_rms"] = rows[-1]["error_rms"]*.01
    return config, report


def modern_fixture(method="BDF", adaptive=True):
    config, report = fixture("ROS3" if method == "BDF" else method, legacy=False)
    config.update(method=method, adaptive=adaptive)
    if adaptive:
        config.update(relative_tolerance=1e-4, absolute_tolerance=1e-6,
                      initial_step=.02, maximum_order=5, maximum_steps=20000)
    primary = runner.METHODS[method]
    report.update(format="ch.energy.visualization.v3", method=method, configuration=config,
                  primary_order=primary, expected_order=None if adaptive else primary,
                  validation_mode="tolerance_refinement" if adaptive else "temporal_refinement",
                  energy_policy="convex_split_euler" if method == "IMEX_EULER" else "observe_only")
    for k, row in enumerate(report["history"][1:], 1):
        row.update(order=min(k, 3) if method == "BDF" else primary, attempts=1)
    def summary(row, order):
        row.update(minimum_step=.2/row["steps"], maximum_step=.2/row["steps"], maximum_order=order,
                   rejected_attempts=0, order_counts=[row["steps"] if i == order else 0 for i in range(6)])
        return row
    for level, row in enumerate(report["refinements"]):
        if adaptive:
            row.update(steps=4+2*level, step=None, error_rms=1e-5*.1**level,
                       relative_tolerance=config["relative_tolerance"]*.1**level,
                       absolute_tolerance=config["absolute_tolerance"]*.1**level)
        summary(row, min(level+1, 5) if method == "BDF" else primary)
    trajectory = report["trajectory"]
    summary(trajectory, max(r["order"] for r in report["history"][1:]))
    trajectory.update(step=None if adaptive else .1, minimum_step=.1, maximum_step=.1,
                      order_counts=[sum(r["order"] == i for r in report["history"][1:]) for i in range(6)])
    if adaptive:
        trajectory.update(relative_tolerance=config["relative_tolerance"], absolute_tolerance=config["absolute_tolerance"])
    for frame in report["snapshots"]:
        source = frame.pop("step_index")
        frame.update(source_step_index=source, evaluation_time=frame["time"])
        frame.update({key: report["history"][source][key] for key in ("energy", "mass", "physical_dissipation")})
    report["reference"]["difference_rms"] = report["refinements"][-1]["error_rms"]*.01
    return config, report


class CHBenchmarksTest(unittest.TestCase):
    def test_default_configuration(self):
        args = runner.arguments([])
        self.assertEqual(args.grid, 16)
        self.assertEqual(args.frames, 5)
        self.assertEqual(args.methods, list(runner.METHODS))
        self.assertEqual(runner.configuration(args)["amplitude"], .12)

    def test_help_explains_step_counts_defaults_and_separate_order_study(self):
        text = io.StringIO()
        with contextlib.redirect_stdout(text), self.assertRaises(SystemExit) as exit:
            runner.arguments(["--help"])
        self.assertEqual(exit.exception.code, 0)
        help_text = text.getvalue()
        for phrase in ("ceil(end/step)", "20000", "--end 6 --step 0.01 for 600 steps",
                       "Separate temporal-order study", "5, 10, 20, 40", "320, 640", "default: 0.1"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, help_text)

    def test_more_main_steps_do_not_change_refinement_configuration(self):
        defaults = runner.configuration(runner.arguments([]))
        more_steps = runner.configuration(runner.arguments(["--end", "6", "--step", "0.01"]))
        self.assertEqual(math.ceil(more_steps["end"]/more_steps["step"]-1e-12), 600)
        for key in ("order_end", "base_steps", "levels", "frames"):
            self.assertEqual(more_steps[key], defaults[key])
        extended = runner.arguments(["--end", "60", "--step", "0.01"])
        self.assertEqual(math.ceil(extended.end/extended.step-1e-12), 6000)
        limit = runner.arguments(["--end", "20", "--step", "0.001"])
        self.assertEqual(math.ceil(limit.end/limit.step-1e-12), 20000)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.arguments(["--end", "21", "--step", "0.001"])

    def test_invalid_budgets_and_parameters(self):
        for values in (["--grid", "7"], ["--grid", "65"], ["--levels", "2"], ["--epsilon", "nan"],
                       ["--step", "0"], ["--step", "100"], ["--frames", "100"], ["--amplitude", "0"],
                       ["--timeout", "-1"], ["--order-end", "1e-20"], ["--make-option", "BUILD_DIR=x"]):
            with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.arguments(values)

    def test_valid_report(self):
        config, report = fixture()
        metrics = runner.validate_report(report, config)
        self.assertAlmostEqual(metrics["observed_order"], 1)
        self.assertLess(metrics["energy_final"], metrics["energy_initial"])

    def test_all_new_methods_and_energy_policies(self):
        for method, order in runner.METHODS.items():
            if method == "BDF":
                config, report = modern_fixture()
                self.assertIsNone(runner.validate_report(report, config)["observed_order"])
                continue
            config, report = fixture(method, legacy=False)
            with self.subTest(method=method):
                metrics = runner.validate_report(report, config)
                self.assertAlmostEqual(metrics["observed_order"], order)
                self.assertEqual(metrics["expected_order"], order)
                self.assertEqual(metrics["formulation"], "convex_split" if method == "IMEX_EULER" else "imex_split" if method.startswith("IMEX_") else "unsplit")

    def test_adaptive_configuration_and_limits(self):
        args = runner.arguments(["--methods", "ARK324L2SA", "BDF", "--adaptive-methods", "ARK324L2SA"])
        for method in args.methods:
            config = runner.configuration(args, method)
            self.assertTrue(config["adaptive"])
            self.assertEqual(config["maximum_order"], 5)
        self.assertFalse(runner.configuration(runner.arguments(["--methods", "ROS34PW2"]))["adaptive"])
        for values in (["--relative-tolerance", "0"], ["--maximum-steps", "0"], ["--maximum-order", "6"],
                       ["--levels", "-200000"], ["--relative-tolerance", "1e-14"],
                       ["--methods", "CN", "--adaptive-methods", "ARK324L2SA"]):
            with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.arguments(values)

    def test_adaptive_reports_for_embedded_and_variable_order_methods(self):
        for method in ("BDF", *runner.ADAPTIVE_METHODS):
            config, report = modern_fixture(method)
            with self.subTest(method=method):
                metrics = runner.validate_report(report, config)
                self.assertTrue(metrics["adaptive"])
                self.assertIsNone(metrics["expected_order"])
                self.assertIsNone(metrics["observed_order"])
        fixed = modern_fixture("CN", adaptive=False)[1]
        adaptive = modern_fixture()[1]
        self.assertEqual(len(runner.comparison_limits([fixed, adaptive])), 2)

    def test_invalid_adaptive_histories_samples_and_estimates_are_rejected(self):
        mutations = [lambda r: r["history"][1].update(step=-.1),
                     lambda r: r["history"][1].update(order=6),
                     lambda r: r["history"][1].update(attempts=0),
                     lambda r: r["trajectory"].update(steps=20001),
                     lambda r: r["trajectory"].update(rejected_attempts=1),
                     lambda r: r["refinements"][1].update(relative_tolerance=.2),
                     lambda r: r["refinements"][-1].update(error_rms=1e-5),
                     lambda r: r["snapshots"][1].update(evaluation_time=.3),
                     lambda r: r["snapshots"][1].update(source_step_index=99),
                     lambda r: r["snapshots"][1].update(energy=100)]
        for mutate in mutations:
            config, report = modern_fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                runner.validate_report(report, config)

    def test_interior_dense_samples_are_not_endpoint_diagnostics(self):
        config, report = modern_fixture()
        config["step"] = .2
        times = [0., .08, .18, .3, .4]
        initial = report["snapshots"][0]["concentration"]
        for k, row in enumerate(report["history"][1:], 1):
            row.update(time=times[k], step=times[k]-times[k-1])
            values = [c+(c-config["mean"])*.1*times[k]/config["end"] for c in initial]
            energy, mass, mu = runner.field_diagnostics(values, config)
            row.update(energy=energy, mass=mass, physical_dissipation=-config["mobility"]*(2*math.pi/config["grid"])**2*
                       sum(a*b for a, b in zip(mu, runner.periodic_laplacian(mu, config["grid"]))))
        report["trajectory"].update(minimum_step=.08, maximum_step=.12)
        report["snapshots"][1]["source_step_index"] = 3
        self.assertTrue(runner.validate_report(report, config)["adaptive"])
        report["snapshots"][1]["energy"] = report["history"][3]["energy"]
        with self.assertRaisesRegex(ValueError, "energy/mass history"):
            runner.validate_report(report, config)

    def test_unsplit_energy_growth_is_reported_not_rejected(self):
        config, report = fixture("CN", legacy=False, grows_energy=True)
        metrics = runner.validate_report(report, config)
        self.assertEqual(metrics["energy_increase_steps"], 4)
        self.assertGreater(metrics["maximum_energy_increase"], 0)
        report["history"][1]["step_defect_rms"] = 0
        with self.assertRaisesRegex(ValueError, "split-Euler diagnostics"):
            runner.validate_report(report, config)

    def test_wrong_policy_reference_and_physical_dissipation_are_rejected(self):
        mutations = [lambda r: r.update(energy_policy="convex_split_euler"),
                     lambda r: r["reference"].update(method="CN"),
                     lambda r: r["history"][2].update(physical_dissipation=-1),
                     lambda r: r["history"][2].update(physical_dissipation=100)]
        for mutate in mutations:
            config, report = fixture("ROS3", legacy=False)
            mutate(report)
            with self.assertRaises(ValueError):
                runner.validate_report(report, config)

    def test_comparison_requires_common_physics_and_times(self):
        first = fixture("IMEX_EULER", legacy=False)[1]
        second = fixture("CN", legacy=False)[1]
        limits = runner.comparison_limits([first, second])
        self.assertEqual(len(limits), 2)
        for mutate in (lambda r: r["configuration"].update(grid=16),
                       lambda r: r["snapshots"][1].update(time=.3),
                       lambda r: r.update(method="IMEX_EULER")):
            bad = copy.deepcopy(second)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.comparison_limits([first, bad])

    def test_degenerate_and_wrong_order(self):
        for exponent in (0, 2):
            config, report = fixture()
            for row in report["refinements"]:
                row["error_rms"] = .01*row["step"]**exponent
            with self.subTest(exponent=exponent), self.assertRaises(ValueError):
                runner.validate_report(report, config)
        with self.assertRaisesRegex(ValueError, "numerical floor"):
            runner.measured_order([dict(step=.1, error_rms=1e-15)]*3, 1e-11)

    def test_corrupt_numerical_reports_are_rejected(self):
        mutations = [lambda r: r["history"][1].update(energy=100),
                     lambda r: r["history"][1].update(mass=100),
                     lambda r: r["history"][1].update(step_defect_rms=1e-8),
                     lambda r: r["history"][1].update(residual_work=1e-8),
                     lambda r: r["history"][1].update(roundoff_budget=1),
                     lambda r: r["history"][1].update(time=math.nan),
                     lambda r: r["reference"].update(difference_rms=1),
                     lambda r: r["reference"]["fine"].update(steps=3),
                     lambda r: r["refinements"][0].update(backend_allocations_after_warmup=1),
                     lambda r: r.update(integration_bulk_host_transfer_bytes=512),
                     lambda r: r.update(export_bulk_host_transfer_bytes=0),
                     lambda r: r.update(norm="unknown"),
                     lambda r: r["snapshots"][1].update(step_index=1),
                     lambda r: r["snapshots"][1]["concentration"].pop(),
                     lambda r: r["snapshots"][1]["concentration"].__setitem__(0, 99),
                     lambda r: r["history"].pop()]
        for index, mutate in enumerate(mutations):
            config, report = fixture()
            mutate(report)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                runner.validate_report(report, config)

    def test_physical_chemical_potential_not_stage_potential(self):
        config, _ = fixture()
        values = [.2]*config["grid"]**2
        energy, mass, mu = runner.field_diagnostics(values, config)
        self.assertAlmostEqual(energy, (2*math.pi)**2*(.2**2-1)**2/4)
        self.assertAlmostEqual(mass, (2*math.pi)**2*.2)
        self.assertTrue(all(abs(v-(.2**3-.2)) < 1e-14 for v in mu))

    def test_periodic_field_stencil_matches_discrete_fourier_eigenvalue(self):
        config, _ = fixture()
        n = config["grid"]
        dx = 2*math.pi/n
        mode = [.1*math.cos(i*dx) for i in range(n) for j in range(n)]
        values = [.2+v for v in mode]
        eigenvalue = -4*math.sin(math.pi/n)**2/dx**2
        energy, _, mu = runner.field_diagnostics(values, config)
        expected_mu = [c**3-c-config["epsilon"]**2*eigenvalue*v for c, v in zip(values, mode)]
        expected_energy = dx**2*sum((c*c-1)**2/4-config["epsilon"]**2*c*eigenvalue*v/2
                                   for c, v in zip(values, mode))
        self.assertLess(max(abs(a-b) for a, b in zip(mu, expected_mu)), 1e-14)
        self.assertAlmostEqual(energy, expected_energy)

    def test_stamp_and_incremental_build_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.arguments(["--build-dir", tmp])
            with patch.object(runner.subprocess, "run") as run:
                runner.build(args, runner.configuration(args))
                stamp = Path(tmp)/"bin/cpu_build_options.json"
                first = stamp.stat().st_mtime_ns
                runner.build(args, runner.configuration(args))
                self.assertEqual(stamp.stat().st_mtime_ns, first)
                self.assertNotIn("-B", run.call_args.args[0])
                args.make_option = ["G++=clang++"]
                runner.build(args, runner.configuration(args))
                self.assertIn("G++ = clang++", run.call_args.args[0])
                self.assertEqual(json.loads(stamp.read_text())["options"], args.make_option)
                args.build_only = True
                runner.build(args, runner.configuration(args))
                self.assertIn("-B", run.call_args.args[0])

    def test_build_only_does_not_import_plotting_or_run(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "ensure_plotting") as plotting, \
                patch.object(runner, "build") as build, patch.object(runner.subprocess, "run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["-B", "--build-dir", tmp]), 0)
            plotting.assert_not_called()
            build.assert_called_once()
            run.assert_not_called()
            self.assertFalse((Path(tmp)/"runs").exists())

    def test_plot_only_validates_without_building(self):
        config, report = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            runner.write_if_changed(path/"manifest.json", dict(returncode=0))
            runner.write_if_changed(path/"config.json", config)
            runner.write_if_changed(path/"report.json", report)
            with patch.object(runner, "build") as build, patch.object(runner, "ensure_plotting"), \
                    patch.object(runner, "plot") as plot, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--plot-only", tmp]), 0)
                build.assert_not_called()
                plot.assert_called_once()
            self.assertTrue(json.loads((path/"validation.json").read_text())["passed"])
            self.assertTrue((path/"history.csv").exists())

    def test_failure_and_timeout_never_plot_or_reuse_report(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as tmp:
                outcome = subprocess.CompletedProcess([], 1)
                error = subprocess.TimeoutExpired([], .01) if timeout else None
                with patch.object(runner, "build", return_value=Path(tmp)/"binary"), \
                        patch.object(runner, "ensure_plotting"), patch.object(runner, "plot") as plot, \
                        patch.object(runner.subprocess, "run", return_value=outcome, side_effect=error), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(runner.main(["--build-dir", tmp]), 1)
                    plot.assert_not_called()
                directory = next((Path(tmp)/"runs").iterdir())
                self.assertFalse(json.loads((directory/"validation.json").read_text())["passed"])
                cases = json.loads((directory/"manifest.json").read_text())["cases"]
                self.assertEqual(len(cases), len(runner.METHODS))
                for case in cases:
                    self.assertEqual(case["returncode"], 124 if timeout else 1)
                    self.assertTrue((directory/case["directory"]/"run.log").exists())

    def test_multimethod_execution_continues_after_a_failure(self):
        def execute(command, **kwargs):
            config = json.loads(Path(command[1]).read_text())
            if config["method"] == "IE":
                return subprocess.CompletedProcess(command, 3)
            _, report = fixture(config["method"], legacy=False)
            report["configuration"] = config
            runner.write_if_changed(Path(command[2]), report)
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "build", return_value=Path(tmp)/"binary"), \
                patch.object(runner, "ensure_plotting"), patch.object(runner, "plot") as plot, \
                patch.object(runner, "plot_comparison") as comparison, \
                patch.object(runner.subprocess, "run", side_effect=execute), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.main(["--build-dir", tmp, "--methods", "IMEX_EULER", "IE", "CN",
                "--grid", "8", "--end", ".4", "--step", ".1", "--levels", "3", "--base-steps", "4", "--frames", "3"]), 1)
            self.assertEqual(plot.call_count, 2)
            comparison.assert_called_once()
            self.assertEqual(plot.call_args_list[0].args[3], plot.call_args_list[1].args[3])
            directory = next((Path(tmp)/"runs").iterdir())
            result = json.loads((directory/"validation.json").read_text())
            self.assertFalse(result["passed"])
            self.assertEqual(result["included_methods"], ["IMEX_EULER", "CN"])
            self.assertEqual(result["excluded_methods"], ["IE"])
            self.assertTrue((directory/"comparison.csv").exists())

    def test_interrupted_batch_is_not_reported_as_complete(self):
        config, report = fixture(legacy=False)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "plot"), \
                patch.object(runner, "plot_comparison"), contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp)
            case = path/"IMEX_EULER"
            runner.write_if_changed(case/"config.json", config)
            runner.write_if_changed(case/"report.json", report)
            runner.write_if_changed(path/"manifest.json", dict(methods=["IMEX_EULER", "CN"],
                cases=[dict(method="IMEX_EULER", directory="IMEX_EULER", returncode=0)]))
            self.assertEqual(runner.summarize(path), 1)
            result = json.loads((path/"validation.json").read_text())
            self.assertFalse(result["passed"])
            self.assertEqual(result["excluded_methods"], ["CN"])

    def test_ci_and_all_backend_build_targets(self):
        workflow = (ROOT/".github/workflows/operations.yml").read_text()
        self.assertIn("test_ch_benchmarks.py", workflow)
        self.assertIn("source/nonlinear_operators/cahn_hilliard/tests/run_all.sh", workflow)
        self.assertIn("ch-energy-timestepper-validation", workflow)
        makefile = (ROOT/"Makefile").read_text()
        for backend in ("cpu", "cpu_omp", "cuda", "hip"):
            self.assertIn(f"$(BUILD_DIR)/ch_energy_benchmark_{backend}.bin $(BUILD_DIR)/ch_fourier_benchmark_{backend}.bin:", makefile)

    def test_fourier_selects_separate_binary_and_directory(self):
        args = runner.arguments(["--spatial", "fourier", "-B"])
        self.assertEqual(args.build_dir.name, "ch_fourier_benchmarks")
        self.assertEqual(runner.configuration(args)["spatial"], "fourier")
        with tempfile.TemporaryDirectory() as path, patch.object(subprocess, "run") as command:
            args.build_dir = Path(path)
            result = runner.build(args, runner.configuration(args))
            self.assertEqual(result.name, "ch_fourier_benchmark_cpu.bin")
            self.assertIn(result.name, command.call_args.args[0])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Fourier postprocessing needs NumPy")
    def test_fourier_potential_and_quartic_energy_at_cutoff(self):
        d, a, eps = .2, .3, .2
        for n in (8, 9):
            k = (n-1)//2
            wave = [math.cos(2*math.pi*(k*i-2*j)/n) for i in range(n) for j in range(n)]
            c = [d+a*v for v in wave]
            e, mass, mu = runner.field_diagnostics(c, dict(grid=n,epsilon=eps,spatial="fourier"))
            volume = (2*math.pi)**2
            fourth = d**4 + 3*d*d*a*a + 3*a**4/8
            expected = (fourth-2*(d*d+a*a/2)+1)/4+eps**2*(k*k+4)*a*a/4
            self.assertAlmostEqual(e, volume*expected)
            self.assertAlmostEqual(mass, volume*d)
            for w, value in zip(wave,mu):
                self.assertAlmostEqual(value, d**3+1.5*d*a*a-d+
                                       a*(3*d*d+.75*a*a-1+eps**2*(k*k+4))*w)


if __name__ == "__main__":
    unittest.main()
