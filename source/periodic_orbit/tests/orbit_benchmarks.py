#!/usr/bin/env python3
"""Build, independently validate and plot seeded periodic-orbit correction."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ODE_CASES = {
    "rossler_periodic": dict(period=6.0309863678295104, steps={"RK45": .01, "DOP853": .04},
                             parameters=dict(a=.1, b=.1, c=6)),
    "lorenz_periodic": dict(period=1.6443075739726336, steps={"RK45": .002, "DOP853": .01},
                            parameters=dict(sigma=10, rho=28, beta=8/3, epsilon=.0055, delta=0)),
}
BUILD_ENVIRONMENT = (
    "PATH", "G++", "GCC_ROOT_PATH", "TARGET_GCC", "CPPSTD", "NVCC", "CUDA_ROOT_PATH",
    "CUDA_ARCH_LIST", "TARGET_NVCC", "HIPCC", "HIPFLAGS", "HIP_ROOT_PATH", "HIP_PATH",
    "HIP_ARCH_LIST", "TARGET_HIPCC", "NMFD_TIMESTEPPERS_ROOT", "CONTRIB_NMFD_OPERATIONS", "CONTRIB_SCFD",
)


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite value")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-B", dest="build_only", action="store_true", help="Force build and assembly only")
    mode.add_argument("--plot-only", type=Path, help="Revalidate and plot an existing run, without building")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/periodic_orbit_benchmarks")
    parser.add_argument("--systems", "--system", nargs="+", choices=("analytic_cycle", "analytic_cycle_newton", *ODE_CASES), default=["analytic_cycle"])
    parser.add_argument("--sections", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--methods", nargs="+", choices=("RK45", "DOP853"), default=["DOP853"])
    parser.add_argument("--backend", choices=("cpu", "cpu_omp", "cuda", "hip"), default="cpu")
    parser.add_argument("--step", type=positive, help="Override the system/method-specific default step")
    parser.add_argument("--start", type=float, default=.37)
    parser.add_argument("--perturbation", type=positive, default=.02)
    parser.add_argument("--matching-tolerance", type=positive, default=1e-6)
    parser.add_argument("--newton-tolerance", type=positive, default=1e-10)
    parser.add_argument("--newton-max-iterations", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=positive, default=300)
    parser.add_argument("--config-file", type=Path, default=ROOT / "build_configs/config_github_ubuntu_cpu.inc")
    parser.add_argument("--make-option", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.start) or (args.step is not None and args.start + args.step == args.start) or
            args.perturbation > .1 or args.max_steps < 1 or args.threads < 1 or args.newton_max_iterations < 1 or
            any(n < 1 or n > 64 for n in args.sections)):
        parser.error("Invalid start, section count, perturbation or execution budget")
    for option in args.make_option:
        key, equal, _ = option.partition("=")
        if not equal or key in ("BUILD_DIR", "CONFIG_FILE", "PERIODIC_BENCHMARK_OPTIONS") or not key.replace("_", "").replace("+", "").isalnum():
            parser.error("Use --make-option KEY=VALUE; build directory/configuration are separate options")
    args.build_dir = args.build_dir.resolve()
    args.config_file = args.config_file.resolve()
    args.sections = list(dict.fromkeys(args.sections))
    args.methods = list(dict.fromkeys(args.methods))
    args.systems = list(dict.fromkeys(args.systems))
    if any(args.start + c["step"] == args.start for c in configurations(args)):
        parser.error("The chosen step cannot advance the starting time")
    return args


def configurations(args):
    configs = [dict(system=system, backend=args.backend, method=method, sections=count,
                 step=args.step or (ODE_CASES[system]["steps"][method] if system in ODE_CASES else .08),
                 start_time=args.start, perturbation=args.perturbation,
                 matching_tolerance=args.matching_tolerance, maximum_steps=args.max_steps)
            for system in args.systems for method in args.methods for count in args.sections]
    for config in configs:
        if config["system"] != "analytic_cycle":
            config.update(newton_tolerance=args.newton_tolerance, newton_max_iterations=args.newton_max_iterations)
    return configs


def write_if_changed(path, value):
    text = json.dumps(value, indent=2) + "\n"
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def build(args, configs):
    binary_dir = args.build_dir / "bin"
    options = binary_dir / f"{args.backend}_build_options.json"
    write_if_changed(options, dict(root=str(ROOT), configuration=str(args.config_file),
        configuration_text=args.config_file.read_text(), options=args.make_option,
        environment={name: os.environ.get(name) for name in BUILD_ENVIRONMENT}))
    targets = {system: f"periodic_orbit_benchmark_{system}_{args.backend}.bin" for system in args.systems}
    command = ["make", "--no-print-directory", "-C", str(ROOT)]
    if args.build_only:
        command.append("-B")
    command.extend([*targets.values(), f"BUILD_DIR={binary_dir}", f"CONFIG_FILE={args.config_file}",
                    f"PERIODIC_BENCHMARK_OPTIONS={options}",
                    # Spaces prevent G++=... being parsed as G+ += ... by Make.
                    *(" = ".join(option.split("=", 1)) for option in args.make_option)])
    log_path = args.build_dir / f"build_{args.backend}.log"
    with log_path.open("w") as log:
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Build failed; see {log_path}") from error
    for config in configs:
        write_if_changed(args.build_dir / "cases" / case_name(config) / "config.json", config)
    return {system: binary_dir / target for system, target in targets.items()}


def case_name(config):
    return f"{config['system']}_{config['method']}_M{config['sections']}"


def stage(configs):
    stages = {"matching_only" if c["system"] == "analytic_cycle" else "periodic_newton" for c in configs}
    return next(iter(stages)) if len(stages) == 1 else "mixed"


def section_samples(directory, filename, config, part):
    import numpy as np
    samples = np.genfromtxt(directory / filename, delimiter=",", names=True, ndmin=1)
    count = config["sections"]
    if len(samples) != count or not np.array_equal(samples["section"], np.arange(count)):
        raise ValueError("Incomplete section export")
    if not all(np.isfinite(samples[key]).all() for key in samples.dtype.names):
        raise ValueError("Nonfinite section export")
    axes = "xyz" if config["system"] in ODE_CASES else "xy"
    def values(prefix):
        return np.column_stack([samples[prefix + axis] for axis in axes])
    state, center, normal, arrival, target = (values(p) for p in ("", "c", "n", "arrival_", "target_"))
    gap = np.linalg.norm(arrival - target, axis=1)
    normal_error = np.sum(normal * (state - center), axis=1)
    arrival_error = np.sum(np.roll(normal, -1, axis=0) * (arrival - np.roll(center, -1, axis=0)), axis=1)
    target_error = np.linalg.norm(target - np.roll(state, -1, axis=0), axis=1)
    if (np.max(np.abs(gap - samples["gap"])) > 1e-10 or np.max(np.abs(normal_error)) > 1e-8 or
            np.max(np.abs(arrival_error)) > 1e-8 or np.max(target_error) > 1e-10 or
            not np.allclose(samples["flight_time"], [s["flight_time"] for s in part["segments"]], atol=1e-12, rtol=1e-12) or
            not np.array_equal(samples["steps"], [s["steps"] for s in part["segments"]]) or
            not np.allclose(gap, [s["gap_norm"] for s in part["segments"]], atol=1e-10, rtol=1e-9)):
        raise ValueError("Exported geometry disagrees with physical matching gaps")
    return samples


def validate_case(directory, config):
    if config["system"] != "analytic_cycle":
        return validate_newton_case(directory, config)
    report = json.loads((directory / "report.json").read_text())
    if (report.get("format") != "periodic_orbit.matching.v1" or report.get("stage") != "matching_only" or
            report.get("orbit_corrected") is not False or report.get("configuration") != config or report["passed"] is not True):
        raise ValueError("Missing, inconsistent or failed matching report")
    import numpy as np
    tolerance, count = config["matching_tolerance"], config["sections"]
    for name in ("baseline", "refined", "perturbed"):
        part = report[name]
        if not part["succeeded"] or len(part["segments"]) != count:
            raise ValueError(f"Incomplete {name} itinerary")
        times = [segment["flight_time"] for segment in part["segments"]]
        if not all(math.isfinite(t) and t > 0 for t in times) or not math.isclose(sum(times), part["period"], abs_tol=1e-12):
            raise ValueError("Invalid flight-time accounting")
        for segment in part["segments"]:
            if (segment["steps"] < 1 or not math.isfinite(segment["crossing_speed"]) or
                    segment["crossing_speed"] <= 1e-10 or abs(segment["section_value"]) > 1e-8):
                raise ValueError("Invalid section crossing")
    for name in ("baseline", "refined"):
        if abs(report[name]["period"] - 2 * math.pi) >= tolerance or not 0 <= report[name + "_norm"] < tolerance:
            raise ValueError("Cycle period/matching accuracy failed")
    if (report["matching_host_bytes"] != 0 or report["perturbed_norm"] <= config["perturbation"] * .01 or
            not 0 <= report["closure_error"] < tolerance or not 0 <= report["trajectory_error"] < tolerance):
        raise ValueError("Transfer, perturbation or independent closure check failed")
    trajectory = np.genfromtxt(directory / "trajectory.csv", delimiter=",", names=True, ndmin=1)
    time_values = trajectory["time"]
    if (len(trajectory) != report["trajectory_rows"] or len(trajectory) < 2 or not np.all(np.diff(time_values) > 0) or
            time_values[0] != config["start_time"] or time_values[-1] != config["start_time"] + 2 * math.pi):
        raise ValueError("Incomplete trajectory/time accounting")
    error = np.hypot(trajectory["x"] - np.cos(time_values - time_values[0]),
                     trajectory["y"] - np.sin(time_values - time_values[0]))
    if not np.isfinite(error).all() or np.max(error) >= tolerance:
        raise ValueError("Trajectory disagrees with analytical cycle")
    for filename, name in (("sections.csv", "baseline"), ("perturbed_sections.csv", "perturbed")):
        section_samples(directory, filename, config, report[name])
    return report


def validate_newton_case(directory, config):
    report = json.loads((directory / "report.json").read_text())
    ode = config["system"] in ODE_CASES
    expected_format = "periodic_orbit.ode_newton.v1" if ode else "periodic_orbit.newton.v1"
    reference_period = ODE_CASES[config["system"]]["period"] if ode else 2 * math.pi
    if (report.get("format") != expected_format or report.get("stage") != "periodic_newton" or
            report.get("orbit_corrected") is not True or report.get("configuration") != config or
            report["passed"] is not True or report["result_status"] != 0):
        raise ValueError("Missing, inconsistent or failed Newton report")
    import numpy as np
    tolerance = config["matching_tolerance"]
    for name in ("initial", "corrected", "refined"):
        part = report[name]
        if not part["succeeded"] or len(part["segments"]) != config["sections"]:
            raise ValueError(f"Incomplete {name} itinerary")
        times = [s["flight_time"] for s in part["segments"]]
        if not all(math.isfinite(t) and t > 0 for t in times) or not math.isclose(sum(times), part["period"], abs_tol=1e-12):
            raise ValueError("Invalid flight-time accounting")
        for s in part["segments"]:
            if (s["steps"] < 1 or not math.isfinite(s["crossing_speed"]) or s["crossing_speed"] <= 1e-10 or
                    not math.isfinite(s["gap_norm"]) or s["gap_norm"] < 0 or not abs(s["section_value"]) <= 1e-8):
                raise ValueError("Invalid section crossing")
    if (report["matching_host_bytes"] != 0 or not report["initial_norm"] > config["perturbation"] * .01 or
            not 0 <= report["corrected_norm"] <= config["newton_tolerance"] or
            not 0 <= report["refined_norm"] < tolerance or
            any(not abs(report[name]["period"] - reference_period) < tolerance for name in ("corrected", "refined")) or
            len(report["closure_errors"]) != 2 or any(not 0 <= e < tolerance for e in report["closure_errors"])):
        raise ValueError("Correction, refinement, closure or transfer check failed")
    for filename, name in (("initial_sections.csv", "initial"), ("corrected_sections.csv", "corrected")):
        samples = section_samples(directory, filename, config, report[name])
        if not ode and name == "corrected" and np.max(np.abs(np.hypot(samples["x"], samples["y"]) - 1)) >= tolerance:
            raise ValueError("Corrected sections disagree with isolated cycle")
        axes = "xyz" if ode else "xy"
        gap = np.column_stack([samples["arrival_" + a] - samples["target_" + a] for a in axes])
        normal = np.roll(np.column_stack([samples["n" + a] for a in axes]), -1, axis=0)
        if not np.allclose(np.linalg.norm(normal, axis=1), 1, rtol=1e-12, atol=1e-12):
            raise ValueError("Section normals are not normalized")
        # Newton uses reduced coordinates. The separate physical section check
        # bounds the normal defect; do not confuse that defect with tangent gaps.
        projected = gap - normal * np.sum(normal * gap, axis=1)[:, None]
        if not math.isclose(np.linalg.norm(projected), report[name + "_norm"], rel_tol=1e-6, abs_tol=1e-10):
            raise ValueError("Reported norm disagrees with exported section-coordinate gaps")
    trajectories = []
    for filename in ("trajectory.csv", "refined_trajectory.csv"):
        trajectory = np.genfromtxt(directory / filename, delimiter=",", names=True, ndmin=1)
        t = trajectory["time"]
        if (len(t) < 2 or not np.all(np.diff(t) > 0) or t[0] != config["start_time"] or
                t[-1] != config["start_time"] + report["corrected"]["period"]):
            raise ValueError("Independent trajectory is incomplete")
        if not all(np.isfinite(trajectory[key]).all() for key in trajectory.dtype.names):
            raise ValueError("Nonfinite independent trajectory")
        if not ode:
            error = np.hypot(trajectory["x"] - np.cos(t - t[0]), trajectory["y"] - np.sin(t - t[0]))
            if np.max(error) >= tolerance:
                raise ValueError("Independent trajectory disagrees with analytical cycle")
        trajectories.append(trajectory)
    if ode:
        if (report["parameters"] != ODE_CASES[config["system"]]["parameters"] or
                report["reference_period"] != reference_period or not math.isfinite(report["rate_norm"]) or
                report["rate_norm"] <= 1 or not math.isfinite(report["excursion"]) or report["excursion"] <= 1):
            raise ValueError("ODE parameter/reference or non-equilibrium check failed")
        states = [np.column_stack([t[axis] for axis in "xyz"]) for t in trajectories]
        if (any(len(t) != 401 for t in trajectories) or
                not np.array_equal(trajectories[0]["time"], trajectories[1]["time"]) or
                np.max(np.linalg.norm(states[0] - states[1], axis=1)) >= tolerance):
            raise ValueError("Independent half/quarter-step trajectories disagree")
        for i, state in enumerate(states):
            closure = np.linalg.norm(state[-1] - state[0])
            if not math.isclose(closure, report["closure_errors"][i], rel_tol=1e-6, abs_tol=1e-10):
                raise ValueError("Exported trajectory disagrees with reported closure")
    history = np.genfromtxt(directory / "newton_history.csv", delimiter=",", names=True, ndmin=1)
    norms = np.array(report["history"])
    if (len(norms) < 2 or not np.isfinite(norms).all() or np.any(norms < 0) or
            len(norms) > config["newton_max_iterations"] + 2 or norms[0] != report["initial_norm"] or
            not np.array_equal(history["iteration"], np.arange(len(norms))) or
            not np.allclose(history["residual"], norms, rtol=1e-12, atol=1e-15)):
        raise ValueError("Invalid Newton convergence history")
    return report


def plot_newton_case(directory, config):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = validate_newton_case(directory, config)
    trajectory = np.genfromtxt(directory / "trajectory.csv", delimiter=",", names=True, ndmin=1)
    initial = np.genfromtxt(directory / "initial_sections.csv", delimiter=",", names=True, ndmin=1)
    solved = np.genfromtxt(directory / "corrected_sections.csv", delimiter=",", names=True, ndmin=1)
    figure = plt.figure(figsize=(8, 6))
    ode = config["system"] in ODE_CASES
    if ode:
        ax = figure.add_subplot(111, projection="3d")
        reference = np.genfromtxt(directory / "refined_trajectory.csv", delimiter=",", names=True, ndmin=1)
        guess = np.genfromtxt(directory / "initial_trajectory.csv", delimiter=",", names=True, ndmin=1)
        ax.plot(guess["x"], guess["y"], guess["z"], color="#be493e", alpha=.5, label="Initial-state trajectory")
        ax.plot(reference["x"], reference["y"], reference["z"], "--", color="0.5", label="Quarter-step verification")
        ax.plot(trajectory["x"], trajectory["y"], trajectory["z"], color="#147a78", label="Corrected orbit")
        ax.scatter(initial["x"], initial["y"], initial["z"], marker="x", color="#be493e", label="Initial section states")
        ax.scatter(solved["x"], solved["y"], solved["z"], color="#284c98", label="Corrected section states")
        ax.set(zlabel="z")
    else:
        ax = figure.add_subplot(111)
        theta = np.linspace(0, 2 * np.pi, 500)
        ax.plot(np.cos(theta), np.sin(theta), "--", color="0.6", label="Exact cycle")
        ax.plot(trajectory["x"], trajectory["y"], color="#147a78", label="Corrected, independently integrated")
        ax.scatter(initial["x"], initial["y"], marker="x", color="#be493e", label="Initial section states")
        ax.scatter(solved["x"], solved["y"], color="#284c98", label="Corrected section states")
        ax.set(aspect="equal")
    title = "Lorenz (cubic damping)" if config["system"] == "lorenz_periodic" else config["system"].replace("_", " ")
    ax.set(xlabel="x", ylabel="y", title=f"{title}: {config['method']}, M={config['sections']}, T={report['corrected']['period']:.8f}")
    ax.legend(fontsize="small")
    figure.savefig(directory / "phase.png", dpi=150, bbox_inches="tight",
                   bbox_extra_artists=(ax.zaxis.label,) if ode else None)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(trajectory["time"], trajectory["x"], label="x")
    axes[0, 0].plot(trajectory["time"], trajectory["y"], label="y")
    if ode:
        axes[0, 0].plot(trajectory["time"], trajectory["z"], label="z")
    axes[0, 0].set(xlabel="Time", title="Independent one-period trajectory")
    axes[0, 0].legend()
    axes[0, 1].semilogy(np.maximum(report["history"], 1e-17), "o-", color="#147a78")
    axes[0, 1].set(xlabel="Newton iteration", ylabel="Matching residual norm")
    for name in ("initial", "corrected", "refined"):
        axes[1, 0].semilogy(np.maximum([s["gap_norm"] for s in report[name]["segments"]], 1e-17), "o-", label=name)
    axes[1, 0].set(xlabel="Segment", ylabel="Physical matching gap")
    axes[1, 0].legend()
    axes[1, 1].bar(np.arange(config["sections"]), solved["steps"], color="#284c98")
    axes[1, 1].set(xlabel="Segment", ylabel="Accepted steps")
    figure.savefig(directory / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return dict(passed=True, corrected_norm=report["corrected_norm"], period=report["corrected"]["period"])


def plot_case(directory, config):
    if config["system"] != "analytic_cycle":
        return plot_newton_case(directory, config)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = validate_case(directory, config)
    trajectory = np.genfromtxt(directory / "trajectory.csv", delimiter=",", names=True, ndmin=1)
    sections = np.genfromtxt(directory / "sections.csv", delimiter=",", names=True, ndmin=1)
    perturbed = np.genfromtxt(directory / "perturbed_sections.csv", delimiter=",", names=True, ndmin=1)
    figure, ax = plt.subplots(figsize=(7, 6))
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="0.6", label="Exact cycle")
    ax.plot(trajectory["x"], trajectory["y"], color="#147a78", label="Independent integration")
    ax.scatter(sections["cx"], sections["cy"], marker="o", color="#284c98", label="Section centers")
    ax.scatter(perturbed["x"], perturbed["y"], marker="x", color="#be493e", label="Perturbed section states")
    for s in sections:
        tangent = np.array([-s["ny"], s["nx"]]) * .13
        ax.plot([s["cx"] - tangent[0], s["cx"] + tangent[0]], [s["cy"] - tangent[1], s["cy"] + tangent[1]], color="#284c98")
        if len(sections) <= 16:
            ax.annotate(str(int(s["section"])), (s["cx"], s["cy"]), xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="x", ylabel="y", title=f"Analytical cycle: {config['method']}, M={config['sections']} (matching only)", aspect="equal")
    ax.legend(fontsize="small")
    figure.savefig(directory / "phase.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(trajectory["time"], trajectory["x"], label="x")
    axes[0, 0].plot(trajectory["time"], trajectory["y"], label="y")
    axes[0, 0].set(xlabel="Time", title="Independent one-period trajectory")
    axes[0, 0].legend()
    index = np.arange(config["sections"])
    for name in ("baseline", "refined", "perturbed"):
        axes[0, 1].semilogy(index, np.maximum([s["gap_norm"] for s in report[name]["segments"]], 1e-17), "o-", label=name)
    axes[0, 1].set(xlabel="Segment", ylabel="Physical matching gap")
    axes[0, 1].legend()
    axes[1, 0].bar(index, sections["flight_time"], color="#147a78")
    axes[1, 0].set(xlabel="Segment", ylabel="Flight time")
    axes[1, 1].bar(index, sections["steps"], color="#284c98")
    axes[1, 1].set(xlabel="Segment", ylabel="Accepted steps")
    figure.savefig(directory / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return dict(passed=True, baseline_norm=report["baseline_norm"], perturbed_norm=report["perturbed_norm"], period=report["baseline"]["period"])


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    results = {}
    for case in manifest["cases"]:
        name = case_name(case["configuration"])
        try:
            if case["returncode"]:
                raise ValueError(f"Executable failed with status {case['returncode']}; see {name}/run.log")
            results[name] = plot_case(directory / name, case["configuration"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            results[name] = dict(passed=False, error=str(error))
        print(f"{name}: {'PASS' if results[name]['passed'] else 'FAIL'}")
    passed = bool(results) and all(case["passed"] for case in results.values())
    (directory / "summary.json").write_text(json.dumps(dict(passed=passed, stage=stage([c["configuration"] for c in manifest["cases"]]), cases=results), indent=2) + "\n")
    return 0 if passed else 1


def main(argv=None):
    args = arguments(argv)
    if not args.build_only:
        cache_root = args.plot_only.resolve().parent if args.plot_only else args.build_dir
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    if args.plot_only:
        return summarize(args.plot_only.resolve())
    configs = configurations(args)
    executables = build(args, configs)
    if args.build_only:
        print(f"Built and assembled {len(configs)} cases in {args.build_dir}; no cases executed")
        return 0
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns()}"
    directory = args.build_dir / "runs" / batch
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(stage=stage(configs), executables={k: str(v) for k, v in executables.items()}, cases=[])
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
    for config in configs:
        case_dir = directory / case_name(config)
        case_dir.mkdir()
        path = case_dir / "config.json"
        write_if_changed(path, config)
        before = time.perf_counter()
        with (case_dir / "run.log").open("w") as log:
            try:
                process = subprocess.run([str(executables[config["system"]]), str(path), str(case_dir)], env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=args.timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                log.write("\nBenchmark timeout reached\n")
                code = 124
        manifest["cases"].append(dict(configuration=config, returncode=code, seconds=time.perf_counter() - before))
        write_if_changed(directory / "manifest.json", manifest)
    result = summarize(directory)
    print(f"Results: {directory}")
    return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
