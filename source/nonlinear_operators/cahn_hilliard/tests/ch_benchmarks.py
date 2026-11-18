#!/usr/bin/env python3
"""Compare split and unsplit Cahn-Hilliard timesteppers on the same periodic grid."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUILD_ENVIRONMENT = ("PATH", "G++", "G++FLAGS", "GCC_ROOT_PATH", "CPPSTD", "NVCC", "NVCCFLAGS", "CUDA_ROOT_PATH", "CUDA_ARCH_LIST",
                     "HIPCC", "HIPFLAGS", "HIP_ROOT_PATH", "HIP_PATH", "HIP_ARCH_LIST",
                     "NMFD_TIMESTEPPERS_ROOT", "CONTRIB_NMFD_OPERATIONS", "CONTRIB_SCFD")
PARAMETERS = ("backend", "grid", "end", "step", "order_end", "base_steps", "levels", "frames",
              "mobility", "epsilon", "mean", "amplitude")
METHODS = {"IMEX_EULER": 1, "IE": 1, "CN": 2, "SDIRK2(1)2": 2, "BDF2": 2, "ROS3": 3,
           "ARK324L2SA": 3, "ROS34PW2": 3, "BDF": None, "IMEX_ARK324L2SA": 3}
ADAPTIVE_METHODS = ("ARK324L2SA", "ROS34PW2", "IMEX_ARK324L2SA")
ADAPTIVE_KEYS = ("adaptive", "relative_tolerance", "absolute_tolerance", "initial_step", "maximum_order", "maximum_steps")


def method_label(method):
    if method == "IMEX_EULER":
        return "IMEX Euler (convex split)"
    return method + (" (split)" if method.startswith("IMEX_") else " (unsplit)")


def case_label(report):
    return (method_label(report["method"]) +
            (" [adaptive]" if report.get("validation_mode") == "tolerance_refinement" else ""))


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter,
        epilog="""Step counts:
  The main trajectory starts at t=0. Its uniform step count is ceil(end/step),
  capped at 20000; the nominal timestep is end/count (floating-point roundoff applies).
  Defaults: 6 / 0.1 = 60 steps. Use --end 6 --step 0.01 for 600 steps.
  More steps at the same --end improve time resolution; increasing --end instead
  simulates longer physical time. Neither changes the separate order study.

Order study:
  Each level doubles --base-steps over [0, --order-end]. Defaults: 5, 10, 20, 40
  steps. Common ROS3 references use 8 and 16 times the finest count (320, 640).
  These measure temporal convergence on the selected grid, not spatial order.
  Adaptive cases instead tighten rtol and atol by 10 per level. Adaptive BDF
  selects orders 1-5; tolerance response is not a fixed order-five claim.

Examples (append to run_all.sh):
  --grid 32 --end 6 --step 0.01 --timeout 1800
  --end 60 --step 0.01 --frames 9 --timeout 3600
  --order-end 0.2 --base-steps 10 --levels 4
  --backend cuda --grid 32 --end 6 --step 0.01
  --methods IMEX_EULER IE CN 'SDIRK2(1)2' BDF2 ROS3
  --methods ARK324L2SA ROS34PW2 BDF IMEX_ARK324L2SA
  --methods ARK324L2SA BDF --adaptive-methods ARK324L2SA --relative-tolerance 1e-5

Results are kept in timestamped runs under --build-dir. Bulk field exports occur
only after integration. Finer grids/longer runs may require more solver work;
this runner does not relax Newton/GMRES tolerances automatically.""")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-B", dest="build_only", action="store_true", help="Force rebuild and assemble only")
    mode.add_argument("--plot-only", type=Path, help="Revalidate and plot an existing run; no integration")
    trajectory = parser.add_argument_group("Main trajectory: energy and physical-field plots")
    trajectory.add_argument("--spatial", choices=("finite_difference", "fourier"), default="finite_difference",
                            help="Spatial operator; Fourier uses cubic-dealiased Galerkin projection")
    trajectory.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS),
                            help="Compared implicit/IMEX methods; BDF is adaptive, others fixed unless selected below. Only IMEX_EULER enforces its discrete energy bound")
    trajectory.add_argument("--grid", type=int, default=16, help="N for an N x N periodic grid; range 8-64")
    trajectory.add_argument("--end", type=float, default=6., help="Final physical time, starting at t=0")
    trajectory.add_argument("--step", type=float, default=.1,
                            help="Fixed timestep request, or adaptive maximum step; also sets the common snapshot-time grid (maximum 20000 fixed steps)")
    trajectory.add_argument("--frames", type=int, default=5,
                            help="Common physical-time field snapshots, including endpoints; range 2-9. Adaptive cases use dense output without forcing steps")
    order = parser.add_argument_group("Separate temporal-order study (does not set main-trajectory steps)")
    order.add_argument("--order-end", type=float, default=.2, help="Refinement final time, starting at t=0")
    order.add_argument("--base-steps", type=int, default=5, help="Coarsest fixed-step refinement count; also sizes the common references for adaptive cases; range 4-64")
    order.add_argument("--levels", type=int, default=4, help="Levels: double fixed step counts, or tighten adaptive rtol/atol tenfold; range 3-6")
    adaptation = parser.add_argument_group("Adaptive implicit/IMEX control (BDF is always adaptive)")
    adaptation.add_argument("--adaptive-methods", nargs="*", choices=ADAPTIVE_METHODS, default=[],
                            help="Use embedded error control for selected higher-order methods; BDF needs no selection")
    adaptation.add_argument("--relative-tolerance", type=float, default=1e-4, help="Adaptive relative state-error tolerance")
    adaptation.add_argument("--absolute-tolerance", type=float, default=1e-6, help="Adaptive absolute state-error tolerance")
    adaptation.add_argument("--initial-step", type=float, default=.02, help="Initial adaptive step, capped by --step")
    adaptation.add_argument("--maximum-order", type=int, choices=range(1, 6), default=5, help="Maximum adaptive BDF order")
    adaptation.add_argument("--maximum-steps", type=int, default=20000, help="Accepted-step budget per adaptive trajectory; range 1-20000")
    physics = parser.add_argument_group("Cahn-Hilliard model and initial concentration")
    physics.add_argument("--mobility", type=float, default=.5, help="Positive constant mobility m")
    physics.add_argument("--epsilon", type=float, default=.2, help="Positive interface parameter epsilon (equation uses epsilon squared)")
    physics.add_argument("--mean", type=float, default=.2, help="Mean initial concentration")
    physics.add_argument("--amplitude", type=float, default=.12,
                        help="Positive initial amplitude A in c=mean+A*(cos(x)+0.4*sin(y)+0.25*cos(2*x)*cos(y))")
    execution = parser.add_argument_group("Execution and output")
    execution.add_argument("--backend", choices=("cpu", "cpu_omp", "cuda", "hip"), default="cpu",
                           help="Numerical backend; CUDA/HIP require a matching compiler and usable GPU")
    execution.add_argument("--threads", type=int, default=1, help="OpenMP thread count (cpu_omp backend)")
    execution.add_argument("--timeout", type=float, default=600,
                           help="Wall-clock seconds per method, including its reference/refinement solves; excludes build and plots")
    execution.add_argument("--build-dir", type=Path,
                           help="Build/cache root (build/ch_energy_benchmarks or build/ch_fourier_benchmarks); results go under runs/")
    execution.add_argument("--config-file", type=Path, default=ROOT / "build_configs/config_github_ubuntu_cpu.inc",
                           help="Parent-project Make configuration")
    execution.add_argument("--make-option", action="append", default=[], metavar="KEY=VALUE",
                           help="Additional Make variable; repeat for multiple settings, e.g. --make-option G++=clang++")
    args = parser.parse_args(argv)
    args.methods = list(dict.fromkeys(args.methods))
    args.adaptive_methods = list(dict.fromkeys(args.adaptive_methods))
    if any(m not in args.methods for m in args.adaptive_methods):
        parser.error("Every --adaptive-methods selection must also occur in --methods")
    if (not all(math.isfinite(v) and v > 0 for v in (args.relative_tolerance, args.absolute_tolerance, args.initial_step)) or
            not 1 <= args.maximum_steps <= 20000 or
            not 3 <= args.levels <= 6 or
            args.relative_tolerance*.1**(args.levels-1) < 100*sys.float_info.epsilon):
        parser.error("Invalid adaptive tolerances, refinement floor or step budget")
    positive = (args.end, args.step, args.order_end, args.mobility, args.epsilon, args.amplitude, args.timeout)
    if (not all(math.isfinite(x) and x > 0 for x in positive) or not math.isfinite(args.mean) or
            not 8 <= args.grid <= 64 or not 3 <= args.levels <= 6 or not 4 <= args.base_steps <= 64 or
            not 2 <= args.frames <= 9 or args.threads < 1):
        parser.error("Invalid physical parameters, grid, refinement or storage budget")
    ratio = args.end / args.step
    if not math.isfinite(ratio) or not args.frames-1 <= math.ceil(ratio-1e-12) <= 20000:
        parser.error("Choose between frames-1 and 20000 uniform trajectory steps")
    if args.order_end / (args.base_steps * 2**(args.levels-1) * 16) < 1e-14:
        parser.error("Reference timestep is too small")
    for option in args.make_option:
        key, equal, _ = option.partition("=")
        if (not equal or key in ("BUILD_DIR", "CONFIG_FILE", "CH_BENCHMARK_OPTIONS") or
                not key.replace("_", "").replace("+", "").isalnum()):
            parser.error("Use --make-option KEY=VALUE; build paths have separate options")
    args.build_dir = (args.build_dir or ROOT / "build" / (
        "ch_fourier_benchmarks" if args.spatial == "fourier" else "ch_energy_benchmarks")).resolve()
    args.config_file = args.config_file.resolve()
    return args


def configuration(args, method=None):
    method = method or args.methods[0]
    config = dict({key: getattr(args, key) for key in PARAMETERS}, method=method,
                  adaptive=method == "BDF" or method in args.adaptive_methods)
    if args.spatial == "fourier":
        config["spatial"] = "fourier"
    if config["adaptive"]:
        config.update({key: getattr(args, key) for key in ADAPTIVE_KEYS if key != "adaptive"})
    return config


def write_if_changed(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def build(args, config):
    binary_dir = args.build_dir / "bin"
    options = binary_dir / f"{args.backend}_build_options.json"
    write_if_changed(options, dict(root=str(ROOT), configuration=str(args.config_file),
        configuration_text=args.config_file.read_text(), options=args.make_option,
        environment={key: os.environ.get(key) for key in BUILD_ENVIRONMENT}))
    family = "fourier" if args.spatial == "fourier" else "energy"
    target = f"ch_{family}_benchmark_{args.backend}.bin"
    command = ["make", "--no-print-directory", "-C", str(ROOT)]
    if args.build_only:
        command.append("-B")
    command.extend([target, f"BUILD_DIR={binary_dir}", f"CONFIG_FILE={args.config_file}",
                    f"CH_BENCHMARK_OPTIONS={options}",
                    *(" = ".join(option.split("=", 1)) for option in args.make_option)])
    log_path = args.build_dir / f"build_{args.backend}.log"
    with log_path.open("w") as log:
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Build failed; see {log_path}") from error
    write_if_changed(args.build_dir / "cases" / args.backend / config["method"] / "config.json", config)
    return binary_dir / target


def measured_order(rows, floor):
    eligible = [r for r in rows if r["error_rms"] > floor]
    if len(eligible) < 3:
        raise ValueError("Insufficient refinement points above the numerical floor")
    selected = eligible[-3:]
    x = [math.log(r["step"]) for r in selected]
    y = [math.log(r["error_rms"]) for r in selected]
    mx, my = sum(x)/3, sum(y)/3
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / sum((a-mx)**2 for a in x)


def field_diagnostics(values, config):
    """Independent postprocessing on exported fields, using their spatial discretization."""
    n = config["grid"]
    if len(values) != n*n or not all(math.isfinite(x) for x in values):
        raise ValueError("Invalid concentration snapshot")
    dx = 2*math.pi/n
    if config.get("spatial") == "fourier":
        return fourier_diagnostics(values, config)
    lap = periodic_laplacian(values, n)
    energy = dx**2 * math.fsum((c*c-1)**2/4 - config["epsilon"]**2*c*l/2 for c, l in zip(values, lap))
    mass = dx**2 * math.fsum(values)
    chemical_potential = [c**3-c-config["epsilon"]**2*l for c, l in zip(values, lap)]
    return energy, mass, chemical_potential


def periodic_laplacian(values, n):
    dx = 2*math.pi/n
    return [(values[((i+1) % n)*n+j] + values[((i-1) % n)*n+j] +
             values[i*n+(j+1) % n] + values[i*n+(j-1) % n] - 4*values[i*n+j]) / dx**2
            for i in range(n) for j in range(n)]


def fourier_diagnostics(values, config):
    # Full complex FFT on an odd oversized grid, independent of the C++ R2C implementation.
    import numpy as np
    n = config["grid"]
    m = 2*n+1
    c = np.asarray(values).reshape(n,n)
    z = np.fft.fft2(c, norm="forward")
    k = np.arange(-((n-1)//2), (n-1)//2+1)
    band = np.ix_(k % n, k % n)
    fine = np.ix_(k % m, k % m)
    padded = np.zeros((m,m), dtype=complex)
    padded[fine] = z[band]
    if n % 2 == 0:
        require(np.max(np.abs(z[n//2,:])) < 1e-11 and np.max(np.abs(z[:,n//2])) < 1e-11,
                "State outside the retained Fourier band")
    u = np.fft.ifft2(padded, norm="forward").real
    k2 = k[:,None]**2+k[None,:]**2
    eps2 = config["epsilon"]**2
    volume = (2*np.pi)**2
    energy = volume*(np.mean((u*u-1)**2)/4 + eps2*np.sum(k2*np.abs(z[band])**2)/2)
    mu = np.zeros_like(z)
    mu[band] = np.fft.fft2(u**3, norm="forward")[fine]+(eps2*k2-1)*z[band]
    return float(energy), float(volume*c.mean()), np.fft.ifft2(mu,norm="forward").real.ravel().tolist()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report(report, config):
    legacy = report["format"] == "ch.energy.visualization.v1"
    modern = report["format"] == "ch.energy.visualization.v3"
    method = config.get("method", "IMEX_EULER")
    require(method in METHODS, "Unknown CH method")
    adaptive = modern and (method == "BDF" or config.get("adaptive", False))
    expected_order = None if adaptive else METHODS[method]
    split = method == "IMEX_EULER"
    require(report["format"] in ("ch.energy.visualization.v1", "ch.energy.visualization.v2", "ch.energy.visualization.v3") and
            (not legacy or split) and report["configuration"] == config and
            report["method"] == method and report["expected_order"] == expected_order and
            report["norm"] == "physical_concentration_rms" and report["fit_error_floor"] == 1e-11,
            "Unexpected report contract")
    if modern:
        require(report["validation_mode"] == ("tolerance_refinement" if adaptive else "temporal_refinement") and
                report["primary_order"] == METHODS[method], "Incorrect refinement/primary-order metadata")
        require(method != "BDF" or adaptive, "Variable-order BDF requires adaptive control")
    if not legacy:
        require(report["energy_policy"] == ("convex_split_euler" if split else "observe_only") and
                report["reference"]["method"] == "ROS3", "Incorrect energy policy or common reference")
    sample_count = math.ceil(config["end"]/config["step"]-1e-12)
    count = report["trajectory"]["steps"] if adaptive else sample_count
    h = config["end"]/sample_count
    budget = config["maximum_steps"] if adaptive else count
    require(isinstance(count, int) and 1 <= count <= budget, "Invalid accepted-step count")
    time_roundoff = 8*sys.float_info.epsilon*budget*max(1., config["end"])
    history, rows, snapshots = report["history"], report["refinements"], report["snapshots"]
    require(len(history) == count+1 and len(rows) == config["levels"] and len(snapshots) == config["frames"],
            "Incomplete history, refinement or snapshots")
    initial = history[0]
    require(initial["time"] == 0 and all(math.isfinite(initial[k]) for k in ("energy", "mass")),
            "Invalid initial diagnostics")
    for k, row in enumerate(history[1:], 1):
        require(all(math.isfinite(row[key]) for key in ("time", "step", "energy", "mass")), "Nonfinite history")
        if adaptive:
            require(row["time"] > history[k-1]["time"] and 0 < row["step"] <= h+time_roundoff and
                    abs(row["time"]-history[k-1]["time"]-row["step"]) <= time_roundoff,
                    "Invalid adaptive accepted step")
        else:
            require(abs(row["time"]-k*h) <= time_roundoff and
                    abs(row["step"]-h) <= time_roundoff, "Incorrect accepted step or endpoint")
        previous = history[k-1]["energy"]
        if split:
            require(all(math.isfinite(row[key]) for key in ("dissipation", "residual_work", "bound_defect",
                        "roundoff_budget", "step_defect_rms")), "Nonfinite split diagnostics")
            budget = 256*sys.float_info.epsilon*(1+abs(previous)+abs(row["energy"])+
                                            abs(row["step"]*row["dissipation"])+abs(row["residual_work"]))
            require(math.isclose(row["roundoff_budget"], budget, rel_tol=1e-12) and
                math.isclose(row["bound_defect"], row["energy"]-previous+row["step"]*row["dissipation"],
                             rel_tol=1e-10, abs_tol=1e-13), "Inconsistent energy diagnostics")
            require(0 <= row["step_defect_rms"] < 2e-10 and abs(row["residual_work"]) < 2e-10 and
                row["dissipation"] >= -budget and row["bound_defect"] <= row["residual_work"]+budget and
                row["energy"] <= previous+abs(row["residual_work"])+budget, "Energy or nonlinear accuracy gate failed")
        else:
            require(not any(key in row for key in ("step_defect_rms", "residual_work", "bound_defect",
                                                   "roundoff_budget", "dissipation")),
                    "Unsplit history incorrectly uses split-Euler diagnostics")
        require(abs(row["mass"]-initial["mass"]) < 2e-10, "Mass conservation gate failed")
    if not legacy:
        require(all(math.isfinite(r["physical_dissipation"]) and r["physical_dissipation"] >=
                    -256*sys.float_info.epsilon*(1+abs(r["physical_dissipation"])) for r in history),
                "Invalid physical dissipation")
    if split:
        require(history[-1]["energy"] < initial["energy"], "No energy decay for nonconstant initial data")
    require(abs(history[-1]["time"]-config["end"]) <= time_roundoff, "Incomplete physical trajectory")
    allocations = [report["trajectory"], report["reference"]["coarse"], report["reference"]["fine"], *rows]
    require(all(r["backend_allocations_after_warmup"] == 0 for r in allocations), "Backend allocation in stepping")
    require(report["trajectory"]["steps"] == count and
            (report["trajectory"]["step"] is None if adaptive else math.isclose(report["trajectory"]["step"], h)),
            "Incorrect trajectory summary")
    for level, row in enumerate(rows):
        require(math.isfinite(row["error_rms"]) and row["error_rms"] > 0, "Invalid refinement error")
        if adaptive:
            require(row["step"] is None and 1 <= row["steps"] <= budget and
                    math.isclose(row["relative_tolerance"], config["relative_tolerance"]*.1**level) and
                    math.isclose(row["absolute_tolerance"], config["absolute_tolerance"]*.1**level),
                    "Invalid tolerance-refinement result")
        else:
            steps = config["base_steps"]*2**level
            require(row["steps"] == steps and math.isclose(row["step"], config["order_end"]/steps), "Invalid refinement mesh")
    if modern:
        maximum_order = config.get("maximum_order", 5) if method == "BDF" else METHODS[method]
        for row in [report["trajectory"], *rows]:
            require(0 < row["minimum_step"] <= row["maximum_step"] and
                    1 <= row["maximum_order"] <= maximum_order and len(row["order_counts"]) == 6 and
                    sum(row["order_counts"]) == row["steps"] and row["order_counts"][0] == 0 and
                    row["rejected_attempts"] >= 0, "Invalid step/order diagnostics")
        counts = [0]*6
        for row in history[1:]:
            require(isinstance(row["order"], int) and 1 <= row["order"] <= maximum_order and
                    isinstance(row["attempts"], int) and row["attempts"] >= 1, "Invalid accepted order/attempts")
            counts[row["order"]] += 1
        require(counts == report["trajectory"]["order_counts"] and
                max(r["order"] for r in history[1:]) == report["trajectory"]["maximum_order"] and
                sum(r["attempts"]-1 for r in history[1:]) == report["trajectory"]["rejected_attempts"] and
                math.isclose(min(r["step"] for r in history[1:]), report["trajectory"]["minimum_step"]) and
                math.isclose(max(r["step"] for r in history[1:]), report["trajectory"]["maximum_step"]),
                "Trajectory summaries disagree with accepted-step history")
    for key, factor in (("coarse", 8), ("fine", 16)):
        ref = report["reference"][key]
        steps = config["base_steps"]*2**(config["levels"]-1)*factor
        require(ref["steps"] == steps and math.isclose(ref["step"], config["order_end"]/steps), "Invalid reference mesh")
    ref_error = report["reference"]["difference_rms"]
    require(math.isfinite(ref_error) and 0 <= ref_error < .15*rows[-1]["error_rms"],
            "Temporal reference is not sufficiently resolved")
    if adaptive:
        order = None
        require(rows[-1]["error_rms"] < .6*rows[0]["error_rms"], "Tightening tolerances did not improve accuracy")
    else:
        order = measured_order(rows, report["fit_error_floor"])
        low, high = (.7, 1.35) if split else (expected_order-.5, expected_order+.5)
        require(low < order < high and
                all(a["error_rms"] > b["error_rms"] for a, b in zip(rows, rows[1:])),
                "Temporal refinement gate failed")
    require(report["integration_bulk_host_transfer_bytes"] == 0 and
            report["export_bulk_host_transfer_bytes"] == len(snapshots)*config["grid"]**2*8,
            "Unexpected integration/export bulk transfer count")
    for frame, snapshot in enumerate(snapshots):
        if modern:
            index = snapshot["source_step_index"]
            target = (frame*sample_count//(config["frames"]-1))*h
            require(isinstance(index, int) and 0 <= index <= count and abs(snapshot["time"]-target) <= time_roundoff and
                    abs(snapshot["evaluation_time"]-target) <= time_roundoff and
                    history[max(0,index-1)]["time"]-time_roundoff <= target <= history[index]["time"]+time_roundoff,
                    "Invalid physical sample time or source step")
            diagnostic = snapshot
        else:
            index = frame*count//(config["frames"]-1)
            require(snapshot["step_index"] == index and snapshot["time"] == history[index]["time"], "Incorrect snapshot itinerary")
            diagnostic = history[index]
        energy, mass, mu = field_diagnostics(snapshot["concentration"], config)
        require(math.isclose(energy, diagnostic["energy"], rel_tol=1e-11, abs_tol=1e-11) and
                math.isclose(mass, diagnostic["mass"], rel_tol=1e-11, abs_tol=1e-11) and
                abs(mass-initial["mass"]) < 2e-10,
                "Exported fields disagree with energy/mass history")
        if not legacy:
            if config.get("spatial") == "fourier":
                import numpy as np
                n = config["grid"]
                k = np.fft.fftfreq(n)*n
                z = np.fft.fft2(np.asarray(mu).reshape(n,n), norm="forward")
                dissipation = config["mobility"]*(2*np.pi)**2*np.sum((k[:,None]**2+k[None,:]**2)*abs(z)**2)
            else:
                dissipation = -config["mobility"]*(2*math.pi/config["grid"])**2 * math.fsum(
                    a*b for a, b in zip(mu, periodic_laplacian(mu, config["grid"])))
            require(math.isclose(dissipation, diagnostic["physical_dissipation"], rel_tol=1e-10, abs_tol=1e-11),
                    "Exported physical dissipation disagrees with history")
    n = config["grid"]
    expected = [config["mean"] + config["amplitude"]*(math.cos(i*2*math.pi/n) +
                .4*math.sin(j*2*math.pi/n) + .25*math.cos(2*i*2*math.pi/n)*math.cos(j*2*math.pi/n))
                for i in range(n) for j in range(n)]
    require(max(abs(a-b) for a, b in zip(expected, snapshots[0]["concentration"])) < 1e-13,
            "Incorrect initial physical field")
    increases = [b["energy"]-a["energy"] for a, b in zip(history, history[1:])]
    growing = sum(delta > 256*sys.float_info.epsilon*(1+abs(a["energy"])+abs(b["energy"]))
                  for delta, a, b in zip(increases, history, history[1:]))
    return dict(method=method, formulation="convex_split" if split else "imex_split" if method.startswith("IMEX_") else "unsplit",
                adaptive=adaptive,
                observed_order=order, expected_order=expected_order, energy_initial=initial["energy"],
                energy_final=history[-1]["energy"], accepted_steps=count,
                energy_increase_steps=growing, maximum_energy_increase=max(0., *increases),
                maximum_mass_drift=max(abs(r["mass"]-initial["mass"]) for r in history),
                maximum_step_defect_rms=max(r["step_defect_rms"] for r in history[1:]) if split else None,
                reference_difference_rms=ref_error, integration_bulk_host_transfer_bytes=0,
                minimum_step=report["trajectory"].get("minimum_step", h),
                maximum_step=report["trajectory"].get("maximum_step", h),
                maximum_order=report["trajectory"].get("maximum_order", expected_order),
                rejected_attempts=report["trajectory"].get("rejected_attempts", 0),
                export_bulk_host_transfer_bytes=report["export_bulk_host_transfer_bytes"])


def plot(directory, report, metrics, shared_limits=None):
    import numpy as np
    import matplotlib.pyplot as plt
    history = report["history"]
    method = report["method"]
    expected = report["expected_order"]
    split = method == "IMEX_EULER"
    times = [r["time"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes[0, 0].plot(times, [r["energy"] for r in history], color="#276749")
    spatial = "Fourier-Galerkin" if report["configuration"].get("spatial") == "fourier" else "Finite difference"
    axes[0, 0].set(ylabel="Discrete free energy", title=f"{spatial}: {case_label(report)}")
    drop = [(a["energy"]-b["energy"])/b["step"] for a, b in zip(history, history[1:])]
    axes[0, 1].plot(times[1:], drop, label=r"$-\Delta E_h/h$")
    dissipation_key = "dissipation" if split else "physical_dissipation"
    axes[0, 1].plot(times[1:], [r[dissipation_key] for r in history[1:]], "--",
                    label=r"$D(\mu_*)$" if split else r"$D(\mu(c))$")
    axes[0, 1].set(ylabel="Energy loss rate", title="Split bound, not equality" if split else "Physical dissipation (not a discrete bound)")
    axes[0, 1].legend()
    axes[1, 0].plot(times, [abs(r["mass"]-history[0]["mass"]) for r in history])
    axes[1, 0].axhline(2e-10, color="#a52834", linestyle=":", label="Validation limit")
    axes[1, 0].set_ylabel("Absolute mass drift")
    axes[1, 0].set_yscale("symlog", linthresh=1e-15)
    axes[1, 0].legend()
    if split:
        axes[1, 1].plot(times[1:], [r["step_defect_rms"] for r in history[1:]])
        axes[1, 1].axhline(2e-10, color="#a52834", linestyle=":")
        axes[1, 1].set_ylabel("Split step defect (RMS)")
        axes[1, 1].set_yscale("symlog", linthresh=1e-15)
    else:
        axes[1, 1].plot(times[1:], [-d for d in drop])
        axes[1, 1].axhline(0, color="#a52834", linestyle=":")
        axes[1, 1].set(ylabel="Energy change / timestep", title=f"Energy-increase steps: {metrics['energy_increase_steps']}")
    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.grid(alpha=.2)
    save_figure(fig, directory / "energy")

    rows = report["refinements"]
    errors = np.array([r["error_rms"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    adaptive = metrics["adaptive"]
    if adaptive:
        tolerances = [r["relative_tolerance"] for r in rows]
        axes[0].loglog(tolerances, errors, "o-", label="Error vs common ROS3 reference")
        axes[0].set(xlabel="Relative tolerance (atol scaled with it)", ylabel="Concentration error (RMS)", title="Tolerance refinement, not fixed order")
        axes[1].loglog(tolerances, [r["steps"] for r in rows], "o-", label="Accepted steps")
        axes[1].set(xlabel="Relative tolerance", ylabel="Accepted steps", title=case_label(report))
        order_axis = axes[1].twinx()
        order_axis.semilogx(tolerances, [r["maximum_order"] for r in rows], "s--", color="#a52834")
        order_axis.set(ylabel="Maximum accepted order", ylim=(.5, 5.5), yticks=range(1, 6))
    else:
        hs = np.array([r["step"] for r in rows])
        axes[0].loglog(hs, errors, "o-", label="Error vs refined spatial-system solution")
        axes[0].loglog(hs, errors[-1]*(hs/hs[-1])**expected, "--", label=f"O(h^{expected}) guide")
        axes[0].set(xlabel="Timestep", ylabel="Concentration error (RMS)", title="Temporal refinement")
        orders = np.log(errors[:-1]/errors[1:])/np.log(hs[:-1]/hs[1:])
        axes[1].semilogx(hs[1:], orders, "o-", label=f"Fitted order: {metrics['observed_order']:.3f}")
        axes[1].axhline(expected, color="#a52834", linestyle="--", label=f"Expected order: {expected}")
        axes[1].set(xlabel="Finer timestep", ylabel="Pairwise observed order", title=case_label(report))
    for ax in axes:
        ax.grid(alpha=.2, which="both")
        ax.legend(fontsize=8)
    save_figure(fig, directory / ("tolerance" if adaptive else "order"))

    if report["format"] == "ch.energy.visualization.v3":
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
        axes[0].plot(times[1:], [r["step"] for r in history[1:]])
        axes[0].set(ylabel="Accepted timestep", title=case_label(report), yscale="log")
        axes[1].step(times[1:], [r["order"] for r in history[1:]], where="pre")
        axes[1].set(ylabel="Accepted order", ylim=(.5, 5.5), yticks=range(1, 6))
        axes[2].plot(times[1:], [r["attempts"]-1 for r in history[1:]], "o", markersize=3)
        axes[2].set(xlabel="Accepted endpoint time", ylabel="Rejected attempts")
        for ax in axes:
            ax.grid(alpha=.2)
        save_figure(fig, directory / "adaptation")

    config = report["configuration"]
    frames = report["snapshots"]
    fields = [[np.array(f["concentration"]).reshape(config["grid"], -1).T for f in frames],
              [np.array(field_diagnostics(f["concentration"], config)[2]).reshape(config["grid"], -1).T
               for f in frames]]
    fig, axes = plt.subplots(2, len(frames), figsize=(3*len(frames)+1, 6), layout="constrained", squeeze=False)
    xy = np.linspace(0, 2*np.pi, config["grid"]+1)
    for row, (group, label) in enumerate(zip(fields, ("Concentration c", "Physical chemical potential mu(c)"))):
        low, high = shared_limits[row] if shared_limits else (min(float(a.min()) for a in group), max(float(a.max()) for a in group))
        for column, (array, frame) in enumerate(zip(group, frames)):
            image = axes[row, column].pcolormesh(xy, xy, np.pad(array, ((0, 1), (0, 1)), mode="wrap"),
                         shading="nearest", cmap="RdBu_r", vmin=low, vmax=high, rasterized=True)
            axes[row, column].set(xlabel="x", ylabel="y" if column == 0 else "", aspect="equal",
                                  xlim=(0, 2*np.pi), ylim=(0, 2*np.pi), title=f"t = {frame['time']:.3g}")
        fig.colorbar(image, ax=list(axes[row]), label=label, shrink=.8)
    save_figure(fig, directory / "solution")


def save_figure(fig, stem):
    import matplotlib.pyplot as plt
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix("."+suffix), dpi=160)
    plt.close(fig)


def comparison_limits(reports):
    """Reject mismatched experiments before putting their fields on common axes."""
    first = reports[0]
    physical = {k: v for k, v in first["configuration"].items() if k not in ("method", *ADAPTIVE_KEYS)}
    itinerary = [f["time"] for f in first["snapshots"]]
    methods = []
    values = [[], []]
    for report in reports:
        require({k: v for k, v in report["configuration"].items() if k not in ("method", *ADAPTIVE_KEYS)} == physical and
                len(report["snapshots"]) == len(itinerary) and
                all(math.isclose(f["time"], t, rel_tol=1e-12, abs_tol=1e-13) for f, t in zip(report["snapshots"], itinerary)),
                "Comparison requires the same physical configuration and sampling times")
        require(report["method"] not in methods, "Duplicate comparison method")
        methods.append(report["method"])
        for frame in report["snapshots"]:
            for row, field in enumerate((frame["concentration"], field_diagnostics(frame["concentration"], physical)[2])):
                values[row].extend((min(field), max(field)))
    return [(min(v), max(v)) for v in values]


def plot_comparison(directory, cases, limits, excluded_methods=()):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    title = ("Validated subset; excluded: " + ", ".join(excluded_methods)) if excluded_methods else "All selected methods validated"
    if cases[0][1]["configuration"].get("spatial") == "fourier":
        title = "Fourier-Galerkin CH: " + title
    fig.suptitle(title)
    colors = plt.get_cmap("tab10")
    for index, (_, report, metrics) in enumerate(cases):
        history = report["history"]
        times = [r["time"] for r in history]
        color, label = colors(index), case_label(report)
        axes[0, 0].plot(times, [r["energy"] for r in history], color=color, label=label)
        axes[0, 1].plot(times[1:], [(b["energy"]-a["energy"])/b["step"]
                                    for a, b in zip(history, history[1:])], color=color,
                        label=f"{label}: {metrics['energy_increase_steps']} increases")
        axes[1, 0].plot(times, [abs(r["mass"]-history[0]["mass"]) for r in history], color=color)
        axes[1, 1].plot(times, [r["physical_dissipation"] for r in history], color=color)
    axes[0, 0].set(ylabel="Discrete free energy", title="Same grid, initial state and physical sampling")
    axes[0, 1].set(ylabel="Energy change / timestep", title="Positive values indicate energy growth")
    axes[0, 1].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[1, 0].set_ylabel("Absolute mass drift")
    axes[1, 0].set_yscale("symlog", linthresh=1e-15)
    axes[1, 0].axhline(2e-10, color="#a52834", linestyle=":")
    axes[1, 1].set(ylabel="Physical dissipation", title=r"$D(c)=-m\langle\mu(c),\Delta_h\mu(c)\rangle_h$")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.grid(alpha=.2)
    save_figure(fig, directory / "energy_comparison")

    for adaptive in (False, True):
        selected = [(i, r, m) for i, (_, r, m) in enumerate(cases) if m["adaptive"] == adaptive]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        fig.suptitle(title)
        for index, report, metrics in selected:
            rows = report["refinements"]
            hs = np.array([r["relative_tolerance"] if adaptive else r["step"] for r in rows])
            errors = np.array([r["error_rms"] for r in rows])
            color = colors(index)
            axes[0].loglog(hs, errors, "o-", color=color, label=case_label(report))
            if adaptive:
                axes[1].loglog(hs, [r["steps"] for r in rows], "o-", color=color, label=case_label(report))
            else:
                axes[0].loglog(hs, errors[-1]*(hs/hs[-1])**report["expected_order"], ":", color=color, alpha=.6)
                axes[1].semilogx(hs[1:], np.log(errors[:-1]/errors[1:])/np.log(hs[:-1]/hs[1:]), "o-", color=color,
                                 label=f"{report['method']}: fit {metrics['observed_order']:.3f}")
        if adaptive:
            axes[0].set(xlabel="Relative tolerance (atol scaled with it)", ylabel="Concentration error (RMS)", title="Common refined ROS3 reference")
            axes[1].set(xlabel="Relative tolerance", ylabel="Accepted steps", title="Tolerance response is not fixed order")
        else:
            for order in sorted({r["expected_order"] for _, r, _ in selected}):
                axes[1].axhline(order, color="grey", linestyle=":", alpha=.6)
            axes[0].set(xlabel="Timestep", ylabel="Concentration error (RMS)", title="Common refined ROS3 reference; dotted order guides")
            axes[1].set(xlabel="Finer timestep", ylabel="Pairwise observed order", title="Temporal order, not a spatial-error estimate")
        for ax in axes:
            ax.grid(alpha=.2, which="both")
            ax.legend(fontsize=8)
        axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
        save_figure(fig, directory / ("adaptive_accuracy_comparison" if adaptive else "order_comparison"))

    n = cases[0][1]["configuration"]["grid"]
    frames = cases[0][1]["snapshots"]
    xy = np.linspace(0, 2*np.pi, n+1)
    for field, name, label in ((0, "concentration_comparison", "Concentration c"),
                               (1, "chemical_potential_comparison", "Physical chemical potential mu(c)")):
        fig, axes = plt.subplots(len(cases), len(frames), figsize=(3*len(frames)+1, 2.7*len(cases)),
                                 layout="constrained", squeeze=False)
        fig.suptitle(title)
        for row, (_, report, _) in enumerate(cases):
            for column, frame in enumerate(report["snapshots"]):
                values = frame["concentration"] if field == 0 else field_diagnostics(frame["concentration"], report["configuration"])[2]
                array = np.array(values).reshape(n, n).T
                image = axes[row, column].pcolormesh(xy, xy, np.pad(array, ((0, 1), (0, 1)), mode="wrap"),
                            shading="nearest", cmap="RdBu_r", vmin=limits[field][0], vmax=limits[field][1], rasterized=True)
                axes[row, column].set(aspect="equal", xlim=(0, 2*np.pi), ylim=(0, 2*np.pi),
                    xlabel="x" if row == len(cases)-1 else "", ylabel=case_label(report)+"\ny" if column == 0 else "")
                if row == 0:
                    axes[row, column].set_title(f"t = {frame['time']:.3g}")
        fig.colorbar(image, ax=list(axes.flat), label=label, shrink=.7)
        save_figure(fig, directory / name)


def summarize_case(directory, manifest, shared_limits=None):
    method = manifest.get("method", "IMEX_EULER")
    try:
        require(manifest["returncode"] == 0, f"Executable returned {manifest['returncode']}; see run.log")
        config = json.loads((directory / "config.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        metrics = validate_report(report, config)
        plot(directory, report, metrics, shared_limits)
        for name in ("history", "refinements"):
            keys = list(dict.fromkeys(key for row in report[name] for key in row))
            with (directory / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(report[name])
        result = dict(passed=True, **metrics)
        accuracy = (f"adaptive, {metrics['accepted_steps']} steps, maximum order {metrics['maximum_order']}" if metrics["adaptive"]
                    else f"order {metrics['observed_order']:.3f}")
        print(f"PASS {report['method']}: {accuracy}; energy {metrics['energy_initial']:.6g} -> {metrics['energy_final']:.6g}")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result = dict(method=method, passed=False, error=str(error))
        print(f"FAIL: {error}", file=sys.stderr)
    write_if_changed(directory / "validation.json", result)
    return result


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if "cases" not in manifest:
        return 0 if summarize_case(directory, manifest)["passed"] else 1
    write_if_changed(directory/"validation.json", dict(passed=False, status="validating"))
    expected = manifest.get("methods", [case["method"] for case in manifest["cases"]])
    require(expected and len(set(expected)) == len(expected) and all(m in METHODS for m in expected),
            "Invalid requested comparison methods")
    valid, results = [], []
    seen = set()
    for case in manifest["cases"]:
        require(case["method"] in expected and case["method"] not in seen and
                case["directory"] == case["method"], "Invalid comparison case path")
        seen.add(case["method"])
        path = directory / case["directory"]
        try:
            require(case["returncode"] == 0, f"Executable returned {case['returncode']}; see {path/'run.log'}")
            config = json.loads((path/"config.json").read_text())
            report = json.loads((path/"report.json").read_text())
            require(report["method"] == case["method"], "Method differs from manifest")
            valid.append((path, report, validate_report(report, config)))
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            result = dict(method=case["method"], passed=False, error=str(error))
            results.append(result)
            write_if_changed(path/"validation.json", result)
            print(f"FAIL {case['method']}: {error}", file=sys.stderr)
    for method in expected:
        if method not in seen:
            results.append(dict(method=method, passed=False, error="Selected method was not executed; interrupted batch"))
    limits = comparison_limits([report for _, report, _ in valid]) if valid else None
    plotted = []
    for path, report, metrics in valid:
        result = summarize_case(path, dict(returncode=0, method=report["method"]), limits)
        results.append(result)
        if result["passed"]:
            plotted.append((path, report, metrics))
    if len(plotted) > 1:
        plot_comparison(directory, plotted, limits, [r["method"] for r in results if not r["passed"]])
    passed = bool(results) and all(r["passed"] for r in results)
    write_if_changed(directory/"validation.json", dict(passed=passed, cases=results,
        included_methods=[r["method"] for _, r, _ in plotted],
        excluded_methods=[r["method"] for r in results if not r["passed"]], shared_field_limits=limits))
    with (directory/"comparison.csv").open("w", newline="") as stream:
        keys = list(dict.fromkeys(key for row in results for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    return 0 if passed else 1


def ensure_plotting():
    try:
        import numpy
        import matplotlib
    except ImportError as error:
        raise RuntimeError("NumPy and Matplotlib required; select an installed interpreter with PYTHON") from error


def main(argv=None):
    args = arguments(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(args.build_dir / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    if not args.build_only:
        ensure_plotting()
    if args.plot_only:
        return summarize(args.plot_only.resolve())
    configs = [configuration(args, method) for method in args.methods]
    executable = build(args, configs[0])
    for config in configs:
        write_if_changed(args.build_dir/"cases"/args.backend/config["method"]/"config.json", config)
    if args.build_only:
        print(f"Built and assembled in {args.build_dir}; no integrations or plots executed")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f".{time.time_ns()}"
    directory = args.build_dir / "runs" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
    manifest = dict(format="ch.comparison.manifest.v1", executable=str(executable), methods=args.methods, cases=[])
    write_if_changed(directory/"manifest.json", manifest)
    if args.spatial == "fourier":
        log = directory/"spatial_validation.log"
        with log.open("w") as stream:
            code = subprocess.run([str(executable), "--verify-spatial"], env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=args.timeout).returncode
        if code != 0:
            raise RuntimeError(f"Fourier spatial validation failed; see {log}")
    for config in configs:
        path = directory/config["method"]
        path.mkdir()
        write_if_changed(path/"config.json", config)
        before = time.perf_counter()
        with (path/"run.log").open("w") as log:
            try:
                code = subprocess.run([str(executable), str(path/"config.json"), str(path/"report.json")],
                                      stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                log.write("\nBenchmark timeout reached\n")
                code = 124
        manifest["cases"].append(dict(method=config["method"], directory=config["method"], returncode=code,
                                      seconds=time.perf_counter()-before))
        write_if_changed(path/"manifest.json", manifest["cases"][-1])
        write_if_changed(directory/"manifest.json", manifest)
    result = summarize(directory)
    print(f"Results: {directory}")
    return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
