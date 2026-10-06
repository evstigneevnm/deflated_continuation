#!/usr/bin/env python3
"""Compare a completed NMFD Lorenz trajectory with the matching SciPy RK method."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

SCIPY_METHODS = {"DOPRI54": "RK45", "BS32": "RK23"}


def read_trajectory(path):
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
        last = ""
        for line in stream:
            if line.strip():
                last = line
    if not first.startswith("# ") or not last.startswith("# "):
        raise ValueError("Missing trajectory metadata or completion footer")
    metadata = json.loads(first[2:])
    footer = json.loads(last[2:])
    if not isinstance(metadata, dict) or not isinstance(footer, dict):
        raise ValueError("Trajectory metadata and footer must be objects")
    if (metadata.get("format") != "nmfd.lorenz_trajectory.v1"
            or metadata.get("problem") != "lorenz"
            or not isinstance(metadata.get("method"), str) or metadata["method"] not in SCIPY_METHODS):
        raise ValueError("Expected a Lorenz trajectory using DOPRI54 or BS32")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[0] < 2 or data.shape[1] != 4 or not np.isfinite(data).all():
        raise ValueError("Expected finite time/x/y/z rows, including both endpoints")
    if footer.get("status") != "completed" or footer.get("rows") != len(data):
        raise ValueError("Incomplete trajectory or incorrect completion row count")
    initial = np.asarray(metadata["initial_state"], dtype=float)
    interval = np.asarray([metadata["start_time"], metadata["end_time"]], dtype=float)
    parameters = [metadata["parameters"][key] for key in ("sigma", "rho", "beta", "epsilon", "delta")]
    if (initial.shape != (3,) or not np.isfinite(initial).all()
            or not np.isfinite(interval).all() or interval[1] <= interval[0]
            or not np.isfinite(parameters).all()):
        raise ValueError("Invalid initial state, interval, or Lorenz parameters")
    if (data[0, 0] != interval[0] or data[-1, 0] != interval[1]
            or not np.array_equal(data[0, 1:], initial) or not (np.diff(data[:, 0]) > 0).all()):
        raise ValueError("Trajectory endpoints, initial state, or time ordering are inconsistent")
    adaptation = metadata["adaptation"]
    settings = [adaptation[key] for key in
                ("relative_tolerance", "absolute_tolerance", "initial_step", "minimum_step", "maximum_step")]
    if not np.isfinite(settings).all() or min(settings) <= 0 or settings[3] > settings[4]:
        raise ValueError("Invalid timestep adaptation metadata")
    return metadata, data


def lorenz_rhs(time, state, sigma, rho, beta, epsilon, delta):
    x, y, z = state
    return [-sigma * x + sigma * y - epsilon * x * x * x,
            rho * x - y - x * z + delta,
            -beta * z + x * y - delta]


def save_plots(base, time, actual, reference, error, method, scipy_method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for component, axis in enumerate(axes):
        axis.plot(time, actual[:, component], label=f"NMFD {method}", color="black")
        axis.plot(time, reference[:, component], label=f"SciPy {scipy_method}", color="tab:blue", linestyle="--")
        axis.set_ylabel("xyz"[component])
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Time")
    figure.tight_layout()
    figure.savefig(base.with_name(base.name + "_comparison.png"), dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    for component in range(3):
        axis.plot(time, np.abs(error[:, component]), label="xyz"[component])
    axis.set_xlabel("Time")
    axis.set_ylabel(f"Absolute difference from SciPy {scipy_method}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(base.with_name(base.name + "_errors.png"), dpi=150)
    plt.close(figure)


def compare(args):
    metadata, data = read_trajectory(args.trajectory)
    time, actual = data[:, 0], data[:, 1:]
    parameters = tuple(metadata["parameters"][key] for key in ("sigma", "rho", "beta", "epsilon", "delta"))
    adaptation = metadata["adaptation"]
    scipy_method = SCIPY_METHODS[metadata["method"]]
    reference = solve_ivp(
        lorenz_rhs, (time[0], time[-1]), metadata["initial_state"], args=parameters,
        method=scipy_method, t_eval=time, rtol=args.reference_rtol, atol=args.reference_atol,
        first_step=min(adaptation["initial_step"], time[-1] - time[0]),
        max_step=adaptation["maximum_step"])
    if not reference.success or reference.y.shape != actual.T.shape or not np.isfinite(reference.y).all():
        raise ValueError("SciPy reference integration failed: " + reference.message)
    expected = reference.y.T
    error = actual - expected
    scale = args.check_atol + args.check_rtol * np.maximum(np.abs(actual), np.abs(expected))
    maximum_scaled_error = float(np.max(np.abs(error) / scale))
    within_tolerance = maximum_scaled_error <= 1
    summary = {
        "cpp_metadata": metadata,
        "reference_method": f"SciPy {scipy_method}",
        "reference_rtol": args.reference_rtol,
        "reference_atol": args.reference_atol,
        "reference_rhs_calls": reference.nfev,
        "accepted_cpp_steps": len(data) - 1,
        "check_rtol": args.check_rtol,
        "check_atol": args.check_atol,
        "maximum_absolute_error": np.max(np.abs(error), axis=0).tolist(),
        "rms_error": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "maximum_scaled_error": maximum_scaled_error,
        "within_tolerance": within_tolerance,
        "diagnostic_only": args.diagnostic_only
    }
    base = args.trajectory.with_suffix("")
    np.savetxt(base.with_name(base.name + "_python.dat"), np.column_stack((time, expected)),
               fmt="%.17g", header="time x y z")
    np.savetxt(base.with_name(base.name + "_error.dat"), np.column_stack((time, error)),
               fmt="%.17g", header="time error_x error_y error_z")
    base.with_name(base.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_plots:
        save_plots(base, time, actual, expected, error, metadata["method"], scipy_method)
    status = "DIAGNOSTIC" if args.diagnostic_only else "PASS" if within_tolerance else "FAIL"
    print(f"Lorenz {status}: {len(data)} samples, max scaled difference {maximum_scaled_error:.6g}")
    print("Maximum absolute differences (x, y, z):", *summary["maximum_absolute_error"])
    if not within_tolerance:
        print("Pointwise tolerance exceeded; long chaotic runs can diverge. "
              "Use --diagnostic-only only for exploratory comparisons.", file=sys.stderr)
    return 0 if within_tolerance or args.diagnostic_only else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--reference-rtol", type=float, default=1e-12)
    parser.add_argument("--reference-atol", type=float, default=1e-14)
    parser.add_argument("--check-rtol", type=float, default=1e-7)
    parser.add_argument("--check-atol", type=float, default=1e-8)
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Report differences without enforcing pointwise agreement")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    for key in ("reference_rtol", "reference_atol", "check_rtol", "check_atol"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error("All tolerances must be positive and finite")
    try:
        return compare(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Lorenz comparison: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
