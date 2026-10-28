#!/usr/bin/env python3
"""Verify symmetric KS1D NMFD Newton, continuation, stability, restart, and discovery."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bd-executable", required=True, type=Path)
    parser.add_argument("--stability-executable", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    config = args.work_dir / "config.json"
    project = args.work_dir / "project"
    fixtures = Path("source/models/KS_1D/tests/data")

    def run(*command: object, solver: bool = False) -> None:
        result = subprocess.run(
            [str(value) for value in command], timeout=args.timeout, text=True,
            stdout=subprocess.PIPE if solver else None, stderr=subprocess.STDOUT if solver else None,
        )
        if solver:
            print(result.stdout, end="")
        result.check_returncode()
        if solver and "Newton solver: NMFD" not in result.stdout:
            raise RuntimeError("production symmetric KS1D did not select NMFD Newton")
        if solver and Path(command[0]).resolve() == args.bd_executable.resolve() and \
                "Continuation linear solver: augmented BiCGStab(L)" not in result.stdout:
            raise RuntimeError("production symmetric KS1D did not select augmented BiCGStab(L)")

    run(sys.executable, "scripts/write_ks1d_ci_config.py", config, project,
        "--grid-size", 128, "--knots", 5.9, 6, 6.1, "--steps", 6,
        "--step-size", 0.02, "--max-step-size", 0.03, "--clean-project")
    device = [args.device] if args.device else []
    run(args.bd_executable.resolve(), config, "--seed-state", fixtures / "symmetric_lambda6.dat",
        6, "--continue-seed-only", "--quiet", *device, solver=True)

    curve = project / "0/debug_curve.dat"
    points = [[float(field) for field in line.split()] for line in curve.read_text().splitlines() if line.strip()]
    reference = [[float(field) for field in line.split()]
                 for line in (fixtures / "symmetric_lambda6_curve.dat").read_text().splitlines()]
    if len(points) != 14 or len(list(project.glob("*/debug_curve.dat"))) != 1:
        raise RuntimeError("expected 14 points on one nonzero curve")
    if any(not all(math.isfinite(value) for value in row) or row[1] < 10 for row in points):
        raise RuntimeError("nonfinite or zero-branch point in nonzero continuation")
    # The archive covers (lambda, RMS norm) on the decreasing-parameter semicurve.
    if len(points) != 2 * len(reference):
        raise RuntimeError("reference semicurve length changed")
    for actual, expected in zip(points[:len(reference)], reference):
        if len(actual) < 2 or len(expected) != 2:
            raise RuntimeError("reference semicurve columns changed")
        for value, baseline in zip(actual[:2], expected):
            if not math.isclose(value, baseline, rel_tol=2e-7, abs_tol=2e-8):
                raise RuntimeError(f"reference mismatch: {value} != {baseline}")
    if not min(row[0] for row in points) < 5.99 < 6.01 < max(row[0] for row in points):
        raise RuntimeError("both semicurves must advance from the seed")

    run(args.stability_executable.resolve(), config, "--quiet", *device, solver=True)
    run(sys.executable, "scripts/validate_stability_diagram.py", project,
        "--expected-curves", 1, "--min-points", 14, "--require-complete", "--require-plot-sidecar",
        "--expected-point-type", "stable", "--expected-unstable-real", 0,
        "--expected-unstable-complex-pairs", 0)
    paths = [project / "stability_diagram.dat", project / "0/debug_curve_stability.dat",
             project / "0/debug_curve_stability_plot.dat"]
    before = [path.read_bytes() for path in paths]
    run(args.stability_executable.resolve(), config, "--quiet", *device, solver=True)
    if before != [path.read_bytes() for path in paths]:
        raise RuntimeError("stability restart changed committed results")

    # Force a real Newton correction before spectrum classification, not just
    # immediate acceptance of an already converged archived equilibrium.
    perturbed = args.work_dir / "perturbed_state.dat"
    values = [float(value) for value in (fixtures / "symmetric_lambda6.dat").read_text().split()]
    perturbed.write_text("\n".join(f"{1.001 * value:.17e}" for value in values) + "\n")
    run(args.stability_executable.resolve(), config, "--state-file", perturbed, "--parameter", 6,
        "--confirm", "--quiet", *device, solver=True)

    deflation_config = args.work_dir / "deflation.json"
    deflation_project = args.work_dir / "deflation"
    run(sys.executable, "scripts/write_ks1d_ci_config.py", deflation_config, deflation_project,
        "--grid-size", 32, "--knots", 5.9, 6, 6.1, "--steps", 6,
        "--step-size", 0.02, "--max-step-size", 0.03, "--deflation-attempts", 1, "--clean-project")
    run(args.bd_executable.resolve(), deflation_config, "--quiet", *device, solver=True)
    expected_mode = abs(float((fixtures / "symmetric_lambda6.dat").read_text().split()[0])) / 4
    discovered = False
    for path in deflation_project.glob("*/debug_curve.dat"):
        rows = [[float(field) for field in line.split()] for line in path.read_text().splitlines() if line.strip()]
        if len(rows) >= 10 and any(math.isclose(row[0], 6, abs_tol=1e-9)
                                  and math.isclose(abs(row[2]), expected_mode, rel_tol=1e-7) for row in rows):
            discovered = True
    if not discovered:
        raise RuntimeError("deflation did not discover and continue the nonzero reference branch")
    print("Symmetric KS1D NMFD replay: reference curve, stable spectrum, restart, and deflation passed")


if __name__ == "__main__":
    main()
