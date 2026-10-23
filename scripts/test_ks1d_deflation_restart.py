#!/usr/bin/env python3
"""Reject approximate known roots at a KS bifurcation and preserve discovery on restart."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import sys


def snapshot(project: Path) -> dict[str, str]:
    paths = [project / "bifurcation_diagram.dat"]
    paths.extend(path for path in project.glob("*/*")
                 if path.parent.name.isdigit() and path.is_file())
    return {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bd-executable", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    config, project = args.work_dir / "config.json", args.work_dir / "project"
    subprocess.run([
        sys.executable, "scripts/write_ks1d_ci_config.py", str(config), str(project),
        "--grid-size", "32", "--knots", "3.9", "4", "6", "6.1",
        "--steps", "1000", "--step-size", ".02", "--max-step-size", ".25",
        "--deflation-attempts", "5", "--high-precision-reduction", "--clean-project",
    ], check=True)

    def run(label: str) -> None:
        result = subprocess.run([str(args.bd_executable.resolve()), str(config), "--quiet"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=args.timeout)
        (args.work_dir / f"{label}.log").write_text(result.stdout)
        result.check_returncode()
        if "Newton solver: NMFD" not in result.stdout:
            raise RuntimeError("KS1D did not select NMFD Newton")
        if "exception init_tangent" in result.stdout:
            raise RuntimeError("deflation accepted a candidate that cannot start a curve")

    # At lambda=4 the zero branch is singular. The nonzero branch discovered at
    # lambda=6 crosses the pitchfork and covers both signs before ending at 6.1.
    run("initial")
    curves = sorted(project.glob("*/metadata_curve.dat"))
    if len(curves) != 2:
        raise RuntimeError(f"expected zero and nonzero curves, got {len(curves)}")
    nonzero = project / "1/debug_curve_all.dat"
    points = [[float(value) for value in line.split()] for line in nonzero.read_text().splitlines() if line.strip()]
    if not points or not math.isclose(points[0][0], 6., abs_tol=1e-12):
        raise RuntimeError("an approximate zero root at lambda=4 was accepted as a new branch")
    if not all(math.isfinite(value) for row in points for value in row):
        raise RuntimeError("nonfinite curve output")
    expected_mode = abs(float(Path("source/models/KS_1D/tests/data/symmetric_lambda6.dat").read_text().split()[0])) / 4
    if not math.isclose(abs(points[0][2]), expected_mode, rel_tol=1e-7):
        raise RuntimeError("discovered branch disagrees with the archived equilibrium")
    if not min(row[2] for row in points) < -expected_mode < expected_mode < max(row[2] for row in points):
        raise RuntimeError("both symmetry-related intersections must be present before restart")

    before = snapshot(project)
    run("restart")
    if snapshot(project) != before:
        raise RuntimeError("restart changed committed curves or added overlapping curves")
    print("KS1D deflation: singular-knot rejection, reference discovery, both signs and unchanged restart passed")


if __name__ == "__main__":
    main()
