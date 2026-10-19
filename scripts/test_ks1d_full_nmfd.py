#!/usr/bin/env python3
"""Verify full KS1D NMFD continuation, stability, restart, and deflation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def read_rows(path: Path) -> list[list[float]]:
    rows = [[float(field) for field in line.split()]
            for line in path.read_text().splitlines() if line.strip()]
    if any(not all(math.isfinite(value) for value in row) for row in rows):
        raise RuntimeError(f"nonfinite values in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bd-executable", required=True, type=Path)
    parser.add_argument("--stability-executable", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    fixtures = Path("source/models/KS_1D/tests/data")
    manifest = json.loads((fixtures / "stability_replay/manifest.json").read_text())
    seed = next(state for state in manifest["states"] if state["id"] == "curve03_state058")
    config = args.work_dir / "config.json"
    project = args.work_dir / "project"
    device = [args.device] if args.device else []

    def run(*command: object) -> None:
        subprocess.run([str(value) for value in command], check=True, timeout=args.timeout)

    run(sys.executable, "scripts/write_ks1d_ci_config.py", config, project,
        "--base-config", "json_project_files/KS1D_test_full.json", "--grid-size", 128,
        "--knots", 16, 16.2, "--steps", 4, "--step-size", 0.02, "--max-step-size", 0.03, "--clean-project")
    run(args.bd_executable.resolve(), config, "--seed-file", seed["parameter"],
        fixtures / "stability_replay" / seed["file"], "--continue-seed-only", "--quiet", *device)
    points = read_rows(project / "0/debug_curve.dat")
    reference = read_rows(fixtures / "full_nmfd_continuation_reference.dat")
    if len(points) != 10 or len(reference) != 10 or len(list(project.glob("*/debug_curve.dat"))) != 1:
        raise RuntimeError("expected ten nonzero points on one full-KS1D curve")
    for actual, expected in zip(points, reference):
        if len(actual) < 2 or actual[1] < 1:
            raise RuntimeError("nonzero continuation returned a zero state")
        if any(not math.isclose(value, baseline, rel_tol=2e-7, abs_tol=2e-8)
               for value, baseline in zip(actual[:2], expected)):
            raise RuntimeError(f"continuation reference mismatch: {actual[:2]} != {expected}")
    if not min(row[0] for row in points) < seed["parameter"] - 1e-4:
        raise RuntimeError("negative semicurve did not advance")
    if not max(row[0] for row in points) > seed["parameter"] + 1e-4:
        raise RuntimeError("positive semicurve did not advance")

    run(args.stability_executable.resolve(), config, "--quiet", *device)
    run(sys.executable, "scripts/validate_stability_diagram.py", project,
        "--expected-curves", 1, "--min-points", 10, "--require-complete", "--require-plot-sidecar",
        "--expected-point-type", "unstable", "--expected-unstable-real", seed["unstable"][0],
        "--expected-unstable-complex-pairs", seed["unstable"][1])
    paths = [project / "stability_diagram.dat", project / "0/debug_curve_stability.dat",
             project / "0/debug_curve_stability_plot.dat"]
    before = [path.read_bytes() for path in paths]
    run(args.stability_executable.resolve(), config, "--quiet", *device)
    if before != [path.read_bytes() for path in paths]:
        raise RuntimeError("full-KS1D stability restart changed committed results")

    deflation_config = args.work_dir / "deflation.json"
    deflation_project = args.work_dir / "deflation"
    run(sys.executable, "scripts/write_ks1d_ci_config.py", deflation_config, deflation_project,
        "--base-config", "json_project_files/KS1D_test_full.json", "--grid-size", 32,
        "--knots", 5.9, 6, 6.1, "--steps", 4, "--step-size", 0.02, "--max-step-size", 0.03,
        "--deflation-attempts", 1, "--clean-project")
    run(args.bd_executable.resolve(), deflation_config, "--quiet", *device)
    expected_mode = abs(read_rows(fixtures / "symmetric_lambda6.dat")[0][0]) / 4
    discovered = False
    for path in deflation_project.glob("*/debug_curve.dat"):
        rows = read_rows(path)
        if len(rows) >= 10 and any(len(row) >= 3 and math.isclose(row[0], 6, abs_tol=1e-9)
                                  and math.isclose(row[2], expected_mode, rel_tol=1e-7) for row in rows):
            discovered = True
    if not discovered:
        raise RuntimeError("full-KS1D deflation did not discover and continue the reference orbit")
    print("Full KS1D NMFD: reference curve, unstable signature, restart, and deflation passed")


if __name__ == "__main__":
    main()
