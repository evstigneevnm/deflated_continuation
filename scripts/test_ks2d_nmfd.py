#!/usr/bin/env python3
"""Replay symmetry-restricted KS2D against the pre-migration solver baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys


def read_rows(path: Path) -> list[list[float]]:
    rows = [[float(value) for value in line.split()]
            for line in path.read_text().splitlines() if line.strip()]
    if any(not all(math.isfinite(value) for value in row) for row in rows):
        raise RuntimeError(f"nonfinite data in {path}")
    return rows


def diagram_snapshot(project: Path) -> dict[Path, bytes]:
    paths = [project / name for name in (
        "bifurcation_diagram.dat", "branch_topology.json", "symmetry_group.json")]
    paths.extend(path for directory in project.iterdir()
                 if directory.is_dir() and directory.name.isdigit()
                 for path in directory.iterdir() if path.is_file())
    paths.extend((project / "branch_topology" / "endpoints").glob("*.dat"))
    return {path.relative_to(project): path.read_bytes() for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bd-executable", required=True, type=Path)
    parser.add_argument("--stability-executable", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    fixtures = Path("source/models/KS_2D/tests/data")
    config = args.work_dir / "config.json"
    project = args.work_dir / "project"
    device = [args.device] if args.device else []

    def run(*command: object, capture: bool = False) -> str:
        command = [str(value) for value in command]
        completed = subprocess.run(command, check=False,
                                   timeout=args.timeout, capture_output=capture, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n"
                               f"{completed.stdout or ''}{completed.stderr or ''}")
        return completed.stdout if capture else ""

    run(sys.executable, "scripts/write_ks2d_ci_config.py", config, project,
        "--clean-project", "--nx", 12, "--ny", 12, "--steps", 8,
        "--step-size", 0.02, "--max-step-size", 0.03,
        "--seed-parameter", 4.333, "--parameter-min", 4.1, "--parameter-max", 4.5,
        "--deflation-attempts", 1, "--newton-deflation-iterations", 80,
        "--newton-deflation-weight", 0.5, "--newton-deflation-tolerance", 1e-8,
        "--linear-solver-iterations", 150, "--linear-solver-tolerance", 1e-3,
        "--max-failed-continuations-per-knot", 1)
    run(args.bd_executable.resolve(), config, "--quiet", *device)
    points = read_rows(project / "0/debug_curve.dat")
    reference = read_rows(fixtures / "nmfd_continuation_reference.dat")
    if len(points) != 18 or len(reference) != 18 or len(list(project.glob("*/debug_curve.dat"))) != 1:
        raise RuntimeError("expected one nonzero KS2D curve with 18 saved points")
    for actual, expected in zip(points, reference):
        if len(actual) < 4 or len(expected) != 4 or actual[1] < 1:
            raise RuntimeError("invalid nonzero continuation data")
        if any(not math.isclose(value, baseline, rel_tol=2e-7, abs_tol=2e-8)
               for value, baseline in zip(actual[:4], expected)):
            raise RuntimeError(f"KS2D reference mismatch: {actual[:4]} != {expected}")
    if not min(row[0] for row in points) < 4.332 or not max(row[0] for row in points) > 4.334:
        raise RuntimeError("both semicurves must advance")
    run(sys.executable, "scripts/validate_symmetry_group_manifest.py", project, "--expected-order", 8)
    diagram = diagram_snapshot(project)
    run(args.bd_executable.resolve(), config, "--quiet", *device)
    if diagram != diagram_snapshot(project):
        raise RuntimeError("deflation/continuation restart changed committed KS2D results")
    run(args.stability_executable.resolve(), config, "--quiet", *device)
    run(sys.executable, "scripts/validate_stability_diagram.py", project,
        "--expected-curves", 1, "--min-points", 18, "--require-complete", "--require-plot-sidecar",
        "--expected-point-type", "stable", "--expected-unstable-real", 0,
        "--expected-unstable-complex-pairs", 0)
    paths = [project / "bifurcation_diagram.dat", project / "branch_topology.json", project / "symmetry_group.json",
             project / "stability_diagram.dat", project / "0/debug_curve_stability.dat",
             project / "0/debug_curve_stability_plot.dat"]
    before = [path.read_bytes() for path in paths]
    run(args.stability_executable.resolve(), config, "--quiet", *device)
    if before != [path.read_bytes() for path in paths]:
        raise RuntimeError("stability restart changed committed KS2D results")

    matrix_free_config = args.work_dir / "matrix_free.json"
    matrix_free_settings = json.loads(config.read_text())
    small_system = matrix_free_settings["stability_continuation"]["matrix_free_eigensolver"]["small_system"]
    small_system["enabled"] = False
    small_system["prefer"] = False
    matrix_free_config.write_text(json.dumps(matrix_free_settings, indent=4) + "\n")
    output = run(args.stability_executable.resolve(), matrix_free_config,
                 "--state-file", fixtures / "symmetric_lambda4_333.dat", "--parameter", 4.333,
                 "--confirm", "--quiet", *device, capture=True)
    if "Single-state stability result: status=complete, unstable=(0,0)" not in output:
        raise RuntimeError(f"paired matrix-free probes failed: {output}")

    fallback_config = args.work_dir / "fallback.json"
    settings = json.loads(config.read_text())
    settings["stability_continuation"]["linear_solver"]["maximum_iterations"] = 0
    fallback_config.write_text(json.dumps(settings, indent=4) + "\n")
    for replay_config, fallback in ((config, 0), (fallback_config, 1)):
        output = run(args.stability_executable.resolve(), replay_config,
                     "--state-file", fixtures / "symmetric_lambda4_333.dat", "--parameter", 4.333,
                     "--newton-target", 4.334, "--quiet", *device, capture=True)
        match = re.search(r"converged=(\d), fallback=(\d).*final_target_residual=([^,]+)", output)
        if match is None or int(match[1]) != 1 or int(match[2]) != fallback:
            raise RuntimeError(f"unexpected Newton recovery: {output}")
        residual = float(match[3])
        if not math.isfinite(residual) or residual > 1e-9:
            raise RuntimeError(f"Newton recovery residual is too large: {output}")

    zero = args.work_dir / "zero_state.dat"
    zero.write_text("0\n" * len(read_rows(fixtures / "symmetric_lambda4_333.dat")))
    output = run(args.stability_executable.resolve(), config,
                 "--state-file", zero, "--parameter", 3.9, "--state-file-2", zero,
                 "--parameter-2", 4.1, "--confirm", "--quiet", *device, capture=True)
    match = re.search(r"status=success, lambda=([^,]+), before=\(0,0\), after=\(2,0\)", output)
    if match is None or not math.isclose(float(match[1]), 4.0000976562500004, rel_tol=0, abs_tol=2e-8):
        raise RuntimeError(f"paired stability transition differs from baseline: {output}")
    if before != [path.read_bytes() for path in paths]:
        raise RuntimeError("targeted replays changed committed stability results")
    print("KS2D NMFD: reference curve, symmetry, matrix-free stability, restart, Newton fallback, and transition passed")


if __name__ == "__main__":
    main()
