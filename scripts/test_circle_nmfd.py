#!/usr/bin/env python3
"""Verify unseeded NMFD circle deflation discovers both bounded solution branches."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from validate_circle_diagram import curve_files, read_curve_points


ROOT = Path(__file__).resolve().parents[1]


def verify(executable, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    config = work_dir / "circle.json"
    project = work_dir / "project"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/write_circle_ci_config.py"), str(config), str(project),
         "--clean-project", "--steps", "80", "--deflation-attempts", "12", "--knots", "-.8", "0", ".8"],
        cwd=ROOT, check=True, timeout=30,
    )
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    result = subprocess.run(
        [str(executable), str(config), "--quiet"], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=120,
    )
    (work_dir / "discovery.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stdout + result.stderr, file=sys.stderr)
    result.check_returncode()
    if "Newton solver: NMFD" not in result.stdout:
        raise RuntimeError("Production workflow did not select NMFD nonlinear solvers")
    if "Analytical seed: disabled" not in result.stdout or "Exact regular-continuation seed:" in result.stdout:
        raise RuntimeError("Discovery test unexpectedly used an exact seed")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_circle_diagram.py"), str(project),
         "--tolerance", "1e-5", "--min-points", "100", "--expected-curves", "2"],
        cwd=ROOT, check=True, timeout=30,
    )
    signs = []
    for curve in curve_files(project):
        values = [point[3] for point in read_curve_points(curve)]
        if values and min(values) > .1:
            signs.append(1)
        elif values and max(values) < -.1:
            signs.append(-1)
        else:
            raise RuntimeError(f"Expected a single bounded branch in {curve}")
    if sorted(signs) != [-1, 1]:
        raise RuntimeError("Deflation did not discover distinct positive and negative branches")
    print("NMFD circle discovery: both unseeded branches verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bd-executable", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.bd_executable.resolve(), args.work_dir.resolve())


if __name__ == "__main__":
    main()
