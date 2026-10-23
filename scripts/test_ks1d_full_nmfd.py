#!/usr/bin/env python3
"""Verify full KS1D NMFD continuation, stability, restart, and deflation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
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

    def run(*command: object, log_name: str | None = None) -> None:
        # Keep partial diagnostics even if the process reaches its timeout.
        with ((args.work_dir / log_name).open("w") if log_name else nullcontext(None)) as output:
            result = subprocess.run([str(value) for value in command], timeout=args.timeout,
                                    stdout=output, stderr=subprocess.STDOUT)
        if log_name:
            text = (args.work_dir / log_name).read_text()
            print(text, end="")
        result.check_returncode()
        if log_name and "Newton solver: NMFD" not in text:
            raise RuntimeError(f"Full KS1D did not select NMFD Newton: {log_name}")

    def snapshot(directory: Path, ignored: tuple[str, ...] = ()) -> dict[str, str]:
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file() and str(path.relative_to(directory)) not in ignored}

    run(sys.executable, "scripts/write_ks1d_ci_config.py", config, project,
        "--base-config", "json_project_files/KS1D_test_full.json", "--grid-size", 128,
        "--knots", 16, 16.2, "--steps", 4, "--step-size", 0.02, "--max-step-size", 0.03, "--clean-project")
    run(args.bd_executable.resolve(), config, "--seed-file", seed["parameter"],
        fixtures / "stability_replay" / seed["file"], "--continue-seed-only", "--quiet", *device,
        log_name="continuation.log")
    # Seed-only continuation omits discovery registries. Complete the configured
    # zero-attempt workflow before comparing repeated executions of that workflow.
    run(args.bd_executable.resolve(), config, "--quiet", *device, log_name="discovery.log")
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

    before_restart = snapshot(project)
    run(args.bd_executable.resolve(), config, "--quiet", *device, log_name="continuation_restart.log")
    if snapshot(project) != before_restart:
        raise RuntimeError("full-KS1D continuation restart changed committed results")

    run(args.stability_executable.resolve(), config, "--quiet", *device, log_name="stability.log")
    run(sys.executable, "scripts/validate_stability_diagram.py", project,
        "--expected-curves", 1, "--min-points", 10, "--require-complete", "--require-plot-sidecar",
        "--expected-point-type", "unstable", "--expected-unstable-real", seed["unstable"][0],
        "--expected-unstable-complex-pairs", seed["unstable"][1])
    paths = [project / "stability_diagram.dat", project / "0/debug_curve_stability.dat",
             project / "0/debug_curve_stability_plot.dat"]
    before = [path.read_bytes() for path in paths]
    run(args.stability_executable.resolve(), config, "--quiet", *device, log_name="stability_restart.log")
    if before != [path.read_bytes() for path in paths]:
        raise RuntimeError("full-KS1D stability restart changed committed results")

    deflation_config = args.work_dir / "deflation.json"
    deflation_project = args.work_dir / "deflation"
    run(sys.executable, "scripts/write_ks1d_ci_config.py", deflation_config, deflation_project,
        "--base-config", "json_project_files/KS1D_test_full.json", "--grid-size", 32,
        "--knots", 5.9, 6, 6.1, "--steps", 4, "--step-size", 0.02, "--max-step-size", 0.03,
        "--deflation-attempts", 1, "--clean-project")
    run(args.bd_executable.resolve(), deflation_config, "--quiet", *device, log_name="deflation.log")
    expected_mode = abs(read_rows(fixtures / "symmetric_lambda6.dat")[0][0]) / 4
    discovered = False
    for path in deflation_project.glob("*/debug_curve.dat"):
        rows = read_rows(path)
        if len(rows) >= 10 and any(len(row) >= 3 and math.isclose(row[0], 6, abs_tol=1e-9)
                                  and math.isclose(row[2], expected_mode, rel_tol=1e-7) for row in rows):
            discovered = True
    if not discovered:
        raise RuntimeError("full-KS1D deflation did not discover and continue the reference orbit")
    # A restart intentionally advances the enabled seed-attempt ledger, even
    # when no new root is found. All scientific results and other files must stay unchanged.
    policy = json.loads(deflation_config.read_text())["deflation_continuation"]["restart_policy"]["seed_schedule"]
    seed_ledger = (policy["registry_file"],) if policy["enabled"] else ()
    before_restart = snapshot(deflation_project, seed_ledger)
    (args.work_dir / "deflation_before_restart.json").write_text(json.dumps(before_restart, indent=2) + "\n")
    run(args.bd_executable.resolve(), deflation_config, "--quiet", *device, log_name="deflation_restart.log")
    if snapshot(deflation_project, seed_ledger) != before_restart:
        after_restart = snapshot(deflation_project, seed_ledger)
        changed = sorted(name for name in set(before_restart) | set(after_restart)
                         if before_restart.get(name) != after_restart.get(name))
        raise RuntimeError(f"full-KS1D deflation restart changed files: {changed}")
    print("Full KS1D NMFD: reference curve, unstable signature, restart, and deflation passed")


if __name__ == "__main__":
    main()
