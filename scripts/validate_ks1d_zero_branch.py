#!/usr/bin/env python3
"""Validate the symmetrized KS1D zero-branch bifurcation output."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def curve_files(project_dir: Path) -> list[Path]:
    return sorted(project_dir.glob("*/debug_curve_all.dat"))


def read_curve_points(curve_file: Path):
    points = []
    with curve_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise RuntimeError(
                    f"{curve_file}:{line_number}: expected lambda and at least one norm column"
                )
            try:
                lambda_value = float(fields[0])
                reduced_norm = float(fields[1])
            except ValueError as exc:
                raise RuntimeError(
                    f"{curve_file}:{line_number}: failed to parse numeric columns"
                ) from exc
            points.append((curve_file, line_number, lambda_value, reduced_norm))
    return points


def validate(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir)
    if not project_dir.is_dir():
        raise RuntimeError(f"project directory does not exist: {project_dir}")

    files = curve_files(project_dir)
    if args.expected_curves is not None and len(files) != args.expected_curves:
        names = ", ".join(str(path.parent.name) for path in files) or "<none>"
        raise RuntimeError(
            f"expected {args.expected_curves} curve files, found {len(files)}: {names}"
        )

    point_count = 0
    max_norm = -1.0
    min_lambda = math.inf
    max_lambda = -math.inf
    worst_point = None
    for curve_file in files:
        for curve_file, line_number, lambda_value, reduced_norm in read_curve_points(curve_file):
            point_count += 1
            if not (math.isfinite(lambda_value) and math.isfinite(reduced_norm)):
                raise RuntimeError(
                    f"{curve_file}:{line_number}: non-finite point "
                    f"(lambda={lambda_value}, norm={reduced_norm})"
                )
            min_lambda = min(min_lambda, lambda_value)
            max_lambda = max(max_lambda, lambda_value)
            if abs(reduced_norm) > max_norm:
                max_norm = abs(reduced_norm)
                worst_point = (curve_file, line_number, lambda_value, reduced_norm)

    if point_count < args.min_points:
        raise RuntimeError(
            f"only {point_count} KS1D zero-branch points were found; "
            f"expected at least {args.min_points}"
        )
    if max_norm > args.max_reduced_norm:
        curve_file, line_number, lambda_value, reduced_norm = worst_point
        raise RuntimeError(
            f"max reduced norm {max_norm:.6e} exceeds tolerance "
            f"{args.max_reduced_norm:.6e} at {curve_file}:{line_number}: "
            f"lambda={lambda_value:.17g}, norm={reduced_norm:.17g}"
        )
    if args.min_lambda_span > 0.0 and max_lambda - min_lambda < args.min_lambda_span:
        raise RuntimeError(
            f"lambda span {max_lambda - min_lambda:.6e} is below "
            f"{args.min_lambda_span:.6e}; interval=[{min_lambda:.6e}, {max_lambda:.6e}]"
        )

    print(
        f"Validated {point_count} KS1D zero-branch points in {project_dir}: "
        f"max norm={max_norm:.6e}, lambda span={max_lambda - min_lambda:.6e}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", help="Root project directory produced by KS1D_bd.")
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--expected-curves", type=int)
    parser.add_argument("--max-reduced-norm", type=float, default=1.0e-9)
    parser.add_argument("--min-lambda-span", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(validate(parse_args()))
