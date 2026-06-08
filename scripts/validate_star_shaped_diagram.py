#!/usr/bin/env python3
"""Validate star-shaped bifurcation output against the scalar residual."""

from __future__ import annotations

import argparse
import json
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
                    f"{curve_file}:{line_number}: expected at least lambda and x columns"
                )
            try:
                lambda_value = float(fields[0])
                x_value = float(fields[1])
            except ValueError as exc:
                raise RuntimeError(
                    f"{curve_file}:{line_number}: failed to parse numeric columns"
                ) from exc
            points.append((curve_file, line_number, lambda_value, x_value))
    return points


def read_curvature(args: argparse.Namespace) -> float:
    if args.config is None:
        return args.curvature
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    values = config.get("nonlinear_operator", {}).get("problem_real_parameters_vector", [])
    if not values:
        return args.curvature
    return float(values[0])


def residual(lambda_value: float, x_value: float, curvature: float) -> float:
    r2 = x_value * x_value + lambda_value * lambda_value
    if r2 <= 0.0:
        return math.inf
    return (
        math.sqrt(r2)
        - 1.0
        - 4.0
        * curvature
        * x_value
        * lambda_value
        * (x_value * x_value - lambda_value * lambda_value)
        / (r2 * r2)
    )


def expected_axis_extent(curvature: float) -> float:
    samples = 20000
    extent = 0.0
    for i in range(samples):
        theta = 2.0 * math.pi * i / samples
        radius = 1.0 + curvature * math.sin(4.0 * theta)
        extent = max(extent, abs(radius * math.cos(theta)), abs(radius * math.sin(theta)))
    return extent


def angle_coverage(points) -> float:
    if len(points) < 2:
        return 0.0
    angles = [math.atan2(lambda_value, x_value) for _, _, lambda_value, x_value in points]
    unwrapped = [angles[0]]
    for angle in angles[1:]:
        previous = unwrapped[-1]
        while angle - previous > math.pi:
            angle -= 2.0 * math.pi
        while angle - previous < -math.pi:
            angle += 2.0 * math.pi
        unwrapped.append(angle)
    return max(unwrapped) - min(unwrapped)


def validate(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir)
    if not project_dir.is_dir():
        raise RuntimeError(f"project directory does not exist: {project_dir}")

    curvature = read_curvature(args)
    files = curve_files(project_dir)
    if args.expected_curves is not None and len(files) != args.expected_curves:
        names = ", ".join(str(path.parent.name) for path in files) or "<none>"
        raise RuntimeError(
            f"expected {args.expected_curves} curve files, found {len(files)}: {names}"
        )

    point_count = 0
    max_residual = -1.0
    worst_residual_point = None
    all_points = []
    for curve_file in files:
        for curve_file, line_number, lambda_value, x_value in read_curve_points(curve_file):
            point_count += 1
            all_points.append((curve_file, line_number, lambda_value, x_value))
            if not (math.isfinite(lambda_value) and math.isfinite(x_value)):
                raise RuntimeError(
                    f"{curve_file}:{line_number}: non-finite point ({lambda_value}, {x_value})"
                )
            res = abs(residual(lambda_value, x_value, curvature))
            if res > max_residual:
                max_residual = res
                worst_residual_point = (curve_file, line_number, lambda_value, x_value)

    if point_count < args.min_points:
        raise RuntimeError(
            f"only {point_count} star-shaped points were found; expected at least {args.min_points}"
        )
    if max_residual > args.residual_tolerance:
        curve_file, line_number, lambda_value, x_value = worst_residual_point
        raise RuntimeError(
            f"max residual {max_residual:.6e} exceeds tolerance {args.residual_tolerance:.6e} "
            f"at {curve_file}:{line_number}: lambda={lambda_value:.17g}, x={x_value:.17g}"
        )

    if args.require_closed_star:
        if not all_points:
            raise RuntimeError("closed-star validation requested, but no points were found")
        min_lambda = min(point[2] for point in all_points)
        max_lambda = max(point[2] for point in all_points)
        min_x = min(point[3] for point in all_points)
        max_x = max(point[3] for point in all_points)
        coverage = angle_coverage(all_points)
        first = all_points[0]
        last = all_points[-1]
        closure = math.hypot(first[2] - last[2], first[3] - last[3])
        extent_threshold = expected_axis_extent(curvature) * args.min_axis_extent_fraction
        if min_lambda > -extent_threshold or max_lambda < extent_threshold:
            raise RuntimeError(
                f"closed-star lambda extent is incomplete: [{min_lambda:.6e}, {max_lambda:.6e}]"
            )
        if min_x > -extent_threshold or max_x < extent_threshold:
            raise RuntimeError(f"closed-star x extent is incomplete: [{min_x:.6e}, {max_x:.6e}]")
        if coverage < args.min_angle_coverage:
            raise RuntimeError(
                f"closed-star angular coverage {coverage:.6e} is below "
                f"{args.min_angle_coverage:.6e}"
            )
        if closure > args.max_closure_distance:
            raise RuntimeError(
                f"closed-star endpoint distance {closure:.6e} exceeds "
                f"{args.max_closure_distance:.6e}"
            )

    print(
        f"Validated {point_count} star-shaped points in {project_dir}: "
        f"C={curvature:.6g}, max residual={max_residual:.6e}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", help="Root project directory produced by star_shaped_bd.")
    parser.add_argument("--config", help="Star-shaped config used by the solver.")
    parser.add_argument("--curvature", type=float, default=0.2)
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--expected-curves", type=int)
    parser.add_argument(
        "--require-closed-star",
        action="store_true",
        help="Require one continuous point set to cover and close the full star-shaped curve.",
    )
    parser.add_argument("--min-axis-extent-fraction", type=float, default=0.9)
    parser.add_argument("--min-angle-coverage", type=float, default=6.0)
    parser.add_argument("--max-closure-distance", type=float, default=1.0e-7)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(validate(parse_args()))
