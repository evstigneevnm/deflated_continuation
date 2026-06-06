#!/usr/bin/env python3
"""Validate Bratu bifurcation output using the collocation residual."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def curve_files(project_dir: Path) -> list[Path]:
    return sorted(project_dir.glob("*/debug_curve_all.dat"))


def read_vector(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if not fields:
                continue
            try:
                values.append(float(fields[0]))
            except ValueError as exc:
                raise RuntimeError(f"{path}:{line_number}: failed to parse vector value") from exc
    return values


def read_curve_entries(curve_file: Path):
    entries = []
    with curve_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 5:
                raise RuntimeError(
                    f"{curve_file}:{line_number}: expected lambda, max_u, norm, midpoint, id columns"
                )
            try:
                lambda_value = float(fields[0])
                max_u = float(fields[1])
                midpoint_u = float(fields[3])
                vector_id = int(float(fields[4]))
            except ValueError as exc:
                raise RuntimeError(f"{curve_file}:{line_number}: failed to parse numeric columns") from exc
            entries.append((curve_file, line_number, lambda_value, max_u, midpoint_u, vector_id))
    return entries


def chebyshev_d2_interior(n_interior: int) -> list[list[float]]:
    n_total = n_interior + 2
    last = n_total - 1
    z = [math.cos(math.pi * i / last) for i in range(n_total)]
    c = [1.0] * n_total
    c[0] = 2.0
    c[-1] = 2.0
    d = [[0.0 for _ in range(n_total)] for _ in range(n_total)]
    for i in range(n_total):
        row_sum = 0.0
        for j in range(n_total):
            if i == j:
                continue
            sign = 1.0 if (i + j) % 2 == 0 else -1.0
            value = sign * c[i] / (c[j] * (z[i] - z[j]))
            d[i][j] = value
            row_sum += value
        d[i][i] = -row_sum

    d2_full = [[0.0 for _ in range(n_total)] for _ in range(n_total)]
    for i in range(n_total):
        for j in range(n_total):
            d2_full[i][j] = 4.0 * sum(d[i][k] * d[k][j] for k in range(n_total))

    return [
        [d2_full[i + 1][j + 1] for j in range(n_interior)]
        for i in range(n_interior)
    ]


def fd3_d2_interior(n_interior: int) -> list[list[float]]:
    h = 1.0 / (n_interior + 1)
    inv_h2 = 1.0 / (h * h)
    d2 = [[0.0 for _ in range(n_interior)] for _ in range(n_interior)]
    for row in range(n_interior):
        d2[row][row] = -2.0 * inv_h2
        if row > 0:
            d2[row][row - 1] = inv_h2
        if row + 1 < n_interior:
            d2[row][row + 1] = inv_h2
    return d2


def normalize_spatial_discretization(value: object) -> str:
    if isinstance(value, str):
        name = value.lower()
        aliases = {
            "cheb": "chebyshev",
            "chebyshev_lobatto": "chebyshev",
            "chebyshev-lobatto": "chebyshev",
            "fd": "fd3",
            "finite_difference": "fd3",
            "finite-difference": "fd3",
        }
        return aliases.get(name, name)
    if isinstance(value, int):
        if value == 0:
            return "chebyshev"
        if value == 1:
            return "fd3"
    raise RuntimeError(f"unknown Bratu spatial discretization: {value!r}")


def read_spatial_discretization(args: argparse.Namespace) -> str:
    if args.config is None:
        return normalize_spatial_discretization(args.spatial_discretization)

    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    nonlinear_operator = config.get("nonlinear_operator", {})
    if "spatial_discretization" in nonlinear_operator:
        return normalize_spatial_discretization(nonlinear_operator["spatial_discretization"])

    problem_ints = nonlinear_operator.get("problem_int_parameters_vector", [0])
    if not problem_ints:
        return "chebyshev"
    return normalize_spatial_discretization(int(problem_ints[0]))


def d2_interior(n_interior: int, spatial_discretization: str) -> list[list[float]]:
    if spatial_discretization == "chebyshev":
        return chebyshev_d2_interior(n_interior)
    if spatial_discretization == "fd3":
        return fd3_d2_interior(n_interior)
    raise RuntimeError(f"unknown Bratu spatial discretization: {spatial_discretization}")


def bratu_lambda_from_max(max_u: float) -> float:
    if max_u <= 0.0:
        return 0.0
    theta = 2.0 * math.acosh(math.exp(0.5 * max_u))
    return 2.0 * theta * theta * math.exp(-max_u)


def residual_inf(d2: list[list[float]], u: list[float], lambda_value: float) -> float:
    worst = 0.0
    for row, row_values in enumerate(d2):
        value = sum(a * b for a, b in zip(row_values, u)) + lambda_value * math.exp(u[row])
        worst = max(worst, abs(value))
    return worst


def validate(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir)
    if not project_dir.is_dir():
        raise RuntimeError(f"project directory does not exist: {project_dir}")

    spatial_discretization = read_spatial_discretization(args)
    files = curve_files(project_dir)
    if args.expected_curves is not None and len(files) != args.expected_curves:
        names = ", ".join(path.parent.name for path in files) or "<none>"
        raise RuntimeError(f"expected {args.expected_curves} curve files, found {len(files)}: {names}")

    point_count = 0
    saved_count = 0
    max_residual = -1.0
    max_relation_error = -1.0
    max_lambda = -math.inf
    max_u_seen = -math.inf
    d2_cache: dict[int, list[list[float]]] = {}

    for curve_file in files:
        for curve_file, line_number, lambda_value, max_u, midpoint_u, vector_id in read_curve_entries(curve_file):
            point_count += 1
            max_lambda = max(max_lambda, lambda_value)
            max_u_seen = max(max_u_seen, max_u)
            if not all(math.isfinite(v) for v in (lambda_value, max_u, midpoint_u)):
                raise RuntimeError(f"{curve_file}:{line_number}: non-finite Bratu diagram point")
            if max_u < -args.max_u_tolerance:
                raise RuntimeError(f"{curve_file}:{line_number}: negative max_u={max_u:.6e}")

            relation_error = abs(lambda_value - bratu_lambda_from_max(max_u))
            max_relation_error = max(max_relation_error, relation_error)
            if relation_error > args.relation_tolerance:
                raise RuntimeError(
                    f"{curve_file}:{line_number}: lambda/max_u relation error "
                    f"{relation_error:.6e} exceeds {args.relation_tolerance:.6e}"
                )

            if vector_id <= 0:
                continue
            vector_path = curve_file.parent / str(vector_id)
            u = read_vector(vector_path)
            n = len(u)
            if n == 0:
                raise RuntimeError(f"{vector_path}: empty saved vector")
            d2 = d2_cache.setdefault(n, d2_interior(n, spatial_discretization))
            saved_count += 1
            res = residual_inf(d2, u, lambda_value)
            max_residual = max(max_residual, res)
            if res > args.residual_tolerance:
                raise RuntimeError(
                    f"{curve_file}:{line_number}: residual {res:.6e} exceeds "
                    f"{args.residual_tolerance:.6e} for saved vector {vector_path}"
                )

    if point_count < args.min_points:
        raise RuntimeError(f"only {point_count} Bratu points were found; expected at least {args.min_points}")
    if saved_count < args.min_saved_points:
        raise RuntimeError(
            f"only {saved_count} saved Bratu vectors were found; expected at least {args.min_saved_points}"
        )
    if args.require_fold and max_lambda < args.min_fold_lambda:
        raise RuntimeError(
            f"maximum lambda {max_lambda:.6e} is below required fold threshold {args.min_fold_lambda:.6e}"
        )
    if args.min_max_u is not None and max_u_seen < args.min_max_u:
        raise RuntimeError(f"maximum u {max_u_seen:.6e} is below required {args.min_max_u:.6e}")

    print(
        f"Validated {point_count} Bratu points in {project_dir}: "
        f"spatial={spatial_discretization}, saved={saved_count}, max residual={max_residual:.6e}, "
        f"max relation error={max_relation_error:.6e}, max lambda={max_lambda:.6e}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", help="Root project directory produced by bratu_bd.")
    parser.add_argument("--config", help="Bratu config used by the solver.")
    parser.add_argument(
        "--spatial-discretization",
        choices=("chebyshev", "fd3"),
        default="chebyshev",
        help="Spatial operator to use if --config is not provided.",
    )
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--relation-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--max-u-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--min-saved-points", type=int, default=20)
    parser.add_argument("--expected-curves", type=int)
    parser.add_argument("--require-fold", action="store_true")
    parser.add_argument("--min-fold-lambda", type=float, default=3.45)
    parser.add_argument("--min-max-u", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(validate(parse_args()))
