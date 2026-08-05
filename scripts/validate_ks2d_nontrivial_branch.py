#!/usr/bin/env python3
"""Validate that a KS2D project contains a nonzero branch near a seed knot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def read_curve(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise RuntimeError(
                    f"{path}:{line_number}: expected lambda and norm columns"
                )
            parameter = float(fields[0])
            norm = abs(float(fields[1]))
            if not math.isfinite(parameter) or not math.isfinite(norm):
                raise RuntimeError(f"{path}:{line_number}: non-finite curve value")
            points.append((parameter, norm))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--seed-parameter", type=float, default=4.333)
    parser.add_argument("--parameter-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--minimum-norm", type=float, default=1.0e-3)
    parser.add_argument("--minimum-nontrivial-curves", type=int, default=1)
    args = parser.parse_args()

    qualifying: list[tuple[int, float, int]] = []
    for curve_dir in sorted(
        (path for path in args.project_dir.iterdir() if path.is_dir()),
        key=lambda path: (
            (0, int(path.name)) if path.name.isdigit() else (1, path.name)
        ),
    ):
        curve_file = curve_dir / "debug_curve_all.dat"
        if not curve_file.is_file():
            continue
        points = read_curve(curve_file)
        maximum_norm = max((norm for _, norm in points), default=0.0)
        contains_seed = any(
            abs(parameter - args.seed_parameter) <= args.parameter_tolerance
            for parameter, _ in points
        )
        if contains_seed and maximum_norm >= args.minimum_norm:
            curve_number = int(curve_dir.name) if curve_dir.name.isdigit() else -1
            qualifying.append((curve_number, maximum_norm, len(points)))

    if len(qualifying) < args.minimum_nontrivial_curves:
        raise SystemExit(
            "FAILED: expected at least "
            f"{args.minimum_nontrivial_curves} nontrivial KS2D curves near "
            f"lambda={args.seed_parameter}, found {len(qualifying)}"
        )

    summary = ", ".join(
        f"curve {number}: max norm={maximum_norm:.6e}, points={count}"
        for number, maximum_norm, count in qualifying
    )
    print(f"Validated {len(qualifying)} nontrivial KS2D curves: {summary}")


if __name__ == "__main__":
    main()
