#!/usr/bin/env python3
"""Generate an offline sparse-direct reference catalog for S80PI_n1."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import eigs


@dataclass
class Observation:
    value: complex
    relative_residual: float
    shift_frequency: float


def parse_frequencies(text: str) -> list[float]:
    values = [float(item) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one frequency is required")
    return values


def relative_residual(matrix, value: complex, vector) -> float:
    applied = matrix @ vector
    residual = applied - value * vector
    denominator = np.linalg.norm(applied) + abs(value) * np.linalg.norm(vector)
    if denominator == 0.0:
        return float(np.linalg.norm(residual))
    return float(np.linalg.norm(residual) / denominator)


def insert_observation(
    catalog: list[Observation],
    candidate: Observation,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    for index, current in enumerate(catalog):
        tolerance = absolute_tolerance + relative_tolerance * max(
            1.0, abs(current.value), abs(candidate.value)
        )
        if abs(current.value - candidate.value) <= tolerance:
            if candidate.relative_residual < current.relative_residual:
                catalog[index] = candidate
            return
    catalog.append(candidate)


def build_catalog(args: argparse.Namespace) -> list[Observation]:
    # Complex storage forces scipy/ARPACK to use the full complex shift-inverse
    # path. Its real-matrix OPpart path is inappropriate for this reference.
    matrix = mmread(
        args.matrix,
        spmatrix=True,
    ).tocsc().astype(np.complex128)
    if matrix.shape != (4028, 4028):
        raise ValueError(f"unexpected S80PI_n1 shape: {matrix.shape}")

    catalog: list[Observation] = []
    for frequency in args.frequencies:
        values, vectors = eigs(
            matrix,
            k=args.eigenvalues_per_shift,
            sigma=complex(0.0, frequency),
            which="LM",
            tol=args.solver_tolerance,
            maxiter=args.maximum_iterations,
        )
        for index, value in enumerate(values):
            if value.imag < 0.0:
                continue
            observation = Observation(
                value=complex(value),
                relative_residual=relative_residual(
                    matrix, value, vectors[:, index]
                ),
                shift_frequency=frequency,
            )
            insert_observation(
                catalog,
                observation,
                args.dedup_absolute_tolerance,
                args.dedup_relative_tolerance,
            )

    catalog.sort(key=lambda item: (item.value.imag, item.value.real))
    return catalog


def write_catalog(
    catalog: list[Observation],
    stream,
) -> None:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "real",
            "imaginary",
            "relative_residual",
            "shift_frequency",
        ]
    )
    for observation in catalog:
        writer.writerow(
            [
                f"{observation.value.real:.17e}",
                f"{observation.value.imag:.17e}",
                f"{observation.relative_residual:.17e}",
                f"{observation.shift_frequency:.17e}",
            ]
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an independent sparse-direct eigenvalue catalog for "
            "the standard S80PI_n1 A matrix."
        )
    )
    parser.add_argument(
        "matrix",
        type=Path,
        nargs="?",
        default=Path(
            "data/external/suitesparse/Rommes/S80PI_n1/S80PI_n1.mtx"
        ),
    )
    parser.add_argument(
        "--frequencies",
        type=parse_frequencies,
        default=parse_frequencies("1.55,1.60,1.65,1.70,1.75,1.80,1.85,1.90"),
        help="comma-separated positive imaginary shifts",
    )
    parser.add_argument("--eigenvalues-per-shift", type=int, default=8)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--maximum-iterations", type=int, default=10000)
    parser.add_argument("--dedup-absolute-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--dedup-relative-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.eigenvalues_per_shift <= 0:
        raise ValueError("--eigenvalues-per-shift must be positive")
    catalog = build_catalog(args)
    if args.output is None:
        write_catalog(catalog, sys.stdout)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="ascii") as stream:
            write_catalog(catalog, stream)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"S80 reference generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
