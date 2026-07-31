#!/usr/bin/env python3
"""Run and collate the CPU-only exact S80 transformed-angle experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AngleResult:
    angle: float
    status: str
    iterations: int
    restarts: int
    transformed_calls: int
    solve_seconds: float
    eigenvalues: list[complex]


def parse_fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in line.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


def run_angle(args: argparse.Namespace, angle: float) -> tuple[AngleResult, str]:
    command = [
        str(args.binary),
        str(args.matrix),
        f"{angle:.17g}",
        str(args.maximum_restarts),
        str(args.krylov_dimension),
        str(args.restart_dimension),
        str(args.desired_eigenvalues),
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.omp_threads)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"angle {angle:.17g} failed with rc={completed.returncode}\n"
            f"{output}"
        )

    summary: dict[str, str] | None = None
    eigenvalues: list[complex] = []
    for line in completed.stdout.splitlines():
        if line.startswith("case=S80PI_n1 "):
            summary = parse_fields(line)
        elif line.startswith("recovered_eigenvalue="):
            fields = parse_fields(line)
            value = complex(
                float(fields["value_real"]),
                float(fields["value_imag"]),
            )
            if value.imag >= 0.0:
                eigenvalues.append(value)
    if summary is None:
        raise RuntimeError(f"angle {angle:.17g} produced no summary")

    return (
        AngleResult(
            angle=angle,
            status=summary["status"],
            iterations=int(summary["iterations"]),
            restarts=int(summary["restarts"]),
            transformed_calls=int(summary["transformed_calls"]),
            solve_seconds=float(summary["solve_seconds"]),
            eigenvalues=eigenvalues,
        ),
        output,
    )


def insert_unique(
    catalog: list[dict],
    angle: float,
    value: complex,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    for observation in catalog:
        current = complex(observation["real"], observation["imaginary"])
        tolerance = absolute_tolerance + relative_tolerance * max(
            1.0, abs(current), abs(value)
        )
        if abs(current - value) <= tolerance:
            observation["angles"].append(angle)
            return
    catalog.append(
        {
            "real": value.real,
            "imaginary": value.imag,
            "angles": [angle],
        }
    )


def scan_angles(args: argparse.Namespace) -> list[float]:
    first = args.first_angle_index
    last = args.last_angle_index
    return [
        index * math.pi / args.angle_denominator
        for index in range(first, last + 1)
    ]


def serializable_result(result: AngleResult) -> dict:
    data = asdict(result)
    data["eigenvalues"] = [
        {"real": value.real, "imaginary": value.imag}
        for value in result.eigenvalues
    ]
    return data


def write_outputs(
    args: argparse.Namespace,
    results: list[AngleResult],
    catalog: list[dict],
) -> None:
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with (args.output_directory / "scan.json").open(
        "w", encoding="ascii"
    ) as stream:
        json.dump(
            {
                "matrix": str(args.matrix),
                "angles": [serializable_result(result) for result in results],
                "catalog": catalog,
            },
            stream,
            indent=2,
        )
        stream.write("\n")

    with (args.output_directory / "catalog.csv").open(
        "w", newline="", encoding="ascii"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["real", "imaginary", "angles"])
        for observation in catalog:
            writer.writerow(
                [
                    f"{observation['real']:.17e}",
                    f"{observation['imaginary']:.17e}",
                    ";".join(
                        f"{angle:.17e}" for angle in observation["angles"]
                    ),
                ]
            )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(
            "build/eigensolver_stage5_exact/"
            "characterize_transformed_krylov_schur_"
            "S80PI_n1_exact_cpu_omp.bin"
        ),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "data/external/suitesparse/Rommes/S80PI_n1/S80PI_n1.mtx"
        ),
    )
    parser.add_argument("--angle-denominator", type=int, default=50)
    parser.add_argument("--first-angle-index", type=int, default=1)
    parser.add_argument("--last-angle-index", type=int, default=16)
    parser.add_argument("--maximum-restarts", type=int, default=25)
    parser.add_argument("--krylov-dimension", type=int, default=50)
    parser.add_argument("--restart-dimension", type=int, default=20)
    parser.add_argument("--desired-eigenvalues", type=int, default=2)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--dedup-absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--dedup-relative-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("build/eigensolver_stage5_exact/angle_scan"),
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.angle_denominator <= 0:
        raise ValueError("--angle-denominator must be positive")
    if args.first_angle_index > args.last_angle_index:
        raise ValueError("invalid angle index interval")

    results: list[AngleResult] = []
    catalog: list[dict] = []
    raw_directory = args.output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    for index, angle in enumerate(scan_angles(args), start=args.first_angle_index):
        result, output = run_angle(args, angle)
        results.append(result)
        (raw_directory / f"angle_{index:02d}.txt").write_text(
            output, encoding="ascii"
        )
        for value in result.eigenvalues:
            insert_unique(
                catalog,
                angle,
                value,
                args.dedup_absolute_tolerance,
                args.dedup_relative_tolerance,
            )
        print(
            f"angle_index={index} phi={angle:.9f} "
            f"status={result.status} modes={len(result.eigenvalues)} "
            f"iterations={result.iterations}"
        )

    catalog.sort(key=lambda item: item["imaginary"])
    write_outputs(args, results, catalog)
    converged = sum(result.status == "success" for result in results)
    print(
        f"angles={len(results)} converged={converged} "
        f"unique_positive_modes={len(catalog)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"S80 exact angle scan failed: {error}", file=sys.stderr)
        raise SystemExit(1)
