#!/usr/bin/env python3
"""Validate text output produced by the stability-diagram postprocessor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StabilityRecord:
    parameter: float
    point_type: str
    unstable_real: int
    unstable_complex_pairs: int
    solution_id: int


@dataclass(frozen=True)
class StabilityPlotRecord:
    source_index: int
    parameter: float
    point_type: str
    unstable_real: int
    unstable_complex_pairs: int
    before_real: int
    before_complex_pairs: int
    after_real: int
    after_complex_pairs: int
    event_type: str
    event_id: int
    norms: tuple[float, ...]


def numeric_curve_directories(project: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in project.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )


def nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_records(path: Path) -> list[StabilityRecord]:
    records: list[StabilityRecord] = []
    for line_number, line in enumerate(nonempty_lines(path), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(
                f"{path}:{line_number}: expected 5 fields, got {len(fields)}"
            )
        try:
            records.append(
                StabilityRecord(
                    parameter=float(fields[0]),
                    point_type=fields[1],
                    unstable_real=int(fields[2]),
                    unstable_complex_pairs=int(fields[3]),
                    solution_id=int(fields[4]),
                )
            )
        except ValueError as error:
            raise RuntimeError(
                f"{path}:{line_number}: invalid stability record"
            ) from error
    return records


def read_plot_records(path: Path) -> list[StabilityPlotRecord]:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")
    records: list[StabilityPlotRecord] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 12:
            raise RuntimeError(
                f"{path}:{line_number}: expected at least 12 fields"
            )
        try:
            norm_count = int(fields[11])
            if norm_count <= 0 or len(fields) != 12 + norm_count:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid norm count "
                    f"{norm_count}"
                )
            records.append(
                StabilityPlotRecord(
                    source_index=int(fields[0]),
                    parameter=float(fields[1]),
                    point_type=fields[2],
                    unstable_real=int(fields[3]),
                    unstable_complex_pairs=int(fields[4]),
                    before_real=int(fields[5]),
                    before_complex_pairs=int(fields[6]),
                    after_real=int(fields[7]),
                    after_complex_pairs=int(fields[8]),
                    event_type=fields[9],
                    event_id=int(fields[10]),
                    norms=tuple(float(value) for value in fields[12:]),
                )
            )
        except ValueError as error:
            raise RuntimeError(
                f"{path}:{line_number}: invalid plot record"
            ) from error
    return records


def validate_plot_records(
    curve: Path,
    records: list[StabilityRecord],
    continuation_count: int,
) -> None:
    sidecar = curve / "debug_curve_stability_plot.dat"
    plot_records = read_plot_records(sidecar)
    if len(plot_records) != len(records):
        raise RuntimeError(
            f"{curve}: plot records ({len(plot_records)}) do not "
            f"match stability records ({len(records)})"
        )
    if (curve / "debug_curve_stability_plot.dat.tmp").exists():
        raise RuntimeError(f"{curve}: plot sidecar temporary file remains")

    previous_source_index = -1
    for record, plot_record in zip(records, plot_records):
        if (
            plot_record.source_index < previous_source_index
            or plot_record.source_index < 0
            or plot_record.source_index >= continuation_count
        ):
            raise RuntimeError(
                f"{curve}: invalid plot source index "
                f"{plot_record.source_index}"
            )
        previous_source_index = plot_record.source_index
        if (
            plot_record.point_type != record.point_type
            or plot_record.unstable_real != record.unstable_real
            or plot_record.unstable_complex_pairs
            != record.unstable_complex_pairs
            or plot_record.event_id != record.solution_id
        ):
            raise RuntimeError(
                f"{curve}: plot record does not match stability archive "
                f"text output"
            )
        if record.point_type == "bifurcation":
            if (
                plot_record.event_type
                not in {"steady", "hopf", "multiple"}
                or (
                    plot_record.before_real,
                    plot_record.before_complex_pairs,
                )
                == (
                    plot_record.after_real,
                    plot_record.after_complex_pairs,
                )
            ):
                raise RuntimeError(
                    f"{curve}: inconsistent bifurcation plot metadata"
                )
        elif plot_record.event_type != "none":
            raise RuntimeError(
                f"{curve}: non-bifurcation plot record has event type "
                f"{plot_record.event_type!r}"
            )


def validate_record(
    curve: Path,
    record: StabilityRecord,
    args: argparse.Namespace,
) -> None:
    if record.unstable_real < 0 or record.unstable_complex_pairs < 0:
        raise RuntimeError(f"{curve}: negative unstable dimension")
    if record.point_type == "stable":
        if (
            record.unstable_real != 0
            or record.unstable_complex_pairs != 0
            or record.solution_id != 0
        ):
            raise RuntimeError(f"{curve}: inconsistent stable record")
    elif record.point_type == "unstable":
        if (
            record.unstable_real + record.unstable_complex_pairs == 0
            or record.solution_id != 0
        ):
            raise RuntimeError(f"{curve}: inconsistent unstable record")
    elif record.point_type == "bifurcation":
        if record.solution_id <= 0:
            raise RuntimeError(f"{curve}: bifurcation record has no solution")
        solution_path = curve / f"s{record.solution_id}"
        if not solution_path.is_file():
            raise RuntimeError(
                f"{curve}: missing bifurcation solution {solution_path.name}"
            )
    else:
        raise RuntimeError(
            f"{curve}: unknown point type {record.point_type!r}"
        )

    if (
        args.expected_point_type is not None
        and record.point_type != args.expected_point_type
    ):
        raise RuntimeError(
            f"{curve}: expected point type {args.expected_point_type}, "
            f"got {record.point_type}"
        )
    if (
        args.expected_unstable_real is not None
        and record.unstable_real != args.expected_unstable_real
    ):
        raise RuntimeError(
            f"{curve}: expected unstable real dimension "
            f"{args.expected_unstable_real}, got {record.unstable_real}"
        )
    if (
        args.expected_unstable_complex_pairs is not None
        and record.unstable_complex_pairs
        != args.expected_unstable_complex_pairs
    ):
        raise RuntimeError(
            f"{curve}: expected unstable complex-pair dimension "
            f"{args.expected_unstable_complex_pairs}, "
            f"got {record.unstable_complex_pairs}"
        )


def validate(args: argparse.Namespace) -> None:
    project = Path(args.project_dir)
    if not project.is_dir():
        raise RuntimeError(f"project directory does not exist: {project}")
    archive = project / "stability_diagram.dat"
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError(f"missing or empty stability archive: {archive}")

    curves = numeric_curve_directories(project)
    if args.expected_curves is not None and len(curves) != args.expected_curves:
        raise RuntimeError(
            f"expected {args.expected_curves} curves, found {len(curves)}"
        )
    if not curves:
        raise RuntimeError("stability diagram has no curve directories")

    total_records = 0
    stable_records = 0
    unstable_records = 0
    bifurcation_records: list[StabilityRecord] = []
    for curve in curves:
        stability_path = curve / "debug_curve_stability.dat"
        records = read_records(stability_path)
        if len(records) < args.min_points:
            raise RuntimeError(
                f"{curve}: expected at least {args.min_points} stability "
                f"records, found {len(records)}"
            )
        if args.require_complete:
            continuation_records = nonempty_lines(
                curve / "debug_curve_all.dat"
            )
            if len(records) != len(continuation_records):
                raise RuntimeError(
                    f"{curve}: stability records ({len(records)}) do not "
                    f"cover continuation records ({len(continuation_records)})"
                )
        else:
            continuation_records = nonempty_lines(
                curve / "debug_curve_all.dat"
            )
        if args.require_plot_sidecar:
            validate_plot_records(
                curve,
                records,
                len(continuation_records),
            )
        for record in records:
            validate_record(curve, record, args)
            if record.point_type == "stable":
                stable_records += 1
            elif record.point_type == "unstable":
                unstable_records += 1
            else:
                bifurcation_records.append(record)
        total_records += len(records)

    if (
        args.expected_bifurcations is not None
        and len(bifurcation_records) != args.expected_bifurcations
    ):
        raise RuntimeError(
            f"expected {args.expected_bifurcations} bifurcations, found "
            f"{len(bifurcation_records)}"
        )
    if stable_records < args.min_stable_points:
        raise RuntimeError(
            f"expected at least {args.min_stable_points} stable records, "
            f"found {stable_records}"
        )
    if unstable_records < args.min_unstable_points:
        raise RuntimeError(
            f"expected at least {args.min_unstable_points} unstable records, "
            f"found {unstable_records}"
        )

    unmatched = list(bifurcation_records)
    for expected_parameter in args.expected_bifurcation_parameter:
        if not unmatched:
            raise RuntimeError(
                f"no bifurcation remains to match parameter "
                f"{expected_parameter:.16g}"
            )
        closest = min(
            unmatched,
            key=lambda record: abs(record.parameter - expected_parameter),
        )
        error = abs(closest.parameter - expected_parameter)
        if error > args.bifurcation_parameter_tolerance:
            raise RuntimeError(
                f"expected bifurcation near {expected_parameter:.16g}, "
                f"closest is {closest.parameter:.16g} (error {error:.3e}, "
                f"tolerance {args.bifurcation_parameter_tolerance:.3e})"
            )
        unmatched.remove(closest)

    print(
        f"Validated stability diagram: curves={len(curves)}, "
        f"records={total_records}, stable={stable_records}, "
        f"unstable={unstable_records}, "
        f"bifurcations={len(bifurcation_records)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir")
    parser.add_argument("--expected-curves", type=int)
    parser.add_argument("--min-points", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-plot-sidecar", action="store_true")
    parser.add_argument(
        "--expected-point-type",
        choices=("stable", "unstable", "bifurcation"),
    )
    parser.add_argument("--expected-unstable-real", type=int)
    parser.add_argument("--expected-unstable-complex-pairs", type=int)
    parser.add_argument("--expected-bifurcations", type=int)
    parser.add_argument("--min-stable-points", type=int, default=0)
    parser.add_argument("--min-unstable-points", type=int, default=0)
    parser.add_argument(
        "--expected-bifurcation-parameter",
        action="append",
        default=[],
        type=float,
    )
    parser.add_argument(
        "--bifurcation-parameter-tolerance",
        type=float,
        default=1.0e-3,
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        validate(parse_args())
    except RuntimeError as error:
        raise SystemExit(f"FAILED: {error}") from error
