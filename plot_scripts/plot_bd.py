#!/usr/bin/env python3
"""Plot bifurcation diagram debug curves from a project JSON file."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ANACONDA_PYTHON = Path("/home/noctum/anaconda3/bin/python")

EVENT_STYLES = {
    "steady": ("D", "tab:red", "steady-state transition"),
    "hopf": ("^", "tab:purple", "Hopf transition"),
    "multiple": ("X", "black", "multiple transition"),
    "unknown": ("P", "tab:orange", "unclassified transition"),
}


@dataclass
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
    norms: list[float]
    exact_coordinates: bool = True

    @property
    def unstable_dimension(self) -> int:
        return self.unstable_real + 2 * self.unstable_complex_pairs


class CurveRows(list):
    """Curve rows with optional per-row continuation segment identifiers."""

    def __init__(
        self,
        rows: list[list[float]],
        segment_ids: list[int] | None = None,
    ) -> None:
        super().__init__(rows)
        self.segment_ids = (
            segment_ids
            if segment_ids is not None and len(segment_ids) == len(rows)
            else [0] * len(rows)
        )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_font_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--font-scale",
        type=positive_float,
        default=1.0,
        help="Scale all default figure fonts by this factor (default: 1.0).",
    )
    parser.add_argument(
        "--title-font-size",
        type=positive_float,
        metavar="POINTS",
        help="Override the figure-title font size in points.",
    )
    parser.add_argument(
        "--axis-label-font-size",
        type=positive_float,
        metavar="POINTS",
        help="Override the x/y-axis label font size in points.",
    )
    parser.add_argument(
        "--tick-label-font-size",
        type=positive_float,
        metavar="POINTS",
        help="Override the axis tick-label font size in points.",
    )
    parser.add_argument(
        "--legend-font-size",
        type=positive_float,
        metavar="POINTS",
        help="Override the legend font size in points.",
    )


def add_branch_display_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--uniform-branch-color",
        nargs="?",
        const="gray",
        metavar="COLOR",
        help="Draw every BD branch in one color while retaining stability-event markers. If COLOR is omitted, gray is used.",
    )
    parser.add_argument(
        "--mark-branch-endpoints",
        action="store_true",
        help="Mark both the first and last point of every branch.",
    )
    parser.add_argument(
        "--mark-branch-starts",
        action="store_true",
        help="Mark the first point of every branch.",
    )
    parser.add_argument(
        "--mark-branch-ends",
        action="store_true",
        help="Mark the last point of every branch.",
    )
    parser.add_argument(
        "--branch-endpoint-color",
        default="black",
        metavar="COLOR",
        help="Color for branch start/end markers (default: black).",
    )


def add_stability_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--disable-stability",
        action="store_true",
        help="Ignore stability files and use ordinary branch styling.",
    )
    parser.add_argument(
        "--stability-color-mode",
        choices=("dimension", "class"),
        default="dimension",
        help="Color by unstable-manifold dimension or stable/unstable class.",
    )
    parser.add_argument(
        "--stability-cmap",
        default="viridis",
        help="Matplotlib colormap for unstable dimensions (default: viridis).",
    )
    parser.add_argument(
        "--stable-color",
        default="tab:blue",
        help="Stable-branch color in class mode.",
    )
    parser.add_argument(
        "--unstable-color",
        default="tab:red",
        help="Unstable-branch color in class mode.",
    )
    parser.add_argument(
        "--disable-stability-colorbar",
        action="store_true",
        help="Do not draw the unstable-dimension colorbar.",
    )
    parser.add_argument(
        "--disable-bifurcation-markers",
        action="store_true",
        help="Do not mark refined stability transitions.",
    )
    parser.add_argument(
        "--disable-bifurcation-legend",
        action="store_true",
        help="Draw transition markers without their legend entries.",
    )
    parser.add_argument(
        "--bifurcation-marker-size",
        type=positive_float,
        default=42.0,
        metavar="POINTS2",
        help="Area of stability-transition markers (default: 42).",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot debug_curve*.dat bifurcation-diagram norms."
    )
    parser.add_argument("config", help="JSON project config file")
    parser.add_argument(
        "--project-dir",
        help="Override the project directory from the JSON path_to_project field",
    )
    parser.add_argument(
        "--config-relative",
        action="store_true",
        help="Resolve a relative path_to_project against the config file directory",
    )
    parser.add_argument(
        "--norm",
        action="append",
        default=[],
        help="Norm label or zero-based norm index to plot. Can be used more than once.",
    )
    parser.add_argument(
        "--output",
        help="Output image path. Defaults to bd_all_norms.png or bd_<norm>.png in the project directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window instead of only saving an image.",
    )
    parser.add_argument(
        "--style",
        choices=("lines", "points", "both"),
        default="points",
        help="Curve drawing style.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Saved image DPI")
    parser.add_argument("--title", help="Plot title")
    parser.add_argument(
        "--legend",
        help="Override legend.dat path. One non-comment norm label per line.",
    )
    parser.add_argument(
        "--disable-legend",
        action="store_true",
        help="Do not draw branch legends on figures.",
    )
    add_branch_display_arguments(parser)
    add_stability_arguments(parser)
    add_font_arguments(parser)
    return parser.parse_args()


def import_pyplot(show: bool):
    if "MPLCONFIGDIR" not in os.environ:
        default_config = Path.home() / ".config" / "matplotlib"
        if not default_config.exists() or not os.access(default_config, os.W_OK):
            cache_dir = Path(tempfile.gettempdir()) / f"matplotlib-bd-{os.getuid()}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        if (
            ANACONDA_PYTHON.exists()
            and Path(sys.executable).resolve() != ANACONDA_PYTHON.resolve()
            and os.environ.get("BD_PLOTTER_REEXECED") != "1"
        ):
            env = dict(os.environ)
            env["BD_PLOTTER_REEXECED"] = "1"
            os.execve(str(ANACONDA_PYTHON), [str(ANACONDA_PYTHON), *sys.argv], env)
        raise


def apply_font_scale(plt, scale: float) -> None:
    if scale == 1.0:
        return

    from matplotlib.font_manager import FontProperties

    font_keys = (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    )
    resolved_sizes = {
        key: FontProperties(size=plt.rcParams[key]).get_size_in_points()
        for key in font_keys
    }
    for key, size in resolved_sizes.items():
        plt.rcParams[key] = size * scale


def apply_axis_font_overrides(axis, args: argparse.Namespace) -> None:
    if args.axis_label_font_size is not None:
        axis.xaxis.label.set_size(args.axis_label_font_size)
        axis.yaxis.label.set_size(args.axis_label_font_size)
    if args.tick_label_font_size is not None:
        axis.tick_params(axis="both", labelsize=args.tick_label_font_size)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def project_directory(args: argparse.Namespace, config_path: Path, config: dict) -> Path:
    if args.project_dir:
        return Path(args.project_dir).expanduser().resolve()

    raw_path = config.get("path_to_project")
    if raw_path is None:
        raise ValueError("JSON config does not contain path_to_project")

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    if args.config_relative:
        return (config_path.parent / path).resolve()

    cwd_relative = path.resolve()
    config_relative = (config_path.parent / path).resolve()
    if not cwd_relative.exists() and config_relative.exists():
        return config_relative
    return cwd_relative


def branch_sort_key(path: Path):
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def read_curve_file(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            try:
                rows.append([float(field) for field in fields])
            except ValueError:
                continue
    return rows


def read_curve_segment_ids(path: Path, row_count: int) -> list[int]:
    if not path.exists():
        return [0] * row_count

    segment_ids: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 6:
                return [0] * row_count
            try:
                source_index = int(fields[0])
                segment_id = int(fields[4])
            except ValueError:
                return [0] * row_count
            if source_index != len(segment_ids):
                return [0] * row_count
            segment_ids.append(segment_id)

    return segment_ids if len(segment_ids) == row_count else [0] * row_count


def load_curves(project_dir: Path) -> dict[str, list[list[float]]]:
    curves: dict[str, list[list[float]]] = {}
    for branch_dir in sorted((path for path in project_dir.iterdir() if path.is_dir()), key=branch_sort_key):
        curve_file = branch_dir / "debug_curve_all.dat"
        if not curve_file.exists():
            curve_file = branch_dir / "debug_curve.dat"
        if not curve_file.exists():
            continue
        raw_rows = read_curve_file(curve_file)
        rows = CurveRows(
            raw_rows,
            read_curve_segment_ids(
                branch_dir / "metadata_curve.dat",
                len(raw_rows),
            ),
        )
        if rows:
            curves[branch_dir.name] = rows
    return curves


def classify_event(
    before_real: int,
    before_complex_pairs: int,
    after_real: int,
    after_complex_pairs: int,
) -> str:
    real_change = after_real - before_real
    complex_change = after_complex_pairs - before_complex_pairs
    if real_change != 0 and complex_change == 0:
        return "steady"
    if real_change == 0 and complex_change != 0:
        return "hopf"
    if real_change != 0 or complex_change != 0:
        return "multiple"
    return "unknown"


def read_stability_plot_file(path: Path) -> list[StabilityPlotRecord]:
    records: list[StabilityPlotRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 12:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 12 fields"
                )
            try:
                norm_count = int(fields[11])
                if norm_count < 0 or len(fields) != 12 + norm_count:
                    raise ValueError(
                        f"norm_count={norm_count}, fields={len(fields)}"
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
                        norms=[float(value) for value in fields[12:]],
                        exact_coordinates=True,
                    )
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid stability plot row: {error}"
                ) from error
    return records


def read_legacy_stability_file(path: Path) -> list[StabilityPlotRecord]:
    records: list[StabilityPlotRecord] = []
    previous_dimension: tuple[int, int] | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 5:
                raise ValueError(
                    f"{path}:{line_number}: expected five stability fields"
                )
            try:
                unstable_real = int(fields[2])
                unstable_complex = int(fields[3])
                current_dimension = (unstable_real, unstable_complex)
                point_type = fields[1]
                before_dimension = (
                    previous_dimension
                    if (
                        point_type == "bifurcation"
                        and previous_dimension is not None
                    )
                    else current_dimension
                )
                event_type = (
                    classify_event(
                        before_dimension[0],
                        before_dimension[1],
                        current_dimension[0],
                        current_dimension[1],
                    )
                    if point_type == "bifurcation"
                    else "none"
                )
                records.append(
                    StabilityPlotRecord(
                        source_index=-1,
                        parameter=float(fields[0]),
                        point_type=point_type,
                        unstable_real=unstable_real,
                        unstable_complex_pairs=unstable_complex,
                        before_real=before_dimension[0],
                        before_complex_pairs=before_dimension[1],
                        after_real=current_dimension[0],
                        after_complex_pairs=current_dimension[1],
                        event_type=event_type,
                        event_id=int(fields[4]),
                        norms=[],
                        exact_coordinates=False,
                    )
                )
                previous_dimension = current_dimension
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid legacy stability row: {error}"
                ) from error
    return records


def align_legacy_stability_records(
    rows: list[list[float]],
    records: list[StabilityPlotRecord],
) -> None:
    cursor = 0
    for record in records:
        if not rows:
            record.source_index = 0
            continue
        candidates = range(cursor, len(rows))
        source_index = min(
            candidates,
            key=lambda index: abs(rows[index][0] - record.parameter),
            default=len(rows) - 1,
        )
        record.source_index = source_index
        cursor = min(source_index + 1, len(rows))


def load_stability_curves(
    project_dir: Path,
    curves: dict[str, list[list[float]]],
    disabled: bool = False,
) -> dict[str, list[StabilityPlotRecord]]:
    if disabled:
        return {}

    stability: dict[str, list[StabilityPlotRecord]] = {}
    for branch_name, rows in curves.items():
        branch_dir = project_dir / branch_name
        sidecar = branch_dir / "debug_curve_stability_plot.dat"
        legacy = branch_dir / "debug_curve_stability.dat"
        if sidecar.exists():
            records = read_stability_plot_file(sidecar)
        elif legacy.exists():
            records = read_legacy_stability_file(legacy)
            align_legacy_stability_records(rows, records)
        else:
            continue
        if records:
            stability[branch_name] = records
    return stability


def read_labels(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []

    labels: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                labels.append(stripped)
    return labels


def norm_count(curves: dict[str, list[list[float]]]) -> int:
    count = 0
    for rows in curves.values():
        for row in rows:
            if len(row) >= 3:
                count = max(count, len(row) - 2)
            elif len(row) == 2:
                count = max(count, 1)
    return count


def complete_labels(labels: list[str], count: int) -> list[str]:
    result = list(labels[:count])
    for index in range(len(result), count):
        result.append(f"norm_{index}")
    return result


def resolve_norms(requested: list[str], labels: list[str]) -> list[int]:
    if not requested:
        return list(range(len(labels)))

    normalized = {label: index for index, label in enumerate(labels)}
    selected: list[int] = []
    for item in requested:
        if re.fullmatch(r"\d+", item):
            index = int(item)
        elif item in normalized:
            index = normalized[item]
        else:
            raise ValueError(
                f"Unknown norm '{item}'. Available norms: {', '.join(labels)}"
            )
        if index < 0 or index >= len(labels):
            raise ValueError(f"Norm index {index} is outside 0..{len(labels)-1}")
        if index not in selected:
            selected.append(index)
    return selected


def row_norm_value(row: list[float], norm_index: int) -> float | None:
    value_index = norm_index + 1
    if len(row) >= norm_index + 3:
        return row[value_index]
    if len(row) == 2 and norm_index == 0:
        return row[1]
    return None


def line_style(style: str) -> dict:
    if style == "lines":
        return {"marker": None, "linewidth": 1.2}
    if style == "points":
        return {"marker": ".", "linestyle": "None", "markersize": 3.0}
    return {"marker": ".", "linewidth": 1.0, "markersize": 2.5}


def curve_coordinates(rows: list[list[float]], norm_index: int) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    segment_ids = getattr(rows, "segment_ids", [0] * len(rows))
    previous_segment: int | None = None
    for source_index, row in enumerate(rows):
        value = row_norm_value(row, norm_index)
        if value is None:
            continue
        segment_id = segment_ids[source_index]
        if previous_segment is not None and segment_id != previous_segment:
            xs.append(math.nan)
            ys.append(math.nan)
        xs.append(row[0])
        ys.append(value)
        previous_segment = segment_id
    return xs, ys


def curve_coordinates_with_indices(
    rows: list[list[float]],
    norm_index: int,
) -> tuple[list[float], list[float], list[int]]:
    xs: list[float] = []
    ys: list[float] = []
    indices: list[int] = []
    segment_ids = getattr(rows, "segment_ids", [0] * len(rows))
    previous_segment: int | None = None
    for source_index, row in enumerate(rows):
        value = row_norm_value(row, norm_index)
        if value is None:
            continue
        segment_id = segment_ids[source_index]
        if previous_segment is not None and segment_id != previous_segment:
            xs.append(math.nan)
            ys.append(math.nan)
            indices.append(source_index)
        xs.append(row[0])
        ys.append(value)
        indices.append(source_index)
        previous_segment = segment_id
    return xs, ys, indices


def stability_dimensions_for_rows(
    row_count: int,
    records: list[StabilityPlotRecord],
) -> list[int]:
    if row_count <= 0 or not records:
        return []

    ordered = sorted(records, key=lambda record: record.source_index)
    dimensions = [ordered[0].unstable_dimension] * row_count
    for record_index, record in enumerate(ordered):
        first = min(max(record.source_index, 0), row_count - 1)
        if record_index + 1 < len(ordered):
            last = min(
                max(ordered[record_index + 1].source_index, first + 1),
                row_count,
            )
        else:
            last = row_count
        for source_index in range(first, last):
            dimensions[source_index] = record.unstable_dimension
    return dimensions


def make_stability_color_mapping(
    plt,
    stability_curves: dict[str, list[StabilityPlotRecord]],
    args: argparse.Namespace,
):
    if not stability_curves:
        return None
    if args.uniform_branch_color is not None:
        return {
            "mode": "uniform",
            "color": args.uniform_branch_color,
        }
    if args.stability_color_mode == "class":
        return {
            "mode": "class",
            "stable_color": args.stable_color,
            "unstable_color": args.unstable_color,
        }

    from matplotlib.colors import BoundaryNorm

    maximum_dimension = max(
        record.unstable_dimension
        for records in stability_curves.values()
        for record in records
    )
    color_count = max(maximum_dimension + 1, 1)
    cmap = plt.get_cmap(args.stability_cmap, color_count)
    boundaries = [value - 0.5 for value in range(color_count + 1)]
    norm = BoundaryNorm(boundaries, color_count)
    return {
        "mode": "dimension",
        "cmap": cmap,
        "norm": norm,
        "maximum_dimension": maximum_dimension,
    }


def stability_color(dimension: int, mapping) -> object:
    if mapping["mode"] == "uniform":
        return mapping["color"]
    if mapping["mode"] == "class":
        return (
            mapping["stable_color"]
            if dimension == 0
            else mapping["unstable_color"]
        )
    return mapping["cmap"](mapping["norm"](dimension))


def plot_colored_line_runs(
    axis,
    xs: list[float],
    ys: list[float],
    dimensions: list[int],
    mapping,
    linewidth: float,
) -> None:
    if len(xs) < 2:
        return
    run_start = 0
    for index in range(1, len(xs)):
        if dimensions[index] == dimensions[index - 1]:
            continue
        axis.plot(
            xs[run_start:index + 1],
            ys[run_start:index + 1],
            color=stability_color(dimensions[index - 1], mapping),
            linewidth=linewidth,
        )
        run_start = index
    axis.plot(
        xs[run_start:],
        ys[run_start:],
        color=stability_color(dimensions[-1], mapping),
        linewidth=linewidth,
    )


def event_norm_value(
    record: StabilityPlotRecord,
    rows: list[list[float]],
    norm_index: int,
) -> float | None:
    if norm_index < len(record.norms):
        return record.norms[norm_index]
    if not rows:
        return None
    source_index = min(max(record.source_index, 0), len(rows) - 1)
    return row_norm_value(rows[source_index], norm_index)


def plot_stability_events(
    axis,
    rows: list[list[float]],
    norm_index: int,
    records: list[StabilityPlotRecord],
    args: argparse.Namespace,
    event_labels: set[str],
) -> None:
    if args.disable_bifurcation_markers:
        return
    for record in records:
        if record.point_type != "bifurcation":
            continue
        value = event_norm_value(record, rows, norm_index)
        if value is None:
            continue
        event_type = (
            record.event_type
            if record.event_type in EVENT_STYLES
            else "unknown"
        )
        marker, color, label = EVENT_STYLES[event_type]
        show_label = (
            not args.disable_legend
            and not args.disable_bifurcation_legend
            and event_type not in event_labels
        )
        axis.scatter(
            [record.parameter],
            [value],
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            s=args.bifurcation_marker_size,
            zorder=7,
            label=label if show_label else None,
        )
        event_labels.add(event_type)


def plot_branch(
    axis,
    branch_name: str,
    rows: list[list[float]],
    norm_index: int,
    args: argparse.Namespace,
    stability_records: list[StabilityPlotRecord] | None,
    stability_mapping,
    event_labels: set[str],
) -> tuple[list[float], list[float]]:
    xs, ys, source_indices = curve_coordinates_with_indices(
        rows,
        norm_index,
    )
    if not xs:
        return xs, ys

    if not stability_records or stability_mapping is None:
        branch_style = dict(line_style(args.style))
        if args.uniform_branch_color is not None:
            branch_style["color"] = args.uniform_branch_color
        axis.plot(
            xs,
            ys,
            label=f"branch {branch_name}",
            **branch_style,
        )
        return xs, ys

    if stability_mapping["mode"] == "uniform":
        branch_style = dict(line_style(args.style))
        branch_style["color"] = stability_mapping["color"]
        axis.plot(
            xs,
            ys,
            label=f"branch {branch_name}",
            **branch_style,
        )
        plot_stability_events(
            axis,
            rows,
            norm_index,
            stability_records,
            args,
            event_labels,
        )
        return xs, ys

    source_dimensions = stability_dimensions_for_rows(
        len(rows),
        stability_records,
    )
    dimensions = [
        source_dimensions[source_index]
        for source_index in source_indices
    ]
    if args.style in ("lines", "both"):
        plot_colored_line_runs(
            axis,
            xs,
            ys,
            dimensions,
            stability_mapping,
            1.2 if args.style == "lines" else 1.0,
        )
    if args.style in ("points", "both"):
        if stability_mapping["mode"] == "dimension":
            axis.scatter(
                xs,
                ys,
                c=dimensions,
                cmap=stability_mapping["cmap"],
                norm=stability_mapping["norm"],
                marker=".",
                s=9.0 if args.style == "points" else 6.25,
                linewidths=0,
                zorder=3,
            )
        else:
            axis.scatter(
                xs,
                ys,
                color=[
                    stability_color(dimension, stability_mapping)
                    for dimension in dimensions
                ],
                marker=".",
                s=9.0 if args.style == "points" else 6.25,
                linewidths=0,
                zorder=3,
            )
    plot_stability_events(
        axis,
        rows,
        norm_index,
        stability_records,
        args,
        event_labels,
    )
    return xs, ys


def add_stability_class_legend(
    axis,
    stability_mapping,
    args: argparse.Namespace,
) -> None:
    if (
        stability_mapping is None
        or stability_mapping["mode"] != "class"
        or args.disable_legend
    ):
        return
    axis.scatter(
        [],
        [],
        color=stability_mapping["stable_color"],
        marker=".",
        label="stable",
    )
    axis.scatter(
        [],
        [],
        color=stability_mapping["unstable_color"],
        marker=".",
        label="unstable",
    )


def add_stability_colorbar(
    figure,
    axes,
    stability_mapping,
    args: argparse.Namespace,
) -> None:
    if (
        stability_mapping is None
        or stability_mapping["mode"] != "dimension"
        or args.disable_stability_colorbar
    ):
        return

    from matplotlib.cm import ScalarMappable

    mappable = ScalarMappable(
        norm=stability_mapping["norm"],
        cmap=stability_mapping["cmap"],
    )
    mappable.set_array([])
    colorbar = figure.colorbar(
        mappable,
        ax=axes,
        ticks=range(stability_mapping["maximum_dimension"] + 1),
        pad=0.02,
        fraction=0.045,
    )
    colorbar.set_label(
        "unstable manifold dimension",
        fontsize=args.axis_label_font_size,
    )
    if args.tick_label_font_size is not None:
        colorbar.ax.tick_params(labelsize=args.tick_label_font_size)


def plot_branches_on_axis(
    axis,
    curves: dict[str, list[list[float]]],
    norm_index: int,
    args: argparse.Namespace,
    stability_curves: dict[str, list[StabilityPlotRecord]],
    stability_mapping,
) -> None:
    event_labels: set[str] = set()
    for branch_index, (branch_name, rows) in enumerate(curves.items()):
        xs, ys = plot_branch(
            axis,
            branch_name,
            rows,
            norm_index,
            args,
            stability_curves.get(branch_name),
            stability_mapping,
            event_labels,
        )
        if xs:
            plot_branch_endpoints(
                axis,
                xs,
                ys,
                args,
                branch_index == 0,
            )
    add_stability_class_legend(axis, stability_mapping, args)


def plot_branch_endpoints(
    axis,
    xs: list[float],
    ys: list[float],
    args: argparse.Namespace,
    include_labels: bool,
) -> None:
    mark_start = args.mark_branch_endpoints or args.mark_branch_starts
    mark_end = args.mark_branch_endpoints or args.mark_branch_ends
    if mark_start:
        axis.scatter(
            [xs[0]],
            [ys[0]],
            color=args.branch_endpoint_color,
            marker="o",
            s=25,
            zorder=5,
            label="branch start" if include_labels else None,
        )
    if mark_end:
        axis.scatter(
            [xs[-1]],
            [ys[-1]],
            color=args.branch_endpoint_color,
            marker="s",
            s=25,
            zorder=5,
            label="branch end" if include_labels else None,
        )


def default_output(project_dir: Path, labels: list[str], selected_norms: list[int]) -> Path:
    if len(selected_norms) == len(labels):
        return project_dir / "bd_all_norms.png"
    label = labels[selected_norms[0]]
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "norm"
    return project_dir / f"bd_{safe_label}.png"


def plot_curves(
    args: argparse.Namespace,
    plt,
    project_dir: Path,
    curves: dict[str, list[list[float]]],
    labels: list[str],
    selected_norms: list[int],
    stability_curves: dict[str, list[StabilityPlotRecord]],
) -> Path | None:
    if not selected_norms:
        raise ValueError("No norms selected")

    fig_height = max(3.0, 2.6 * len(selected_norms))
    figure, axes = plt.subplots(
        len(selected_norms),
        1,
        sharex=True,
        figsize=(8.0, fig_height),
        squeeze=False,
    )
    axes_flat = [axis for row in axes for axis in row]
    stability_mapping = make_stability_color_mapping(
        plt,
        stability_curves,
        args,
    )

    for axis, norm_index in zip(axes_flat, selected_norms):
        plot_branches_on_axis(
            axis,
            curves,
            norm_index,
            args,
            stability_curves,
            stability_mapping,
        )
        axis.set_ylabel(labels[norm_index])
        apply_axis_font_overrides(axis, args)
        axis.grid(True, alpha=0.3)
        if not args.disable_legend:
            handles, legend_labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(
                    handles,
                    legend_labels,
                    loc="best",
                    fontsize=args.legend_font_size or "small",
                )

    axes_flat[-1].set_xlabel("lambda")
    title_options = {}
    if args.title_font_size is not None:
        title_options["fontsize"] = args.title_font_size
    figure.suptitle(
        args.title or project_dir.name or "Bifurcation diagram",
        **title_options,
    )
    figure.tight_layout(
        rect=(0.0, 0.0, 0.94, 0.96)
        if (
            stability_mapping is not None
            and stability_mapping["mode"] == "dimension"
            and not args.disable_stability_colorbar
        )
        else (0.0, 0.0, 1.0, 0.96)
    )
    add_stability_colorbar(
        figure,
        axes_flat,
        stability_mapping,
        args,
    )

    output_path: Path | None = None
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    elif not args.show:
        output_path = default_output(project_dir, labels, selected_norms)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=args.dpi)

    if args.show:
        plt.show()
    else:
        plt.close(figure)

    return output_path


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    project_dir = project_directory(args, config_path, config)

    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")

    curves = load_curves(project_dir)
    if not curves:
        raise FileNotFoundError(f"No debug_curve*.dat files found under: {project_dir}")

    count = norm_count(curves)
    if count == 0:
        raise ValueError(f"No norm columns found under: {project_dir}")

    legend_path = Path(args.legend).expanduser().resolve() if args.legend else project_dir / "legend.dat"
    labels = complete_labels(read_labels(legend_path), count)
    selected_norms = resolve_norms(args.norm, labels)

    plt = import_pyplot(args.show)
    apply_font_scale(plt, args.font_scale)
    stability_curves = load_stability_curves(
        project_dir,
        curves,
        args.disable_stability,
    )
    output_path = plot_curves(
        args,
        plt,
        project_dir,
        curves,
        labels,
        selected_norms,
        stability_curves,
    )

    print(f"project: {project_dir}")
    print(f"curves: {len(curves)}")
    print("norms: " + ", ".join(labels))
    if stability_curves:
        print(f"stability curves: {len(stability_curves)}")
    if output_path is not None:
        print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
