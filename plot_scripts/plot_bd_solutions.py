#!/usr/bin/env python3
"""Plot a bifurcation diagram with solution-profile insets from manifest.jsonl."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import plot_bd as bd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a BD norm and overlay saved solution profiles from visualization/manifest.jsonl."
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
        "--visualization-dir",
        help="Directory containing manifest.jsonl. Defaults to <project-dir>/visualization.",
    )
    parser.add_argument(
        "--manifest",
        help="Override manifest path. Relative paths are resolved against --visualization-dir.",
    )
    parser.add_argument(
        "--norm",
        default=None,
        help="Norm label or zero-based norm index to plot. Defaults to the first norm.",
    )
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        help="Only use thumbnails from this branch. Can be used more than once, or pass 'all' to write one plot per branch.",
    )
    parser.add_argument(
        "--max-thumbnails",
        type=int,
        default=10,
        help="Maximum number of solution insets to draw.",
    )
    parser.add_argument(
        "--one-point-per-branch",
        "--one-per-branch",
        dest="one_point_per_branch",
        action="store_true",
        help="Draw the middle available saved solution from every selected branch.",
    )
    parser.add_argument(
        "--fit-solutions",
        "--scale-to-fit-solutions",
        dest="fit_solutions",
        action="store_true",
        help="Grow the figure and place solution profiles above the BD without overlap.",
    )
    parser.add_argument(
        "--thumbnail-stride",
        type=int,
        default=1,
        help="Keep only every Nth manifest record before max-thumbnails sampling.",
    )
    parser.add_argument(
        "--thumbnail-columns",
        type=int,
        default=5,
        help="Number of inset columns in overlay layout.",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=float,
        default=0.16,
        help="Inset width in main-axis coordinates.",
    )
    parser.add_argument(
        "--thumbnail-height",
        type=float,
        default=0.16,
        help="Inset height in main-axis coordinates.",
    )
    parser.add_argument(
        "--style",
        choices=("lines", "points", "both"),
        default="points",
        help="BD curve drawing style.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Saved image DPI")
    parser.add_argument("--title", help="Plot title")
    parser.add_argument(
        "--x-label",
        "--xlabel",
        dest="x_label",
        help="Override the main bifurcation-diagram x-axis label.",
    )
    parser.add_argument(
        "--y-label",
        "--ylabel",
        dest="y_label",
        help="Override the main bifurcation-diagram y-axis label.",
    )
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window")
    parser.add_argument(
        "--legend",
        help="Override legend.dat path. One non-comment norm label per line.",
    )
    parser.add_argument(
        "--disable-legend",
        action="store_true",
        help="Do not draw the branch legend.",
    )
    parser.add_argument(
        "--profile-title-font-size",
        "--thumbnail-title-font-size",
        dest="profile_title_font_size",
        type=bd.positive_float,
        metavar="POINTS",
        help="Override the title font size above solution profiles.",
    )
    parser.add_argument(
        "--point-label-font-size",
        "--annotation-font-size",
        dest="point_label_font_size",
        type=bd.positive_float,
        metavar="POINTS",
        help="Override the numbered BD point-label font size.",
    )
    bd.add_branch_display_arguments(parser)
    bd.add_font_arguments(parser)
    return parser.parse_args()


def visualization_directory(project_dir: Path, args: argparse.Namespace) -> Path:
    if args.visualization_dir:
        return Path(args.visualization_dir).expanduser().resolve()
    return project_dir / "visualization"


def manifest_path(visualization_dir: Path, args: argparse.Namespace) -> Path:
    if not args.manifest:
        return visualization_dir / "manifest.jsonl"
    path = Path(args.manifest).expanduser()
    if path.is_absolute():
        return path
    return visualization_dir / path


def load_manifest(path: Path, visualization_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            data_file = Path(record.get("data_file", "")).expanduser()
            if not data_file.is_absolute():
                data_file = visualization_dir / data_file
            record["resolved_data_file"] = data_file
            record["manifest_line"] = line_number
            records.append(record)
    return records


def select_norm(args: argparse.Namespace, labels: list[str]) -> int:
    requested = [args.norm] if args.norm is not None else ["0"]
    return bd.resolve_norms(requested, labels)[0]


def manifest_norm_value(record: dict[str, Any], norm_index: int) -> float | None:
    norms = record.get("norms", [])
    if norm_index < 0 or norm_index >= len(norms):
        return None
    try:
        return float(norms[norm_index])
    except (TypeError, ValueError):
        return None


def parse_branch_tokens(tokens: list[str]) -> tuple[bool, list[int]]:
    all_branches = False
    branches: list[int] = []
    for token in tokens:
        if token.lower() == "all":
            all_branches = True
            continue
        try:
            branch = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid --branch value '{token}'. Use an integer branch id or 'all'.") from exc
        if branch < 0:
            raise ValueError(f"Invalid --branch value '{token}'. Branch ids must be non-negative.")
        if branch not in branches:
            branches.append(branch)
    return all_branches, branches


def available_branches(records: list[dict[str, Any]]) -> list[int]:
    branches: set[int] = set()
    for record in records:
        try:
            branches.add(int(record.get("branch")))
        except (TypeError, ValueError):
            continue
    return sorted(branches)


def filter_manifest_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    norm_index: int,
    branches: list[int],
) -> list[dict[str, Any]]:
    branch_set = set(branches)
    stride = max(args.thumbnail_stride, 1)
    filtered: list[dict[str, Any]] = []
    for record in records:
        if branch_set and int(record.get("branch", -1)) not in branch_set:
            continue
        if manifest_norm_value(record, norm_index) is None:
            continue
        data_file = record.get("resolved_data_file")
        if not isinstance(data_file, Path) or not data_file.exists():
            continue
        filtered.append(record)

    filtered.sort(
        key=lambda item: (
            int(item.get("branch", 0)),
            int(item.get("point_index", 0)),
            float(item.get("lambda", 0.0)),
        )
    )

    if args.one_point_per_branch:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for record in filtered:
            grouped.setdefault(int(record.get("branch", -1)), []).append(record)
        return [branch_records[len(branch_records) // 2] for branch_records in grouped.values()]

    filtered = filtered[::stride]

    limit = max(args.max_thumbnails, 0)
    if limit == 0 or len(filtered) <= limit:
        return filtered

    if limit == 1:
        return [filtered[0]]

    selected: list[dict[str, Any]] = []
    for index in range(limit):
        source_index = round(index * (len(filtered) - 1) / (limit - 1))
        selected.append(filtered[source_index])
    return selected


def read_profile(path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                continue
            try:
                xs.append(float(fields[0]))
                ys.append(float(fields[1]))
            except ValueError:
                continue
    return xs, ys


def default_output(project_dir: Path, label: str, branches: list[int]) -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "norm"
    branch_suffix = ""
    if branches:
        branch_suffix = "_" + "_".join(str(branch) for branch in sorted(set(branches)))
    return project_dir / f"bd_solutions_{safe_label}{branch_suffix}.png"


def branch_suffix(branches: list[int]) -> str:
    if not branches:
        return ""
    return "_" + "_".join(str(branch) for branch in sorted(set(branches)))


def output_for_plot(
    project_dir: Path,
    label: str,
    branches: list[int],
    args: argparse.Namespace,
    all_mode: bool,
) -> Path:
    if not args.output:
        return default_output(project_dir, label, branches)

    output = Path(args.output).expanduser().resolve()
    if not all_mode:
        return output

    suffix = branch_suffix(branches)
    if "{branch}" in str(output):
        replacement = suffix[1:] if suffix.startswith("_") else suffix
        return Path(str(output).replace("{branch}", replacement))
    return output.with_name(f"{output.stem}{suffix}{output.suffix}")


def plot_bd_base(axis, curves: dict[str, list[list[float]]], norm_index: int, labels: list[str], args: argparse.Namespace):
    style = bd.line_style(args.style)
    for branch_index, (branch_name, rows) in enumerate(curves.items()):
        xs, ys = bd.curve_coordinates(rows, norm_index)
        if xs:
            branch_style = dict(style)
            if args.uniform_branch_color is not None:
                branch_style["color"] = args.uniform_branch_color
            axis.plot(xs, ys, label=f"branch {branch_name}", **branch_style)
            bd.plot_branch_endpoints(axis, xs, ys, args, branch_index == 0)

    axis.set_xlabel(args.x_label or "lambda")
    axis.set_ylabel(args.y_label or labels[norm_index])
    bd.apply_axis_font_overrides(axis, args)
    axis.grid(True, alpha=0.3)
    if not args.disable_legend:
        axis.legend(
            loc="best",
            fontsize=args.legend_font_size or "small",
        )


def inset_position(index: int, count: int, columns: int, width: float, height: float) -> list[float]:
    columns = max(columns, 1)
    rows = max(math.ceil(count / columns), 1)
    gap_x = 0.012
    gap_y = 0.018
    total_width = columns * width + (columns - 1) * gap_x
    total_height = rows * height + (rows - 1) * gap_y
    left = max(0.01, 0.5 - 0.5 * total_width)
    top = 0.955
    row = index // columns
    col = index % columns
    x0 = left + col * (width + gap_x)
    y0 = top - total_height + (rows - row - 1) * (height + gap_y)
    return [x0, y0, width, height]


def fitted_inset_position(index: int, count: int, columns: int) -> list[float]:
    columns = min(max(columns, 1), max(count, 1))
    rows = max(math.ceil(count / columns), 1)
    gap_x = 0.018
    gap_y = 0.035
    width = (1.0 - (columns - 1) * gap_x) / columns
    height = (1.0 - (rows - 1) * gap_y) / rows
    row = index // columns
    col = index % columns
    x0 = col * (width + gap_x)
    y0 = 1.0 - (row + 1) * height - row * gap_y
    return [x0, y0, width, height]


def add_solution_insets(
    figure,
    axis,
    inset_parent,
    records: list[dict[str, Any]],
    norm_index: int,
    args: argparse.Namespace,
):
    from matplotlib.patches import ConnectionPatch

    color_cycle = itertools.cycle(
        [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
    )
    profile_title_font_size = args.profile_title_font_size or 6.0 * args.font_scale
    point_label_font_size = args.point_label_font_size or 7.0 * args.font_scale

    for plot_index, record in enumerate(records, start=1):
        if args.fit_solutions:
            position = fitted_inset_position(
                plot_index - 1,
                len(records),
                args.thumbnail_columns,
            )
        else:
            position = inset_position(
                plot_index - 1,
                len(records),
                args.thumbnail_columns,
                args.thumbnail_width,
                args.thumbnail_height,
            )
        inset = inset_parent.inset_axes(position)
        xs, ys = read_profile(record["resolved_data_file"])
        color = next(color_cycle)
        inset.plot(xs, ys, color=color, linewidth=0.9)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(
            f"{plot_index}: b{record.get('branch')}  λ={float(record.get('lambda')):.4g}",
            fontsize=profile_title_font_size,
            pad=1.5,
        )
        for spine in inset.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor(color)

        marker_x = float(record.get("lambda"))
        marker_y = manifest_norm_value(record, norm_index)
        if marker_y is None:
            continue
        axis.scatter([marker_x], [marker_y], color=color, s=24, zorder=5)
        axis.annotate(
            str(plot_index),
            xy=(marker_x, marker_y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=point_label_font_size,
            color=color,
            zorder=6,
        )
        connection = ConnectionPatch(
            xyA=(0.5, 0.0),
            coordsA=inset.transAxes,
            xyB=(marker_x, marker_y),
            coordsB=axis.transData,
            color=color,
            linewidth=0.45,
            alpha=0.55,
        )
        figure.add_artist(connection)


def render_plot(
    plt,
    project_dir: Path,
    manifest: Path,
    curves: dict[str, list[list[float]]],
    labels: list[str],
    norm_index: int,
    records: list[dict[str, Any]],
    branches: list[int],
    args: argparse.Namespace,
    all_mode: bool,
) -> Path:
    if args.fit_solutions:
        columns = min(max(args.thumbnail_columns, 1), max(len(records), 1))
        solution_rows = max(math.ceil(len(records) / columns), 1)
        solution_height = max(1.6, 1.45 * solution_rows)
        diagram_height = 5.5
        figure = plt.figure(figsize=(11.5, diagram_height + solution_height))
        grid = figure.add_gridspec(
            2,
            1,
            height_ratios=(solution_height, diagram_height),
            hspace=0.08,
        )
        inset_parent = figure.add_subplot(grid[0])
        inset_parent.set_axis_off()
        axis = figure.add_subplot(grid[1])
    else:
        figure, axis = plt.subplots(figsize=(11.5, 7.0))
        inset_parent = axis

    plot_bd_base(axis, curves, norm_index, labels, args)
    add_solution_insets(figure, axis, inset_parent, records, norm_index, args)
    branch_title = f" branch {branches[0]}" if len(branches) == 1 else ""
    title_options = {}
    if args.title_font_size is not None:
        title_options["fontsize"] = args.title_font_size
    figure.suptitle(
        args.title or f"{project_dir.name}: {labels[norm_index]}{branch_title}",
        **title_options,
    )
    if args.fit_solutions:
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.94)
    else:
        figure.tight_layout()

    output_path = output_for_plot(project_dir, labels[norm_index], branches, args, all_mode)
    if not args.show:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=args.dpi)

    if args.show:
        plt.show()
    else:
        plt.close(figure)

    print(f"project: {project_dir}")
    print(f"manifest: {manifest}")
    print(f"norm: {labels[norm_index]}")
    if branches:
        print("branches: " + ", ".join(str(branch) for branch in branches))
    print(f"thumbnails: {len(records)}")
    if not args.show:
        print(f"wrote: {output_path}")
    return output_path


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = bd.load_config(config_path)
    project_dir = bd.project_directory(args, config_path, config)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")

    curves = bd.load_curves(project_dir)
    if not curves:
        raise FileNotFoundError(f"No debug_curve*.dat files found under: {project_dir}")

    count = bd.norm_count(curves)
    if count == 0:
        raise ValueError(f"No norm columns found under: {project_dir}")

    legend_path = Path(args.legend).expanduser().resolve() if args.legend else project_dir / "legend.dat"
    labels = bd.complete_labels(bd.read_labels(legend_path), count)
    norm_index = select_norm(args, labels)

    vis_dir = visualization_directory(project_dir, args)
    manifest = manifest_path(vis_dir, args)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    manifest_records = load_manifest(manifest, vis_dir)

    all_mode, requested_branches = parse_branch_tokens(args.branch)
    if all_mode:
        branch_groups = [[branch] for branch in available_branches(manifest_records)]
    else:
        branch_groups = [requested_branches]
    if not branch_groups:
        raise FileNotFoundError(f"No branches found in: {manifest}")

    plt = bd.import_pyplot(args.show)
    bd.apply_font_scale(plt, args.font_scale)
    wrote_any = False
    for branches in branch_groups:
        records = filter_manifest_records(manifest_records, args, norm_index, branches)
        if not records:
            if all_mode:
                continue
            raise FileNotFoundError(f"No plottable solution records found in: {manifest}")
        render_plot(plt, project_dir, manifest, curves, labels, norm_index, records, branches, args, all_mode)
        wrote_any = True

    if not wrote_any:
        raise FileNotFoundError(f"No plottable solution records found in: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
