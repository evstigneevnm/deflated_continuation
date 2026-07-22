#!/usr/bin/env python3
"""Plot bifurcation diagram debug curves from a project JSON file."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path


ANACONDA_PYTHON = Path("/home/noctum/anaconda3/bin/python")


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
        help="Draw every BD branch in one color. If COLOR is omitted, gray is used.",
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


def load_curves(project_dir: Path) -> dict[str, list[list[float]]]:
    curves: dict[str, list[list[float]]] = {}
    for branch_dir in sorted((path for path in project_dir.iterdir() if path.is_dir()), key=branch_sort_key):
        curve_file = branch_dir / "debug_curve_all.dat"
        if not curve_file.exists():
            curve_file = branch_dir / "debug_curve.dat"
        if not curve_file.exists():
            continue
        rows = read_curve_file(curve_file)
        if rows:
            curves[branch_dir.name] = rows
    return curves


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
    for row in rows:
        value = row_norm_value(row, norm_index)
        if value is None:
            continue
        xs.append(row[0])
        ys.append(value)
    return xs, ys


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


def plot_curves(args: argparse.Namespace, plt, project_dir: Path, curves: dict[str, list[list[float]]], labels: list[str], selected_norms: list[int]) -> Path | None:
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
    style = line_style(args.style)

    for axis, norm_index in zip(axes_flat, selected_norms):
        for branch_index, (branch_name, rows) in enumerate(curves.items()):
            xs, ys = curve_coordinates(rows, norm_index)
            if xs:
                branch_style = dict(style)
                if args.uniform_branch_color is not None:
                    branch_style["color"] = args.uniform_branch_color
                axis.plot(xs, ys, label=f"branch {branch_name}", **branch_style)
                plot_branch_endpoints(axis, xs, ys, args, branch_index == 0)
        axis.set_ylabel(labels[norm_index])
        apply_axis_font_overrides(axis, args)
        axis.grid(True, alpha=0.3)
        if not args.disable_legend:
            axis.legend(
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
    figure.tight_layout()

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
    output_path = plot_curves(args, plt, project_dir, curves, labels, selected_norms)

    print(f"project: {project_dir}")
    print(f"curves: {len(curves)}")
    print("norms: " + ", ".join(labels))
    if output_path is not None:
        print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
