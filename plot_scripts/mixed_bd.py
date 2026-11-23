"""Render a mixed diagram from stationary scalar exports and v1 periodic archives."""
from __future__ import annotations

import argparse
from itertools import groupby
import math
from pathlib import Path

import plot_bd as steady


IDENTITY = ("model", "discretization", "parameter", "observable", "normalization")
INCOMPLETE_ENDPOINTS = {"max_steps", "no_progress", "unresolved_branch_intersection",
                        "hard_failure", "knot_interpolation_failure"}


def nonempty(data, key):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {key}")
    return value


def finite(row, key):
    value = row.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {key} at source index {row.get('source_index')}")
    return value


def periodic_archive(path, identity):
    data = steady.load_config(path)
    try:
        if (data["format"], data["version"], data["kind"], data["closure"]) != (
            "periodic_branch", 1, "periodic", "identity"
        ):
            raise ValueError("Expected an ordinary identity-closure v1 periodic branch")
        for key in IDENTITY:
            if nonempty(data, key) != identity[key]:
                raise ValueError(f"Incompatible {key}: {data[key]!r} vs stationary {identity[key]!r}")
        nonempty(data, "id")
        nonempty(data, "method")
        if type(data["sections"]) is not int or data["sections"] < 1:
            raise ValueError("Invalid section count")
        if type(data["trace_complete"]) is not bool or type(data["all_points_verified"]) is not bool:
            raise ValueError("Missing trace or verification status")
        if "restarted_from" in data:
            ref = data["restarted_from"]
            nonempty(ref, "branch")
            if (type(ref["segment"]) is not int or ref["segment"] < 1 or
                    type(ref["source_index"]) is not int or ref["source_index"] < 0):
                raise ValueError("Invalid restart source")
        index, verified = 0, 0
        if not isinstance(data["segments"], list):
            raise ValueError("Expected an array of segments")
        for segment, arm in enumerate(data["segments"], 1):
            if type(arm["id"]) is not int or arm["id"] != segment or not isinstance(arm["points"], list):
                raise ValueError("Invalid segment ID or points")
            nonempty(arm, "failure")
            nonempty(arm, "endpoint")
            for p in arm["points"]:
                if type(p["source_index"]) is not int or p["source_index"] != index or type(p["verified"]) is not bool:
                    raise ValueError("Invalid point index or verification status")
                finite(p, "parameter")
                if p["verified"]:
                    if (finite(p, "period") <= 0 or finite(p, "observable_rms") < 0 or
                            not 0 <= finite(p, "observable_min") <= finite(p, "observable_max") or
                            finite(p, "closure_error") < 0):
                        raise ValueError(f"Invalid physical measurement at point {index}")
                    verified += 1
                index += 1
        if data["all_points_verified"] != (index > 0 and verified == index):
            raise ValueError("Inconsistent verification status")
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{path}: {error}") from error
    return data


def draw(manifest_path, plt, *, envelope=False):
    manifest = steady.load_config(manifest_path)
    if (not isinstance(manifest, dict) or manifest.get("format") != "mixed_branch_plot"
            or type(manifest.get("version")) is not int or manifest["version"] != 1):
        raise ValueError("Expected a v1 mixed_branch_plot manifest")
    source = manifest["stationary"]
    if not isinstance(source, dict):
        raise ValueError("Expected a stationary identity and project")
    identity = {key: nonempty(source, key) for key in IDENTITY}
    project = manifest_path.parent / nonempty(source, "project")
    labels = steady.read_labels(project / "legend.dat")
    column = nonempty(source, "column")
    if labels.count(column) != 1:
        raise ValueError(f"{project}: missing or ambiguous stationary column {column!r}")
    column_index = labels.index(column)
    curves = steady.load_curves(project, strict=True)
    if not curves:
        raise ValueError(f"{project}: no stationary curves")
    paths = manifest["periodic"]
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p.strip() for p in paths):
        raise ValueError("List at least one periodic archive path")
    traces = [periodic_archive(manifest_path.parent / path, identity) for path in paths]
    if len({t["id"] for t in traces}) != len(traces):
        raise ValueError("Duplicate periodic trace ID")

    stationary = []
    for branch, rows in curves.items():
        # Without segment metadata the points are useful, but their connections are unknown.
        joined = (project / branch / "metadata_curve.dat").is_file()
        groups = groupby(zip(rows.segment_ids, rows), key=lambda item: item[0])
        for i, (seg, group) in enumerate(groups):
            values = [row for _, row in group]
            x = [row[0] for row in values]
            y = [steady.row_norm_value(row, column_index) for row in values]
            if any(v is None or not math.isfinite(v) or v < 0 for v in y) or not all(map(math.isfinite, x)):
                raise ValueError(f"Invalid stationary {column} data in branch {branch}")
            stationary.append((branch, seg, i, joined, x, y))

    fig, (height, period) = plt.subplots(2, 1, sharex=True, figsize=(8.4, 6.3),
                                        gridspec_kw={"height_ratios": (1.4, 1)}, constrained_layout=True)
    for branch, seg, i, joined, x, y in stationary:
        height.plot(x, y, "k.--" if joined else "k.", label=f"Stationary {branch}" if i == 0 else "_",
                    gid=f"stationary:{branch}:{seg}")

    for n, trace in enumerate(traces):
        trace_id = trace["id"]
        label = f"{trace_id} (M={trace['sections']})"
        ref = trace.get("restarted_from")
        if ref:
            label += f", restart of {ref['branch']} [{ref['segment']}:{ref['source_index']}]"
        if not trace["trace_complete"]:
            label += ", incomplete trace"
        if not trace["all_points_verified"]:
            label += ", verification incomplete"
        color = f"C{n % 10}"
        if not trace["segments"]:
            height.plot([], [], color=color, label=f"{label}, no arms")
        for i, arm in enumerate(trace["segments"]):
            points = arm["points"]
            x = [p["parameter"] for p in points]

            def measured(key):
                return [p[key] if p["verified"] else math.nan for p in points]

            legend = label if i == 0 else "_"
            if arm["failure"] != "none" or arm["endpoint"] in INCOMPLETE_ENDPOINTS:
                legend = f"{label}, arm {arm['id']}: {arm['endpoint']}"
                if arm["failure"] != "none":
                    legend += f" ({arm['failure']})"
            style = dict(color=color, marker=("o", "x", "+")[n % 3], fillstyle="none",
                         linestyle=":" if ref else "-", linewidth=1, markersize=5, label=legend)
            gid = f"{trace_id}:{arm['id']}"
            height.plot(x, measured("observable_rms"), gid=f"rms:{gid}", **style)
            period.plot(x, measured("period"), gid=f"period:{gid}", **style)
            if envelope:
                height.fill_between(x, measured("observable_min"), measured("observable_max"),
                                    color=color, alpha=.13,
                                    label="Sampled orbit min/max" if n == i == 0 else "_")
    height.set(ylabel=f"{identity['observable']}\n(stationary / orbit time-RMS)",
               title=f"{identity['model']}   |   {identity['discretization']}")
    height.legend(fontsize="small")
    period.set(xlabel=identity["parameter"], ylabel="Orbit period")
    for axis in (height, period):
        axis.grid(alpha=.22)
        axis.ticklabel_format(useOffset=False)
    return fig


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, help="Default: mixed_bd.png next to the manifest")
    parser.add_argument("--envelope", action="store_true", help="Shade sampled orbit minima and maxima")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    plt = steady.import_pyplot(False)
    try:
        fig = draw(args.manifest, plt, envelope=args.envelope)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Cannot render {args.manifest}: {error}\n")
    output = args.output or args.manifest.with_name("mixed_bd.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote: {output}")
    return 0
