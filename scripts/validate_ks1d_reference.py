#!/usr/bin/env python3
"""Replay the archived N=64 KS1D diagram, compare branches, and verify restart."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

from download_reference_data import sha256, validate_members


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numbers(path: Path) -> list[list[float]]:
    rows = [[float(v) for v in line.split()] for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    require(bool(rows) and all(math.isfinite(v) for row in rows for v in row),
            f"Empty or nonfinite data: {path}")
    return rows


@dataclass
class State:
    parameter: float
    segment: tuple[int, int]
    path: Path
    values: list[float]


@dataclass
class Curve:
    name: str
    points: list[list[float]]
    states: list[State]
    endpoints: list[str]
    point_segments: list[tuple[int, int]] | None = None

    @property
    def analytical(self) -> bool:
        return max(abs(point[1]) for point in self.points) < 1e-10


def read_curve(path: Path, state_size: int, interval: list[float]) -> Curve:
    metadata = [line.split() for line in (path / "metadata_curve.dat").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    points = numbers(path / "debug_curve_all.dat")
    require(len(metadata) == len(points), f"Mismatched curve rows: {path}")
    states, endpoints, saved_names, segments = [], [], set(), []
    for index, (row, point) in enumerate(zip(metadata, points)):
        require(len(row) == 8 and len(point) == 5 and int(row[0]) == index,
                f"Invalid metadata index/columns: {path}:{index}")
        parameter = float(row[1])
        require(math.isfinite(parameter) and math.isclose(parameter, point[0], rel_tol=1e-14),
                f"Parameter mismatch: {path}:{index}")
        require(interval[0] - 1e-8 <= parameter <= interval[1] + 1e-8,
                f"Parameter outside reference interval: {path}:{index}")
        require(row[2] in ("0", "1") and row[3].isdigit() and int(row[3]) == point[4],
                f"Invalid saved-state flag/name: {path}:{index}")
        require((row[2] == "1") == (int(row[3]) > 0), f"Inconsistent saved-state flag: {path}:{index}")
        segments.append((int(row[4]), int(row[5])))
        if row[2] == "1":
            require(row[3] != "0" and row[3] not in saved_names, f"Duplicate saved state: {path}:{index}")
            saved_names.add(row[3])
            vector = path / row[3]
            values = [v for values in numbers(vector) for v in values]
            require(len(values) == state_size, f"Wrong state dimension: {vector}")
            observables = (math.hypot(*values) / math.sqrt(state_size), values[0], values[1])
            require(all(math.isclose(value, recorded, rel_tol=1e-12, abs_tol=1e-12)
                        for value, recorded in zip(observables, point[1:4])),
                    f"Saved state disagrees with diagram: {vector}")
            states.append(State(parameter, (int(row[4]), int(row[5])), vector, values))
        if row[7] != "none":
            require(row[7] in ("boundary_min", "boundary_max", "self_intersection") and row[2] == "1",
                    f"Invalid or unsaved endpoint: {path}:{index}: {row[7]}")
            if row[7].startswith("boundary_"):
                boundary = interval[0 if row[7] == "boundary_min" else 1]
                require(abs(parameter - boundary) <= 1e-7, f"Endpoint misses boundary: {path}:{index}")
            endpoints.append(row[7])
    require(metadata[0][2] == "1", f"Missing discovery state: {path}")
    require(bool(states) and len(endpoints) == 2, f"Incomplete curve: {path}")
    return Curve(path.name, points, states, endpoints, segments)


def inventory(project: Path, manifest: dict) -> list[Curve]:
    curves = [read_curve(path.parent, manifest["state_size"], manifest["parameter_interval"])
              for path in sorted(project.glob("*/metadata_curve.dat")) if path.parent.name.isdigit()]
    require(len(curves) == manifest["curve_directories"], f"Expected nine curves, got {len(curves)} in {project}")
    require(sum(curve.analytical for curve in curves) == manifest["analytical_curves"],
            f"Wrong analytical/nonanalytical branch count: {project}")
    require(Counter(reason for curve in curves for reason in curve.endpoints) == manifest["endpoint_reasons"],
            f"Unexpected terminal outcomes: {project}")
    return curves


def summarize(curves: list[Curve]) -> dict:
    return {"curve_count": len(curves), "metadata_points": sum(len(c.points) for c in curves),
            "saved_vectors": sum(len(c.states) for c in curves),
            "endpoint_reasons": dict(Counter(reason for c in curves for reason in c.endpoints))}


def verify_reference(reference: Path, destination: Path) -> tuple[dict, dict, Path]:
    verified = set()
    for line in (reference / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        require(Path(name).name == name, f"Unsafe checksum path: {name}")
        require(sha256(reference / name) == digest, f"Reference checksum mismatch: {name}")
        verified.add(name)
    require({"config.json", "manifest.json", "project.tar.xz"} <= verified, "Missing reference checksums")
    manifest = json.loads((reference / "manifest.json").read_text())
    config = json.loads((reference / "config.json").read_text())
    require(manifest["spatial_resolution"] == 64 and manifest["curve_directories"] == 9,
            "This regression requires the archived nine-curve N=64 case")
    require(config["nonlinear_operator"]["discrete_problem_dimensions"] == [64], "Reference grid mismatch")
    with tarfile.open(reference / "project.tar.xz", "r:xz") as archive:
        members = validate_members(archive, PurePosixPath("KS1D_test_sym"))
        require(all(member.isfile() or member.isdir() for member in members), "Unsupported reference member")
        archive.extractall(destination, members=members, filter="data")
    return manifest, config, destination / "KS1D_test_sym"


def distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))) / max(1., math.hypot(*right))


def predictor(curve: Curve, target: State) -> tuple[float, list[float]]:
    candidates = [state.values for state in curve.states if abs(state.parameter - target.parameter) < .2]
    for left, right in zip(curve.states, curve.states[1:]):
        a, b, parameter = left.parameter, right.parameter, target.parameter
        if left.segment == right.segment and a != b and min(a, b) <= parameter <= max(a, b):
            weight = (parameter - a) / (b - a)
            candidates.append([u + weight * (v - u) for u, v in zip(left.values, right.values)])
    # A subperiodic odd state may be stored at an equivalent discrete phase.
    # Only use translations that retain the inversion-odd real-vector layout.
    threshold = 1e-12 * max(1., max(map(abs, target.values)))
    active = [k for k, value in enumerate(target.values, 1) if abs(value) > threshold]
    fundamental = math.gcd(*active)
    for values in list(candidates):
        phase = [k * math.pi / fundamental for k in range(1, len(values) + 1)]
        if math.hypot(*(v * math.sin(p) for v, p in zip(values, phase))) <= 1e-10 * max(1., math.hypot(*values)):
            candidates.append([v * math.cos(p) for v, p in zip(values, phase)])
    if not candidates:
        return math.inf, []
    values = min(candidates, key=lambda values: distance(values, target.values))
    return distance(values, target.values), values


def assign_branches(costs: list[list[float]]) -> tuple[int, ...]:
    size = len(costs)
    require(size > 0 and all(len(row) == size for row in costs), "Branch matching requires a square cost matrix")
    assignment = min(itertools.permutations(range(size)),
                     key=lambda order: sum(costs[i][j] for i, j in enumerate(order)))
    require(all(math.isfinite(costs[i][j]) and costs[i][j] < 1e-3 for i, j in enumerate(assignment)),
            "Missing or duplicated reference branch: no accurate one-to-one state matching")
    return assignment


def diagram_coverage(reference: Curve, fresh: Curve, scales: tuple[float, float]) -> float:
    """Maximum sampled reference-to-polyline distance in normalized (lambda, RMS)."""
    points = [(point[0] / scales[0], point[1] / scales[1]) for point in fresh.points]
    segments = []
    for i, ((x, y), (u, v)) in enumerate(zip(points, points[1:])):
        if fresh.point_segments is None or fresh.point_segments[i] == fresh.point_segments[i + 1]:
            dx, dy = u - x, v - y
            segments.append((x, y, dx, dy, dx * dx + dy * dy))
    require(bool(segments), f"No continuous segments in curve {fresh.name}")
    errors = []
    for index in sorted({i * (len(reference.points) - 1) // 200 for i in range(201)}):
        point = reference.points[index]
        x, y = point[0] / scales[0], point[1] / scales[1]
        best = math.inf
        for u, v, dx, dy, length2 in segments:
            weight = max(0., min(1., ((x - u) * dx + (y - v) * dy) / length2)) if length2 else 0.
            best = min(best, math.hypot(x - u - weight * dx, y - v - weight * dy))
        errors.append(best)
    return max(errors)


def branch_inputs(reference: list[Curve], fresh: list[Curve], work: Path) -> tuple[list[dict], list[dict]]:
    targets = [c for c in reference if not c.analytical]
    candidates = [c for c in fresh if not c.analytical]
    require(len(targets) == len(candidates), "Missing reference branches")
    predictions = [[predictor(c, target.states[0]) for c in candidates] for target in targets]
    assignment = assign_branches([[prediction[0] for prediction in row] for row in predictions])
    scales = (max(p[0] for c in reference for p in c.points) - min(p[0] for c in reference for p in c.points),
              max(1., max(p[1] for c in reference for p in c.points)))
    inputs, matches = [], []
    for i, j in enumerate(assignment):
        target = targets[i].states[0]
        error, values = predictions[i][j]
        coverage = diagram_coverage(targets[i], candidates[j], scales)
        require(coverage <= 1e-3, f"Incomplete reference-branch coverage: {targets[i].name}: {coverage}")
        path = work / f"predictor_{i}.dat"
        path.write_text("\n".join(f"{v:.17e}" for v in values) + "\n")
        inputs.append({"path": str(path), "parameter": target.parameter, "reference": str(target.path)})
        matches.append({"reference_curve": targets[i].name, "fresh_curve": candidates[j].name,
                        "parameter": target.parameter, "predictor_relative_difference": error,
                        "diagram_coverage_error": coverage})
    return inputs, matches


def snapshot(project: Path) -> dict[str, str]:
    return {str(path.relative_to(project)): sha256(path) for path in project.rglob("*") if path.is_file()}


def run(command: list[str], label: str, work: Path, timeout: float, report: dict) -> None:
    environment = dict(os.environ, OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.monotonic()
    record = report.setdefault("runs", {})[label] = {"command": command}
    try:
        with (work / f"{label}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=timeout)
        record["returncode"] = result.returncode
        result.check_returncode()
    finally:
        record["seconds"] = time.monotonic() - start
    if label in ("initial", "restart"):
        log = (work / f"{label}.log").read_text()
        require("Newton solver: NMFD" in log and "ERROR:" not in log and "exception init_tangent" not in log,
                f"Missing NMFD selection or numerical errors in {label}.log")


def validate(args, report: dict) -> None:
    work = args.work_dir.resolve()
    manifest, config, reference_project = verify_reference(args.reference_dir, work / "reference")
    reference = inventory(reference_project, manifest)
    report["reference"] = summarize(reference)
    require(all(report["reference"][key] == manifest[key] for key in ("metadata_points", "saved_vectors")),
            "Reference archive inventory disagrees with manifest")
    report["reference_checksums_verified"] = True
    (work / "project").mkdir()
    config["path_to_project"] = str(work / "project") + "/"
    config_path = work / "config.json"
    write_json(config_path, config)
    command = [str(args.bd_executable.resolve()), str(config_path), "--quiet"]
    run(command, "initial", work, args.timeout, report)
    fresh = inventory(work / "project", manifest)
    report["fresh"] = summarize(fresh)
    inputs, matches = branch_inputs(reference, fresh, work)
    report["branch_matches"] = matches
    entries = [{"path": str(s.path), "parameter": s.parameter} for c in fresh for s in c.states]
    write_json(work / "states.json", entries + inputs)
    run([str(args.checker_executable.resolve()), str(config_path), str(work / "states.json"),
         str(work / "state_checks.json")], "residuals", work, args.timeout, report)
    checks = json.loads((work / "state_checks.json").read_text())
    require(len(checks) == len(entries) + len(inputs), "Incomplete physical/branch report")
    for entry, check in zip(entries + inputs, checks):
        require(check["path"] == entry["path"], "Physical report ordering mismatch")
        require(isinstance(check["physical_residual"], (float, int)) and math.isfinite(check["physical_residual"]),
                "Nonfinite physical residual")
    physical_max = max(check["physical_residual"] for check in checks[:len(entries)])
    report["fresh"]["physical_residual_max"] = physical_max
    require(physical_max <= config["deflation_continuation"]["newton_deflation"]["tolerance"],
            f"Saved physical residual exceeds tolerance: {physical_max}")
    for match, check in zip(matches, checks[len(entries):]):
        match.update(check)
        require(check["converged"] is True and isinstance(check["relative_difference"], (float, int))
                and math.isfinite(check["relative_difference"]) and check["relative_difference"] <= 1e-6
                and isinstance(check["corrected_residual"], (float, int))
                and math.isfinite(check["corrected_residual"])
                and check["corrected_residual"] <= config["nonlinear_operator"]["newton"]["tolerance"],
                f"Reference branch correction failed: {match}")
    before = snapshot(work / "project")
    write_json(work / "before_restart_hashes.json", before)
    run(command, "restart", work, args.timeout, report)
    after = snapshot(work / "project")
    report["restart_changed_files"] = sorted(name for name in before if after.get(name) != before[name])
    report["restart_added_files"] = sorted(set(after) - set(before))
    report["restart"] = summarize(inventory(work / "project", manifest))
    require(not report["restart_changed_files"] and not report["restart_added_files"],
            "Restart changed committed data or added overlapping curves")
    require("bifurcation_diagram.dat" in before, "Missing serialized diagram")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bd-executable", required=True, type=Path)
    parser.add_argument("--checker-executable", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path, default=Path("data/reference/KS1D/N64/symmetric"))
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    require(args.timeout > 0 and math.isfinite(args.timeout), "Timeout must be finite and positive")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "threads": 1}
    try:
        validate(args, report)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
    finally:
        write_json(args.work_dir / "comparison.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
