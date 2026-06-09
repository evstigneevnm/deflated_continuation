#!/usr/bin/env python3
"""Write a star-shaped bifurcation-diagram config for local CI."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _is_safe_to_clean(path: Path, cwd: Path) -> bool:
    resolved = path.resolve()
    cwd_resolved = cwd.resolve()
    tmp_resolved = Path("/tmp").resolve()
    try:
        resolved.relative_to(cwd_resolved / "build")
        return True
    except ValueError:
        pass
    try:
        resolved.relative_to(tmp_resolved)
        return True
    except ValueError:
        return False


def default_knots(curvature: float) -> list[float]:
    endpoint = max(1.25, 1.0 + abs(curvature) + 0.05)
    return [
        -endpoint,
        -1.05,
        -0.9,
        -0.6,
        -0.25,
        0.0,
        0.25,
        0.6,
        0.9,
        1.05,
        endpoint,
    ]


def write_config(args: argparse.Namespace) -> None:
    base_config = Path(args.base_config)
    output_config = Path(args.output_config)
    project_dir = Path(args.project_dir)

    with base_config.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    if args.clean_project and project_dir.exists():
        if not _is_safe_to_clean(project_dir, Path.cwd()):
            raise RuntimeError(
                f"refusing to clean project directory outside build/ or /tmp: {project_dir}"
            )
        shutil.rmtree(project_dir)

    project_dir.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)

    config["path_to_project"] = str(project_dir)
    config["use_high_precision_reduction"] = args.high_precision_reduction
    config["bifurcaiton_diagram_file_name"] = "bifurcation_diagram.dat"

    continuation = config["deflation_continuation"]
    continuation["maximum_continuation_steps"] = args.steps
    continuation["step_size"] = args.step_size
    continuation["max_step_size"] = args.max_step_size
    continuation["deflation_attempts"] = args.deflation_attempts
    continuation["continuation_fail_attempts"] = args.continuation_fail_attempts
    continuation["initial_direciton"] = args.initial_direction
    continuation["skip_file_output"] = 1
    continuation["deflation_knots"] = args.knots or default_knots(args.curvature)

    continuation["linear_solver_extended"]["save_convergence_history"] = False
    continuation["linear_solver_extended"]["verbose"] = False
    continuation["newton_continuation"]["save_norms_history"] = False
    continuation["newton_continuation"]["verbose"] = False
    continuation["newton_deflation"]["save_norms_history"] = False
    continuation["newton_deflation"]["verbose"] = False

    config["nonlinear_operator"]["discrete_problem_dimensions"] = [1]
    config["nonlinear_operator"]["problem_real_parameters_vector"] = [args.curvature]
    config["nonlinear_operator"]["linear_solver"]["verbose"] = False
    config["nonlinear_operator"]["newton"]["save_norms_history"] = False
    config["nonlinear_operator"]["newton"]["verbose"] = False
    config["stability_continuation"]["linear_solver"]["save_convergence_history"] = False
    config["stability_continuation"]["linear_solver"]["verbose"] = False
    config["stability_continuation"]["newton"]["save_norms_history"] = False
    config["stability_continuation"]["newton"]["verbose"] = False

    with output_config.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=4)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_config", help="Config file to write.")
    parser.add_argument("project_dir", help="Root project directory used by the solver.")
    parser.add_argument(
        "--base-config",
        default="json_project_files/star_shaped_test.json",
        help="Base star-shaped JSON file.",
    )
    parser.add_argument("--curvature", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--max-step-size", type=float, default=0.02)
    parser.add_argument("--deflation-attempts", type=int, default=0)
    parser.add_argument("--continuation-fail-attempts", type=int, default=12)
    parser.add_argument("--initial-direction", type=int, default=-1)
    parser.add_argument("--knots", type=float, nargs="+")
    parser.add_argument("--high-precision-reduction", action="store_true")
    parser.add_argument(
        "--clean-project",
        action="store_true",
        help="Remove the project directory before writing the config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    write_config(parse_args())
