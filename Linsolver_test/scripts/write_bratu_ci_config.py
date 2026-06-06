#!/usr/bin/env python3
"""Write a Bratu bifurcation-diagram config for local CI."""

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

    config["path_to_prject"] = str(project_dir)
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
    continuation["deflation_knots"] = args.knots

    continuation["linear_solver_extended"]["save_convergence_history"] = False
    continuation["newton_continuation"]["save_norms_history"] = False
    continuation["newton_continuation"]["verbose"] = False
    continuation["newton_deflation"]["save_norms_history"] = False
    continuation["newton_deflation"]["verbose"] = False

    discretization_ids = {
        "chebyshev": 0,
        "fd3": 1,
    }
    config["nonlinear_operator"]["discrete_problem_dimensions"] = [args.interior_size]
    config["nonlinear_operator"]["spatial_discretization"] = args.spatial_discretization
    config["nonlinear_operator"]["problem_int_parameters_vector"] = [
        discretization_ids[args.spatial_discretization]
    ]
    config["nonlinear_operator"]["newton"]["save_norms_history"] = False
    config["nonlinear_operator"]["newton"]["verbose"] = False
    config["stability_continuation"]["linear_solver"]["save_convergence_history"] = False
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
        default="json_project_files/bratu_test.json",
        help="Base Bratu JSON file.",
    )
    parser.add_argument("--interior-size", type=int, default=31)
    parser.add_argument(
        "--spatial-discretization",
        choices=("chebyshev", "fd3"),
        default="chebyshev",
        help="Spatial operator for the Bratu problem.",
    )
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--step-size", type=float, default=0.025)
    parser.add_argument("--max-step-size", type=float, default=0.05)
    parser.add_argument("--deflation-attempts", type=int, default=0)
    parser.add_argument("--continuation-fail-attempts", type=int, default=8)
    parser.add_argument("--initial-direction", type=int, default=1)
    parser.add_argument(
        "--knots",
        type=float,
        nargs="+",
        default=[0.1, 3.6],
        help="Continuation interval/deflation knots.",
    )
    parser.add_argument("--high-precision-reduction", action="store_true")
    parser.add_argument(
        "--clean-project",
        action="store_true",
        help="Remove the project directory before writing the config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    write_config(parse_args())
