#!/usr/bin/env python3
"""Write a compact symmetric KS2D config from the current parameter schema."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _safe_to_clean(path: Path) -> bool:
    resolved = path.resolve()
    for root in (Path.cwd().resolve() / "build", Path("/tmp").resolve()):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            pass
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_config")
    parser.add_argument("project_dir")
    parser.add_argument("--base-config", default="json_project_files/KS2D_test_sym.json")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--max-step-size", type=float, default=0.03)
    parser.add_argument("--skip-file-output", type=int, default=1)
    parser.add_argument("--seed-parameter", type=float, default=2.0)
    parser.add_argument("--parameter-min", type=float, default=None)
    parser.add_argument("--parameter-max", type=float, default=None)
    parser.add_argument("--deflation-attempts", type=int, default=0)
    parser.add_argument("--newton-deflation-iterations", type=int, default=None)
    parser.add_argument("--newton-deflation-weight", type=float, default=None)
    parser.add_argument("--newton-deflation-tolerance", type=float, default=None)
    parser.add_argument("--linear-solver-iterations", type=int, default=None)
    parser.add_argument("--linear-solver-tolerance", type=float, default=None)
    parser.add_argument("--linear-solver-verbose", action="store_true")
    parser.add_argument("--add-analytical-solution", action="store_true")
    parser.add_argument("--enable-branch-intersections", action="store_true")
    parser.add_argument("--enable-seed-schedule", action="store_true")
    parser.add_argument("--max-failed-continuations-per-knot", type=int, default=None)
    parser.add_argument("--clean-project", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_config)
    project = Path(args.project_dir)
    with Path(args.base_config).open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    if args.clean_project and project.exists():
        if not _safe_to_clean(project):
            raise RuntimeError(f"refusing to clean project directory outside build/ or /tmp: {project}")
        shutil.rmtree(project)
    project.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    config["path_to_project"] = str(project)
    config["use_high_precision_reduction"] = False
    config["nonlinear_operator"]["discrete_problem_dimensions"] = [args.nx, args.ny]
    config["nonlinear_operator"]["problem_real_parameters_vector"] = [2.0, 4.0]

    continuation = config["deflation_continuation"]
    continuation["maximum_continuation_steps"] = args.steps
    continuation["step_size"] = args.step_size
    continuation["max_step_size"] = args.max_step_size
    continuation["deflation_attempts"] = args.deflation_attempts
    continuation["continuation_fail_attempts"] = 4
    continuation["skip_file_output"] = args.skip_file_output
    parameter_min = args.parameter_min
    parameter_max = args.parameter_max
    if parameter_min is None:
        parameter_min = args.seed_parameter - 0.5
    if parameter_max is None:
        parameter_max = args.seed_parameter + 0.5
    if not parameter_min < args.seed_parameter < parameter_max:
        raise ValueError("expected parameter-min < seed-parameter < parameter-max")
    continuation["deflation_knots"] = [args.seed_parameter]
    continuation["continuation_parameter_bounds"] = {
        "enabled": True,
        "minimum": parameter_min,
        "maximum": parameter_max,
        "resolve_with_knot_registry": True,
    }
    continuation["add_analytical_solution_to_diagram"] = args.add_analytical_solution
    continuation["branch_intersection_policy"]["enabled"] = \
        args.enable_branch_intersections
    continuation["self_intersection_policy"]["enabled"] = False
    continuation["restart_policy"]["seed_schedule"] = {
        "enabled": args.enable_seed_schedule,
        "registry_file": "deflation_seed_registry.json",
        "save_registry": True,
    }
    if args.max_failed_continuations_per_knot is not None:
        continuation["restart_policy"]["max_failed_continuations_per_knot"] = \
            args.max_failed_continuations_per_knot
    if args.newton_deflation_iterations is not None:
        continuation["newton_deflation"]["maximum_iterations"] = \
            args.newton_deflation_iterations
    if args.newton_deflation_weight is not None:
        continuation["newton_deflation"]["update_wight_maximum"] = \
            args.newton_deflation_weight
    if args.newton_deflation_tolerance is not None:
        continuation["newton_deflation"]["tolerance"] = \
            args.newton_deflation_tolerance
    linear_solver = continuation["linear_solver_extended"]
    if args.linear_solver_iterations is not None:
        linear_solver["maximum_iterations"] = args.linear_solver_iterations
    if args.linear_solver_tolerance is not None:
        linear_solver["tolerance"] = args.linear_solver_tolerance
    if args.linear_solver_verbose:
        linear_solver["verbose"] = True

    matrix_free = config["stability_continuation"]["matrix_free_eigensolver"]
    matrix_free["small_system"] = {
        "enabled": True,
        "maximum_dimension": 512,
        "prefer": True,
        "absolute_residual_tolerance": 1.0e-9,
        "relative_residual_tolerance": 1.0e-8,
    }

    with output.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=4)
        stream.write("\n")


if __name__ == "__main__":
    main()
