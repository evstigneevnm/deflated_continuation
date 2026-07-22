#!/usr/bin/env python3
"""Write a KS1D bifurcation-diagram config for local/GitHub CI."""

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
    continuation["deflation_knots"] = args.knots
    continuation["add_analytical_solution_to_diagram"] = True
    continuation.setdefault("analytical_solution_branches", [])
    restart_policy = continuation.setdefault("restart_policy", {})
    restart_policy.setdefault("allow_incomplete_restart_intersections", False)
    restart_policy.setdefault("allow_knot_interpolation_failure", False)
    restart_policy.setdefault("allow_failed_continuation_curve_save", False)
    restart_policy.setdefault("check_duplicate_after_deflation", True)
    restart_policy.setdefault("duplicate_after_deflation_retries", 2)
    restart_policy.setdefault("duplicate_after_deflation_tolerance", 1.0e-8)
    restart_policy.setdefault("max_failed_continuations_per_knot", 3)
    restart_policy.setdefault("failed_continuation_rejection_tolerance", 1.0e-8)
    knot_relocation = restart_policy.setdefault("knot_relocation", {})
    knot_relocation.setdefault("enabled", False)
    knot_relocation.setdefault("registry_file", "knot_registry.json")
    knot_relocation.setdefault("min_shift_abs", 1.0e-5)
    knot_relocation.setdefault("max_shift_abs", 0.05)
    knot_relocation.setdefault("candidate_count", 12)
    knot_relocation.setdefault("prefer_positive_shift", True)
    knot_relocation.setdefault("require_all_intersections", True)
    knot_relocation.setdefault("save_registry", True)

    continuation["linear_solver_extended"]["save_convergence_history"] = False
    continuation["linear_solver_extended"]["verbose"] = False
    continuation["newton_continuation"]["save_norms_history"] = False
    continuation["newton_continuation"]["verbose"] = False
    continuation["newton_deflation"]["save_norms_history"] = False
    continuation["newton_deflation"]["verbose"] = False

    config["nonlinear_operator"]["discrete_problem_dimensions"] = [args.grid_size]
    config["nonlinear_operator"]["problem_real_parameters_vector"] = [args.a_val, args.b_val]
    config["nonlinear_operator"]["linear_solver"]["save_convergence_history"] = False
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
        default="json_project_files/KS1D_test_sym.json",
        help="Base KS1D JSON file.",
    )
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--a-val", type=float, default=2.0)
    parser.add_argument("--b-val", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--max-step-size", type=float, default=0.03)
    parser.add_argument("--deflation-attempts", type=int, default=0)
    parser.add_argument("--continuation-fail-attempts", type=int, default=8)
    parser.add_argument("--initial-direction", type=int, default=-1)
    parser.add_argument("--knots", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5])
    parser.add_argument("--high-precision-reduction", action="store_true")
    parser.add_argument(
        "--clean-project",
        action="store_true",
        help="Remove the project directory before writing the config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    write_config(parse_args())
