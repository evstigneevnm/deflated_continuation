"""Keep refactored workflows independent of archived project linear solvers."""

import argparse
from pathlib import Path
import re


ACTIVE_DIRECTORIES = (
    "source/continuation/", "source/deflation/", "source/discretization/",
    "source/main/", "source/stability/", "source/symmetry/",
    "source/time_stepper/integration/", "source/time_stepper/runge_kutta/",
    "source/time_stepper/tests/", "source/numerical_algos/tests/",
    "source/nonlinear_operators/bratu/", "source/nonlinear_operators/circle/",
    "source/nonlinear_operators/star_shaped/",
    "source/nonlinear_operators/Kuramoto_Sivashinskiy_1D/",
    "source/nonlinear_operators/Kuramoto_Sivashinskiy_2D/",
    "source/models/bratu/", "source/models/star_shaped/",
)
ACTIVE_FILES = {"source/nonlinear_operators/projected_system_operator.h"}
ACTIVE_MODELS = re.compile(
    r"^source/models/(?:circle/circle_(?:bd|stability|prepare_visualization|linear_solvers|backend_typedefs)"
    r"|KS_[12]D/(?:KS[12]D_|test_KS[12]D_))"
)
PARITY_TESTS = {
    "source/numerical_algos/tests/test_nmfd_short_recurrence.cpp",
    "source/numerical_algos/tests/test_legacy_linear_solve_recovery.cpp",
}
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def violations(path, source):
    if path in PARITY_TESTS or not (
        path.startswith(ACTIVE_DIRECTORIES) or path in ACTIVE_FILES or ACTIVE_MODELS.match(path)
    ):
        return []
    errors = []
    for match in INCLUDE.finditer(source):
        header = match.group(1)
        if "lin_solvers/" in header or "numerical_algos/legacy/" in header:
            line = source.count("\n", 0, match.start()) + 1
            errors.append(f"{path}:{line}: archived solver dependency: {header}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    errors = []
    if any(path.is_file() for path in (root / "source/numerical_algos/lin_solvers").rglob("*")):
        errors.append("source/numerical_algos/lin_solvers: retired solver tree exists")
    if not (root / "source/numerical_algos/linear_solve_recovery.h").is_file():
        errors.append("Active linear_solve_recovery.h is missing")
    for path in sorted((root / "source").rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(("source/contrib/", "source/numerical_algos/legacy/")):
            continue
        if path.is_file() and path.suffix in {".h", ".hpp", ".cuh", ".cpp", ".cu"}:
            errors.extend(violations(relative, path.read_text()))
    if errors:
        print("\n".join(errors))
        return 1
    print("Linear-solver dependency boundaries: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
