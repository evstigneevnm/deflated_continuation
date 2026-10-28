"""Keep supported assembly independent of archived project Newton implementations."""

import argparse
from pathlib import Path

from check_linear_solver_dependencies import ACTIVE_DIRECTORIES, ACTIVE_FILES, ACTIVE_MODELS, INCLUDE


def violations(path, source):
    if path == "source/main/plot_diagram_to_pos.hpp" or not (
        path.startswith(ACTIVE_DIRECTORIES) or path in ACTIVE_FILES or ACTIVE_MODELS.match(path)
    ):
        return []
    return [f"{path}:{source.count(chr(10), 0, match.start()) + 1}: archived Newton dependency: {match.group(1)}"
            for match in INCLUDE.finditer(source) if "newton_solvers/" in match.group(1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    errors = []
    if any(p.is_file() for p in (root / "source/numerical_algos/newton_solvers").rglob("*")):
        errors.append("Retired project Newton tree still exists")
    for path in sorted((root / "source").rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(("source/contrib/", "source/numerical_algos/legacy/")):
            continue
        if path.is_file() and path.suffix in {".h", ".hpp", ".cpp", ".cu", ".cuh"}:
            errors.extend(violations(relative, path.read_text()))
    print("\n".join(errors) if errors else "Newton-solver dependency boundaries: PASS")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
