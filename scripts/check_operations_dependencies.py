"""Reject application dependencies on retired operations compatibility APIs."""

import argparse
from pathlib import Path, PurePosixPath
import re


RETIRED_DIRECTORIES = (
    "source/common/NMFD-operations",
    "source/common/scfd_backend_ext",
)
FROZEN_DIRECTORIES = (
    "source/contrib",
    "source/time_stepper/legacy",
)
GLUED_COMPATIBILITY = {
    "source/common/glued_vector_operations.h",
    "source/common/glued_vector_space.h",
}
RETIRED_HEADERS = {
    "scfd_vector_operations.h",
    "scfd_serial_cpu_vector_operations.h",
    "glued_vector_operations.h",
    "glued_vector_space.h",
}
MOVED_COMMON_HEADERS = {
    "color_std_out.h",
    "complex_real_type_cast.hpp",
    "cpu_file_operations.h",
    "cpu_matrix_file_operations.h",
    "cpu_matrix_vector_operations_var_prec.h",
    "dot_product.h",
    "gpu_matrix_file_operations.h",
    "gpu_matrix_vector_operations.h",
    "gpu_matrix_vector_operations_impl.cuh",
    "gpu_matrix_vector_operations_kernels.cu",
    "gpu_reduction.h",
    "gpu_reduction_impl.cuh",
    "gpu_reduction_kernels.cu",
    "gpu_vector_operations.h",
    "gpu_vector_operations_impl.cuh",
    "gpu_vector_operations_kernels.cu",
    "threaded_reduction.h",
    "vector_operations_base.h",
    "vector_pool.h",
    "vector_wrap.h",
}
REFACTORED_DIRECTORIES = (
    "source/continuation/",
    "source/deflation/",
    "source/discretization/",
    "source/main/",
    "source/stability/analysis/",
    "source/stability/eigensolvers/",
    "source/stability/persistence/",
    "source/stability/tracking/",
    "source/symmetry/",
    "source/time_stepper/integration/",
    "source/time_stepper/runge_kutta/",
)
REFACTORED_MODEL = re.compile(
    r"^source/models/(?:(?:bratu|circle|star_shaped)/|KS_[12]D/(?:KS[12]D_|test_KS[12]D_))"
)
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
OLD_TYPES = re.compile(
    r"(?<![\w:])(?:::)?(?:scfd_vector_operations|scfd_serial_cpu_vector_operations)\s*<"
    r"|(?:common|nmfd::operations::detail)::scfd_backend_ext\b"
    r"|scfd::linspace::glued_"
)


def violations(path, source):
    if any(path == prefix or path.startswith(prefix + "/") for prefix in FROZEN_DIRECTORIES):
        return []
    if path in GLUED_COMPATIBILITY:
        return []
    errors = []
    for match in INCLUDE.finditer(source):
        header = match.group(1)
        if (
            "common/NMFD-operations/" in header
            or "common/scfd_backend_ext/" in header
            or "nmfd/operations/detail/scfd_backend_ext/" in header
            or (header.startswith("common/") and header[len("common/"):] in MOVED_COMMON_HEADERS)
            or (
                header.startswith("common/legacy/")
                and (path.startswith(REFACTORED_DIRECTORIES) or REFACTORED_MODEL.match(path))
            )
            or (
                PurePosixPath(header).name in RETIRED_HEADERS
                and not header.startswith("nmfd/operations/")
            )
        ):
            errors.append(f"{path}:{source.count(chr(10), 0, match.start()) + 1}: {header}")
    for match in OLD_TYPES.finditer(source):
        errors.append(f"{path}:{source.count(chr(10), 0, match.start()) + 1}: {match.group()}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    errors = []
    for directory in RETIRED_DIRECTORIES:
        if any(path.is_file() for path in (root / directory).rglob("*")):
            errors.append(f"{directory}: retired compatibility tree exists")
    for name in sorted(MOVED_COMMON_HEADERS):
        if (root / "source/common" / name).exists():
            errors.append(f"source/common/{name}: belongs in source/common/legacy")
    for path in sorted((root / "source").rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(relative.startswith(prefix + "/") for prefix in FROZEN_DIRECTORIES):
            continue
        if path.suffix in {".h", ".hpp", ".cuh", ".cpp", ".cu"} and path.is_file():
            errors.extend(violations(relative, path.read_text()))
    makefile = (root / "Makefile").read_text()
    for directory in RETIRED_DIRECTORIES:
        if directory in makefile:
            errors.append(f"Makefile: dependency on {directory}")
    for header in sorted(MOVED_COMMON_HEADERS | {"scfd_vector_operations.h", "scfd_serial_cpu_vector_operations.h"}):
        if f"source/common/{header}" in makefile:
            errors.append(f"Makefile: dependency on retired {header}")
    if errors:
        print("\n".join(errors))
        return 1
    print("Operations dependency boundaries: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
