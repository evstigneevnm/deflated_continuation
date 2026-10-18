import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "operations_dependencies", Path(__file__).resolve().parents[1] / "check_operations_dependencies.py"
)
dependencies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dependencies)


class OperationsDependenciesTest(unittest.TestCase):
    def test_canonical_interface(self):
        source = """
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/operations/scfd_vector_operations.h>
#include <nmfd/operations/backend/complex.h>
using space = nmfd::operations::scfd_vector_space<Backend, T>;
using algebra = nmfd::operations::scfd_vector_operations<Backend, T>;
using complex = nmfd::operations::backend::complex_t<Backend, T>;
"""
        self.assertEqual(dependencies.violations("source/models/problem.h", source), [])

    def test_removed_headers_and_private_utilities(self):
        for header in (
            "common/NMFD-operations/nmfd/operations/dense_vector_space.h",
            "common/scfd_backend_ext/complex.h",
            "../scfd_serial_cpu_vector_operations.h",
            "common/glued_vector_space.h",
            "nmfd/operations/detail/scfd_backend_ext/math.h",
        ):
            with self.subTest(header=header):
                self.assertTrue(dependencies.violations("source/models/problem.h", f'#include "{header}"'))

    def test_legacy_names(self):
        for source in (
            "using space = scfd_vector_operations<Backend, T>;",
            "using space = scfd_serial_cpu_vector_operations<T>;",
            "using space = ::scfd_vector_operations<Backend, T>;",
            "namespace utilities = nmfd::operations::detail::scfd_backend_ext;",
            "using value = common::scfd_backend_ext::complex_t<Backend, T>;",
            "using value = nmfd::operations::detail::scfd_backend_ext::complex_t<Backend, T>;",
            "using space = scfd::linspace::glued_vector_space<Space, 2>;",
        ):
            with self.subTest(source=source):
                self.assertTrue(dependencies.violations("source/models/problem.h", source))

    def test_moved_common_paths(self):
        for name in sorted(dependencies.MOVED_COMMON_HEADERS):
            with self.subTest(header=name):
                self.assertTrue(
                    dependencies.violations("source/models/problem.h", f'#include <common/{name}>')
                )
                self.assertEqual(
                    dependencies.violations("source/models/KF_3D/problem.h", f'#include <common/legacy/{name}>'),
                    [],
                )

    def test_refactored_code_cannot_use_legacy_common(self):
        source = '#include <common/legacy/gpu_vector_operations.h>'
        for path in (
            "source/continuation/problem.h",
            "source/time_stepper/runge_kutta/explicit_time_step.h",
            "source/models/KS_2D/KS2D_backend_typedefs.h",
            "source/models/KS_1D/test_KS1D_operator.cpp",
            "source/models/bratu/bratu_bd.cpp",
        ):
            with self.subTest(path=path):
                self.assertTrue(dependencies.violations(path, source))

    def test_frozen_code_and_narrow_glued_exceptions(self):
        source = '#include <common/glued_vector_space.h>'
        for path in (
            "source/time_stepper/legacy/detail/problem.h",
            "source/common/glued_vector_space.h",
            "source/common/glued_vector_operations.h",
        ):
            self.assertEqual(dependencies.violations(path, source), [])
        self.assertTrue(dependencies.violations("source/time_stepper/legacy_extra.h", source))


if __name__ == "__main__":
    unittest.main()
