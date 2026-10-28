from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from check_newton_solver_dependencies import violations


class NewtonSolverDependenciesTest(unittest.TestCase):
    def test_supported_code_rejects_archived_newton(self):
        for path in ("source/main/nmfd_newton_types.h", "source/main/deflation_continuation.hpp",
                     "source/deflation/deflation.hpp", "source/continuation/continuation.hpp",
                     "source/models/star_shaped/star_shaped_bd.cpp", "source/models/bratu/bratu_bd.cpp",
                     "source/models/circle/circle_bd.cpp", "source/models/KS_2D/KS2D_bd.cpp"):
            self.assertTrue(violations(path, '#include <numerical_algos/newton_solvers/newton_solver.h>'))
            self.assertFalse(violations(path, '#include <nmfd/solvers/nonlinear_solver.h>'))

    def test_historical_targets_alone_opt_in(self):
        for target, legacy in (("star_shaped_bd_var_prec", False), ("bratu_bd_var_prec", False),
                               ("circle_bd_var_prec", False), ("test_workflow_ownership_cpu.bin", False),
                               ("cont_def_circle_cpu_omp", True)):
            result = subprocess.run(["make", "--no-print-directory", "-n", target,
                "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc", "BUILD_DIR=build/newton_dependency_probe"],
                cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual("-DPROJECT_ENABLE_LEGACY_NEWTON" in result.stdout, legacy)
            self.assertEqual("-I source/numerical_algos/legacy/include" in result.stdout, legacy)

    def test_repository_boundary(self):
        subprocess.run([sys.executable, "scripts/check_newton_solver_dependencies.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
