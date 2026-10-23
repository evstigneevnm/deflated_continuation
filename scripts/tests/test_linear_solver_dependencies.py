import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from check_linear_solver_dependencies import violations


class LinearSolverDependenciesTest(unittest.TestCase):
    def test_active_components_reject_archived_solver_headers(self):
        for path in ("source/continuation/system_operator_continuation.h",
                     "source/continuation/corrector_operator.h",
                     "source/continuation/bordered_linear_solver.h",
                     "source/deflation/system_operator_deflation.h",
                     "source/deflation/deflated_operator.h",
                     "source/deflation/rank_one_linear_solver.h",
                     "source/nonlinear_operators/projected_system_operator.h",
                     "source/nonlinear_operators/fixed_parameter_operator.h",
                     "source/nonlinear_operators/tests/test_fixed_parameter_newton.cpp",
                     "source/numerical_algos/recovering_linear_solver.h",
                     "source/numerical_algos/parameterized_correction.h",
                     "source/main/nmfd_newton_types.h",
                     "source/models/circle/circle_bd.cpp",
                     "source/models/KS_1D/KS1D_bd.cpp",
                     "source/models/KS_1D/KS1D_stability.cpp",
                     "source/models/KS_1D/test_KS1D_newton.cpp",
                     "source/models/KS_1D/KS1D_full_bd.cpp",
                     "source/models/KS_2D/KS2D_stability.cpp",
                     "source/numerical_algos/tests/test_linear_solve_recovery.cpp"):
            for header in ("numerical_algos/lin_solvers/cgs.h",
                           "numerical_algos/legacy/include/numerical_algos/lin_solvers/cgs.h",
                           "../legacy/include/numerical_algos/lin_solvers/cgs.h"):
                with self.subTest(path=path, header=header):
                    self.assertTrue(violations(path, f'#include "{header}"\n'))

    def test_canonical_solvers_and_neutral_recovery_are_allowed(self):
        for header in ("nmfd/solvers/cgs.h", "nmfd/solvers/sherman_morrison.h",
                       "numerical_algos/linear_solve_recovery.h"):
            self.assertFalse(violations("source/continuation/system_operator_continuation.h",
                                        f"#include <{header}>\n"))

    def test_historical_clients_and_deliberate_comparisons_remain_supported(self):
        for path in ("source/time_stepper/legacy/tests/vdp_1.cpp",
                     "source/models/circle/circle_test_deflation_continuation.cpp",
                     "source/models/KF_3D/KF3D_bd.cpp",
                     "source/numerical_algos/tests/test_nmfd_short_recurrence.cpp",
                     "source/numerical_algos/tests/test_legacy_linear_solve_recovery.cpp"):
            self.assertFalse(violations(path, "#include <numerical_algos/lin_solvers/cgs.h>\n"))

    def test_legacy_include_root_is_enabled_only_for_explicit_targets(self):
        env = os.environ.copy()
        for name in ("MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "IPROJECT"):
            env.pop(name, None)
        for target, archived in (
            ("test_linear_solve_recovery.bin", False),
            ("test_nmfd_gmres_baseline_cpu.bin", False),
            ("test_nmfd_gmres_shared_krylov_cpu_omp.bin", False),
            ("test_fixed_parameter_newton_cpu.bin", False),
            ("test_fixed_parameter_newton_cpu_omp.bin", False),
            ("test_fixed_parameter_newton_cuda.bin", False),
            ("test_continuation_operator_cpu.bin", False),
            ("test_continuation_newton_cpu_omp.bin", False),
            ("test_continuation_newton_cuda.bin", False),
            ("test_deflation_operator_cpu.bin", False),
            ("test_deflation_newton_cpu_omp.bin", False),
            ("test_deflation_newton_cuda.bin", False),
            ("test_nmfd_newton_assembly_cpu.bin", False),
            ("test_nmfd_newton_assembly_cpu_omp.bin", False),
            ("test_nmfd_newton_assembly_cuda.bin", False),
            ("circle_bd_cpu_omp", False), ("circle_bd_var_prec", False),
            ("KS1D_bd_cpu_omp", False), ("KS1D_bd_cuda", False),
            ("KS1D_stability_cpu_omp", False), ("KS1D_stability_cuda", False),
            ("KS1D_bd_hip", False), ("KS1D_stability_hip", False),
            ("test_KS1D_newton_cpu_omp.bin", False), ("test_KS1D_newton_cuda.bin", False),
            ("test_KS1D_newton_hip.bin", False),
            ("KS1D_full_bd_cpu_omp", False), ("KS2D_stability_cpu_omp", False),
            ("test_explicit_lorenz_cpu.bin", False),
            ("test_legacy_linear_solve_recovery.bin", True),
            ("test_nmfd_short_recurrence_baseline_cpu.bin", True),
            ("test_vdp_time_stepping", True), ("cont_def_circle_cpu_omp", True),
        ):
            with self.subTest(target=target):
                result = subprocess.run(
                    ["make", "--no-print-directory", "-n", target,
                     "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc",
                     "BUILD_DIR=build/dependency_probe"],
                    cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual("-I source/numerical_algos/legacy/include" in result.stdout, archived)
                if target.startswith(("KS1D_bd", "KS1D_stability", "test_KS1D_newton")):
                    self.assertIn("-I source/contrib/nmfd-newton/include", result.stdout)


if __name__ == "__main__":
    unittest.main()
