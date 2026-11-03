from pathlib import Path
import os
import shlex
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "source/periodic_orbit/legacy/include"
HISTORICAL_TARGETS = (
    "test_rossler_to_section", "test_glued_nonlinear_operator",
    "test_glued_poincare_map_linear_operator", "test_rossler_periodic_orbit_cpu",
    "test_lorentz_periodic_orbit_cpu", "test_rossler_periodic_continuate_cpu",
    "test_rossler_periodic_orbit_gpu", "test_lorentz_time_stepping",
    "abc_flow_perodic_orbit_stabilization", "Kolmogorov_3D_perodic_orbit_stabilization",
)


def dry_run(*targets):
    return subprocess.run(
        ["make", "--no-print-directory", "-n", *targets,
         "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc",
         "BUILD_DIR=build/periodic_orbit_archive_probe"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=30).stdout


class PeriodicOrbitArchiveTest(unittest.TestCase):
    def test_historical_targets_select_archived_sources_and_headers(self):
        for target in HISTORICAL_TARGETS:
            with self.subTest(target=target):
                output = dry_run(target)
                self.assertIn(f"-I {ARCHIVE}", output)
                self.assertLess(output.index(f"-I {ARCHIVE}"), output.index("-I source/ "))
                self.assertNotIn(" source/periodic_orbit/tests/", output)
                sources = [token for token in shlex.split(output)
                           if token.endswith((".cpp", ".cu"))]
                self.assertTrue(sources)
                for source in sources:
                    self.assertTrue((ROOT / source).is_file(), source)
                if target.startswith("test_") and target != "test_lorentz_time_stepping":
                    self.assertTrue(all(source.startswith(f"{ARCHIVE}/periodic_orbit/tests/")
                                        for source in sources))

    def test_supported_targets_do_not_resolve_archived_headers(self):
        for target in ("circle_bd_cpu_omp", "test_rk_tables_implicit.bin",
                       "test_implicit_dense_output_cpu.bin"):
            with self.subTest(target=target):
                self.assertNotIn(ARCHIVE, dry_run(target))

    def test_legacy_selection_does_not_leak_between_goals(self):
        targets = ("test_glued_nonlinear_operator", "test_implicit_dense_output_cpu.bin")
        for order in (targets, targets[::-1]):
            with self.subTest(order=order):
                output = dry_run(*order)
                modern = [line for line in output.splitlines()
                          if "tests/test_implicit_dense_output.cpp" in line]
                self.assertEqual(len(modern), 1)
                self.assertNotIn(ARCHIVE, modern[0])

    def test_frozen_lorenz_include_requires_explicit_archive_root(self):
        source = "#include <periodic_orbit/tests/lorentz_operator.h>\n"
        for enabled in (False, True):
            with self.subTest(archive_enabled=enabled):
                command = shlex.split(os.environ.get("CXX", "c++"))
                command += ["-std=c++17", "-E", "-x", "c++", "-I", "source"]
                if enabled:
                    command += ["-I", ARCHIVE]
                result = subprocess.run(
                    [*command, "-o", os.devnull, "-"], input=source, cwd=ROOT,
                    capture_output=True, text=True, timeout=30)
                if enabled:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("periodic_orbit/tests/lorentz_operator.h", result.stderr)


if __name__ == "__main__":
    unittest.main()
