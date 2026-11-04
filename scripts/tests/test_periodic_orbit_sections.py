from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODERN = ROOT / "source/periodic_orbit"
SYSTEMS = (
    "section_geometry", "event_locator", "poincare_harmonic", "poincare_implicit",
    "poincare_dae", "section_map_lifecycle", "section_layout_variable_speed", "section_rpo_target",
    "section_coordinates", "coordinate_solvers", "multiple_section_matching",
    "prepared_section_derivatives", "multiple_section_newton",
)


class PeriodicSectionContractTest(unittest.TestCase):
    def test_backend_local_public_headers(self):
        headers = [path for path in MODERN.rglob("*.h")
                   if "legacy" not in path.parts and "tests" not in path.parts]
        self.assertEqual(len(headers), 13)
        for path in headers:
            with self.subTest(header=path.name):
                text = path.read_text()
                self.assertNotRegex(text, r"->(?:view|get|set|get_value_at_point|set_value_at_point)\s*\(")
                self.assertNotRegex(text, r"cudaMemcpy|hipMemcpy|copy_to_host|copy_from_host")
                if path.name != "scfd_chart_transport.h":
                    self.assertNotIn("raw_ptr", text)
                self.assertNotIn("nmfd/solvers", text)
                self.assertNotIn("legacy/", text)

    def test_make_targets_use_modern_sources_and_flat_dependencies(self):
        targets = [f"test_periodic_orbit_{name}_{backend}.bin"
                   for name in SYSTEMS for backend in ("cpu", "cpu_omp", "cuda", "hip")]
        result = subprocess.run(
            ["make", "-n", *targets, "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc",
             "BUILD_DIR=build/periodic_section_contract", "HIPCC=true", "HIP_EXTENDED_LAMBDA="],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
        self.assertNotIn("periodic_orbit/legacy", result.stdout)
        for name in SYSTEMS:
            self.assertIn(f"source/periodic_orbit/tests/test_{name}.cpp", result.stdout)
        self.assertNotIn("nmfd-timesteppers/contrib", result.stdout)

    def test_every_system_has_every_ci_backend_and_required_gate(self):
        workflow = (ROOT / ".github/workflows/operations.yml").read_text()
        for suffix in ("cpu", "sanitizers", "cuda", "hip"):
            name = f"periodic-orbit-sections-{suffix}"
            job = re.search(rf"^  {name}:\n(.*?)(?=^  [\w-]+:|\Z)", workflow, re.M | re.S)
            self.assertIsNotNone(job, name)
            for system in SYSTEMS:
                self.assertIn(system, job.group(1))
            if suffix != "cpu":
                gate = workflow.split("  all-tests-passed:", 1)[1].split("    steps:", 1)[0]
                self.assertIn(name, gate)
        self.assertEqual({p.stem.removeprefix("test_") for p in (MODERN / "tests").glob("test_*.cpp")},
                         set(SYSTEMS))


if __name__ == "__main__":
    unittest.main()
