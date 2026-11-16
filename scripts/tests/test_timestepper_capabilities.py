"""Keep capability evidence and the parent validation gate connected."""

from pathlib import Path
import sys
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_timestepper_capabilities as capabilities
import run_local_ci as ci


class CapabilityContractTest(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / capabilities.MATRIX).read_text()

    def test_matrix_has_linked_evidence(self):
        self.assertEqual(capabilities.validate()["methods"], 28)

    def test_stale_test_link_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing capability evidence"):
            capabilities.validate(text=self.text.replace("test_explicit_rk.cpp", "missing_test.cpp"))

    def test_missing_evidence_and_duplicate_method_are_rejected(self):
        row = next(line for line in self.text.splitlines() if line.startswith("| EE |"))
        for text in (self.text.replace(row, "| EE | 1 | 0 | 1 | F | none |"),
                     self.text.replace(row, row + "\n" + row)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                capabilities.validate(text=text)

    def test_compiled_metadata_must_agree(self):
        rows = capabilities.method_rows(self.text)
        metadata = "\n".join("|".join((name, *values)) for name, values in rows.items())
        self.assertTrue(capabilities.validate(metadata=metadata)["compiled_metadata_checked"])
        with self.assertRaisesRegex(ValueError, "differ from compiled"):
            capabilities.validate(metadata=metadata.replace("RK4|4|0|3", "RK4|3|0|3"))

    def test_pde_jobs_and_failure_artifacts_are_present(self):
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        gate = jobs["all-tests-passed"]["needs"]
        for name in ("timestepper-capability-contract", "timestepper-pde-cpu",
                     "timestepper-pde-cuda", "timestepper-pde-hip"):
            with self.subTest(job=name):
                self.assertIn(name, jobs)
                if name != "timestepper-capability-contract":
                    uploads = [step for step in jobs[name]["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")]
                    self.assertTrue(uploads)
                    self.assertEqual(uploads[0]["if"], "always()")
        for name in ("timestepper-pde-cpu", "timestepper-pde-cuda", "timestepper-pde-hip"):
            self.assertIn(name, gate)

    def test_job_shells_parse_and_accelerator_skips_are_explicit(self):
        jobs = ci.load_workflow(ROOT / ".github/workflows/operations.yml")["jobs"]
        for name in ("timestepper-capability-contract", "timestepper-pde-cpu", "timestepper-pde-cuda", "timestepper-pde-hip"):
            commands = "\n".join(step.get("run", "") for step in jobs[name]["steps"])
            with self.subTest(job=name):
                subprocess.run(["bash", "-n"], input=commands, text=True, check=True, capture_output=True)
                if name.endswith(("-cuda", "-hip")):
                    self.assertIn('"executed":false', commands)
                    self.assertIn('"executed":true', commands)


if __name__ == "__main__":
    unittest.main()
