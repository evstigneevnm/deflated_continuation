#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import validate_stability_diagram as validator


class StabilityDiagramValidatorTests(unittest.TestCase):
    def make_project(self, root: Path, last_source_index: int = 3) -> Path:
        project = root / "project"
        curve = project / "0"
        curve.mkdir(parents=True)
        (project / "stability_diagram.dat").write_text(
            "archive\n",
            encoding="utf-8",
        )
        (curve / "debug_curve_all.dat").write_text(
            "0.0 1.0 0\n"
            "0.5 2.0 1\n"
            "1.0 3.0 2\n"
            "1.5 4.0 3\n",
            encoding="utf-8",
        )
        (curve / "debug_curve_stability.dat").write_text(
            "0.0 unstable 2 0 0\n"
            "0.25 bifurcation 0 0 1\n"
            "1.0 unstable 2 0 0\n"
            "1.25 bifurcation 4 0 2\n"
            "1.375 topology_break 5 0 3\n"
            "1.5 unstable 5 0 0\n",
            encoding="utf-8",
        )
        for event_id in (1, 2, 3):
            (curve / f"s{event_id}").write_text(
                f"event {event_id}\n",
                encoding="utf-8",
            )
        (curve / "debug_curve_stability_plot.dat").write_text(
            "# stability_plot_v1\n"
            "0 0.0 unstable 2 0 2 0 2 0 none 0 1 1.0\n"
            "1 0.25 bifurcation 0 0 2 0 0 0 steady 1 1 1.5\n"
            "2 1.0 unstable 2 0 2 0 2 0 none 0 1 3.0\n"
            "2 1.25 bifurcation 4 0 2 0 4 0 steady 2 1 3.5\n"
            "1 1.375 topology_break 5 0 4 0 5 0 topology 3 1 3.75\n"
            f"{last_source_index} 1.5 unstable 5 0 5 0 5 0 "
            "none 0 1 4.0\n",
            encoding="utf-8",
        )
        return project

    @staticmethod
    def args(project: Path) -> SimpleNamespace:
        return SimpleNamespace(
            project_dir=str(project),
            expected_curves=1,
            min_points=1,
            require_complete=True,
            require_plot_sidecar=True,
            expected_point_type=None,
            expected_unstable_real=None,
            expected_unstable_complex_pairs=None,
            expected_bifurcations=2,
            expected_topology_breaks=1,
            min_stable_points=0,
            min_unstable_points=2,
            expected_bifurcation_parameter=[],
            bifurcation_parameter_tolerance=1.0e-3,
        )

    def test_complete_sampled_traversal_accepts_refined_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.make_project(Path(temporary))
            validator.validate(self.args(project))

    def test_complete_traversal_rejects_missing_terminal_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = self.make_project(
                Path(temporary),
                last_source_index=2,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "expected \\[0, 3\\]",
            ):
                validator.validate(self.args(project))


if __name__ == "__main__":
    unittest.main()
