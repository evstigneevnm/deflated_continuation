#!/usr/bin/env python3

import math
import sys
import tempfile
import unittest
from pathlib import Path


PLOT_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLOT_SCRIPTS))

import plot_bd as bd


class StabilityPlotDataTests(unittest.TestCase):
    def test_curve_segments_insert_line_breaks(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            branch = project / "0"
            branch.mkdir()
            (branch / "debug_curve_all.dat").write_text(
                "0.0 1.0 0\n"
                "0.5 2.0 1\n"
                "0.0 1.0 2\n"
                "-0.5 3.0 3\n",
                encoding="utf-8",
            )
            (branch / "metadata_curve.dat").write_text(
                "# index lambda saved file segment semicurve forced reason\n"
                "0 0.0 1 1 1 1 1 none\n"
                "1 0.5 1 2 1 1 0 none\n"
                "2 0.0 1 3 2 2 1 none\n"
                "3 -0.5 1 4 2 2 0 none\n",
                encoding="utf-8",
            )

            rows = bd.load_curves(project)["0"]
            xs, ys, source_indices = bd.curve_coordinates_with_indices(
                rows,
                0,
            )
            self.assertEqual(len(xs), 5)
            self.assertTrue(math.isnan(xs[2]))
            self.assertTrue(math.isnan(ys[2]))
            self.assertEqual(source_indices, [0, 1, 2, 2, 3])

    def test_exact_event_coordinates_and_source_index_coloring(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            branch = project / "0"
            branch.mkdir()
            (branch / "debug_curve_all.dat").write_text(
                "0.0 1.0 0\n"
                "0.5 2.0 1\n"
                "1.0 9.0 2\n",
                encoding="utf-8",
            )
            (branch / "debug_curve_stability_plot.dat").write_text(
                "# stability_plot_v1\n"
                "0 0.0 stable 0 0 0 0 0 0 none 0 1 1.0\n"
                "2 0.75 bifurcation 1 0 0 0 1 0 steady 1 1 0.125\n",
                encoding="utf-8",
            )

            curves = bd.load_curves(project)
            stability = bd.load_stability_curves(project, curves)
            records = stability["0"]

            self.assertEqual(
                bd.stability_dimensions_for_rows(3, records),
                [0, 0, 1],
            )
            self.assertEqual(records[1].event_type, "steady")
            self.assertTrue(records[1].exact_coordinates)
            self.assertAlmostEqual(
                bd.event_norm_value(records[1], curves["0"], 0),
                0.125,
            )
            self.assertNotAlmostEqual(
                bd.event_norm_value(records[1], curves["0"], 0),
                curves["0"][2][1],
            )

    def test_legacy_stability_alignment_is_available_as_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            branch = project / "0"
            branch.mkdir()
            (branch / "debug_curve_all.dat").write_text(
                "0.0 1.0 0\n"
                "0.5 2.0 1\n"
                "1.0 3.0 2\n",
                encoding="utf-8",
            )
            (branch / "debug_curve_stability.dat").write_text(
                "0.0 stable 0 0 0\n"
                "0.6 bifurcation 0 1 1\n",
                encoding="utf-8",
            )

            curves = bd.load_curves(project)
            records = bd.load_stability_curves(project, curves)["0"]

            self.assertEqual(records[0].source_index, 0)
            self.assertEqual(records[1].source_index, 1)
            self.assertEqual(records[1].event_type, "hopf")
            self.assertFalse(records[1].exact_coordinates)
            self.assertAlmostEqual(
                bd.event_norm_value(records[1], curves["0"], 0),
                2.0,
            )

    def test_invalid_sidecar_norm_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = Path(temporary) / "debug_curve_stability_plot.dat"
            sidecar.write_text(
                "0 0 stable 0 0 0 0 0 0 none 0 2 1.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                bd.read_stability_plot_file(sidecar)


if __name__ == "__main__":
    unittest.main()
