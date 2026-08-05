#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PLOT_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLOT_SCRIPTS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_bd_solutions as solutions


def field_arguments(**overrides):
    values = {
        "field_color_scale": "selected",
        "field_vmin": None,
        "field_vmax": None,
        "field_colorbar": False,
        "field_render": "imshow",
        "field_cmap": "RdBu_r",
        "contour_levels": 12,
        "fit_solutions": False,
        "thumbnail_columns": 1,
        "thumbnail_width": 0.4,
        "thumbnail_height": 0.4,
        "profile_title_font_size": None,
        "point_label_font_size": None,
        "font_scale": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BDSolutionFieldTests(unittest.TestCase):
    def make_record(self, directory: Path, field: np.ndarray) -> dict:
        path = directory / "field.npy"
        np.save(path, field)
        return {
            "branch": 2,
            "point_index": 5,
            "lambda": 4.5,
            "norms": [2.0],
            "kind": "physical_scalar_2d",
            "format": "npy",
            "shape": list(field.shape),
            "origin": [0.0, 1.0],
            "spacing": [0.5, 0.25],
            "field_name": "u",
            "resolved_data_file": path,
            "manifest_line": 1,
        }

    def test_field_loading_shape_and_extent(self):
        with tempfile.TemporaryDirectory() as temporary:
            field = np.arange(12, dtype=np.float64).reshape(3, 4)
            record = self.make_record(Path(temporary), field)
            loaded = solutions.read_scalar_field(record)
            np.testing.assert_array_equal(loaded, field)
            self.assertEqual(
                solutions.scalar_field_extent(record, loaded.shape),
                (0.0, 1.5, 1.0, 2.0),
            )

    def test_field_shape_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            field = np.arange(12, dtype=np.float64).reshape(3, 4)
            record = self.make_record(Path(temporary), field)
            record["shape"] = [4, 3]
            with self.assertRaises(ValueError):
                solutions.read_scalar_field(record)

    def test_field_is_transposed_for_matplotlib_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            field = np.arange(12, dtype=np.float64).reshape(3, 4)
            record = self.make_record(directory, field)
            figure, axis = plt.subplots(figsize=(5, 4))
            solutions.add_solution_insets(
                figure,
                axis,
                axis,
                [record],
                0,
                field_arguments(),
            )
            inset = axis.child_axes[-1]
            self.assertEqual(len(inset.images), 1)
            np.testing.assert_array_equal(np.asarray(inset.images[0].get_array()), field.T)
            output = directory / "render.png"
            figure.savefig(output, dpi=80)
            plt.close(figure)
            self.assertGreater(output.stat().st_size, 1000)

    def test_existing_one_dimensional_profile_path_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.dat"
            profile.write_text("0.0 1.0\n1.0 -2.0\n", encoding="utf-8")
            record = {
                "branch": 0,
                "point_index": 0,
                "lambda": 1.0,
                "norms": [2.0],
                "kind": "physical_scalar_1d",
                "resolved_data_file": profile,
            }
            figure, axis = plt.subplots(figsize=(5, 4))
            solutions.add_solution_insets(
                figure,
                axis,
                axis,
                [record],
                0,
                field_arguments(),
            )
            inset = axis.child_axes[-1]
            self.assertEqual(len(inset.lines), 1)
            self.assertEqual(list(inset.lines[0].get_xdata()), [0.0, 1.0])
            self.assertEqual(list(inset.lines[0].get_ydata()), [1.0, -2.0])
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
