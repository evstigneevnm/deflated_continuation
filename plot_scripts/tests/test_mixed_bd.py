"""Offline regressions for plotting folds, failed verification and incompatible overlays."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PLOTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLOTS))
import mixed_bd


class MixedDiagramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cls.plt = plt

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(self.plt.close, "all")
        self.root = Path(tmp.name) / "bundle"
        fixture = PLOTS.parent / "source/periodic_orbit/tests/fixtures/mixed_bd/analytic-cycle"
        shutil.copytree(fixture, self.root)
        self.manifest = self.root / "manifest.json"
        self.spec = json.loads(self.manifest.read_text())
        self.spec["periodic"] = ["trace.json"]
        self.manifest.write_text(json.dumps(self.spec))
        self.trace = {k: self.spec["stationary"][k] for k in mixed_bd.IDENTITY}
        points = [dict(source_index=i, parameter=x, verified=(i != 2)) for i, x in enumerate([1., 2., 1.5, 1.])]
        for p in points:
            if p["verified"]:
                p.update(period=3., observable_rms=2., observable_min=1., observable_max=2.5, closure_error=.0001)
        self.trace.update(id="fold", format="periodic_branch", version=1, kind="periodic", closure="identity",
                          sections=4, method="RK45", trace_complete=False, all_points_verified=False,
                          restarted_from=dict(branch="original", segment=1, source_index=5),
                          segments=[dict(id=1, failure="none", endpoint="max_steps", points=points)])

    def draw(self, **kwargs):
        (self.root / "trace.json").write_text(json.dumps(self.trace))
        return mixed_bd.draw(self.manifest, self.plt, **kwargs)

    def test_fold_gap_envelope_and_provenance(self):
        top, bottom = self.draw(envelope=True).axes
        lines = {line.get_gid(): line for line in top.lines}
        self.assertEqual(set(lines), {"stationary:0:1", "stationary:0:2", "rms:fold:1"})
        self.assertEqual(list(lines["stationary:0:1"].get_xdata()), [1., .95])
        self.assertEqual(list(lines["stationary:0:2"].get_xdata()), [1., 1.05])
        rms = lines["rms:fold:1"]
        self.assertEqual(len(bottom.lines), 1)
        for line in (rms, bottom.lines[0]):
            self.assertEqual(list(line.get_xdata()), [1., 2., 1.5, 1.])
            self.assertTrue(math.isnan(line.get_ydata()[2]))
        self.assertEqual(rms.get_ydata()[0], 2.)
        self.assertEqual(bottom.lines[0].get_ydata()[0], 3.)
        for phrase in ("incomplete", "max_steps", "restart of original"):
            self.assertIn(phrase, rms.get_label())
        paths = top.collections[0].get_paths()
        self.assertEqual(len(paths), 2)  # No envelope across the failed sample.
        self.assertEqual(set(paths[0].vertices[:, 1]), {1., 2.5})

    def test_reject_mismatch_for_each_identity_field(self):
        for field in mixed_bd.IDENTITY:
            with self.subTest(field=field):
                self.trace[field] += "-different"
                with self.assertRaisesRegex(ValueError, "Incompatible " + field):
                    self.draw()
                self.trace[field] = self.spec["stationary"][field]

    def test_missing_identity_and_invalid_column(self):
        del self.trace["normalization"]
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.draw()
        (self.root / "stationary/legend.dat").write_text("other_norm\n")
        with self.assertRaisesRegex(ValueError, "stationary column"):
            self.draw()

    def test_cli_refuses_incompatible_overlay_without_writing_figure(self):
        self.trace["discretization"] = "wrong grid"
        (self.root / "trace.json").write_text(json.dumps(self.trace))
        output = self.root / "must_not_exist.png"
        run = subprocess.run([sys.executable, str(PLOTS / "plot_bd.py"), "mixed", str(self.manifest),
                              "--output", str(output)], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Incompatible discretization", run.stderr)
        self.assertFalse(output.exists())

    def test_unverified_is_not_a_zero_or_a_verified_point(self):
        self.trace["all_points_verified"] = True
        with self.assertRaisesRegex(ValueError, "verification status"):
            self.draw()
        self.trace["all_points_verified"] = False
        del self.trace["segments"][0]["points"][0]["period"]
        with self.assertRaisesRegex(ValueError, "period"):
            self.draw()

    def test_missing_vs_corrupt_stationary_topology(self):
        path = self.root / "stationary/0/metadata_curve.dat"
        path.write_text("bad row\n")
        with self.assertRaisesRegex(ValueError, "segment metadata"):
            self.draw()
        path.unlink()
        fig = self.draw()
        self.assertEqual(fig.axes[0].lines[0].get_linestyle(), "None")
        self.assertFalse(fig.axes[0].collections)


if __name__ == "__main__":
    unittest.main()
