#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validate_circle_visualization as validator


class CircleVisualizationValidatorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "state"
        self.source.write_text("0.6\n")
        self.output = self.root / "export.dat"
        self.output.write_text("0 0.6\n")
        self.manifest = self.root / "manifest.jsonl"
        self.record = {
            "kind": "state_scalar",
            "points": 1,
            "components": 1,
            "state_file": str(self.source),
            "data_file": self.output.name,
        }
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.record) + "\n")

    def test_matching_export(self):
        validator.validate(self.manifest, 1)

    def test_bad_export(self):
        for output in ("0 0.7", "0 nan", "0 inf", "1 0.6", "0 0.6 0.7", ""):
            with self.subTest(output=output):
                self.output.write_text(output)
                with self.assertRaises(RuntimeError):
                    validator.validate(self.manifest, 1)

    def test_bad_metadata_or_count(self):
        with self.assertRaises(RuntimeError):
            validator.validate(self.manifest, 2)
        self.record["kind"] = "state_vector_1d"
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            validator.validate(self.manifest, 1)


if __name__ == "__main__":
    unittest.main()
