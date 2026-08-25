#!/usr/bin/env python3
"""Unit tests for the compact full-KS1D stability replay plan."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "test_ks1d_full_stability_replay.py"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "source"
    / "models"
    / "KS_1D"
    / "tests"
    / "data"
    / "stability_replay"
    / "manifest.json"
)

SPEC = importlib.util.spec_from_file_location("ks1d_stability_replay", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {SCRIPT_PATH}")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class StabilityReplayPlanTests(unittest.TestCase):
    def test_transition_endpoints_replace_duplicate_state_commands(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        state_ids = {state["id"] for state in manifest["states"]}

        transition_states = REPLAY.transition_state_ids(manifest, state_ids)
        standalone_states = state_ids - transition_states

        self.assertEqual(
            standalone_states,
            {
                "curve19_state084",
                "curve19_state085",
                "curve19_state086",
            },
        )
        self.assertEqual(len(transition_states), 5)
        self.assertEqual(
            len(standalone_states) + len(manifest["transitions"]),
            6,
        )

    def test_unknown_transition_state_is_rejected(self) -> None:
        manifest = {
            "transitions": [
                {
                    "id": "invalid",
                    "first_state": "known",
                    "second_state": "missing",
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "unknown second_state"):
            REPLAY.transition_state_ids(manifest, {"known"})


if __name__ == "__main__":
    unittest.main()
