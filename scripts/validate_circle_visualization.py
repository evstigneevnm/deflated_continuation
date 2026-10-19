#!/usr/bin/env python3
"""Compare exported circle scalar states with their archived source vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate(manifest: Path, expected_points: int) -> None:
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(records) != expected_points:
        raise RuntimeError(f"expected {expected_points} visualization records, found {len(records)}")

    for record in records:
        if record["kind"] != "state_scalar" or record["points"] != 1 or record["components"] != 1:
            raise RuntimeError("circle visualization must contain scalar states")
        source = [float(value) for value in Path(record["state_file"]).read_text().split()]
        output = manifest.parent / record["data_file"]
        exported = [float(value) for value in output.read_text().split()]
        if len(source) != 1 or len(exported) != 2 or exported[0] != 0.0:
            raise RuntimeError(f"invalid scalar state layout: {output}")
        if not all(math.isfinite(value) for value in source + exported):
            raise RuntimeError(f"nonfinite scalar state: {output}")
        if not math.isclose(source[0], exported[1], rel_tol=1e-14, abs_tol=1e-14):
            raise RuntimeError(f"export differs from archived scalar state: {output}")

    print(f"Validated {len(records)} circle visualization records against archived states")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-points", type=int, default=3)
    args = parser.parse_args()
    validate(args.manifest, args.expected_points)
