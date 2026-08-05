#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a finite-symmetry group manifest.")
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--expected-order", type=int, required=True)
    parser.add_argument("--expected-fingerprint")
    arguments = parser.parse_args()

    manifest_path = arguments.project_dir / "symmetry_group.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing symmetry manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)

    if manifest.get("version") != 2:
        raise SystemExit(
            f"unsupported symmetry manifest version: {manifest.get('version')}")
    actions = manifest.get("actions")
    if not isinstance(actions, list):
        raise SystemExit("symmetry manifest actions are not a list")
    if manifest.get("order") != len(actions):
        raise SystemExit("symmetry manifest order does not match its action list")
    if len(actions) != arguments.expected_order:
        raise SystemExit(
            f"expected group order {arguments.expected_order}, got {len(actions)}")
    if len(set(actions)) != len(actions):
        raise SystemExit("symmetry manifest contains duplicate action names")
    if not actions or actions[0] != "identity":
        raise SystemExit("symmetry manifest does not start with identity")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise SystemExit("symmetry manifest fingerprint is missing")
    if (arguments.expected_fingerprint is not None and
            fingerprint != arguments.expected_fingerprint):
        raise SystemExit(
            f"expected fingerprint {arguments.expected_fingerprint}, "
            f"got {fingerprint}")

    print(
        f"symmetry manifest: order={len(actions)}, "
        f"fingerprint={fingerprint}, PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
