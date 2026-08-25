#!/usr/bin/env python3
"""Validate immutable stability assets and compact replay fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "data" / "reference" / "index.json"
DEFAULT_LOCK = (
    PROJECT_ROOT / "data" / "reference" / "stability_regressions.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    require(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA-256 digest",
    )
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_below(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"reference path escapes the project root: {relative}") from error
    return candidate


def validate_summary(dataset: dict[str, Any]) -> None:
    label = dataset.get("path", "stability dataset")
    required_counts = (
        "curves",
        "records",
        "bifurcations",
        "steady_events",
        "hopf_events",
        "maximum_unstable_dimension",
    )
    for field in required_counts:
        value = dataset.get(field)
        require(
            isinstance(value, int) and value >= 0,
            f"{label}: {field} must be a nonnegative integer",
        )
    classified_events = dataset["steady_events"] + dataset["hopf_events"]
    classified_events += dataset.get("multiple_events", 0)
    require(
        classified_events == dataset["bifurcations"],
        f"{label}: bifurcation event counts are inconsistent",
    )


def validate_assets(index: dict[str, Any], lock: dict[str, Any]) -> int:
    require(index.get("schema_version") == 1, "unsupported reference index schema")
    require(lock.get("schema_version") == 1, "unsupported stability lock schema")
    require(
        index.get("latest") == lock.get("latest_asset"),
        "stability lock and reference index disagree about the latest asset",
    )
    indexed_assets = {
        asset.get("version"): asset
        for asset in index.get("assets", [])
        if isinstance(asset, dict)
    }
    locked_assets = lock.get("assets", [])
    require(isinstance(locked_assets, list), "stability assets must be a list")
    dataset_count = 0
    for expected in locked_assets:
        require(isinstance(expected, dict), "stability asset lock must be an object")
        version = expected.get("version")
        require(version in indexed_assets, f"locked asset is absent from index: {version}")
        actual = indexed_assets[version]
        for field in ("release_tag", "file", "sha256", "size_bytes"):
            require(
                actual.get(field) == expected.get(field),
                f"asset {version}: locked {field} differs from index",
            )
        require_sha256(expected.get("sha256"), f"asset {version} sha256")
        require(
            isinstance(expected.get("size_bytes"), int) and expected["size_bytes"] > 0,
            f"asset {version}: invalid size",
        )
        indexed_paths = {
            dataset.get("path")
            for dataset in actual.get("datasets", [])
            if isinstance(dataset, dict)
        }
        for dataset in expected.get("datasets", []):
            require(isinstance(dataset, dict), f"asset {version}: invalid dataset lock")
            path = dataset.get("path")
            require(path in indexed_paths, f"asset {version}: dataset is absent: {path}")
            validate_summary(dataset)
            dataset_count += 1
    return dataset_count


def validate_fixtures(project_root: Path, lock: dict[str, Any]) -> int:
    fixtures = lock.get("compact_fixtures", [])
    require(isinstance(fixtures, list), "compact_fixtures must be a list")
    file_count = 0
    for fixture in fixtures:
        require(isinstance(fixture, dict), "compact fixture lock must be an object")
        root = resolve_below(project_root, fixture["root"])
        manifest_path = resolve_below(root, fixture["manifest"])
        require(manifest_path.is_file(), f"missing fixture manifest: {manifest_path}")
        expected_manifest_hash = require_sha256(
            fixture.get("manifest_sha256"),
            f"fixture {fixture.get('id')} manifest sha256",
        )
        require(
            file_sha256(manifest_path) == expected_manifest_hash,
            f"fixture manifest changed: {manifest_path}",
        )
        manifest = load_object(manifest_path)
        require(
            len(manifest.get("states", [])) == fixture.get("states"),
            f"fixture {fixture.get('id')}: state count changed",
        )
        require(
            len(manifest.get("transitions", [])) == fixture.get("transitions"),
            f"fixture {fixture.get('id')}: transition count changed",
        )
        files = fixture.get("files", {})
        require(isinstance(files, dict) and files, "fixture files must be a nonempty object")
        manifest_files = {
            state.get("file")
            for state in manifest.get("states", [])
            if isinstance(state, dict)
        }
        require(
            manifest_files == set(files),
            f"fixture {fixture.get('id')}: manifest and checksum files differ",
        )
        for relative, expected_hash in files.items():
            path = resolve_below(root, relative)
            require(path.is_file(), f"missing compact fixture: {path}")
            require_sha256(expected_hash, f"fixture {relative} sha256")
            require(file_sha256(path) == expected_hash, f"fixture changed: {path}")
            file_count += 1
    return file_count


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    index = load_object(args.index.resolve())
    lock = load_object(args.lock.resolve())
    datasets = validate_assets(index, lock)
    fixture_files = validate_fixtures(project_root, lock)
    print(
        "Stability reference contract: "
        f"{datasets} datasets, {fixture_files} compact state files, passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
