#!/usr/bin/env python3
"""Build a deterministic archive of the complete reference-data payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "reference"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "build" / "reference_assets"
ARCHIVE_PREFIX = "deflated-continuation-reference-data"
EXCLUDED_ROOT_FILES = {"index.json"}
REQUIRED_PAYLOAD_FILES = (
    "KS1D/N64/symmetric/manifest.json",
    "KS1D/N64/full/manifest.json",
    "KS1D/N128/symmetric/manifest.json",
    "KS1D/N128/full/manifest.json",
    "KS1D/N128/full_stability_20260730/manifest.json",
    "eigensolvers/baseline_20260725/manifest.json",
    "eigensolvers/eigs_paper_stage3_20260725/manifest.json",
    "eigensolvers/eigs_paper_stage4_20260725/manifest.json",
    "eigensolvers/eigs_paper_stage5_20260725/manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="Immutable asset version, normally YYYYMMDD.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Reference directory to package.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output archive path. Defaults below build/reference_assets.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(0, 10),
        default=6,
    )
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=0,
        help="Normalized archive timestamp.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output archive.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_paths(source: Path) -> list[Path]:
    paths = [source]
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if len(relative.parts) == 1 and relative.name in EXCLUDED_ROOT_FILES:
            continue
        if "__pycache__" in relative.parts or relative.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise RuntimeError(f"reference payload contains a symlink: {path}")
        paths.append(path)
    return sorted(paths, key=lambda value: value.as_posix())


def validate_payload(source: Path) -> None:
    missing = [
        relative
        for relative in REQUIRED_PAYLOAD_FILES
        if not (source / relative).is_file()
    ]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise RuntimeError(
            "reference payload is incomplete. Download and extract the "
            "current asset before packaging a replacement. Missing:\n" +
            formatted
        )


def normalized_tar_info(
    archive: tarfile.TarFile,
    path: Path,
    source: Path,
    timestamp: int,
) -> tarfile.TarInfo:
    if path == source:
        relative = Path()
    else:
        relative = path.relative_to(source)
    archive_name = Path("data") / "reference" / relative
    info = archive.gettarinfo(str(path), arcname=archive_name.as_posix())
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = timestamp
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def build_archive(
    source: Path,
    output: Path,
    compression_level: int,
    timestamp: int,
) -> None:
    temporary = output.with_name(output.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(
            temporary,
            mode="w:xz",
            preset=compression_level,
        ) as archive:
            for path in included_paths(source):
                info = normalized_tar_info(
                    archive,
                    path,
                    source,
                    timestamp,
                )
                if info.isfile():
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                else:
                    archive.addfile(info)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise RuntimeError(f"reference directory does not exist: {source}")
    validate_payload(source)
    if args.source_date_epoch < 0:
        raise RuntimeError("--source-date-epoch cannot be negative")

    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_DIRECTORY / (
            f"{ARCHIVE_PREFIX}-{args.version}.tar.xz"
        )
    output = output.resolve()
    if output.exists() and not args.force:
        raise RuntimeError(
            f"output already exists: {output}; use --force to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    build_archive(
        source,
        output,
        args.compression_level,
        args.source_date_epoch,
    )
    result = {
        "version": args.version,
        "file": output.name,
        "path": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "archive_root": "data/reference",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, tarfile.TarError) as error:
        raise SystemExit(f"FAILED: {error}") from error
