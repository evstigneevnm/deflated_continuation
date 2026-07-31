#!/usr/bin/env python3
"""Download, verify, and safely install a versioned reference-data asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "data" / "reference" / "index.json"
DEFAULT_CACHE = PROJECT_ROOT / "build" / "reference_assets" / "cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Asset index JSON.",
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="Asset version or 'latest'.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List indexed assets without downloading.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing local archive instead of downloading.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory below which data/reference will be installed.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Verify the archive but do not extract it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace differing extracted files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload an otherwise valid cached archive.",
    )
    return parser.parse_args()


def load_index(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("schema_version") != 1:
        raise RuntimeError("unsupported reference-asset index schema")
    assets = index.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("reference-asset index has no assets")
    return index


def select_asset(index: dict, requested_version: str) -> dict:
    version = (
        index.get("latest")
        if requested_version == "latest"
        else requested_version
    )
    for asset in index["assets"]:
        if asset.get("version") == version:
            return asset
    raise RuntimeError(f"reference asset version is not indexed: {version}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path: Path, asset: dict) -> None:
    expected_size = int(asset["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"archive size mismatch: expected {expected_size}, "
            f"found {actual_size}"
        )
    actual_digest = sha256(path)
    expected_digest = str(asset["sha256"]).lower()
    if actual_digest != expected_digest:
        raise RuntimeError(
            "archive SHA-256 mismatch: "
            f"expected {expected_digest}, found {actual_digest}"
        )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "deflated-continuation-reference-data/1"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_members(
    archive: tarfile.TarFile,
    expected_root: PurePosixPath,
) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive path: {member.name}")
        if path != expected_root and expected_root not in path.parents:
            raise RuntimeError(
                f"archive member is outside {expected_root}: {member.name}"
            )
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(
                f"unsupported archive member type: {member.name}"
            )
    return members


def files_equal(first: Path, second: Path) -> bool:
    return (
        first.stat().st_size == second.stat().st_size
        and sha256(first) == sha256(second)
    )


def merge_payload(
    extracted_root: Path,
    destination_root: Path,
    force: bool,
) -> tuple[int, int]:
    files = sorted(
        (path for path in extracted_root.rglob("*") if path.is_file()),
        key=lambda value: value.as_posix(),
    )
    conflicts: list[Path] = []
    for source in files:
        target = destination_root / source.relative_to(extracted_root)
        if target.exists() and not files_equal(source, target):
            conflicts.append(target)
    if conflicts and not force:
        displayed = "\n".join(f"  {path}" for path in conflicts[:10])
        suffix = "\n  ..." if len(conflicts) > 10 else ""
        raise RuntimeError(
            "reference extraction would overwrite differing files; "
            "use --force to replace them:\n" + displayed + suffix
        )

    installed = 0
    unchanged = 0
    for source in files:
        target = destination_root / source.relative_to(extracted_root)
        if target.exists() and files_equal(source, target):
            unchanged += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        installed += 1
    return installed, unchanged


def extract_archive(
    archive_path: Path,
    asset: dict,
    destination: Path,
    force: bool,
) -> tuple[int, int]:
    archive_root = PurePosixPath(asset["archive_root"])
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".reference-data-",
        dir=destination,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        with tarfile.open(archive_path, mode="r:xz") as archive:
            members = validate_members(archive, archive_root)
            for member in members:
                try:
                    archive.extract(
                        member,
                        path=temporary_root,
                        filter="data",
                    )
                except TypeError:
                    # Extraction filters were added to maintained Python
                    # versions after the original tarfile API. Member paths
                    # and types have already been validated above.
                    archive.extract(member, path=temporary_root)
        extracted_root = temporary_root / Path(*archive_root.parts)
        if not extracted_root.is_dir():
            raise RuntimeError(
                f"archive does not contain its declared root: {archive_root}"
            )
        destination_root = destination / Path(*archive_root.parts)
        return merge_payload(
            extracted_root,
            destination_root,
            force,
        )


def print_assets(index: dict) -> None:
    latest = index.get("latest")
    for asset in index["assets"]:
        marker = " (latest)" if asset.get("version") == latest else ""
        print(
            f"{asset['version']}{marker}: {asset['file']} "
            f"[{asset['size_bytes']} bytes]"
        )
        for dataset in asset.get("datasets", []):
            print(f"  {dataset['path']}: {dataset['description']}")


def main() -> int:
    args = parse_args()
    index = load_index(args.index.resolve())
    if args.list:
        print_assets(index)
        return 0

    asset = select_asset(index, args.version)
    if args.archive is not None:
        archive_path = args.archive.resolve()
        if not archive_path.is_file():
            raise RuntimeError(f"local archive does not exist: {archive_path}")
    else:
        archive_path = args.cache_dir.resolve() / asset["file"]
        if args.force_download or not archive_path.exists():
            print(f"Downloading {asset['url']}")
            download(asset["url"], archive_path)

    verify_archive(archive_path, asset)
    print(f"Verified {archive_path}")
    if args.download_only:
        return 0

    installed, unchanged = extract_archive(
        archive_path,
        asset,
        args.destination.resolve(),
        args.force,
    )
    print(
        f"Installed reference data: {installed} files written, "
        f"{unchanged} identical files retained"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        RuntimeError,
        tarfile.TarError,
        urllib.error.URLError,
    ) as error:
        raise SystemExit(f"FAILED: {error}") from error
