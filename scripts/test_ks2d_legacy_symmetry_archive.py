#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def sha256(file_name: Path) -> str:
    digest = hashlib.sha256()
    with file_name.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_signature(directory: Path) -> list[tuple[str, int]]:
    return sorted(
        (str(path.relative_to(directory)), path.stat().st_size)
        for path in directory.rglob("*")
        if path.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify transactional rejection of a KS2D archive containing "
            "finite-symmetry duplicate branches."))
    parser.add_argument("executable", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument(
        "--expected-pair",
        type=int,
        nargs=2,
        default=(1, 2),
        metavar=("FIRST", "SECOND"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--omp-threads", type=int, default=8)
    arguments = parser.parse_args()

    executable = arguments.executable.resolve()
    config = arguments.config.resolve()
    project_dir = arguments.project_dir.resolve()
    archive = project_dir / "bifurcation_diagram.dat"
    manifest = project_dir / "symmetry_group.json"
    if not executable.is_file():
        raise SystemExit(f"missing executable: {executable}")
    if not config.is_file():
        raise SystemExit(f"missing config: {config}")
    if not archive.is_file():
        raise SystemExit(f"missing bifurcation archive: {archive}")
    if manifest.exists():
        raise SystemExit(
            "legacy archive regression requires a project without "
            "symmetry_group.json")

    archive_before = sha256(archive)
    tree_before = tree_signature(project_dir)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(arguments.omp_threads)
    with config.open("r", encoding="utf-8") as input_file:
        audit_config = json.load(input_file)
    audit_config["path_to_project"] = str(project_dir) + os.sep

    try:
        with tempfile.TemporaryDirectory(
                prefix="ks2d_legacy_archive_") as temporary_directory:
            audit_config_path = (
                Path(temporary_directory) / "KS2D_legacy_audit.json")
            with audit_config_path.open("w", encoding="utf-8") as output_file:
                json.dump(audit_config, output_file, indent=4)
                output_file.write("\n")
            completed = subprocess.run(
                [str(executable), str(audit_config_path), "--quiet"],
                cwd=Path.cwd(),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=arguments.timeout,
                check=False)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        raise SystemExit(
            "legacy archive audit timed out after "
            f"{arguments.timeout:g} seconds; output was:\n{output}") from error

    if completed.returncode == 0:
        raise SystemExit("legacy duplicate archive was unexpectedly accepted")
    first, second = arguments.expected_pair
    expected = f"curves {first} and {second} have"
    if expected not in completed.stdout:
        raise SystemExit(
            f"expected duplicate diagnostic {expected!r}; output was:\n"
            f"{completed.stdout}")
    if sha256(archive) != archive_before:
        raise SystemExit("legacy archive checksum changed after rejection")
    if tree_signature(project_dir) != tree_before:
        raise SystemExit("legacy project file tree changed after rejection")
    if manifest.exists():
        raise SystemExit("manifest was committed for a rejected archive")
    temporary_files = list(project_dir.rglob("*.tmp"))
    if temporary_files:
        raise SystemExit(
            "temporary files remained after rejection: " +
            ", ".join(str(path) for path in temporary_files))

    diagnostic = next(
        (line for line in completed.stdout.splitlines()
         if expected in line),
        expected)
    print(f"legacy KS2D symmetry archive: PASS\n{diagnostic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
