"""Validate committed NMFD pins in fresh, isolated standalone checkouts."""

import argparse
import configparser
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time


LIBRARIES = ("nmfd-operations", "nmfd-linsolvers", "nmfd-newton")
BUILD_OVERRIDES = {
    "SCFD_ROOT", "NMFD_OPERATIONS_ROOT", "NMFD_LINSOLVERS_ROOT", "CONFIG_FILE", "BUILD_DIR",
    "MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "GNUMAKEFLAGS", "MAKEFILES", "CPPFLAGS",
    "CXXFLAGS", "CFLAGS", "LDFLAGS", "EXTRA_FLAGS", "NVCCFLAGS", "MPI_CXXFLAGS", "MPI_LDFLAGS",
    "INCLUDES", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH",
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
}
TARGETS = {
    "cpu": {
        "nmfd-operations": ["test-cpu", "test-multiprecision", "check-headers"],
        "nmfd-linsolvers": ["run_tests_cpu", "check_public_headers_cpu", "check_multiprecision_cpu",
                            "check_bicgstab_json_cpu", "check_bicgstabl_json_cpu", "check_cgs_json_cpu", "check_jacobi_json_cpu"],
        "nmfd-newton": ["test_cpu", "check_headers_cpu", "check_ch_original"],
    },
    "mpi": {
        "nmfd-operations": ["test-rect-mpi"],
        "nmfd-newton": ["check_ch_mpi"],
    },
    "cuda": {
        "nmfd-operations": ["test-cuda"],
        "nmfd-linsolvers": ["check_operations_cuda", "check_bicgstab_cuda", "check_bicgstabl_cuda",
                            "check_cgs_cuda", "check_jacobi_cuda", "check_sherman_morrison_cuda", "check_direct_solver_cuda"],
        "nmfd-newton": ["check_cuda", "check_headers_cuda"],
    },
}


def clean_environment(environment):
    return {key: value for key, value in environment.items() if key not in BUILD_OVERRIDES}


def git(directory, *arguments, env):
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], env=env, text=True, stderr=subprocess.STDOUT,
    ).strip()


def pinned_revision(repository, path, env):
    entry = git(repository, "ls-tree", "HEAD", "--", path, env=env).split()
    if len(entry) != 4 or entry[:2] != ["160000", "commit"]:
        raise RuntimeError(f"{path}: expected a committed submodule gitlink")
    return entry[2]


def clone_revision(source, revision, destination, env, fallback_url=None):
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", "--no-hardlinks", str(source), str(destination)],
        check=True, env=env,
    )
    present = subprocess.run(
        ["git", "-C", str(destination), "cat-file", "-e", revision + "^{commit}"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not present:
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--quiet", "--no-tags", "--depth=1",
             fallback_url or str(source), revision], check=True, env=env,
        )
    git(destination, "checkout", "--quiet", "--detach", revision, env=env)


def prepare_checkout(root, library, destination, env):
    contrib = root / "source/contrib"
    revision = pinned_revision(root, "source/contrib/" + library, env)
    clone_revision(contrib / library, revision, destination, env)
    modules = configparser.ConfigParser(interpolation=None)
    modules.read(destination / ".gitmodules")
    dependencies = {}
    for section in modules.sections():
        path = Path(modules[section]["path"])
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe dependency path: {path}")
        name = "scfd" if path.name.lower() == "scfd" else path.name
        if name not in (*LIBRARIES, "scfd"):
            raise RuntimeError(f"Unexpected direct dependency: {path}")
        pinned = pinned_revision(destination, path.as_posix(), env)
        clone_revision(contrib / name, pinned, destination / path, env, modules[section]["url"])
        dependencies[path.as_posix()] = pinned
    return {"revision": revision, "dependencies": dependencies}


def make_command(library, compiler, suite, jobs, ranks):
    if library not in TARGETS[suite]:
        raise ValueError(f"{library} has no {suite} validation target")
    if library == "nmfd-linsolvers":
        config = f"build_configs/config_github_ubuntu_{compiler}.inc"
    elif library == "nmfd-newton" and compiler == "clang":
        config = "build_configs/config_clang.inc"
    else:
        config = "build_configs/config_cpu.inc"
    cxx = "g++" if compiler == "gcc" else "clang++"
    return [
        "make", f"-j{jobs}", *TARGETS[suite][library], f"CONFIG_FILE={config}",
        f"CXX={cxx}", f"HOST_CXX={cxx}", "BUILD_DIR=build/validation",
        "OMP_NUM_THREADS=2", "TEST_OMP_NUM_THREADS=2", f"MPI_NP={ranks}",
    ]


def verify_checkout(repository, selected, env):
    for relative, revision in {".": selected["revision"], **selected["dependencies"]}.items():
        path = repository / relative
        if git(path, "rev-parse", "HEAD", env=env) != revision:
            raise RuntimeError(f"{path}: validation changed the selected revision")
        git(path, "diff", "--exit-code", "--ignore-submodules=untracked", "HEAD", env=env)
        git(path, "diff", "--cached", "--exit-code", env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--library", choices=LIBRARIES, required=True)
    parser.add_argument("--compiler", choices=("gcc", "clang"), default="gcc")
    parser.add_argument("--suite", choices=TARGETS, default="cpu")
    parser.add_argument("--work-dir", type=Path, default=Path("build/nmfd_standalone"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--mpi-ranks", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("jobs and timeout must be positive")
    if args.library not in TARGETS[args.suite]:
        parser.error(f"{args.library} has no {args.suite} validation target")

    env = clean_environment(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f"{args.library}-{args.compiler}-{args.suite}-", dir=args.work_dir)).resolve()
    repository = workspace / "repository"
    report = {"library": args.library, "compiler": args.compiler, "suite": args.suite,
              "mpi_ranks": args.mpi_ranks if args.suite == "mpi" else None,
              "parent_revision": git(args.root, "rev-parse", "HEAD", env=env), "status": "running"}
    started = time.monotonic()
    print(f"Standalone workspace: {workspace}", flush=True)
    try:
        report.update(prepare_checkout(args.root.resolve(), args.library, repository, env))
        print(json.dumps(report, indent=2), flush=True)
        command = make_command(args.library, args.compiler, args.suite, args.jobs, args.mpi_ranks)
        report["command"] = command
        (workspace / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=repository, env=env, check=True, timeout=args.timeout)
        verify_checkout(repository, report, env)
        report["status"] = "passed"
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (workspace / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Standalone {args.library}/{args.compiler}/{args.suite}: {report['status'].upper()}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
