"""Directory CI profile. Keep the interactive benchmark defaults in ch_benchmarks.py."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import ch_benchmarks as bench
from ch_benchmarks import platforms

CONTRACTS = ("cahn_hilliard_imex", "cahn_hilliard_energy", "cahn_hilliard_imex_lifecycle", "cahn_hilliard_unsplit")
SANITIZER_FLAGS = "-std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"


def benchmarks(platform):
    # Keep these configurations explicit: adding a supported method is not itself
    # approval to change the coverage or cost of the hosted validation suite.
    fixed = ["IMEX_EULER", "IE", "CN", "SDIRK2(1)2", "BDF2", "ROS3", "ARK324L2SA",
             "ROS34PW2", "BDF", "IMEX_ARK324L2SA"]
    adaptive = ["ARK324L2SA", "ROS34PW2", "IMEX_ARK324L2SA"]
    short = ["--end", ".4", "--step", ".1", "--frames", "3", "--base-steps", "5"]
    yield "fd_methods", ["--grid", "16", *short, "--levels", "4", "--threads", "2", "--methods", *fixed]
    yield "fd_adaptive", ["--grid", "16", *short, "--levels", "4", "--threads", "2",
                          "--methods", *adaptive, "BDF", "--adaptive-methods", *adaptive]
    if platform == "OMP":
        yield "fourier_long_euler", ["--spatial", "fourier", "--grid", "32", "--end", "60", "--step", ".1",
                                     "--frames", "3", "--threads", "1", "--methods", "IE", "IMEX_EULER"]
    else:
        yield "fourier_methods", ["--spatial", "fourier", "--grid", "12", *short, "--levels", "3",
                                  "--methods", "IMEX_EULER", "CN", "BDF", "ROS34PW2"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "contracts", "sanitizers", "benchmarks"), default="all")
    parser.add_argument("--backend", choices=tuple(platforms.BACKENDS.values()))
    parser.add_argument("--build-dir", type=Path, default=bench.ROOT/"build/ch_ci")
    parser.add_argument("--config-file", type=Path, default=bench.ROOT/"build_configs/config_github_ubuntu_cpu.inc")
    parser.add_argument("--make-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-B", action="store_true", dest="build_only", help="Force builds only; no device probe or tests")
    parser.add_argument("--list", action="store_true", help="List selected cases without building or probing")
    args = parser.parse_args(argv)
    try:
        platform, backend = platforms.selection(args.backend)
        platforms.overrides(args.make_option)
    except ValueError as error:
        parser.error(str(error))
    if args.group == "sanitizers" and platform != "SERIAL":
        parser.error("The registered CH sanitizer configuration is SERIAL")

    config = args.config_file.resolve()
    # The caller supplies a separate root per compiler; keep incompatible
    # platforms and sanitizer builds separate even within that root.
    directory = args.build_dir.resolve()/platform/args.group
    options = args.make_option
    if args.group == "sanitizers":
        options = [*options, "G++FLAGS="+SANITIZER_FLAGS]
    env = dict(os.environ, PLATFORM=platform, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
    if args.group == "sanitizers":
        env["ASAN_OPTIONS"] = "detect_leaks=1"
    cases = {}
    if args.group != "benchmarks":
        for name in CONTRACTS:
            binary = f"test_pde_{name}_{backend}.bin"
            build = platforms.make(binary, directory/"bin", config, options)
            if args.build_only:
                build.insert(1, "-B")
            cases[name] = [build] if args.build_only else [build, [str(directory/"bin"/binary)]]
    if args.group in ("all", "benchmarks"):
        for name, flags in benchmarks(platform):
            cmd = [sys.executable, str(bench.HERE/"ch_benchmarks.py"), *flags,
                   "--config-file", str(config), "--build-dir", str(directory/"figures")]
            for option in options:
                cmd.extend(("--make-option", option))
            cases[name] = [[*cmd, "-B"] if args.build_only else cmd]
    if args.list:
        print("\n".join(cases))
        return 0

    directory.mkdir(parents=True, exist_ok=True)
    report = dict(platform=platform, group=args.group, build_only=args.build_only, passed=False,
                  selected=list(cases), results={})
    def save():
        bench.write_if_changed(directory/"summary.json", report)

    save()
    if not args.build_only and args.group != "benchmarks":
        try:
            platforms.probe(platform, directory/"probe", config, options)
        except RuntimeError as error:
            report["error"] = str(error)
            save()
            print(error, file=sys.stderr)
            return 1
    for name, commands in cases.items():
        log = directory/(name+".log")
        with log.open("w") as stream:
            try:
                for cmd in commands:
                    subprocess.run(cmd, cwd=bench.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   check=True, timeout=1800 if name not in CONTRACTS else 300)
                status = "built" if args.build_only else "passed"
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                print(error, file=stream)
                status = "failed"
        report["results"][name] = status
        save()
        print(f"{platform} / {name}: {status} ({log})", flush=True)
    report["passed"] = "failed" not in report["results"].values()
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
