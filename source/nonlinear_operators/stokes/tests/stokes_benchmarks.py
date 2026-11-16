#!/usr/bin/env python3
"""Build, validate and plot manufactured periodic Stokes/Oseen solutions."""

import argparse
import cmath
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
METHODS = {"IE": 1, "IM": 2, "CN": 2, "SDIRK2(1)2": 2, "ESDIRK2(1)3": 2,
           "SDIRK3(1)3": 3, "ARK324L2SA": 3, "IMEX_EULER": 1, "IMEX_HEUN_TR2": 2,
           "IMEX_ARS233": 3, "IMEX_ARS222": 2, "IMEX_ARK324L2SA": 3}
MODE_KEYS = {(0, 0), (1, 1), (-1, 1), (3, 2), (-3, 2)}
BUILD_ENVIRONMENT = ("PATH", "G++", "G++FLAGS", "GCC_ROOT_PATH", "CPPSTD", "NVCC", "NVCCFLAGS",
                     "CUDA_ROOT_PATH", "CUDA_ARCH_LIST", "HIPCC", "HIPFLAGS", "HIP_ROOT_PATH", "HIP_PATH",
                     "HIP_ARCH_LIST", "NMFD_TIMESTEPPERS_ROOT", "CONTRIB_NMFD_OPERATIONS", "CONTRIB_SCFD")


def family(method):
    return "imex" if method.startswith("IMEX_") else "dirk"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-B", dest="build_only", action="store_true", help="Force build and configuration assembly only")
    mode.add_argument("--plot-only", type=Path, help="Revalidate and replot a saved run without integrating")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/stokes_benchmarks")
    parser.add_argument("--systems", nargs="+", choices=("stokes", "oseen"), default=["stokes", "oseen"])
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--backend", choices=("cpu", "cpu_omp", "cuda", "hip"), default="cpu")
    parser.add_argument("--start", type=float, default=.3)
    parser.add_argument("--end", type=float, default=1.3)
    parser.add_argument("--base-steps", type=int, default=16)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--plot-grid", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--config-file", type=Path, default=ROOT / "build_configs/config_github_ubuntu_cpu.inc")
    parser.add_argument("--make-option", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.start) or not math.isfinite(args.end) or args.end <= args.start or
            not math.isfinite(args.timeout) or args.timeout <= 0 or args.threads < 1 or
            not 8 <= args.grid <= 512 or args.grid % 2 or not 16 <= args.plot_grid <= 256 or
            not 4 <= args.base_steps <= 512 or args.base_steps % 2 or not 3 <= args.levels <= 8):
        parser.error("Invalid time interval, grid, refinement or execution budget")
    finest = (args.end - args.start) / (args.base_steps * 2**(args.levels - 1))
    if not math.isfinite(finest) or args.start + finest == args.start:
        parser.error("The finest step cannot advance the initial time")
    for option in args.make_option:
        key, equal, _ = option.partition("=")
        if (not equal or key in ("BUILD_DIR", "CONFIG_FILE", "STOKES_BENCHMARK_OPTIONS") or
                not key.replace("_", "").replace("+", "").isalnum()):
            parser.error("Use --make-option KEY=VALUE; build paths have separate options")
    args.methods = list(dict.fromkeys(args.methods))
    args.systems = list(dict.fromkeys(args.systems))
    args.build_dir = args.build_dir.resolve()
    args.config_file = args.config_file.resolve()
    if not configurations(args):
        parser.error("The Oseen validation requires an IMEX method; DIRK cases cover Stokes")
    return args


def configurations(args):
    return [dict(system=system, method=method, backend=args.backend, start=args.start, end=args.end,
                 grid=args.grid, base_steps=args.base_steps, levels=args.levels)
            for system in args.systems for method in args.methods if system == "stokes" or family(method) == "imex"]


def case_name(config):
    return f"{config['system']}_{config['method']}"


def write_if_changed(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def build(args, configs):
    binary_dir = args.build_dir / "bin"
    options = binary_dir / f"{args.backend}_build_options.json"
    write_if_changed(options, dict(root=str(ROOT), configuration=str(args.config_file),
        configuration_text=args.config_file.read_text(), options=args.make_option,
        environment={name: os.environ.get(name) for name in BUILD_ENVIRONMENT}))
    target = f"stokes_benchmark_{args.backend}.bin"
    command = ["make", "--no-print-directory", "-C", str(ROOT)]
    if args.build_only:
        command.append("-B")
    command.extend([target, f"BUILD_DIR={binary_dir}", f"CONFIG_FILE={args.config_file}",
                    f"STOKES_BENCHMARK_OPTIONS={options}",
                    *(" = ".join(option.split("=", 1)) for option in args.make_option)])
    log_path = args.build_dir / f"build_{args.backend}.log"
    with log_path.open("w") as log:
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Build failed; see {log_path}") from error
    for config in configs:
        write_if_changed(args.build_dir / "cases" / case_name(config) / "config.json", config)
    return binary_dir / target


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def exact_coefficient(kx, ky, t, system):
    if (kx, ky) == (0, 0):
        return complex(.125), complex(-.25)
    amplitude = .25 * math.exp(-t) if ky == 1 else .03 * math.exp(-.7*t)
    phase = cmath.exp(-1j * ((.7*kx - .3*ky) if system == "oseen" else 0) * t)
    return 1j * ky * amplitude * phase, -1j * kx * amplitude * phase


def snapshot_error(frame, config):
    keys = set()
    error_sq = frame["omitted_rms"]**2
    divergence_sq = 0
    if not finite_nonnegative(frame["omitted_rms"]) or frame["omitted_rms"] > 1e-11:
        raise ValueError("Unexported numerical modes exceed the export bound")
    for mode in frame["coefficients"]:
        key = (mode["kx"], mode["ky"])
        values = mode["velocity"]
        if key in keys or key not in MODE_KEYS or len(values) != 4 or not all(math.isfinite(x) for x in values):
            raise ValueError("Invalid Fourier snapshot")
        keys.add(key)
        reference = exact_coefficient(*key, frame["time"], config["system"])
        numerical = (complex(*values[:2]), complex(*values[2:]))
        divergence_sq += (1 if key == (0, 0) else 2) * abs(key[0]*numerical[0]+key[1]*numerical[1])**2
        error_sq += (1 if key == (0, 0) else 2) * sum(abs(a-b)**2 for a, b in zip(numerical, reference))
        if key == (0, 0) and any(abs(a-b) > 1e-12 for a, b in zip(numerical, reference)):
            raise ValueError("Mean velocity was not preserved")
    if keys != MODE_KEYS:
        raise ValueError("Incomplete Fourier snapshot")
    if math.sqrt(divergence_sq) > 1e-11:
        raise ValueError("Exported velocity is not divergence-free")
    return math.sqrt(error_sq)


def measured_order(rows, floor):
    eligible = [r for r in rows if r["error_rms"] > floor]
    if len(eligible) < 3:
        raise ValueError("Insufficient refinement points above the numerical floor")
    selected = eligible[-3:]
    x = [math.log(r["step"]) for r in selected]
    y = [math.log(r["error_rms"]) for r in selected]
    mx, my = sum(x)/len(x), sum(y)/len(y)
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / sum((a-mx)**2 for a in x)


def validate_report(report, config):
    if (config["method"] not in METHODS or config["system"] not in ("stokes", "oseen") or
            (config["system"] == "oseen" and family(config["method"]) != "imex")):
        raise ValueError("Unsupported system/method combination")
    expected = METHODS[config["method"]]
    if (report.get("format") != "stokes.timestepper.validation.v1" or report.get("configuration") != config or
            report.get("passed") is not True or report.get("norm") != "physical_velocity_rms" or
            report.get("expected_order") != expected or report.get("pressure_reconstructed") is not False or
            report.get("bulk_host_transfer_bytes") != 0 or report.get("exported_scalar_values") != 60 or
            report.get("fit_error_floor") != 1e-11):
        raise ValueError("Invalid, failed or incompatible benchmark report")
    rows = report["refinements"]
    if len(rows) != config["levels"]:
        raise ValueError("Incomplete refinement sweep")
    for level, row in enumerate(rows):
        count = config["base_steps"] * 2**level
        if (row["steps"] != count or not math.isclose(row["step"], (config["end"]-config["start"])/count, rel_tol=1e-13) or
                not finite_nonnegative(row["error_rms"]) or row["error_rms"] == 0 or
                not finite_nonnegative(row["maximum_divergence_rms"]) or row["maximum_divergence_rms"] > 1e-11 or
                row["backend_allocations_after_warmup"] != 0):
            raise ValueError("Invalid refinement, physical error, divergence or allocation data")
    for a, b in zip(rows, rows[1:]):
        if b["error_rms"] > report["fit_error_floor"] and b["error_rms"] >= a["error_rms"]:
            raise ValueError("Physical errors do not decrease under refinement")
    observed = measured_order(rows, report["fit_error_floor"])
    if observed < expected - .4:
        raise ValueError(f"Observed order {observed:.3f} is below expected order {expected}")
    history, frames = report["history"], report["snapshots"]
    if len(history) != rows[-1]["steps"] or len(frames) != 3:
        raise ValueError("Incomplete finest trajectory or solution snapshots")
    for index, row in enumerate(history):
        t = config["start"] + (index+1)*rows[-1]["step"]
        if (not math.isclose(row["time"], t, abs_tol=1e-12, rel_tol=1e-13) or
                not finite_nonnegative(row["error_rms"]) or not finite_nonnegative(row["divergence_rms"]) or
                row["divergence_rms"] > 1e-11):
            raise ValueError("Invalid physical trajectory diagnostics")
    for index, frame in enumerate(frames):
        t = config["start"] + index*(config["end"]-config["start"])/2
        if not math.isfinite(frame["time"]) or not math.isclose(frame["time"], t, abs_tol=1e-12, rel_tol=1e-13):
            raise ValueError("Incorrect snapshot time")
        error = snapshot_error(frame, config)
        logged = 0 if index == 0 else history[(len(history)//2 if index == 1 else len(history))-1]["error_rms"]
        if not math.isclose(error, logged, abs_tol=1e-13, rel_tol=1e-6):
            raise ValueError("Exported velocity and physical RMS error disagree")
    if not math.isclose(history[-1]["error_rms"], rows[-1]["error_rms"], abs_tol=1e-14, rel_tol=1e-8):
        raise ValueError("Refinement endpoint and trajectory disagree")
    return dict(expected_order=expected, observed_order=observed,
                finest_step=rows[-1]["step"], finest_error_rms=rows[-1]["error_rms"])


def velocity_fields(frame, system, points):
    import numpy as np
    centers = (np.arange(points)+.5)*2*np.pi/points
    x, y = np.meshgrid(centers, centers, indexing="ij")
    numerical = [np.zeros_like(x), np.zeros_like(x)]
    for mode in frame["coefficients"]:
        phase = np.exp(1j*(mode["kx"]*x+mode["ky"]*y))
        weight = 1 if mode["ky"] == 0 else 2
        for component in range(2):
            value = complex(*mode["velocity"][2*component:2*component+2])
            numerical[component] += weight*(value*phase).real
    t = frame["time"]
    X, Y = (x-.7*t, y+.3*t) if system == "oseen" else (x, y)
    reference = [-np.exp(-t)*np.cos(X)*np.sin(Y)-.24*np.exp(-.7*t)*np.cos(3*X)*np.sin(2*Y)+.125,
                 np.exp(-t)*np.sin(X)*np.cos(Y)+.36*np.exp(-.7*t)*np.sin(3*X)*np.cos(2*Y)-.25]
    return x, y, reference, numerical


def save_figure(fig, path):
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("."+extension), dpi=160)


def plot_solutions(directory, report, points):
    import numpy as np
    import matplotlib.pyplot as plt
    config = report["configuration"]
    fields = [velocity_fields(frame, config["system"], points) for frame in report["snapshots"]]
    vmax = max(float(np.hypot(*data[2]).max()) for data in fields)
    errors = [np.hypot(data[3][0]-data[2][0], data[3][1]-data[2][1]) for data in fields]
    emax = max(1e-15, max(float(e.max()) for e in errors))
    edges = np.linspace(0, 2*np.pi, points+1)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, ((x, y, reference, numerical), error, frame) in enumerate(zip(fields, errors, report["snapshots"])):
        for col, value in enumerate((np.hypot(*reference), np.hypot(*numerical), error)):
            ax = axes[row, col]
            im = ax.pcolormesh(edges, edges, value.T, shading="flat", cmap="magma" if col == 2 else "viridis",
                               vmin=0, vmax=emax if col == 2 else vmax, rasterized=True)
            if col < 2:
                stride = max(1, points//12)
                sample = np.s_[::stride, ::stride]
                u, v = reference if col == 0 else numerical
                ax.quiver(x[sample], y[sample], u[sample], v[sample], color="white", scale=20*vmax, alpha=.85)
            ax.set(xlim=(0, 2*np.pi), ylim=(0, 2*np.pi), aspect="equal", xlabel="x", ylabel=f"t={frame['time']:.3f}\ny" if col == 0 else "y")
            if row == 0:
                ax.set_title(("Manufactured velocity |u|", "Numerical velocity |u|", "Pointwise velocity error")[col])
    fig.colorbar(axes[0, 0].collections[0], ax=axes[:, :2].ravel().tolist(), shrink=.8, label="Speed")
    fig.colorbar(im, ax=axes[:, 2].ravel().tolist(), shrink=.8, label="Absolute velocity error", format="%.1e")
    fig.suptitle(f"Periodic {config['system'].title()} / {config['method']} / h={report['refinements'][-1]['step']:.5g}")
    fig.supxlabel(f"Five exported Fourier modes; maximum omitted-tail RMS = {max(f['omitted_rms'] for f in report['snapshots']):.1e}", fontsize=9)
    save_figure(fig, directory / "solutions")
    plt.close(fig)


def reference_line(steps, anchor_step, anchor_error, order):
    return [anchor_error*(h/anchor_step)**order for h in steps]


def plot_orders(directory, reports):
    import numpy as np
    import matplotlib.pyplot as plt
    groups = {}
    for config, report, metrics in reports:
        groups.setdefault((config["system"], family(config["method"])), []).append((config, report, metrics))
    for (system, engine), cases in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.2, 1]})
        guides = {}
        for index, (config, report, metrics) in enumerate(cases):
            rows = report["refinements"]
            h, error = [r["step"] for r in rows], [r["error_rms"] for r in rows]
            marker = ("o", "s", "^", "D", "v", "P", "X")[index % 7]
            line, = axes[0].loglog(h, error, marker=marker, linestyle="-", markersize=4,
                label=f"{config['method']} (fit {metrics['observed_order']:.2f})")
            local = np.log(np.array(error[:-1])/error[1:])/np.log(np.array(h[:-1])/h[1:])
            above_floor = (np.array(error[:-1]) > report["fit_error_floor"]) & (np.array(error[1:]) > report["fit_error_floor"])
            axes[1].semilogx(np.sqrt(np.array(h[:-1])*h[1:])[above_floor], local[above_floor], marker=marker,
                             linestyle="-", color=line.get_color(), markersize=4)
            order = metrics["expected_order"]
            anchor = [r for r in rows if r["error_rms"] > report["fit_error_floor"]][-1]
            guides.setdefault(order, (h, anchor))
            if any(e <= report["fit_error_floor"] for e in error):
                axes[0].axhline(report["fit_error_floor"], color=".7", linestyle=":", linewidth=1)
        for order, (h, anchor) in sorted(guides.items()):
            axes[0].loglog(h, reference_line(h, anchor["step"], anchor["error_rms"]*.55, order), "--", color=".35", linewidth=1,
                          label=f"Reference h^{order}")
            axes[1].axhline(order, color=".5", linestyle="--", linewidth=1)
        for ax in axes:
            ax.set_xlabel("Timestep h")
            ax.grid(True, which="both", alpha=.2)
        axes[0].set_ylabel("Endpoint velocity RMS error")
        axes[1].set_ylabel("Pairwise observed order")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        fig.suptitle(f"Periodic {system.title()} / {engine.upper()} / fixed-step temporal refinement")
        fig.supxlabel("Dashed reference lines show slopes only; their vertical offsets are arbitrary.", fontsize=9)
        save_figure(fig, directory / f"order_{system}_{engine}")
        plt.close(fig)


def summarize(directory, points=64):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "stokes.benchmark.manifest.v1" or not manifest.get("cases"):
        raise ValueError("Invalid or empty Stokes benchmark manifest")
    names = [case_name(c["configuration"]) for c in manifest["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate Stokes benchmark cases")
    valid, summary = [], []
    for case in manifest["cases"]:
        config = case["configuration"]
        path = directory / case_name(config)
        try:
            if case["returncode"] != 0:
                raise ValueError(f"Executable returned {case['returncode']}; see {path / 'run.log'}")
            report = json.loads((path / "report.json").read_text())
            metrics = validate_report(report, config)
            plot_solutions(path, report, points)
            valid.append((config, report, metrics))
            summary.append(dict(case=case_name(config), passed=True, **metrics))
            print(f"PASS {case_name(config)}: observed order {metrics['observed_order']:.3f}, expected {metrics['expected_order']}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            summary.append(dict(case=case_name(config), passed=False, error=str(error)))
            print(f"FAIL {case_name(config)}: {error}", file=sys.stderr)
    if valid:
        plot_orders(directory, valid)
    write_if_changed(directory / "validation.json", summary)
    with (directory / "order_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "expected_order", "observed_order", "finest_step", "finest_error_rms"))
        writer.writeheader()
        for row in summary:
            if row["passed"]:
                writer.writerow({key: row[key] for key in writer.fieldnames})
    return 0 if summary and all(row["passed"] for row in summary) else 1


def ensure_plotting():
    try:
        import numpy
        import matplotlib
    except ImportError as error:
        raise RuntimeError("NumPy and Matplotlib are required; select an installed interpreter with PYTHON") from error


def main(argv=None):
    args = arguments(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(args.build_dir / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    if not args.build_only:
        ensure_plotting()
    if args.plot_only:
        return summarize(args.plot_only.resolve(), args.plot_grid)
    configs = configurations(args)
    executable = build(args, configs)
    if args.build_only:
        print(f"Built and assembled {len(configs)} cases in {args.build_dir}; no integrations or plots executed")
        return 0
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns()}"
    directory = args.build_dir / "runs" / batch
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(format="stokes.benchmark.manifest.v1", executable=str(executable), cases=[])
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
    for config in configs:
        path = directory / case_name(config)
        path.mkdir()
        write_if_changed(path / "config.json", config)
        before = time.perf_counter()
        with (path / "run.log").open("w") as log:
            try:
                code = subprocess.run([str(executable), str(path / "config.json"), str(path / "report.json")],
                                      stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                log.write("\nBenchmark timeout reached\n")
                code = 124
        manifest["cases"].append(dict(configuration=config, returncode=code, seconds=time.perf_counter()-before))
        write_if_changed(directory / "manifest.json", manifest)
    result = summarize(directory, args.plot_grid)
    print(f"Results: {directory}")
    return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
