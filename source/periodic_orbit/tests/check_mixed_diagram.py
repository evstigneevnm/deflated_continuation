"""Assemble a portable scalar-only plot bundle and check the rendered series."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "plot_scripts"))
import mixed_bd


def close(actual, expected, tolerance, description):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{description}: plotted {actual}, expected {expected} +/- {tolerance}")


def check(case, run, bundle):
    plt = mixed_bd.steady.import_pyplot(False)
    shutil.copytree(HERE / "fixtures" / "mixed_bd" / case, bundle)
    manifest = bundle / "manifest.json"
    paths = json.loads(manifest.read_text())["periodic"]
    for path in paths:
        (bundle / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / path, bundle / path)

    fig = mixed_bd.draw(manifest, plt, envelope=True)
    height, period = fig.axes
    upper = {line.get_gid(): line for line in height.lines}
    lower = {line.get_gid(): line for line in period.lines}
    bounds = (.95, 1.05) if case == "analytic-cycle" else (67.999, 68.001)
    stationary = [line for key, line in upper.items() if key.startswith("stationary:")]
    if len(stationary) != 2 or any(y != 0 for line in stationary for y in line.get_ydata()):
        raise ValueError("Expected both arms of the stationary zero branch")
    if sorted(line.get_xdata()[-1] for line in stationary) != list(bounds):
        raise ValueError("Stationary branch does not reach both reference endpoints")

    # KS periods come from the independent fixed-layout continuation reference.
    ks_periods = (.01515307799761766, .01515430221001677)
    common = {}
    for path in paths:
        archive = json.loads((bundle / path).read_text())
        if not archive["trace_complete"] or not archive["all_points_verified"] or len(archive["segments"]) != 2:
            raise ValueError(f"Expected two complete, verified arms in {path}")
        endpoints = set()
        for arm in archive["segments"]:
            key = f"{archive['id']}:{arm['id']}"
            rms, time = upper[f"rms:{key}"], lower[f"period:{key}"]
            x = list(rms.get_xdata())
            if x != list(time.get_xdata()) or x != [p["parameter"] for p in arm["points"]]:
                raise ValueError(f"Traversal order differs in {key}")
            if list(rms.get_ydata()) != [p["observable_rms"] for p in arm["points"]] or (
                list(time.get_ydata()) != [p["period"] for p in arm["points"]]
            ):
                raise ValueError(f"Wrong physical series in {key}")
            if case == "analytic-cycle":
                for lam, q, t in zip(x, rms.get_ydata(), time.get_ydata()):
                    close(q, math.sqrt(lam), 1e-8, "cycle radius")
                    close(t, 2 * math.pi / (1 + (lam - 1) / 4), 1e-7, "cycle period")
            bound = bounds.index(x[-1])
            endpoints.add((arm["endpoint"], bound))
            if case == "ks1d":
                close(time.get_ydata()[-1], ks_periods[bound], 1e-9, "KS period")
            q = rms.get_ydata()[-1]
            close(q, common.setdefault(bound, q), 1e-7, "section-count/restart agreement")
        if endpoints != {("boundary_min", 0), ("boundary_max", 1)}:
            raise ValueError(f"Missing boundary endpoint in {path}")
    if len(lower) != 2 * len(paths) or len(height.collections) != len(lower):
        raise ValueError("Missing sampled envelopes or unexpected period series")
    output = bundle / "mixed_bd.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Verified {case} diagram: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("analytic-cycle", "ks1d"))
    parser.add_argument("run", type=Path)
    parser.add_argument("new_bundle", type=Path)
    args = parser.parse_args()
    check(args.case, args.run, args.new_bundle)
