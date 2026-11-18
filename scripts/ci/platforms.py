"""Platform contract for directory-owned CI suites (no hardware discovery)."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = {"SERIAL": "cpu", "OMP": "cpu_omp", "CUDA": "cuda", "HIP": "hip"}


def selection(backend=None):
    platform = os.environ.get("PLATFORM")
    if platform is not None:
        if platform not in BACKENDS:
            raise ValueError(f"Unsupported PLATFORM={platform!r}; select {', '.join(BACKENDS)}")
        if backend and backend != BACKENDS[platform]:
            raise ValueError(f"PLATFORM={platform} conflicts with --backend {backend}")
    else:
        platform = next((p for p, b in BACKENDS.items() if b == (backend or "cpu")), None)
        if not platform:
            raise ValueError(f"Unsupported backend: {backend}")
    return platform, BACKENDS[platform]


def overrides(options):
    # GNU Make needs whitespace before '=' for names ending in '+'.
    result = {}
    for option in options:
        key, eq, value = option.partition("=")
        if not eq or not key.replace("_", "").replace("+", "").isalnum() or key in (
                "PLATFORM", "CONFIG_FILE", "BUILD_DIR", "CH_BENCHMARK_OPTIONS"):
            raise ValueError(f"Invalid Make override {option!r}")
        result[key] = value
    for key in ("NVCC", "HIPCC"):
        if os.environ.get(key):
            result.setdefault(key, os.environ[key])
    return [f"{key} = {value}" for key, value in result.items()]


def make(target, directory, config, options):
    return ["make", "--no-print-directory", "-C", str(ROOT), target,
            f"BUILD_DIR={directory}", f"CONFIG_FILE={config}", *overrides(options)]


def probe(platform, directory, config, options):
    if platform not in ("CUDA", "HIP"):
        return
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "probe.log"
    command = [*make("ci-device-probe", directory, config, options), f"PLATFORM={platform}"]
    with log.open("w") as stream:
        try:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Required {platform} compile/link/runtime probe failed; see {log}") from error
