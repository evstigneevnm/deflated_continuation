"""Generic semidiscrete PDE contracts. The split/unsplit CH suite is registered separately."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
MODULE = Path(os.environ.get("NMFD_TIMESTEPPERS_ROOT", ROOT/"source/contrib/nmfd-timesteppers"))
sys.path.insert(0, str(MODULE/"tests"))
import timestepper_ci as suite


def cases(platform, _group):
    for name in ("diffusion", "reaction_diffusion", "cahn_hilliard"):
        yield suite.Case(name, f"test_pde_{name}_{suite.BACKENDS[platform]}.bin",
                         threads=2, timeout=180 if platform in ("SERIAL", "OMP") else 300)


if __name__ == "__main__":
    sys.exit(suite.run_profile(["--build-dir", str(ROOT/"build/pde_ci"), *sys.argv[1:]],
                               ("pde",), cases, ROOT, "ci-device-probe"))
