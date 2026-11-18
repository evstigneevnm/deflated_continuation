"""Run the module's projection contracts together with the parent Stokes integrations."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
MODULE = Path(os.environ.get("NMFD_TIMESTEPPERS_ROOT", ROOT/"source/contrib/nmfd-timesteppers"))
sys.path.insert(0, str(MODULE/"tests"))
import timestepper_ci as suite


def cases(platform, group):
    yield from suite.cases(platform, group)
    for name in ("periodic_stokes_projection", "periodic_stokes_imex_projection",
                 "periodic_stokes_bdf_projection", "periodic_stokes_rosenbrock_projection"):
        yield suite.projection_case(platform, name)


if __name__ == "__main__":
    sys.exit(suite.run_profile(sys.argv[1:], ("projection",), cases, ROOT, "ci-device-probe"))
