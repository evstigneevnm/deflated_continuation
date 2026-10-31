# KS2D Linear-Solver Regressions

These fixtures were generated with the pre-migration binaries from parent
commit `29650b9`, using the project BiCGStab(L) and Sherman-Morrison solvers.
They exercise the existing inversion-odd formulation, not unrestricted KS2D.

`symmetric_lambda4_333.dat` is the first saved nonzero state at lambda=4.333.
The grid is 12x12, a=2, b=4, with 70 packed real coefficients. The generated
finite symmetry group has order eight.

`nmfd_continuation_reference.dat` contains parameter, normalized L2 norm,
and the first two reported components for all 18 saved continuation points.
All points have stability signature (0, 0). The configuration is generated
from `KS2D_test_sym.json` using `scripts/write_ks2d_ci_config.py` with:

```text
--nx 12 --ny 12 --steps 8 --step-size 0.02 --max-step-size 0.03
--seed-parameter 4.333 --parameter-min 4.1 --parameter-max 4.5
--deflation-attempts 1 --newton-deflation-iterations 80
--newton-deflation-weight 0.5 --newton-deflation-tolerance 1e-8
--linear-solver-iterations 150 --linear-solver-tolerance 1e-3
--max-failed-continuations-per-knot 1
```

The replay verifies nonzero deflation and continuation, symmetry manifests,
stable classifications, and byte-identical stability restart. A separate
confirmed state replay disables the dense fallback and exercises the original
matrix-free scans with paired axis-swap probes. Unit tests
use the saved state for true-residual, Newton, and symmetry-orbit checks.
Full-resolution reference archives remain under `data/reference/KS2D`.

The original fixed-parameter correction from lambda=4.333 to 4.334 converges
with residual below 1e-10. Exhausting the primary iteration budget at zero
iterations also converges through the existing GMRES fallback. The zero-branch
transition from 3.9 to 4.1 changes (0, 0) to (2, 0), refined to
4.0000976562500004 using the unchanged default transition tolerance.

## Focused Reference Discovery

`test_KS2D_reference_discovery_cpu_omp.bin` and
`test_KS2D_reference_discovery_cuda.bin` accept three arguments:
`config.json requests.json report.json`. The configuration selects an isolated
copy of the known-branch project and retains its production numerical settings.
The test reconstructs all intersections at each requested parameter, then solves
from radially perturbed reference states using the ordinary NMFD deflation path.
Example request:

```json
{
  "orbit_tolerance": 1e-7,
  "cases": [
    {
      "id": "reference_10",
      "parameter": 17,
      "state": "/absolute/path/to/reference/10/1",
      "perturbations": [0, -1e-4, 1e-4]
    }
  ]
}
```

A novel-root case must recover the specified orbit from a nonzero perturbation;
accepting only the exact state is insufficient. Set `"known": true` for an
existing-root control, with a zero perturbation to check exclusion. Successful
solves must satisfy physical and deflated tolerances, separation, and unchanged
parameters. Failed solves must preserve their input states. JSON reports include
orbit distances and convergence statuses; nonfinite numerical fields serialize
as `null`. The test exits unsuccessfully if a case fails.

The large archives and local discovery/short-continuation launch scripts are
not CI fixtures. Keep their inputs, reports, and isolated projects under `build`.
