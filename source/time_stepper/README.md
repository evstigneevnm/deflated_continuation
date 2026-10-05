# Time Stepping

`legacy/` is frozen. New code lives in `nmfd::time_steppers` and does not include
legacy headers.

Vector ownership uses `nmfd::detail::vector_wrap` from `nmfd-newton`.
Add `source/contrib/nmfd-newton/include` to the include path when building outside
the supplied Makefile. Scratch vectors and RK stages are allocated once and reused.

## Explicit RK

- `runge_kutta/butcher_tables.h`: EE, HE, RK33SSP, RK43SSP, RK64SSP, DOPRI54,
  plus separately tested implicit/IMEX coefficient tables.
- `runge_kutta/explicit_time_step.h`: stage evaluation and bounded retry loop.
- `integration/time_step_adaptation_constant.h`: fixed proposed step magnitude.
- `integration/time_step_adaptation_matlab.h`: component-scaled embedded error
  control, first-rejection formula, repeated halving, and no growth after rejection.
- `integration/time_integrator.h`: interval traversal, external management, commit.

An explicit problem supplies `apply(in, rate)` and optionally `set_time(t)`.
The rate is already mass-inverted/projected when the formulation requires it.
There is no implicit-residual reinterpretation and no general singular-mass solve.

The single-step method returns a pending candidate. The integrator calls
`finalize()` after external processing; failed candidates never replace the
committed output. `apply()` on the integrator starts a fresh trajectory.
External managers own their own reset policy and receive the actual step interval.
Vector operations, problem, adaptation, step, and external manager must outlive
their non-owning consumers. Do not share mutable instances between trajectories.

Primary and embedded orders are separate. The error vector includes the step
size, and `error_order()` is its scaling exponent. `SDIRK3(1)3` retains the legacy
embedded weights, correctly identified as first order. RK64SSP retains published
decimal precision, with a table-specific validation tolerance. Missing stage
times are derived from row sums. Autonomous behavior belongs to the problem.

Table names:
- `DOPRI54` is Dormand-Prince 5(4), stored with seven stages.
- `SDIRK2(1)2`, `ESDIRK2(1)3`, and `SDIRK3(1)3` use `p(q)s`: primary
  order, embedded order, total stage count. `IE`, `IM`, and `CN` are unchanged.
- IMEX names are `IMEX_EULER`, `IMEX_HEUN_TR2` (Heun plus trapezoidal),
  `IMEX_ARS233`, and `IMEX_ARS222`. ARS digits count implicit stages, explicit
  stages, and combined order as in the original paper, not padded table dimensions.
The previous names are retained only by the frozen legacy implementation.

Coefficient sources: [SSP embedded pairs](https://imrefekete.web.elte.hu/files/2022_FCS_JCAM.pdf),
[Dormand-Prince method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html).
Rooted-tree tests check nonlinear order, not just a scalar stability polynomial.

This pass does not implement implicit/IMEX stepping, dense output, or FSAL reuse.
Adaptive initialization uses the configured initial step and bounded rejection,
not an extra initial derivative evaluation.

## Tests

Each Lorenz, Rossler, and Van der Pol test contains its own problem and assembly.
They use the legacy equations/initial values, fixed-step refinement, and adaptive
endpoint references computed using SciPy DOP853 and independently checked with
Radau (`rtol=3e-14`, `atol=1e-14`). SciPy is not needed to run the C++ tests.
The Lorenz problem includes the legacy cubic damping and asymmetry parameters.

```bash
make test_rk_tables_explicit.bin test_rk_tables_implicit.bin \
     test_explicit_rk_cpu_omp.bin test_explicit_lorenz_cpu_omp.bin \
     test_explicit_rossler_cpu_omp.bin test_explicit_vdp_cpu_omp.bin \
     CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_cpu_omp.bin
```

Use `_cpu.bin` for serial or `_cuda.bin` with a CUDA build configuration.
The workflow has separate CPU jobs for explicit tables, implicit tables, RK
lifecycle, and each ODE. CUDA tests are capability-gated.
