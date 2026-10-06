# Time Stepping

`legacy/` is frozen. New code lives in `nmfd::time_steppers` and does not include
legacy headers.

Vector ownership uses `nmfd::detail::vector_wrap` from `nmfd-newton`.
Add `source/contrib/nmfd-newton/include` to the include path when building outside
the supplied Makefile. Scratch vectors and RK stages are allocated once and reused.

## Explicit RK

- `runge_kutta/butcher_tables.h`: EE, HE, BS32, RK33SSP, RK43SSP, RK64SSP, DOPRI54,
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
- `BS32` is Bogacki-Shampine 3(2), stored with four derivatives including the
  endpoint derivative. Its embedded state error scales as `h^3`.
- `DOPRI54` is Dormand-Prince 5(4), stored with seven stages.
- `SDIRK2(1)2`, `ESDIRK2(1)3`, and `SDIRK3(1)3` use `p(q)s`: primary
  order, embedded order, total stage count. `IE`, `IM`, and `CN` are unchanged.
- IMEX names are `IMEX_EULER`, `IMEX_HEUN_TR2` (Heun plus trapezoidal),
  `IMEX_ARS233`, and `IMEX_ARS222`. ARS digits count implicit stages, explicit
  stages, and combined order as in the original paper, not padded table dimensions.
The previous names are retained only by the frozen legacy implementation.

Coefficient sources: [SSP embedded pairs](https://imrefekete.web.elte.hu/files/2022_FCS_JCAM.pdf),
[Bogacki-Shampine method](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_ivp/rk.py),
[Dormand-Prince method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html).
Rooted-tree tests check nonlinear order, not just a scalar stability polynomial.

This pass does not implement implicit/IMEX stepping or FSAL reuse.
Adaptive initialization uses the configured initial step and bounded rejection,
not an extra initial derivative evaluation.

## Dense Output

Enable interpolation using the fourth template argument; the default remains off:

```cpp
using step_type = nmfd::time_steppers::runge_kutta::explicit_time_step<
    vector_operations_type, problem_type, adaptation_type, true>;
// In external_manager::apply(), before finalization:
auto dense = step.get_continuous_integration();
const auto sample_time = dense.evaluate(theta, sample); // theta in [0,1]
```

The manager borrows the stepper. No changes to the problem or integration interface
are required. A view is valid only for its pending step; `finalize()`, `reset()`,
or the next `apply()` invalidate it, including when a newer step is pending.
The stepper must stay alive at the same address while a view is used.

| Method | Interpolant | Dense order | Extra vectors | Extra RHS calls per accepted step |
| --- | --- | --- | --- | --- |
| EE | Native linear | 1 | 1 | 0 |
| HE | Native quadratic | 2 | 1 | 0 |
| BS32 | Native cubic | 3 | 1 | 0 |
| DOPRI54 | Shampine quartic | 4 | 1 | 0 |
| RK33SSP, RK43SSP, RK64SSP | Endpoint cubic Hermite | 3 | 2 | 1 |

Native extension coefficients are stored and validated in the Butcher table.
The fallback is not advertised as a native SSP extension and does not guarantee
SSP/positivity preservation. `step.dense_output_order()` reports the actual order;
in particular RK64SSP remains fourth order at endpoints but has third-order dense
output. Embedded error control still controls the step, not a separate dense error.
Evaluation needs no new vectors or RHS calls; it reuses the pending stages and
endpoint. With dense output disabled, no dense vectors or extra RHS calls exist.
These interpolants apply to explicit ODE/reduced-flow formulations, not arbitrary
algebraic DAE states.

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

Dense-output tests have their own CPU jobs and a capability-gated CUDA job:
`test_explicit_dense_output_{cpu,cpu_omp,cuda}.bin` checks order, lifetime,
storage, rejection, backward integration, and external section refinement.
`test_explicit_{lorenz,rossler,vdp}_dense_output_{cpu,cpu_omp,cuda}.bin` samples
off-step times through an external manager, checks independent 40-digit Decimal
RK4 references, restarts, and commits a shortened endpoint. Build these targets
with the same Makefile configurations as above.

## Lorenz Simulation Comparison

The separate `test_explicit_lorenz_simulation.cpp` executable records accepted
endpoints through `ExternalManagement`, without changing the problem or integrator.
The manager buffers independent host snapshots in a `std::vector`, acquired through
scoped read-only vector views. It creates and writes the output file only after
successful integration. RAM usage therefore grows with the recorded step count.
The user supplies a positive integration end time and an output filename:

```bash
make test_explicit_lorenz_simulation_cpu_omp.bin \
     CONFIG_FILE=build_configs/config_omen_home_release.inc
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_simulation.dat
python3 source/time_stepper/tests/compare_explicit_lorenz_simulation.py \
     build/lorenz_simulation.dat
```

An optional final argument selects `BS32` instead of the default `DOPRI54`:

```bash
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_bs32.dat BS32
python3 source/time_stepper/tests/compare_explicit_lorenz_simulation.py \
     build/lorenz_bs32.dat
```

The same Makefile target supports `_cpu.bin` and `_cuda.bin`. Python requires
NumPy, SciPy, and Matplotlib. The trajectory stores its parameters, initial state,
integration interval, backend, and tolerances in a JSON comment header, followed
by `time x y z` rows. A completion footer identifies successfully finished runs;
the comparator rejects incomplete, nonfinite, or inconsistent data.

The Python script selects SciPy RK45 for DOPRI54 or RK23 for BS32 from the saved
metadata. It uses tighter tolerances and samples at the C++ output times. Adaptive controllers
differ, so matching accepted-step counts or bitwise-identical states is not required.
It writes `_python.dat`, `_error.dat`, `_summary.json`, `_comparison.png`, and
`_errors.png` next to the trajectory. Pointwise absolute/relative checks determine
the exit status; `--check-atol` and `--check-rtol` set their thresholds.

For long chaotic simulations use `--diagnostic-only` to report and plot trajectory
divergence without labeling it a passing accuracy test. Input validation and
reference-solver failures still fail. `--no-plots` skips Matplotlib output.
CI uses a separate `timestepper-lorenz-simulation-python-cpu` job for short serial
and OMP comparisons of both methods, input-validation tests, and plot artifacts.
