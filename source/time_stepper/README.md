# Time Stepping

`legacy/` is frozen. New code lives in `nmfd::time_steppers` and does not include
legacy headers.

Vector ownership uses `nmfd::detail::vector_wrap` from `nmfd-newton`.
Add `source/contrib/nmfd-newton/include` to the include path when building outside
the supplied Makefile. Scratch vectors and RK stages are allocated once and reused.

## Explicit RK

- `runge_kutta/butcher_tables.h`: EE, HE, RK23, RK33SSP, RK43SSP, RK64SSP, RK45,
  DOP853,
  plus separately tested implicit/IMEX coefficient tables.
- `runge_kutta/explicit_time_step.h`: stage evaluation and bounded retry loop.
- `integration/time_step_adaptation_constant.h`: fixed proposed step magnitude.
- `integration/time_step_adaptation_matlab.h`: component-scaled embedded error
  control, first-rejection formula, repeated halving, and no growth after rejection.
- `integration/time_step_adaptation_scipy.h`: additive tolerance scaling, RMS error
  assessment, and SciPy-style power-law step selection with bounded growth/shrinkage.
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
- `RK23` is Bogacki-Shampine 3(2), stored with four derivatives including the
  endpoint derivative. Its embedded state error scales as `h^3`.
- `RK45` is Dormand-Prince 5(4), stored with seven stages.
- `DOP853` supplies the eighth-order primary formula, stored with twelve primary
  stages plus the endpoint derivative, and a combined E5/E3 estimator with error
  exponent eight. Use constant or SciPy adaptation. Its native seventh-order dense
  output adds three optional stages.
- `SDIRK2(1)2`, `ESDIRK2(1)3`, and `SDIRK3(1)3` use `p(q)s`: primary
  order, embedded order, total stage count. `IE`, `IM`, and `CN` are unchanged.
- IMEX names are `IMEX_EULER`, `IMEX_HEUN_TR2` (Heun plus trapezoidal),
  `IMEX_ARS233`, and `IMEX_ARS222`. ARS digits count implicit stages, explicit
  stages, and combined order as in the original paper, not padded table dimensions.
The previous names are retained only by the frozen legacy implementation.

Coefficient sources: [SSP embedded pairs](https://imrefekete.web.elte.hu/files/2022_FCS_JCAM.pdf),
[Bogacki-Shampine method](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_ivp/rk.py),
[DOP853 coefficients](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_ivp/dop853_coefficients.py),
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
| RK23 | Native cubic | 3 | 1 | 0 |
| RK45 | Shampine quartic | 4 | 1 | 0 |
| DOP853 | Native degree-seven polynomial | 7 | 4 | 3 |
| RK33SSP, RK43SSP, RK64SSP | Endpoint cubic Hermite | 3 | 2 | 1 |

Native extension coefficients are stored and validated in the Butcher table.
`dense_outout_stage_count()` counts all interpolant derivatives; `dense_outout_a()`
and `dense_outout_c()` describe ordinary and additional stage rows/times. DOP853
retains thirteen ordinary derivatives and prepares three additional derivatives
only after numerical acceptance, before publishing the pending candidate.
The initial-state snapshot doubles as stage-state scratch. No seven-vector
polynomial cache is needed, and failed extra stages follow the normal retry path.
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
Its stored-reference tests explicitly set `epsilon=0.0055`; the simulation uses
the example's current default (`epsilon=0` for classical Lorenz).

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

## SciPy-Style Adaptation

`time_step_adaptation_scipy<VectorOperations>` uses the same initialization,
assessment, update, and rejection interface as the MATLAB controller. It accepts
only `error_ratio < 1`, uses safety `0.9`, limits growth to `10` and shrinkage to
`0.2`, and prevents immediate growth after a rejected attempt. Ordinary embedded
errors use RMS scaling with `atol + rtol*max(abs(previous), abs(candidate))`.
The controller owns no state vectors; a generic `transform_reduce_sum` reuses
vector-operation scratch storage. The existing MATLAB controller is unchanged.

An optional second error pointer in `assess()` enables the DOP853 combined RMS
assessment, requiring error exponent `8`. Both inputs are state-error vectors,
already multiplied by the attempted step size. DOP853 forms both defects from its
ordinary derivatives and dispatches to this overload through a C++17 capability
check. Controllers without dual assessment cannot adapt DOP853.

The five configuration fields match the MATLAB controller. Relative tolerance is
clamped to at least `100*epsilon(scalar_type)`, exposed by `relative_tolerance()`.
Unlike SciPy's complete solver, initialization uses a configured step rather than
an RHS-based estimate, and the minimum step is configured rather than derived
from ten floating-point ULPs. Failed numerical attempts shrink independently;
an inadmissibly small retry returns failure. Modified endpoints limit the next
step and fresh integrations reset rejection history.

```bash
make test_explicit_scipy_adaptation_cpu_omp.bin \
     CONFIG_FILE=build_configs/config_omen_home_release.inc
OMP_NUM_THREADS=8 ./build/test_explicit_scipy_adaptation_cpu_omp.bin
```

Use `_cpu.bin` or `_cuda.bin` for other backends. Generic reduction tests use
`test_transform_reduce_sum_{cpu,cpu_omp,cuda}.bin`. CI has separate reduction and
SciPy adaptation CPU jobs plus capability-gated CUDA coverage.

## DOP853

Use the existing explicit step with constant adaptation and dense output disabled:

```cpp
using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_constant<vector_operations_type>;
using step_type = nmfd::time_steppers::runge_kutta::explicit_time_step<
    vector_operations_type, problem_type, adaptation_type>;
adaptation_type adaptation({0.25});
step_type step(operations, problem, adaptation, {"DOP853"});
```

The coefficient data is isolated in `runge_kutta/dop853_coefficients.h`, preserving
the published digits as `long double` literals. Rooted-tree tests verify all
conditions through order eight and a failure at order nine, plus the fifth- and
third-order companion formulas. No changes to the
problem, step, adaptation, or integrator interfaces are required.

For adaptive DOP853, replace constant adaptation with
`time_step_adaptation_scipy<vector_operations_type>`. The step owns two reusable
E5/E3 defect buffers only in that configuration. Fixed DOP853 owns neither, and
ordinary embedded methods keep one error buffer. Error estimation adds no RHS
calls; dense-only stages run after acceptance. An incompatible adaptive controller
returns `error_estimate_unavailable` before evaluating the RHS or changing output.
There is no fabricated `embedded_b` pair: `has_error_estimate()` and the estimator
kind distinguish this case. Diagnostic `error_estimate(0)`/`error_estimate(1)`
return raw E5/E3 state defects; neither is the combined error alone. The no-argument
getter remains for ordinary embedded pairs. Native dense output still reports
order seven and remains independently optional.

```bash
make test_explicit_dop853_cpu.bin test_explicit_dop853_cpu_omp.bin \
     CONFIG_FILE=build_configs/config_omen_home_release.inc
OMP_NUM_THREADS=8 ./build/test_explicit_dop853_cpu_omp.bin
```

The same source builds as `test_explicit_dop853_cuda.bin`. Its analytical linear,
non-autonomous, and nonlinear cases check eighth-order refinement above roundoff,
RHS counts, forward/backward endpoints, restart, aliasing, capability guards,
step budgets, and rollback after nonfinite stages. CI has separate
`timestepper-dop853-fixed-cpu` and capability-gated `timestepper-dop853-fixed-cuda`
jobs. Frozen legacy files are not involved.

`test_explicit_dop853_adaptive_{cpu,cpu_omp,cuda}.bin` checks signed E5/E3 defects
against SciPy values, optional buffer counts, tolerance refinement, restart,
backward integration, attempt/minimum-step limits, and dense-stage recovery.
Separate adaptive CPU/CUDA CI jobs also exercise the CUDA Lorenz simulation.

Native DOP853 dense output has separate CPU, SciPy-comparison, and capability-gated
CUDA jobs. Analytical tests verify local eighth-order error, forward/backward
sampling, four optional buffers, three extra RHS calls, view lifetime, shortened
endpoints, aliasing, retry, and rollback on each failed extra stage. The existing
Lorenz, Rossler, and Van der Pol external-manager tests also exercise fixed DOP853.

```bash
make test_explicit_dop853_dense_output_cpu_omp.bin \
     CONFIG_FILE=build_configs/config_omen_home_release.inc
OMP_NUM_THREADS=8 ./build/test_explicit_dop853_dense_output_cpu_omp.bin build/dop853_dense_samples.txt
python3 source/time_stepper/tests/compare_explicit_dop853_dense_output.py build/dop853_dense_samples.txt
```

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

An optional method argument selects `RK23` or `DOP853` instead of the default `RK45`:

```bash
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_rk23.dat RK23
python3 source/time_stepper/tests/compare_explicit_lorenz_simulation.py \
     build/lorenz_rk23.dat
```

An additional controller argument selects SciPy-style adaptation. RK45/RK23
default to MATLAB-style; DOP853 defaults to SciPy and rejects MATLAB-style control:

```bash
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_scipy.dat RK45 scipy
OMP_NUM_THREADS=8 ./build/test_explicit_lorenz_simulation_cpu_omp.bin \
     1 build/lorenz_dop853.dat DOP853 scipy
```

The same Makefile target supports `_cpu.bin` and `_cuda.bin`. Python requires
NumPy, SciPy, and Matplotlib. The trajectory stores its parameters, initial state,
integration interval, backend, and tolerances in a JSON comment header, followed
by `time x y z` rows. A completion footer identifies successfully finished runs;
the comparator rejects incomplete, nonfinite, or inconsistent data.

The Python script uses the saved method name directly: RK45, RK23, or DOP853.
It uses tighter tolerances and samples at the C++ output times. Adaptive controllers
differ, so matching accepted-step counts or bitwise-identical states is not required.
It writes `_python.dat`, `_error.dat`, `_summary.json`, `_comparison.png`, and
`_errors.png` next to the trajectory. Pointwise absolute/relative checks determine
the exit status; `--check-atol` and `--check-rtol` set their thresholds.

For long chaotic simulations use `--diagnostic-only` to report and plot trajectory
divergence without labeling it a passing accuracy test. Input validation and
reference-solver failures still fail. `--no-plots` skips Matplotlib output.
CI uses a separate `timestepper-lorenz-simulation-python-cpu` job for short serial
and OMP comparisons of all supported method/controller combinations, input-validation tests, and
plot artifacts.
