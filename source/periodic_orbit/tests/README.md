# Periodic Orbit Validation

`analytic_cycle` validates multiple-section matching of a known isolated
unit-circle orbit and non-orbit perturbations (`stage: matching_only`).
`analytic_cycle_newton` corrects perturbed section states using NMFD Newton/GMRES
and a prepared matrix-free derivative (`stage: periodic_newton`).
`rossler_periodic` and `lorenz_periodic` add three-dimensional seeded corrections
with the same solver and derivative path. `vaidyanathan_periodic` extends that
path to a five-dimensional unstable cycle in the local chaotic formulation.
These are not general orbit-discovery
algorithms. Each system's equations, parameters and seed provenance are documented
in its separate `*_periodic_problem.h` file.

From the project root:

```bash
bash source/periodic_orbit/tests/run_all.sh
bash source/periodic_orbit/tests/run_all.sh -B
bash source/periodic_orbit/tests/run_all.sh --backend cpu_omp --threads 2 --methods RK45 DOP853
bash source/periodic_orbit/tests/run_all.sh --system analytic_cycle_newton --methods RK45 DOP853
bash source/periodic_orbit/tests/run_all.sh --systems rossler_periodic lorenz_periodic --methods RK45 DOP853
bash source/periodic_orbit/tests/run_all.sh --system vaidyanathan_periodic --methods RK45 DOP853
bash source/periodic_orbit/tests/run_all.sh --systems analytic_cycle_newton rossler_periodic lorenz_periodic vaidyanathan_periodic --methods 'SDIRK3(1)3' ARK324L2SA
bash source/periodic_orbit/tests/run_all.sh --system analytic_cycle_newton --methods IM CN 'SDIRK2(1)2' 'ESDIRK2(1)3'
bash source/periodic_orbit/tests/run_all.sh --system analytic_cycle_newton --methods BDF2
bash source/periodic_orbit/tests/run_all.sh --system vaidyanathan_periodic --methods BDF2
bash source/periodic_orbit/tests/run_all.sh --systems analytic_cycle_newton rossler_periodic lorenz_periodic vaidyanathan_periodic --methods BDF
bash source/periodic_orbit/tests/run_all.sh --plot-only build/periodic_orbit_benchmarks/runs/RUN_DIRECTORY
```

The default system remains `analytic_cycle`. Default section counts are 1, 2, 4,
8; the default dense-output method is DOP853.
Use `--sections`, `--step`, `--start`, `--perturbation`, `--matching-tolerance`,
`--max-steps` and `--timeout` to configure a run. The problem description and
exact radial evolution are in `analytic_cycle_support.h`. The section layout is
generated from a pilot trajectory and is frozen during evaluation.
Correction adds `--newton-tolerance` (default `1e-10`) and
`--newton-max-iterations` (default 12). Its reports include actual matching
residuals, flight times, Newton history and independent half/quarter-step closure
checks. Plots show initial/corrected states and convergence alongside the orbit.
Prepared derivatives replay explicit RK, DIRK, fixed BDF1/BDF2 or adaptive BDF1-5
on a frozen mesh/order schedule and differentiate their native dense section crossing. General DAE tangents
and relative closure are not silently inferred. Core tests include non-autonomous stage times, adaptive mesh
recording, linearity, finite differences, rollback and retained-Jacobian lifetime.

## Optional DIRK Derivatives

`prepared_dirk_derivatives` and `prepared_dirk_matching` are separate core tests.
The visual runner also accepts IM, CN, SDIRK2(1)2, ESDIRK2(1)3, SDIRK3(1)3 and
ARK324L2SA, while its explicit defaults remain unchanged. Build core tests with the ordinary
`test_periodic_orbit_<name>_<cpu|cpu_omp|cuda|hip>.bin` Make targets.
They cover IE, IM, CN, SDIRK2, ESDIRK2, SDIRK3 and ARK324L2SA; fixed/adaptive
non-autonomous flights; coupled/state-dependent masses; a linear-constraint
singular-mass DAE; and Newton correction with 1,2,4,8 sections.

Assembly appends a `dirk_flight_factory` to the defaulted template parameters of
`multiple_section_problem`, then uses its factory constructor. The factory's
provider creator must supply a dedicated problem and injected solvers, plus
`stage_tangent_rhs` and, only when required, `rate_jacobian_apply`.
No sensitivity storage is added to NMFD steppers. Factories allocate no tangent
workspace until Jacobian preparation; repeated Jv allocates no backend vectors.
Tests forbid vector-sized host transfers, check frozen-grid finite differences,
linearity, failed preparation/Jv rollback, independent rebinding and cleanup.
Replay budgets bound scalar schedules; backend vector workspace does not grow
with the number of accepted steps. Factorization/Krylov costs remain those of
the supplied solvers. Nonlinear DAE reconstruction, moving-section derivatives,
relative closure and prepared BDF derivatives are not inferred by this path.

## Optional BDF Derivatives

`prepared_bdf_derivatives` and `prepared_bdf_matching` cover BDF1/BDF2 with
backward-Euler startup independently on each flight. Assembly supplies a
`bdf_flight_factory` through the existing matching factory parameter and records
actual orders, signed steps, candidate times and event truncation. The shared
`implicit_stage_provider` solves the shifted base/tangent systems; it needs no
explicit mass inverse. Ordinary steppers and residual-only evaluations allocate
no sensitivity storage. Replay retains nine/eleven vectors, not a vector history;
per-segment checkpoints and scalar schedules are bounded. Tests cover frozen-mesh
finite differences, Jv linearity, tiny directions, mass/DAE examples, non-autonomous
and backward flights, dense time derivatives, history restart, failed commits,
rollback, retained Jacobians and 1,2,4,8-section Newton solves. Transfer guards
forbid vector-sized host/device copies on CPU/OpenMP/CUDA/HIP.

The visual runner retains fixed BDF1/BDF2. BDF2 default
steps are 0.00025 (cycle), 0.00002 (Rossler), 0.000005 (Lorenz) and 0.000003 (Vaidyanathan),
with budgets including independent half/quarter-step verification. BDF1 requires
`--step`, like IE; strict first-order reference accuracy may require millions of
steps. Neither selection relaxes Newton or reference tolerances. BDF cases default to a 900-second
wall-clock limit (`--timeout` overrides it); other methods retain 300 seconds.
CPU CI runs BDF2 cycle correction
at 1,2,4,8 sections and one-section ODE cases; core tests cover both orders on all
backends. These tiny implicit solves are correctness tests, not GPU speed benchmarks.

Adaptive `BDF` selects orders 1-5 with `--maximum-order` (default 5),
`--relative-tolerance` (default `1e-12`), `--absolute-tolerance` (`1e-13`) and
`--maximum-step` (`0.1`). Here `--step` is the initial step (default `0.001`).
Verification refines local tolerances and the step cap, not merely the initial
step. Reports/`adaptation.png` show accepted orders, step ranges and rejected
attempts; physical reference criteria remain unchanged.
`prepared_bdf_adaptive_derivatives` and `prepared_bdf_adaptive_matching` cover
variable grids, order reduction without history loss, real history resets,
native dense derivatives and multi-section correction. Factories opt in with
`params.adaptive=true` and matching order/ratio bounds. State/tangent histories
use native divided differences with O(qN) backend storage, q<=5. Controllers and
order decisions are frozen during Jv, not differentiated. CPU CI additionally
checks all four corrected benchmark systems at 1,2,4,8 sections.

The ODE benchmark adapter uses identity mass and the existing model RHS/Jv:
`R(Y)=a*(Y-w)-b*(f(t,Y)+r)`, `R_Y*v=a*v-b*J_f*v`, and
`stage_tangent_rhs=a*delta_w`. Each trajectory and prepared Jacobian gets isolated
mutable stage dependencies. These are benchmark-owned; NMFD interfaces and
ordinary users' storage are unchanged. Scalar plotting exports remain outside
the zero-vector-transfer matching checks. Stage linear solves use relative-only
stopping so tiny outer Krylov directions are not discarded by an absolute floor.

The Rossler case uses `a=b=0.1, c=6`, with reference period
`6.0309863678295104`. The Lorenz case retains the historical cubic damping:
`sigma=10, rho=28, beta=8/3, epsilon=0.0055, delta=0`, with reference period
`1.6443075739726336`. It is not the undamped textbook system. The Lorenz seed is
placed at a phase whose flow-normal plane has one increasing return per orbit;
the original shooting phase's plane had two. An independent variational solve
gives a nontrivial unstable multiplier about `5.39008` for this short orbit.

The five-dimensional Vaidyanathan case uses the equations from the local
`PO_tests` model: `x'=10*(y-x)+w+v`, `y'=28*x-x*z-y`,
`z'=x*y-(8/3)*z`, `w'=-x*z+1.3*w`, `v'=2.5*x`.
Its reference period is `2.4064400379694417`; phase-constrained DOP853 shooting
refines a chaotic-trajectory recurrence, with independently verified closure
below `2.1e-13`. Unstable Floquet multipliers include
`0.795584 +/- 1.975593i` and `1.009622`. The reference phase has a single
increasing flow-normal return. The origin is not an acceptable orbit result.

Default fixed steps, overridden by `--step`, are:

| System | RK45 | DOP853 |
| --- | --- | --- |
| Analytical cycle | 0.08 | 0.08 |
| Rossler | 0.01 | 0.04 |
| Cubic-damped Lorenz | 0.002 | 0.01 |
| Vaidyanathan (5D) | 0.002 | 0.01 |

DIRK default fixed steps are calibrated against the same reference tolerances:

| System | IM / CN / SDIRK2 / ESDIRK2 | SDIRK3 | ARK324L2SA |
| --- | --- | --- | --- |
| Analytical cycle | 0.0004 | 0.01 | 0.02 |
| Rossler | 0.00005 | 0.003 | 0.003 |
| Cubic-damped Lorenz | 0.00001 | 0.00025 | 0.00025 |
| Vaidyanathan (5D) | 0.00001 | 0.00015 | 0.00015 |

DIRK integration budgets account for the smaller steps and quarter-step
verification; `--max-steps` explicitly overrides them. IE is also selectable,
but requires `--step`: its first-order reference accuracy can require millions
of steps. Larger steps do not relax validation tolerances. The runner currently
uses fixed steps; adaptive DIRK replay remains covered by the core tests.
CI runs the full third-order DIRK visual matrix on CPU and an analytical-cycle
OpenMP smoke matrix. Core CPU/OpenMP/CUDA/HIP tests check every model's stage
adapter and prepared derivatives. Tiny implicit ODE solves pay substantial
OpenMP/GPU launch overhead; visual runs are validation, not speed benchmarks.

ODE reports validate reference periods, true reduced-coordinate residuals,
physical section defects, and independent half/quarter-step closure. Trajectories
are sampled at the same 401 physical times through dense views; their difference
is checked without requiring long-time shadowing of an unstable orbit. A finite
nonzero physical rate and a nontrivial excursion reject equilibrium collapse.
Plots include the initial/corrected 3D trajectories, section states, time traces,
Newton convergence, matching gaps and segment work. The normal section defect
and tangent-coordinate gap are checked separately, not equated.
For the five-dimensional case, CSV exports and validation include every state
component (`x1` through `x5`); `phase.png` shows the first three and
`projections.png` adds the remaining coordinate projections. Time traces show
all five. The small-system GMRES basis follows the matching dimension, capped
at 64 vectors; neither Newton tolerances nor canonical solver interfaces change.
GPU regression jobs allow this larger case up to 600 seconds for the overhead
of many small stage-kernel launches; the existing cases retain their budgets.

Build products and assembled case configurations go into
`build/periodic_orbit_benchmarks/bin` and `cases`. Each invocation creates a new
timestamped `runs` directory containing its manifest, logs, JSON reports,
trajectory/section CSV files, `phase.png`, and `diagnostics.png`. Old runs are
never removed or overwritten. Replotting updates plots and validation summaries
only, without rebuilding or integrating. `-B` forces a build but executes no cases.
Numerical parameter changes only reassemble configurations; source/header,
compiler-option and Make configuration changes invalidate the relevant build.

CPU/OpenMP need a C++17 compiler. Plotting needs NumPy and Matplotlib, not SciPy.
Set `PYTHON` explicitly if needed. CUDA/HIP use the same source and existing
project build configuration, for example:

```bash
bash source/periodic_orbit/tests/run_all.sh --backend cuda
bash source/periodic_orbit/tests/run_all.sh --backend hip \
  --make-option HIPCC=/opt/rocm-hip-nvidia/bin/hipcc \
  --make-option HIP_ROOT_PATH=/opt/rocm-hip-nvidia --make-option HIP_ARCH_LIST=75
```

Device runtime/library paths must already be configured. Core matching and chart
tests forbid vector-sized host copies. Benchmark plotting explicitly exports scalar
scalar coordinates; this I/O is outside residual evaluation. No legacy headers,
project Newton implementation, or canonical solver modifications are used here.
