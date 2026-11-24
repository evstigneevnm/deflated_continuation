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
bash source/periodic_orbit/tests/run_all.sh --mode continuation --systems rossler_periodic vdp_periodic lorenz_periodic
bash source/periodic_orbit/tests/run_all.sh --system vaidyanathan_periodic --methods RK45 DOP853
bash source/periodic_orbit/tests/run_all.sh --mode continuation --system vaidyanathan_periodic --sections 1 4
bash source/periodic_orbit/tests/run_all.sh --systems analytic_cycle_newton rossler_periodic lorenz_periodic vaidyanathan_periodic --methods 'SDIRK3(1)3' ARK324L2SA
bash source/periodic_orbit/tests/run_all.sh --system analytic_cycle_newton --methods IM CN 'SDIRK2(1)2' 'ESDIRK2(1)3'
bash source/periodic_orbit/tests/run_all.sh --system analytic_cycle_newton --methods BDF2
bash source/periodic_orbit/tests/run_all.sh --system vaidyanathan_periodic --methods BDF2
bash source/periodic_orbit/tests/run_all.sh --systems analytic_cycle_newton rossler_periodic lorenz_periodic vaidyanathan_periodic --methods BDF
bash source/periodic_orbit/tests/run_all.sh --plot-only build/periodic_orbit_benchmarks/runs/RUN_DIRECTORY
```

**Symmetric KS1D (fixed parameter).** This standalone FFT regression exercises
the production stationary operator as a physical rate, `u'=-F(u,lambda)`.
It checks linear mode evolution, nonlinear state and parameter derivatives,
then corrects a seeded ordinary orbit with RK45 and one and four sections.
It compares the phase-fixed anchors and verifies identity closure by integrating
a full period with a four-times finer timestep. No symmetry shift is applied.

```bash
make test_periodic_orbit_ks1d_symmetric_periodic_cpu_omp.bin
OMP_NUM_THREADS=1 build/test_periodic_orbit_ks1d_symmetric_periodic_cpu_omp.bin
```

The seed is for **this discrete system only**: 24 physical grid points,
11 unnormalized imaginary Fourier coefficients, `a=2, b=4, lambda=68`.
It was generated independently with LSODA on the same pseudospectral product:
in rescaled time `s=4*t` and state `v=2*u`, start from `v_k=1/k^2`
at `s=0` with `a=b=1, lambda=17`. At the last two maxima of mode 1
before `s=12`, take half the earlier state and one quarter of the peak spacing.
The resulting period is approximately 0.01515369. The test perturbs the
section coordinates before correction; it is not a continuation or a test
of spatial convergence. Use `--rates-only` for the short evolution and
derivative checks without Newton.

**Analytical-cycle branch archive.** To trace the RK45 cycle using the production
driver and retain its scalar archives, supply a **new** output directory:

```bash
make test_periodic_orbit_periodic_branch_archive_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
OMP_NUM_THREADS=1 build/test_periodic_orbit_periodic_branch_archive_cpu.bin build/analytic_cycle_archive
```

The test reopens `cycle-M1.json`, `cycle-M4.json` and `cycle-capped.json`.
The first two reach both boundaries of [0.95, 1.05]; the capped run records
the last emitted points and the step-limit reason. Without an argument the
test uses a temporary directory.

Version 1 stores ordinary periodic branches with identity closure. Segments
and points are in traversal order (global zero-based `source_index`); do not
sort on the parameter. `trace_complete` reports driver completion, whereas
`all_points_verified` reports physical orbit verification. Only verified
rows have a period, time-RMS, sampled extrema and closure error. The observable
is taken on physical states during independent full-period integration, not
on matching coordinates. These scalar files can be read without a solver;
they are **not** restart snapshots and do not change the stationary archive.

**Restart an interior orbit.** The analytical cycle and symmetric KS1D
have a separate fresh-process restart regression. Run from the repository root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
make test_periodic_orbit_periodic_branch_archive_cpu.bin test_periodic_orbit_ks1d_periodic_continuation_driver_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
python3 source/periodic_orbit/tests/check_branch_restart.py build/test_periodic_orbit_periodic_branch_archive_cpu.bin build/cycle_restart --check-invalid --diagram analytic-cycle
python3 source/periodic_orbit/tests/check_branch_restart.py build/test_periodic_orbit_ks1d_periodic_continuation_driver_cpu.bin build/ks1d_restart --diagram ks1d
```

Supply a new directory. The runner invokes the same executable twice
(`--write DIR` then `--restart DIR`) for M=1 and 4. Each source branch
identifies a verified interior point by segment and global source index.
Its separate `interior.json` stores the binary64 matching coordinates,
parameter, period, ordered Euclidean section centers and signed unit normals,
flight itinerary, and RK/trace configuration. Only ordinary forward
identity closure is supported. The reader restores the chart from these
planes, not from a regenerated pilot, and independently verifies the orbit
**before** starting continuation. `restarted/branch.json` links back to the
source point. This is a new two-arm trace, not an exact predictor-history resume.

The regression compares both lifted endpoints and their periods at the
same parameters; `endpoint_reference.json` is test data, not solver input.
No trajectory is downloaded. The selected state stays on the backend
until the trace returns. Snapshot I/O transfers three blocks per section
(center, normal, coordinates) once per write/read; these bytes and the two
endpoint-reference transfers are counted separately from the guarded solve.

**Mixed stationary and periodic diagrams.** With `--diagram`, the final check
reopens scalar exports after both solver processes have exited. The portable
`diagram/` directory contains a manifest, stationary curve and legend files,
periodic scalar archives, and `mixed_bd.png`; no solution snapshots are needed.
Replot it offline (Python with Matplotlib):

```bash
python3 plot_scripts/plot_bd.py mixed build/ks1d_restart/diagram/manifest.json --envelope
```

For another dataset, use the [KS1D manifest](fixtures/mixed_bd/ks1d/manifest.json)
as a template. Paths are relative to the manifest. Because legacy stationary
exports have no physical identity header, the manifest must declare their
model, discretization, parameter, observable, normalization, and named legend
column. Each periodic archive must match all five identity fields. A shared
filename or a similar norm label alone is not sufficient.

The top panel compares the stationary observable with its orbital time-RMS;
`--envelope` adds the sampled orbit minima and maxima. Only periodic branches
appear in the period panel. Arms retain traversal order through folds;
unverified points leave gaps, and restarts are separate traces. Without
stationary segment metadata, only points are drawn. The analytical fixture
is the exact stationary origin. The KS fixture is the stationary driver's
zero-branch scalar export at N=24, a=2, b=4 on [67.999, 68.001], not the N=32
stationary CI case. These overlays do not imply a detected branch connection.

**Trajectory-derived seed.** The [recurrence regression](test_analytic_cycle_recurrence.cpp)
starts at `(0.001, 0)` near the unstable stationary origin, without using the
known radius or period. It measures consecutive increasing crossings of
`y=0` within a single search deadline. A candidate needs an admissible
flight time, crossing speed, and absolute return distance in the backend
space norm. Short returns are not skipped to obtain a period multiple.
The search reuses two backend vectors and publishes to the caller's buffer
only on success. This is a seed for an ordinary autonomous cycle, not a
primitive-period certificate or a general unstable-orbit finder.

The test builds layouts with one and four sections, corrects and independently
verifies the orbit, then traces both arms with the production driver. To keep
the scalar branches, supply a new directory (omit it for a test-only run):

```bash
make test_periodic_orbit_analytic_cycle_recurrence_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
build/test_periodic_orbit_analytic_cycle_recurrence_cpu.bin build/cycle_recurrence_branches
```

**Comparing verified orbits.** [orbit_comparison](../orbit_comparison.h) compares
ordinary autonomous orbits at one fixed parameter, using physical anchor states,
not shooting coordinates or diagram norms. Bind a dedicated section flight to
that problem and parameter. On a common oriented section, prepare one cached hit
per orbit: its next transverse return must take the supplied period (within the
absolute time tolerance) and close in the physical space norm. This costs at
most two bounded section flights. As with other sampled event tests, the mesh
and event subdivisions must resolve the crossings; this is not a proof of a
primitive period. Failed preparation invalidates the slot.

Cached comparisons take one backend vector subtraction and norm, with no new
vector allocation or integration. Two verified crossings at or above the larger
distance threshold are distinct regardless of period. Crossings at or below the
smaller threshold match only when their periods also agree within tolerance.
All other cases are **inconclusive**; a period difference alone never excludes an orbit.
A custom symmetric distance may be supplied only for actions that preserve the
oriented section. No spatial quotient or Newton deflation is enabled by default.
The [regression](test_orbit_comparison.cpp) covers phase, layout, distinct
equal-period circles, multiple hits, double traversal and failed flights.

```bash
make test_periodic_orbit_orbit_comparison_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
build/test_periodic_orbit_orbit_comparison_cpu.bin
```

**Fixed-parameter deflation.** The [two-cycle benchmark](test_two_cycle_deflation.cpp)
compares ordinary Newton and deflated Newton from identical radial seeds at angular
speed 1, with one and four shooting sections. The isolated analytical cycles
have radii 1 and 2 and period `2*pi`. Each successful correction must satisfy
matching, independent full-period closure and the expected radius and period.
Each experiment freezes the same known inner cycle for its entire batch, so
the printed outcomes are directly comparable across seeds. The final check
rebuilds with both verified cycles and excludes both.

A separate admission check uses the same radial dynamics with radius-dependent
angular speed. Both isolated cycles are admitted at period ratios 3:2 and 2:1.
Reporting the faster cycle with twice its period fails its own return-time check,
even when that number is exactly the slower cycle's true period.

The [collection](../orbit_collection.h) compares each verified candidate once
against each cached orbit until a definite duplicate is found. An inconclusive
comparison never inserts a new solution. Before a Newton batch,
[chart deflation](../matching/chart_deflation.h) recharts the stored cycles and
checks their matching residuals and periods. A failure identifies the index
and publishes no operator. The multiplier uses the existing shifted inverse
distance and rank-one Jacobian in fixed coordinates; zero is not an implicit
root. Changing the layout or parameter requires fresh setup.

```bash
make test_periodic_orbit_two_cycle_deflation_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
OMP_NUM_THREADS=1 build/test_periodic_orbit_two_cycle_deflation_cpu.bin
```

**Discovery from an empty catalog.** At one parameter and fixed layout,
[knot discovery](../knot_discovery.h) uses ordinary Newton until the first
verified orbit is admitted. Each subsequent seed sees all active roots.
Only a newly admitted orbit is recharted; a duplicate or inconclusive return
does not change the exclusions. If a new orbit cannot be activated (including
the configured capacity limit), the batch stops and reports its index and
reason. The catalog retains the verified orbit; do not reuse the stale exclusions.
The [regression](test_knot_discovery.cpp) finds the two analytical cycles for
both section counts, replays the same seeds, and checks failure handling and
backend transfers. Exhausting a seed batch is not a completeness claim.

```bash
make test_periodic_orbit_knot_discovery_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
OMP_NUM_THREADS=1 build/test_periodic_orbit_knot_discovery_cpu.bin
```

**Continue discovered orbits.** `knot.run_and_continue(seeds, anchor, ordinary,
deflated, verify, rechart, trace)` finishes discovery, then calls
`trace(parameter, root_index, coordinates)` only for this batch's activated
roots that pass rechart, matching and period checks. Use ordinary continuation
in `trace`, returning false for an incomplete branch. Inspect
`discovery_complete` (the batch finished, not a proof of completeness) and
the per-root `preparation` and `trace_complete` results separately.

The [two-cycle regression](test_discovered_two_cycle_branches.cpp) checks
radii 1 and 2 along both arms, replay without launches, and a capped trace.
The [KS1D regression](test_ks1d_discovered_branch.cpp) continues one verified
odd-subspace orbit at N=24 from 68 to 67.999 and 68.001. Both section counts
use the existing endpoint Fourier references and independently integrated
closure. Neither test requires downloading an orbit to start a trace.

```bash
make -j2 test_periodic_orbit_discovered_two_cycle_branches_cpu.bin test_periodic_orbit_ks1d_discovered_branch_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
OMP_NUM_THREADS=1 build/test_periodic_orbit_discovered_two_cycle_branches_cpu.bin --write build/two_cycle_traces
OMP_NUM_THREADS=1 build/test_periodic_orbit_ks1d_discovered_branch_cpu.bin --write build/ks1d_traces
```

Omit `--write` for a test-only run. The optional directory must be new; each
JSON archive records its own `discovered_at.parameter` and `root_index`.

**Symmetric KS1D continuation.** Follow this same discrete orbit in physical
`lambda` with the project's pseudo-arclength stepper in the benchmark's own branch loop:

```bash
bash source/periodic_orbit/tests/run_all.sh --mode continuation --system ks1d_symmetric_periodic
```

The [preset](benchmarks/ks1d_symmetric_periodic_continuation.json) starts at
68 and controls the number of RK45-based branch steps on each arm; CI
requests two with `--branch-steps 2`. The runner compares
period, phase-fixed packed Fourier state and time-RMS of the KS `l2_norm`
at 67.999 and 68.001 for one and four sections. Every accepted point is
integrated independently at one quarter of the RK step. Matching uses
backend vectors and matrix-free flight derivatives; plotting samples only
the scalar reduced norm. The two cross-run comparison states and restart
checkpoint are explicit host I/O, counted separately from solver transfers.
No stationary archive is overlaid on this plot.

The separate [driver regression](test_ks1d_periodic_continuation_driver.cpp)
calls `continuate_curve_result()` once per branch, leaving seed tangents,
both semicurves, knot sampling and endpoint refinement to the production driver.
For one and four fixed sections it compares both corrected endpoints against
the step-level benchmark's saved Fourier probes and independently checks closure.
It repeats the branch with an interior knot and checks last-accepted recovery
with a one-step cap. The recorder keeps only scalars per point; selected states
stay on the backend. No archive or generated benchmark data is required.

```bash
make test_periodic_orbit_ks1d_periodic_continuation_driver_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
OMP_NUM_THREADS=1 build/test_periodic_orbit_ks1d_periodic_continuation_driver_cpu.bin
```

**Continuation.** This opt-in benchmark uses fixed-step RK45 and the project's
pseudo-arclength predictor, bordered Newton corrector and tangent updates. It
follows each seeded cycle in both directions: [Rossler](benchmarks/rossler_periodic_continuation.cpp)
in `c` around 6, [Van der Pol](benchmarks/vdp_periodic_continuation.cpp) in `mu`
around 1, [cubic-damped Lorenz](benchmarks/lorenz_periodic_continuation.cpp)
in `rho` around 28, and [five-state Vaidyanathan](benchmarks/vaidyanathan_periodic_continuation.cpp)
in `c` around 28. For Vaidyanathan the seeded unstable orbit has period
2.4064400379694417 at the anchor; the steady reference is the origin. Each
source binds its physical parameter, defines its RHS
parameter derivative, constructs the pilot flights and bordered Newton solver,
and traces both directions from a corrected anchor. Numerical defaults are in
the adjacent `*_continuation.json` preset: edit that file to tune one problem,
or pass CLI overrides for one run. The resolved configuration is saved with
each report. Rossler and Lorenz reuse their fixed-correction model equations
and reference seeds. The [analytical cycle](benchmarks/analytic_cycle_continuation.cpp) uses the
same [branch checks](benchmarks/ode_continuation_benchmark.h) and verifies the radius and period at every point. The
[analytical-cycle regression](test_analytic_cycle_continuation.cpp) separately
checks the exact branch and transverse multiplier. Each solver keeps its pilot
sections fixed throughout a correction. Between steps, the continuation benchmarks
refresh automatic planes when the crossing cosine or the relative spacing of two
sections deteriorates. Suspicious period growth is checked for a shorter closure
before publication. Use `--observe-layout` to log warnings without moving the
planes. The Rossler reference gate (`--require-rossler-reference`) requires reaching
`c=0.4` on the decreasing branch with period 5.53393796. It does not test the
zero-amplitude endpoint. Set `--arclength-step` and `--branch-steps` for continuation;
`--step` controls the ODE timestep, not the arclength.

Each case writes `branch.csv`, `branch.png`, and a full state/tangent
`checkpoint.json`. The run checks the anchor period, physical matching gaps,
full-period closure at one quarter of the RK step, the parameter derivative
against independently relocated crossings, and a restart from a fresh
solver/layout. Different section counts are compared at bracketed parameters
on either side of the anchor (`--probe-offset`), using period and the physical
state on their common first section. Only checkpoint and diagnostic exports
read state entries on the host. This is a local branch validation, not a search
for new orbits or a Floquet stability analysis.

`bifurcation.png` overlays the steady reference height and the periodic time-RMS
of the **same** `norm_bifurcation_diagram` column (`state_euclidean` for these ODEs).
The band is the sampled min/max, not the oscillation amplitude. Only periodic
points appear on the period panel. Scalars are accumulated on the independent
physical trajectory up to the measured period using trapezoidal time weights;
neither the section coordinates nor an unweighted average of timesteps is used.
`steady_reference.csv` is separate from the corrected periodic `branch.csv`:
it samples the analytic origin, or the inner Rossler equilibrium, not a numerically
continued stationary branch. Old v1 exports can still regenerate `branch.png`.

`report.json` lists committed layout changes. The first step following a refresh
also gets a restart check using `refresh_checkpoint.json`. Checkpoints include
the pilot seed, period and parameter needed to reconstruct the active charts.

In default correction mode the system remains `analytic_cycle`. Default section counts are 1, 2, 4,
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
