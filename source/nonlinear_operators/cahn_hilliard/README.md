# Convex-Split Cahn-Hilliard

Project-owned constant-mobility CH on a periodic two-dimensional grid:

```text
c_t = m*Delta_h(mu),  mu = c^3-c-epsilon^2*Delta_h(c),  m>0.
```

`model.h` contains immutable parameters/geometry. `convex_split_problem.h` uses
concentration alone as the timestepper state, with the existing IMEX bundle:

```text
explicit_part.evaluate_rate(c,kE): kE = -m*Delta_h(c)
implicit_part.evaluate_rate(c,kI): kI = m*Delta_h(c^3-epsilon^2*Delta_h(c))
implicit_part.apply(c,R):         R  = a*(c-w)-b*(kI+r)
```

`previous_state()` contains w, not necessarily the old committed concentration.
The implicit facade implements the unchanged time/affine-shift, buffer, residual,
linearization and shared-Jacobian contract. Its prepared action is
`Jv=a*v-b*m*Delta_h(3*c_base^2*v-epsilon^2*Delta_h(v))`. Differential coefficients
are frozen on the backend. Residual and Jacobian scratch are separate; trial
evaluations do not change prepared data. Assembly injects Newton/linear solvers;
production headers do not include or construct them. Each context owns mutable
workspace and must not be shared by concurrent trajectories/Jv evaluations.

## Discrete Energy Law

Select `IMEX_EULER`, positive steps and the existing constant-step controller.
For a solved step,

```text
mu_star = c_new^3-c_old-epsilon^2*Delta_h(c_new)
delta   = c_new-c_old = h*m*Delta_h(mu_star).
```

This is the first-order convex-splitting update. The physical chemical potential
at c_new is different: `mu(c_new)=mu_star-(c_new-c_old)`. Eliminating mu makes
the timestepper mass identity; this is not general singular-mass IMEX support.
The older linear mixed CH problem/tests remain unchanged.

Use the SAME symmetric periodic Laplacian for the energy and dynamics:
`E_h(c)=< (c^2-1)^2/4,1 >_h-epsilon^2*<c,Delta_h(c)>_h/2`.
The volume-weighted inner product gives mass conservation and
`E_h(c_new)-E_h(c_old)+h*D <= 0`, where
`D=-m*<mu_star,Delta_h(mu_star)>_h >= 0`.
Convexity contributes additional numerical dissipation, so this is not an equality.
It follows from implicit convex/explicit concave energy derivatives, as in
[Eyre's convex-splitting analysis](https://www.math.utah.edu/~eyre/research/methods/papers).

`energy_diagnostics.h` is optional and does not control timesteps or commit state.
`evaluate_step(before,after,h)` returns mass, energy, dissipation and the actual
step defect `r_step=delta-h*m*Delta_h(mu_star)`. Inexact solves obey the bound
`delta_E+h*D <= <mu_star,r_step>_h` up to rounding. Reports include this residual
work and a roundoff allowance; tests independently enforce small step defects
and small residual work, rather than letting inaccurate solves hide energy growth.
Invalid steps, nonfinite states and overflowed diagnostics are rejected.

No energy claim is made for higher-order IMEX tables, arbitrary forcing, variable
mobility, other boundary conditions, dense samples or adaptivity. Energy stability
does not guarantee temporal accuracy or nonlinear-solver convergence for large h.

## Unsplit Adapter

`unsplit_problem<Space>` supplies the complete physical equation on the same
concentration space, grid, potential and parameters:

```text
F(c)             = m*Delta_h(c^3-c-epsilon^2*Delta_h(c))
apply(c,R)       : R = a*(c-w)-b*(F(c)+r)
evaluate_rate(c,k): k = F(c)
prepared Jv      = a*v-b*m*Delta_h((3*c_base^2-1)*v-epsilon^2*Delta_h(v))
```

Construct it with shared space/model handles, set time and affine coefficients,
and use the existing `previous_state`, `known_rhs`, `set_linearization_point`
and shared `get_jacobi_operator` contract. Rate evaluation is independent of
these stage buffers/coefficient settings. It returns `true` for completed stencil
evaluation; it does not certify finite output, solver convergence or energy decay.
Timesteppers retain their existing finite-state checks and commit/failure rules.

The adapter works with ordinary DIRK/BDF nonlinear solves and autonomous rate-form
Rosenbrock shifted linear solves. It introduces no singular mass solve, solver
construction, time controller, energy-based acceptance or dense-output policy.
The original split public classes/constructors remain available. Their two-pass
implementation is shared through `detail/concentration_problem.h`, selecting the
convex or full potential at compile time. Both retain five backend vectors;
preparing the full Jacobian adds `-1` without another kernel or vector.

`test_pde_cahn_hilliard_unsplit` checks summed split/full rates, the physical
energy gradient and semidiscrete dissipation, discrete stable/unstable Fourier
linearizations, shifted-residual/Jv consistency, frozen preparation, mass,
equilibria, rollback, restart and ownership. It verifies IE, CN, SDIRK2, BDF2 and
ROS3 temporal orders against a refined numerical solution on the same spatial
grid. Guarded 64x64 smoke trajectories cover DIRK/BDF/Rosenbrock without bulk
host transfers or backend allocations after warmup.

These tests establish adapter correctness, not unconditional energy stability
of those time discretizations. `energy_diagnostics::evaluate_step` remains
specific to convex-split Euler: its stage potential, defect and energy bound
must not be reused as unsplit-method acceptance tests. The graphical runner
below compares split/unsplit methods using common physical diagnostics and
method-specific accuracy/energy checks. Equal-accuracy work-precision studies
remain a separate extension.

```bash
make test_pde_cahn_hilliard_unsplit_cpu.bin \
  CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc BUILD_DIR=build/ch_unsplit
OMP_NUM_THREADS=1 build/ch_unsplit/test_pde_cahn_hilliard_unsplit_cpu.bin
```

The same source has `_cpu_omp`, `_cuda` and `_hip` targets and is included in
the parent PDE CI gates, including sanitizer and public-header checks.

## Tests And Cost

Separate project PDE tests cover reduced/mixed nonlinear equivalence, finite-
difference/linear Jv, frozen preparation, first-order temporal refinement,
unforced energy decay, homogeneous equilibria, mass conservation, rollback,
aliasing, exceptions, restart and ownership. The independent mixed reference
solves for (c_new,mu_star); it never inverts a singular mass matrix.

Every residual, rate, Jv and diagnostic is O(N), with preallocated backend
workspace. The explicit partition owns no field scratch. The implicit partition
retains five vectors (w,r,mu_I,3*c_base^2,Jv scratch); the optional observer adds
three. Storage does not grow with the number of accepted steps. Kernels require
distinct input/output fields. Only scalar reductions reach the host. These are
single-process stencils, not distributed halo-exchange validation.

```bash
make -j2 test_pde_cahn_hilliard_imex_cpu.bin test_pde_cahn_hilliard_energy_cpu.bin \
  test_pde_cahn_hilliard_imex_lifecycle_cpu.bin \
  CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc BUILD_DIR=build/ch_energy
OMP_NUM_THREADS=2 build/ch_energy/test_pde_cahn_hilliard_energy_cpu.bin
```

The same sources have CPU/OpenMP/CUDA/HIP targets. A guarded 64x64 case forbids
bulk host/device transfers and post-warmup backend allocations. Larger timing
or equal-accuracy efficiency comparisons remain separate from this test gate.

## Graphical Validation

```bash
bash source/nonlinear_operators/cahn_hilliard/tests/run_all.sh
# Select a subset (quote method names containing parentheses):
bash source/nonlinear_operators/cahn_hilliard/tests/run_all.sh \
  --methods IMEX_EULER IE CN 'SDIRK2(1)2' BDF2 ROS3
# New higher-order methods and adaptive variable-order BDF:
bash source/nonlinear_operators/cahn_hilliard/tests/run_all.sh \
  --methods ARK324L2SA ROS34PW2 BDF IMEX_ARK324L2SA
# Optional embedded adaptation for the three higher-order pairs:
bash source/nonlinear_operators/cahn_hilliard/tests/run_all.sh \
  --methods ARK324L2SA ROS34PW2 BDF IMEX_ARK324L2SA \
  --adaptive-methods ARK324L2SA ROS34PW2 IMEX_ARK324L2SA \
  --relative-tolerance 1e-4 --absolute-tolerance 1e-6
# Optional: --backend cpu_omp --threads 2, or --backend cuda / hip
# -B rebuilds and assembles only; it does not integrate or import plotting libraries.
# --plot-only build/ch_energy_benchmarks/runs/<run> revalidates and redraws saved data.
```

The default 16x16 experiment runs all ten supported implicit/IMEX methods to time
6 with fixed step/adaptive maximum step 0.1. The new ARK324L2SA, ROS34PW2 and IMEX_ARK324L2SA pairs run at fixed steps
unless selected with `--adaptive-methods`. `BDF` always uses variable steps/order
1-5; `--maximum-order` caps that order. No explicit methods are included.
Each timestamped run under `build/ch_energy_benchmarks/runs` retains a manifest,
per-method configuration/logs/JSON/CSV, validation metrics and PNG/PDF figures.
Within each method's subdirectory:

- `energy`: free energy, energy-loss rate versus the convex-split dissipation
  bound, mass drift and independently measured nonlinear step defects.
- `order`: temporal errors and observed orders versus that method's order guide.
  A separate short interval (default 0.2) uses two finer **unsplit ROS3** references
  on the same spatial grid for every method. This is temporal refinement, not spatial or exact-
  solution validation. Reference disagreement must be below 15% of the finest
  measured error; fits exclude the reported numerical floor.
- `tolerance`: adaptive endpoint errors and accepted step counts under progressively
  tighter relative/absolute tolerances. This replaces the fixed-step `order` plot
  for adaptive cases; no fixed order-five claim is made for adaptive BDF.
- `adaptation`: accepted timesteps, orders and rejected-attempt counts.
- `solution`: actual concentration and reconstructed **physical** chemical
  potential at five common physical times, with a common color scale per field.
  The physical potential is not the convex-split stage potential used in the
  energy inequality. Fixed methods save accepted endpoints; adaptive methods use
  their validated dense output before commitment. No subgrid spatial interpolation
  is claimed, and sampled energies are not acceptance conditions.

The batch directory also contains `energy_comparison`, `order_comparison`,
`adaptive_accuracy_comparison`, `concentration_comparison` and `chemical_potential_comparison` figures, plus
`comparison.csv`. Compared runs share physical parameters, initial data, grid,
fixed-step requests/adaptive maximum steps and snapshot times. Actual adaptive
meshes may differ. All field figures use common color limits across
the selected methods; the physical chemical potential is reconstructed with the
same periodic stencil. Energy overlays show E, energy change per timestep, mass
drift and physical dissipation, distinct from the convex-split stage dissipation.

Only `IMEX_EULER` enforces the convex-split discrete energy bound. Other methods' energy
increases are counted/reported rather than rejected as numerical-solve failures.
This is an observed behavior comparison, not proof of unconditional stability.
All methods must still pass finite-state, mass, solver-success, reference-quality
and fixed-step order or tolerance-refinement gates. `physical_diagnostics.h` provides method-neutral E,
mass and physical dissipation using two optional backend scratch vectors; it
never controls acceptance. The split-only observer retains its separate checks.

Failed methods do not stop the remaining cases. Their logs and failure records
are retained; figures include only validated methods, and the batch reports
failure until every selected case passes. `--plot-only` accepts a batch directory
or a method subdirectory. Older single-method v1 reports remain readable.
Comparisons at the same requested/maximal timestep are not equal-accuracy efficiency rankings;
timings include observer/reference work and are not standalone solver timings.

### More Timesteps

The main trajectory uses `ceil(end/step)` nominally uniform steps, up to 20000.
Its nominal timestep is `end/count`; final times agree up to floating-point
roundoff. The runner carries and commits actual timestepper endpoints so BDF
history is not discarded by separately rounded sampling times. Reduce
`--step` for finer time resolution at the same final time, or increase `--end`
to follow the dynamics longer:

```bash
# 600 main steps instead of the default 60; use --grid 32 for a 32x32 grid.
bash source/nonlinear_operators/cahn_hilliard/tests/run_all.sh \
  --end 6 --step 0.01 --timeout 1800
# 6000 main steps over a longer physical interval; nine field snapshots.
bash source/nonlinear_operators/cahn_hilliard/tests/run_all.sh \
  --end 60 --step 0.01 --frames 9 --timeout 3600
```

`--base-steps` and `--levels` control only the separate accuracy study,
not the energy/field trajectory. Fixed-step defaults use 5, 10, 20 and 40 steps over
`[0,0.2]`; the numerical references use 320 and 640 steps. `--frames` only
changes how many fields are saved, not the number of integration steps.
`--timeout` applies separately to each method's main trajectory and all its
reference/refinement solves, but excludes compilation and plotting. `--help`
lists defaults, limits and examples.

For adaptive cases, `--step` caps the main trajectory's timestep and defines
the common observation grid; it does not prescribe the accepted-step count.
`--initial-step` defaults to 0.02. SciPy-style state-error control uses default
rtol=1e-4 and atol=1e-6; `--maximum-steps` bounds each trajectory (up to 20000).
The tolerance study scales both tolerances by 0.1 per level on `[0,order_end]`,
with a fixed maximum step equal to that interval so step caps do not hide the
controller response. It requires the finest endpoint error to improve on the
coarsest by at least a factor 1/0.6 and verifies the independent reference budget.
Dense sampling does not shorten accepted steps, change committed endpoints or
reset BDF history. Samples retain their source-step index and evaluated time.

Plotting requires NumPy and Matplotlib (`PYTHON` selects the interpreter). Builds
are incremental, including compiler-option stamps and generated header dependencies.
The runner does not overwrite previous runs. Failed solves/invalid reports do not
produce passing plots. Larger steps may fail the unchanged Newton/GMRES gate.

The exporter reuses the tested project assembly and equations. All trajectories,
references and diagnostics run with bulk host-transfer guards enabled. At most
nine preallocated backend snapshots are retained; adaptive interpolation also
writes directly into these backend buffers. Fields are downloaded only
**after all integrations finish**. Reports distinguish zero integration bulk
transfer bytes from these deliberate postprocessing exports. Storage is O(N)
with bounded snapshot/refinement counts; no complete field trajectory is retained.
Scalar histories are bounded by a 20000-step budget. CPU/OpenMP plotting smoke
tests and report/runner unit tests are included in CI; GPU builds use the same
source and remain subject to available hardware.

### Fourier Benchmark

```bash
bash source/nonlinear_operators/cahn_hilliard/fourier/tests/run_all.sh
# Smaller selection; use the same --grid, --end, --step and --adaptive-methods options.
bash source/nonlinear_operators/cahn_hilliard/fourier/tests/run_all.sh --methods IMEX_EULER CN BDF ROS34PW2
```

The separate launcher writes the same energy, convergence, adaptation, concentration
and chemical-potential plots (PNG/PDF) under `build/ch_fourier_benchmarks/runs/`.
Use `-B` to force build only, or `--plot-only RUN_DIRECTORY` to revalidate saved
fields and regenerate figures. CPU/OpenMP require FFTW; CUDA requires cuFFT;
HIP requires hipFFT. No explicit timestepper or coupled algebraic variable is needed.

The state is the real concentration sampled on the periodic grid, including its
conserved mean. The Fourier-Galerkin RHS uses the exact Laplacian symbol
`-(kx*kx+ky*ky)` and the projection of the full cubic. Both the cubic and its
prepared Jacobian action `3*c_base^2*v` are evaluated on a doubled grid before
truncation. In each direction the retained modes satisfy `2*abs(k)<N`:
even-grid Nyquist lines are excluded; odd grids retain both highest modes.
Initialize in this subspace. FFT plans, work arrays and the prepared multiplier
stay on the backend. Each action costs O(N^2 log N) on an N-by-N grid.

For forward implicit stages, the Fourier benchmark right-preconditions GMRES
with the inverse of the shifted principal part: multiply each retained mode by
`1/(a + b*m*epsilon^2*|k|^4)`. The mean and excluded Nyquist lines use
`1/a`. This uses only the base-grid FFT and the prepared stage shift; the full
nonlinear Jacobian and unpreconditioned residual determine convergence. The
finite-difference benchmark retains its original solver.

The Fourier energy integrates the quartic potential without aliasing; diagnostics
use the projected chemical potential, not the five-point FD stencil. The plot
validator independently reconstructs both using NumPy on exported snapshots.
Only convex-split Euler enforces the discrete energy bound.

Each numerical batch first runs cutoff-mode, mean, Nyquist and frozen-Jacobian
checks, then integrates the common nonlinear initial condition to t=0.005.
`spatial_validation.log` reports FD errors on 8, 16 and 32 against a
32-by-32 Fourier endpoint at shared nodes, bounds the reference error by space
and time refinement, and checks for second-order FD convergence. The per-method
order plots instead measure *temporal* error against refined ROS3 on the selected
Fourier grid. Equal-resolution FD and Fourier energies need not agree.
