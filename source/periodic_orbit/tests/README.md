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
Prepared derivatives replay frozen explicit RK stages and differentiate their
dense section crossing. Implicit/DAE tangents and relative closure are not
silently inferred. Core tests include non-autonomous stage times, adaptive mesh
recording, linearity, finite differences, rollback and retained-Jacobian lifetime.

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
