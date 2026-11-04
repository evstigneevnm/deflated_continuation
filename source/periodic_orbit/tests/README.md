# Periodic Orbit Validation

`analytic_cycle` validates multiple-section matching of a known isolated
unit-circle orbit and non-orbit perturbations (`stage: matching_only`).
`analytic_cycle_newton` corrects perturbed section states using NMFD Newton/GMRES
and a prepared matrix-free derivative (`stage: periodic_newton`). Neither claims
general periodic-orbit discovery. Rossler/Lorenz correction follows separately.

From the project root:

```bash
bash source/periodic_orbit/tests/run_all.sh
bash source/periodic_orbit/tests/run_all.sh -B
bash source/periodic_orbit/tests/run_all.sh --backend cpu_omp --threads 2 --methods RK45 DOP853
bash source/periodic_orbit/tests/run_all.sh --system analytic_cycle_newton --methods RK45 DOP853
bash source/periodic_orbit/tests/run_all.sh --plot-only build/periodic_orbit_benchmarks/runs/RUN_DIRECTORY
```

Default section counts are 1, 2, 4, 8; the default dense-output method is DOP853.
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
tests forbid vector-sized host copies. Benchmark plotting explicitly exports two
scalar coordinates; this I/O is outside residual evaluation. No legacy headers,
project Newton implementation, or canonical solver modifications are used here.
