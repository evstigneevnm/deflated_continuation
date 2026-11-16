# Periodic Stokes Projection

`periodic_stokes.h` supplies a matrix-free provisional velocity problem and a
fixed Leray projector on an even, two-dimensional periodic Fourier grid. Each R2C
mode stores `(Re ux, Im ux, Re uy, Im uy)`. Real-field coefficients must obey
Hermitian symmetry. Nyquist rows/columns are excluded, consistently with the
project's translation-equivariant Fourier space. The pressure Poisson solve fixes
its mean to zero; velocity means are retained. The problem injects a backend
forcing functor and NMFD Newton
and linear solvers are selected by assembly, not by the spatial operator.

For identity mass, diffusion commutes with the projector. Projecting provisional
DIRK stage states therefore gives the RK discretization of the incompressible
Stokes equation. This does not validate a nonlinear Navier-Stokes splitting,
wall-boundary pressure corrections, nontrivial mass or constrained DAEs.
The stage projection potential is not automatically the physical stage pressure.

The test uses viscosity `0.15` on `[0,2*pi]^2` with

```text
psi = exp(-t)*cos(x)*cos(y) + .12*exp(-.7*t)*cos(3*x)*cos(2*y)
u   = (d_y psi, -d_x psi) + (.125,-.25)
p   = (1+.2*sin(t))*cos(x)*cos(y)
f   = u_t - nu*Delta(u) + grad(p).
```

It checks temporal refinement, independent pressure elimination, adaptive/dense
results, divergence, gradient removal, projector idempotence and preserved means.
An additional 33280-unknown solve guards vector transfers and allocations after
warmup. Spectral diffusion and the pressure Poisson solve take O(N) work and
storage, with no FFT needed for this linear problem. Wave-number tables upload
once at construction; all solve/verification vectors remain backend-resident.

```bash
make -j2 test_periodic_stokes_projection_cpu.bin \
  test_periodic_stokes_projection_cpu_omp.bin \
  CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc \
  BUILD_DIR=build/stokes_projection
OMP_NUM_THREADS=2 build/stokes_projection/test_periodic_stokes_projection_cpu.bin
OMP_NUM_THREADS=2 build/stokes_projection/test_periodic_stokes_projection_cpu_omp.bin
```

Corresponding `_cuda.bin` and `_hip.bin` targets use the same numerical source.
The projection CI jobs include GCC/Clang serial/OpenMP, sanitizers, and
capability-gated CUDA/HIP coverage.

## IMEX Stokes And Oseen

`periodic_stokes_imex.h` is project-owned assembly with independent partition
contexts and shared immutable Fourier geometry. It reuses `periodic_stokes` with
zero forcing for implicit diffusion; explicit `periodic_transport_forcing` evaluates
`-c.grad(u)+f`. Zero constant transport c gives Stokes. Supply the same existing
Leray projector to IMEX; neither partition performs hidden projection or constructs
Newton/linear solvers. The split adds no field buffers.

`test_periodic_stokes_imex_projection` uses the manufactured solution above with
`X=x-cx*t`, `Y=y-cy*t` replacing x,y in psi and p. It tests c=(0,0) and (.7,-.3),
with forcing `u_t+c.grad(u)-nu*Delta(u)+grad(p)`. This exercises moving phases,
state-dependent explicit transport and pressure-gradient removal. Check all five
IMEX pairs against physical temporal orders and independent analytical pressure
elimination, plus adaptive/dense/backward trajectories. A 33280-component Oseen
case guards all solution-cycle transfers and post-warmup backend allocations.
Work remains O(N) per diffusion, transport or projection application, with only
setup geometry uploads. This is not a general nonlinear Navier-Stokes splitting.

```bash
make -j2 test_periodic_stokes_imex_projection_cpu.bin \
  test_periodic_stokes_imex_projection_cpu_omp.bin \
  CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc \
  BUILD_DIR=build/stokes_imex_projection
OMP_NUM_THREADS=2 build/stokes_imex_projection/test_periodic_stokes_imex_projection_cpu.bin
```

The `_cpu_omp.bin`, `_cuda.bin` and `_hip.bin` variants use identical equations.

## Reference Figures

The project-owned runner builds incrementally, runs fixed-step temporal refinement,
validates the exported data, and creates PNG/PDF figures:

```bash
bash source/nonlinear_operators/stokes/tests/run_all.sh
```

Defaults cover all seven DIRK methods for Stokes and all five IMEX pairs for
Stokes/Oseen: 17 cases, five refinements, 16 initial steps over `[0.3,1.3]`.
The spatial grid is fixed at 12x12 and resolves the manufactured modes exactly;
these are temporal, not spatial, order tests. Each case shows initial/midpoint/final
manufactured and numerical velocity fields, with vector directions and error maps.
Group figures compare endpoint physical velocity RMS errors with `h^p` reference
lines and pairwise observed orders. Reference-line vertical offsets are arbitrary,
not error bounds. Fits use the finest three errors above the declared numerical
floor; insufficient resolved data or degraded order is a failed validation.
Pressure is not reconstructed and no numerical pressure accuracy is claimed.

```bash
# Select methods or another backend.
bash source/nonlinear_operators/stokes/tests/run_all.sh \
  --methods CN ARK324L2SA IMEX_ARK324L2SA --backend cpu_omp --threads 2

# Force building and configuration assembly only; no integration or plotting.
bash source/nonlinear_operators/stokes/tests/run_all.sh -B

# Revalidate and redraw a saved run without building or integrating.
bash source/nonlinear_operators/stokes/tests/run_all.sh \
  --plot-only build/stokes_benchmarks/runs/RUN_DIRECTORY
```

All generated files stay under `build/stokes_benchmarks/`, with separate timestamped
runs. Each run contains `order_*.png/pdf`, `order_validation.csv`, `validation.json`
and per-case `solutions.png/pdf`, numerical reports, configurations and logs.
Change `--build-dir`, `--start/--end`, `--base-steps`, `--levels`, `--grid` and
`--plot-grid` as needed. `--methods` selects only supported system/method combinations;
Oseen is the validated IMEX formulation, not an inferred DIRK transport implementation.

`PYTHON` selects the interpreter; NumPy/Matplotlib are required only for plotting,
not `-B`. CUDA/HIP use the same source and accept existing compiler options via
`--config-file` and repeated `--make-option KEY=VALUE`.

The exporter reads only 20 scalar Fourier values at each of three snapshots and
checks the unexported tail on the backend. Parseval-weighted error/divergence
reductions and all timestepper operations remain backend-resident. Bulk transfers
are guarded during integration and scalar export; backend allocation is checked
after warmup. Python reconstructs display fields afterward from the bounded mode
set. This visualization/accuracy runner is not a fair timing benchmark: diagnostics
add reductions and plotting has its own cost. No canonical NMFD code is changed.
The projection CPU CI job runs figure smoke cases and uploads their reports/plots.
