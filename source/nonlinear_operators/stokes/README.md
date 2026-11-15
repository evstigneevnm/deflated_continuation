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
