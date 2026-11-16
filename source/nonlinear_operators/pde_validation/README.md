# PDE Contract Validation

Three small project-owned problems exercise the unchanged NMFD timestepper
interfaces. Spatial work uses the periodic five-point Laplacian in
`source/discretization/finite_difference`; state layout is interleaved by cell.
Every stencil/rate/residual/Jv is O(N). No assembled N-by-N matrix, FFT dependency,
vector-sized host/device transfer or per-step backend allocation is introduced.
The injected GMRES basis is fixed; its own iteration count is not claimed O(1).
These are single-process grids; distributed halo exchange is not exercised.
Two-argument stencil/rate/residual/Jv evaluations require distinct input/output
storage, as supplied by the timestepper and solver workspaces.

## Equations And Interfaces

- `diffusion.h`: forced heat equation with two exact discrete harmonics and mean
  0.3. Rates contain the physical RHS; affine coefficients never enter them.
- `reaction_diffusion.h`: two-field forced Brusselator. Nonlinear reactions and
  manufactured forcing are explicit in IMEX, diffusion implicit. The unsplit
  problem remains unchanged for other methods. The time derivative is partial
  at fixed state, not a derivative along the exact solution.
- `cahn_hilliard.h`: autonomous linear mixed index-1 CH, with state `(c,mu)`.
  `R_c=a*(c-w_c)-b*(m*Delta_h(mu)+r_c)`;
  `R_mu=mu-kappa*c+epsilon2*Delta_h(c)`. Algebraic rows remain unscaled and ignore
  algebraic reference/RHS slots. Consistent initialization and independent mean/
  constraint checks are problem-owned. No rate provider or mass inverse exists.

`ode_problem.h` shares immutable model data but gives each facade its own stage
reference, forcing buffer, time and frozen Jacobian. `previous_state()` is a
stage reference, not necessarily the last committed state. The IMEX bundle is
retained once and contains independent explicit/implicit contexts. Residual,
rate and partial-time evaluations never mutate prepared Jacobians. Problems do
not construct solvers; assembly injects ordinary NMFD Newton/GMRES.
The standalone rate provider and IMEX explicit partition allocate no implicit
stage/Jacobian workspace, and explicit assembly constructs no linear solver.

Forcing uses analytical discrete Laplacian eigenvalues, not the stencil under
test. Refinement therefore measures temporal error in a fixed semidiscrete PDE,
not a mixture of spatial and temporal error. Small grids exercise genuine
neighbor coupling; larger 64x64 cases guard solution-cycle transfers/allocations.
They are short smoke tests, not stiff-PDE efficiency benchmarks.

## Verification

Each system has its own `tests/test_*.cpp`. Tests check shifted Jacobians,
nonzero times/RHS, frozen preparation, forward/backward trajectories, physical
error, rollback (including aliasing), restart and shortened-view invalidation.
Diffusion/reaction tests cover all five engine families, adaptive refinement and
dense samples. CH covers suitable stiffly accurate DIRK, BDF and the autonomous
ROS34PW2 mass/index-1 selection, with mean and algebraic checks at every endpoint.
ESDIRK without a consistent-rate provider is explicitly rejected. No general
singular-mass IMEX, nonlinear CH energy law or raw DAE dense accuracy is claimed.

```bash
make -j2 test_pde_diffusion_cpu.bin test_pde_reaction_diffusion_cpu.bin \
  test_pde_cahn_hilliard_cpu.bin \
  CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc BUILD_DIR=build/pde_validation
OMP_NUM_THREADS=2 build/pde_validation/test_pde_diffusion_cpu.bin
```

`_cpu_omp`, `_cuda` and `_hip` targets use the same sources. Parent CI uploads
logs and explicit backend execution/skip records even on failure. No downloads
or plotting dependencies are needed for these numerical gates. The authoritative
[capability matrix](../../contrib/nmfd-timesteppers/docs/capabilities.md) links
library evidence separately from project PDE and shooting capabilities.
