# Directory CI suites

## Cahn-Hilliard

The Cahn-Hilliard (CH) [case list](../source/nonlinear_operators/cahn_hilliard/tests/ch_ci.py) is shared by local runs and CI. The default [launcher](../source/nonlinear_operators/cahn_hilliard/tests/run_all.sh) is still the configurable plotting benchmark; put `--profile ci` first to use the registered CI cases.

```bash
suite=source/nonlinear_operators/cahn_hilliard/tests/run_all.sh
PLATFORM=SERIAL bash "$suite" --profile ci --list
PLATFORM=OMP bash "$suite" --profile ci --group benchmarks
make test-ch PLATFORM=CUDA CH_TEST_OPTIONS='--group contracts'
```

| PLATFORM | Existing binary suffix | Hosted pilot coverage |
| --- | --- | --- |
| SERIAL (default) | cpu | Four contracts, GCC and Clang; also ASan/UBSan for both. GCC: two FD batches and the short Fourier batch. |
| OMP | cpu_omp | Same contracts on both compilers. GCC: both FD batches and the long Fourier Euler regression. |
| CUDA | cuda | Four contracts when the hosted compiler and device are available. |
| HIP | hip | Same four contracts; use the configured AMD or NVIDIA HIP toolchain. |

The four contracts are split residual/Jacobian, energy, split lifecycle and unsplit methods. The FD batches cover ten methods and a separate four-method adaptive comparison. Short Fourier covers IMEX Euler, CN, BDF and ROS34PW2. Long Fourier covers IE and IMEX Euler at grid 32, through time 60 with step 0.1. These are numerical validations, not just builds. The generic mixed CH example remains in the PDE tests, and NMFD-newton's own CH examples and MPI jobs are separate.

`--group all` runs contracts and benchmarks for the selected platform. Use `--group sanitizers` separately on SERIAL. `-B` builds only; it neither probes hardware nor counts as an execution pass. Set `--config-file` and repeat `--make-option KEY=VALUE` for the site toolchain. Use a separate build root per compiler. The suite writes a status per case in `PLATFORM/group/summary.json`, keeps logs for failed cases, and exits nonzero if any case fails. The plotting runner still saves timestamped numerical results and figures.

## Explicit Runge-Kutta

The timestepper module owns the [timestepper case inventory](../source/contrib/nmfd-timesteppers/tests/timestepper_ci.py); the parent Make target supplies the same build configuration as its other tests. For example:

```bash
make test-explicit-rk PLATFORM=OMP CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc BUILD_DIR=build/explicit_ci
make test-explicit-rk PLATFORM=OMP EXPLICIT_RK_OPTIONS='--group dop853-reference --list'
```

The default group runs 16 numerical cases on each of SERIAL, OMP and CUDA, plus the coefficient tables on SERIAL and the DOP853 Lorenz trajectory on CUDA. HIP was not in this explicit RK matrix and is rejected rather than silently skipped. CPU and CUDA public-header builds and the SCFD CUDA reduction remain parent checks. Three CPU reference groups (`dop853-reference`, `lorenz-reference`, `rk4-reference`) retain the independent Python comparisons; the mixed Lorenz job also retains its implicit benchmarks.

Each group reports per-case results in `BUILD_DIR/PLATFORM/group/summary.json` and writes a log per case. A failed case makes the group fail without hiding later cases. Core workflow artifacts are named `explicit-rk-SERIAL`, `explicit-rk-OMP` and `explicit-rk-CUDA`. An unavailable optional hosted GPU publishes `backend.json`, not a numerical pass. Use `EXPLICIT_RK_OPTIONS=-B` for a forced build without execution.

## Implicit Runge-Kutta

| CI group | Cases on SERIAL, OMP and CUDA |
| --- | --- |
| `dirk-core` | DIRK, adaptation, mass/DAE, lifecycle, non-autonomous, Lorenz, Rossler and Van der Pol. SERIAL also runs the implicit coefficient tables. |
| `dirk-extensions` | Three ESDIRK, three dense-output, globalization and four steady-state cases. |

```bash
make test-rk PLATFORM=OMP CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc \
    BUILD_DIR=build/dirk_ci RK_TEST_OPTIONS='--group dirk-extensions'
make test-rk PLATFORM=CUDA RK_TEST_OPTIONS='--group dirk-core --list'
```

Both groups use the same per-case logs and summaries as explicit RK. Artifacts are named `<group>-<platform>`. CPU cases use two OpenMP threads and the original 120/180-second budgets; CUDA cases keep their 300-second budgets and require the device probe before execution. HIP was not registered for these cases. The six sanitizer groups remain independent.

## IMEX Runge-Kutta

The `imex` group runs eight cases: fixed and adaptive stepping, mass solves, lifecycle, backend operations, benchmarks, dense output and dense-output lifecycle. GCC and Clang each check the public headers and run the group on SERIAL and OMP. CUDA and HIP each run it once, after a kernel probe. This preserves 48 numerical executions on equipped runners without adding graph nodes. CPU cases use two OpenMP threads and a 180-second timeout; GPU cases retain 300 seconds.

```bash
make test-rk PLATFORM=OMP 'G++ = clang++' CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc \
    BUILD_DIR=build/imex_clang RK_TEST_OPTIONS='--group imex'
make test-rk PLATFORM=HIP RK_TEST_OPTIONS='--group imex'  # use the site's HIP config
```

Keep the space before `=` in `G++ = clang++`: without it Make parses an append assignment. Use a separate build directory for each compiler. The standalone module Makefile provides SERIAL, OMP and CUDA targets; use the parent target for HIP. Reports are under `BUILD_DIR/PLATFORM/imex/`; workflow artifacts are `imex-g++`, `imex-clang++`, `imex-CUDA` and `imex-HIP`. For an optional hosted skip, inspect `backend.json` instead of a numerical summary.

## BDF

The `bdf` group runs 13 cases covering fixed and adaptive stepping, rollback, mass/DAE systems, backend transfer guards, variable order, stability and dense output. GCC and Clang each check the public headers and execute all cases on SERIAL and OMP (two threads, 180 seconds per case). CUDA and HIP each run the same cases with a 300-second limit after a device probe. The six compiler/platform combinations give 78 numerical executions on equipped runners. Sanitizers and the BDF trajectory benchmarks remain separate.

```bash
make test-timesteppers PLATFORM=OMP 'G++ = clang++' CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc \
    BUILD_DIR=build/bdf_clang TIMESTEPPER_TEST_OPTIONS='--group bdf'
make test-timesteppers PLATFORM=CUDA TIMESTEPPER_TEST_OPTIONS='--group bdf --list'
```

Logs and `summary.json` are under `BUILD_DIR/PLATFORM/bdf/`. Workflow artifacts are `bdf-g++`, `bdf-clang++`, `bdf-CUDA` and `bdf-HIP`. An optional hosted GPU skip produces `backend.json`, not a numerical pass. `test-rk` and `test-explicit-rk` retain their original defaults and option variables.

## Required vs optional hardware

Select the expected platform explicitly; never infer the test list from installed compilers. A requested local/self-hosted GPU must compile, link, launch and synchronize a kernel and read the expected value before running numerical cases. A compile or runtime failure is fatal; no CPU fallback is allowed. The direct suite is always strict. Only the optional accelerator steps in the hosted workflow may skip for absent hardware/toolchains, and only with an explicit `optional-on-hosted` policy **and** GitHub's [hosted runner identity](https://docs.github.com/en/actions/reference/workflows-and-actions/variables). Once selected for execution, even a hosted probe or test failure is fatal.

To dispatch [self-hosted CH validation](../.github/workflows/ch-self-hosted.yml), register a Linux runner with a label matching the platform and choose that platform and group in **Run workflow**. Configure `CH_CI_CONFIG` on the runner if toolkit paths or architecture flags differ from the default. No toolchain is installed by the workflow. A missing runner stays queued; a broken installation on a matching runner fails. This workflow never runs automatically on pull requests.

## Migration inventory

At the pilot baseline the parent [Operations workflow](../.github/workflows/operations.yml) had 189 job definitions before matrix expansion, including 70 timestepper jobs. Run `python3 run_local_ci.py --list` for the current inventory. The GCC Stokes/Oseen and CH figures now share a job independent of projection compilation; their artifact names are unchanged. The hosted CPU/compiler, optional GPU and figures jobs all feed `all-tests-passed` before the symmetric KS1D N=64 reference job.

Implicit sanitizer tests run in six method-family groups. Each runner builds at most two binaries concurrently, then runs its cases one at a time with ASan/UBSan and the existing 90-second per-binary timeout. To reproduce a group locally, use its `bash scripts/ci/run_timestepper_sanitizers.sh ...` command and sanitizer environment from the workflow.

The explicit RK batch replaced 27 core jobs with three platform runs under two job definitions. The DIRK batch replaced 23 jobs with six platform/group runs under two definitions, preserving 58 binary executions when CUDA is available. Together these changed Operations from 190 job definitions (222 after matrix expansion) to 144 (181 after expansion). Independent reference, sanitizer and final KS1D gates remain separate. Subsequent batches should inventory cases and compare hosted runtime before regrouping another family. The hardware policy above is not a claim of strict GPU coverage for unmigrated legacy jobs.
