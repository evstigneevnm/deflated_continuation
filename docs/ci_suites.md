# Directory CI suites

The first suite is Cahn-Hilliard (CH). Its [case list](../source/nonlinear_operators/cahn_hilliard/tests/ch_ci.py) is shared by local runs and CI. The default [launcher](../source/nonlinear_operators/cahn_hilliard/tests/run_all.sh) is still the configurable plotting benchmark; put `--profile ci` first to use the registered CI cases.

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

## Required vs optional hardware

Select the expected platform explicitly; never infer the test list from installed compilers. A requested local/self-hosted GPU must compile, link, launch and synchronize a kernel and read the expected value before running numerical cases. A compile or runtime failure is fatal; no CPU fallback is allowed. The direct suite is always strict. Only the optional accelerator steps in the hosted workflow may skip for absent hardware/toolchains, and only with an explicit `optional-on-hosted` policy **and** GitHub's [hosted runner identity](https://docs.github.com/en/actions/reference/workflows-and-actions/variables). Once selected for execution, even a hosted probe or test failure is fatal.

To dispatch [self-hosted CH validation](../.github/workflows/ch-self-hosted.yml), register a Linux runner with a label matching the platform and choose that platform and group in **Run workflow**. Configure `CH_CI_CONFIG` on the runner if toolkit paths or architecture flags differ from the default. No toolchain is installed by the workflow. A missing runner stays queued; a broken installation on a matching runner fails. This workflow never runs automatically on pull requests.

## Migration inventory

At the pilot baseline the parent [Operations workflow](../.github/workflows/operations.yml) has 189 job definitions before matrix expansion, including 70 timestepper jobs. Run `python3 run_local_ci.py --list` for the full job inventory. The CH cases were steps inside shared jobs, not separate graph nodes: the pilot retains those scheduling and artifact boundaries. The hosted CPU/compiler, optional GPU and figures jobs all still feed `all-tests-passed` before the symmetric KS1D N=64 reference job.

Subsequent batches should inventory actual cases inside each job (including compiler variants, sanitizers, Python checks, and backend-specific exceptions) before removing its graph nodes. The CH policy is not yet a claim of strict GPU coverage for unmigrated legacy jobs.
