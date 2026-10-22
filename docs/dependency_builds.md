# Dependency Builds

The parent pins five direct submodules under `source/contrib`: `scfd`,
`nmfd-operations`, `nmfd-linsolvers`, `nmfd-newton`, and `nmfd-timesteppers`.
Parent CI initializes only these checkouts and supplies their absolute paths
to all library builds.
No dependency repository's nested checkout is used by parent integration tests.

## Local Integration

Use current working trees; do not update dependencies automatically:

```sh
export SCFD_ROOT="$PWD/source/contrib/scfd"
export NMFD_OPERATIONS_ROOT="$PWD/source/contrib/nmfd-operations"
export NMFD_LINSOLVERS_ROOT="$PWD/source/contrib/nmfd-linsolvers"
make -C source/contrib/nmfd-operations -j2 test-cpu
make -C source/contrib/nmfd-linsolvers -j2 run_tests_cpu
make -C source/contrib/nmfd-newton -j2 test_cpu check_ch_original
```

The existing local runner also expands these roots from the parent workflow:

```sh
python3 run_local_ci.py --job nmfd-newton-operations-cpu
```

## Standalone CI

Operations, linear-solvers, and Newton retain direct pinned submodules for
standalone builds. Each Makefile defaults to those direct dependencies and accepts
the root overrides above. Library builds do not require recursive initialization.

- Pinned jobs test committed gitlinks, without `--remote`.
- `latest-dependencies` tests the designated `.gitmodules` branch tips. It runs
  on pushes, pull requests, manual dispatch, and daily from the default branch.
- Both modes use fresh CI checkouts. Latest jobs log all direct revisions and
  never stage, commit, or push dependency updates.

To reproduce latest-dependency selection, use an isolated clone of the library:

```sh
git submodule update --init --checkout
git submodule foreach '
  branch=$(git config -f "$toplevel/.gitmodules" "submodule.$name.branch") &&
  git fetch --no-tags --depth=1 origin "+refs/heads/$branch:refs/remotes/origin/$branch"
'
git submodule update --init --remote --no-fetch --checkout
git rev-parse HEAD
git submodule status
```

SCFD tracks `nevstigneev`, operations tracks `master`, and linear solvers tracks
`nevstigneev/sync_with_bifurcator`. Branch choices do not change the committed
pins. Promote tested revisions through normal reviewed commits; commit library
changes before updating their parent-project gitlinks.

## Timesteppers Development

`nmfd-timesteppers` currently uses flat sibling dependencies, without nested
submodules or separate repository workflows. Existing parent timestepper CI jobs
build its tests, including CPU/OpenMP/CUDA, public headers, and Python comparisons.
They do not add NMFD-newton or NMFD-linsolvers include paths to explicit builds.

```sh
make -C source/contrib/nmfd-timesteppers -j2 test_cpu check_headers_cpu
make test_explicit_lorenz_simulation_cpu.bin CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc
```

Set `SCFD_ROOT`, `NMFD_OPERATIONS_ROOT`, or `JSON_INCLUDE` for other flat layouts.
Standalone dependency pins and repository CI are deferred until the interface
stabilizes. Commit and push the extracted module before committing its parent
gitlink; otherwise a fresh checkout cannot obtain the extracted code.

## Migration Validation

Validate committed pins independently of the flat integration environment:

```sh
python3 scripts/validate_nmfd_standalone.py --library nmfd-operations --compiler gcc
python3 scripts/validate_nmfd_standalone.py --library nmfd-linsolvers --compiler clang
python3 scripts/validate_nmfd_standalone.py --library nmfd-newton --suite mpi --mpi-ranks 4
python3 scripts/validate_nmfd_standalone.py --library nmfd-newton --suite cuda
```

Each invocation creates a fresh checkout under `build/nmfd_standalone`, using the
library gitlink committed in the parent and that library's direct dependency pins.
Parent root overrides, inherited Make settings, and implicit compiler include
paths are cleared. Nested checkouts and dirty source trees are never reused or
modified. Missing pinned objects are fetched by exact commit, never by branch tip.
Normal toolchain and optional third-party settings such as `MPICXX`, `MPIEXEC`,
`MPIEXEC_FLAGS`, `BOOST_ROOT`, and `JSON_INCLUDE` may be supplied explicitly.

The checkout and `report.json` remain available for inspection. Reports contain
the selected revisions, command, result, and elapsed time; build-induced changes
to tracked sources or dependency revisions fail validation. A timeout or missing
GPU/MPI runtime is a failure, not a passing numerical result.

Parent CI runs separate pinned CPU jobs for all three libraries with GCC/Clang,
and MPI jobs for operations/Newton with 1, 2, and 4 ranks. These supplement flat
integration and standalone latest-dependency testing. The local workflow runner
does not expand GitHub matrices; invoke the script for each matrix entry locally.
`nmfd-migration-validation-cpu` aggregates the required CPU/MPI jobs and fails if
either required group fails or is skipped. GPU jobs can
skip on hosted CPU runners; a green CPU gate is not proof of GPU execution.

Declare migration closure only after local checks pass, the parent changes are
committed, and CI passes for that exact commit. Keep generated reports in `build`,
not in validation reference assets. Archived historical applications are not part
of the supported gate; the frozen legacy VDP example still references the removed
`common/scfd_serial_cpu_vector_operations.h`.

### Local Checkpoint

Local validation passed for these library pins. Final migration declaration is
pending the parent commit and successful CI for that commit.

| Library | Revision |
| --- | --- |
| SCFD | `8c46aad516987c77267fe76232e88a10a50e520f` |
| NMFD-operations | `245f8e5ce7e50c0ab270064565ce0f1842982f5e` |
| NMFD-linsolvers | `8bb1e1220c167b94091f27109a79186390aad09d` |
| NMFD-newton | `06783b7b478a09289ab808a9f347258aee7b1ce2` |

- Fresh pinned CPU/OpenMP builds: all three libraries with GCC and Clang.
- Fresh pinned MPI builds: operations and original CH tests, each on 1/2/4 ranks.
- Fresh pinned CUDA builds: operations, linear solvers, and Newton/CH compatibility.
- HIP-on-NVIDIA: float/double vector operations, closed-circle continuation,
  stability/restart, and closed star-shaped continuation.
- Parent integration: 22 jobs, including KS references, Bratu FD3, multiprecision,
  bordered/direct solves, ownership, and continuation/stability/restart.
- Newton, workflow ownership, solver-bundle, and recovery sanitizer/leak checks.

Evidence is under `build/migration_final*`; isolated checkouts retain `report.json`.
Newton CUDA had one cuBLAS initialization failure: the failing executable and a
complete fresh serial rerun passed. The first failed report remains preserved;
validation does not retry automatically. Native AMD hardware was not available
locally, so HIP-on-NVIDIA is not claimed as native AMD validation.

The HIP gate exposed and fixed combined source/object compilation and missing
star-shaped HIP link dependencies. Regression checks now enforce separate compile
and link commands. No production numerical formulas, solver tolerances, CH test
logic, reference assets, frozen timestepper files, or archived solver
implementations were changed.
