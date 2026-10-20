# Dependency Builds

The parent pins four direct submodules under `source/contrib`: `scfd`,
`nmfd-operations`, `nmfd-linsolvers`, and `nmfd-newton`. Parent CI initializes
only these checkouts and supplies their absolute paths to all library builds.
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

Each library retains direct pinned submodules for standalone builds. Its
Makefile defaults to those direct dependencies and accepts the root overrides
above. Library builds do not require recursive initialization.

- Pinned jobs test committed gitlinks, without `--remote`.
- `latest-dependencies` tests the designated `.gitmodules` branch tips. It runs
  on pushes, pull requests, manual dispatch, and daily from the default branch.
- Both modes use fresh CI checkouts. Latest jobs log all direct revisions and
  never stage, commit, or push dependency updates.

To reproduce latest-dependency selection, use an isolated clone of the library:

```sh
git submodule update --init --remote --checkout
git rev-parse HEAD
git submodule status
```

SCFD tracks `nevstigneev`, operations tracks `master`, and linear solvers tracks
`nevstigneev/sync_with_bifurcator`. Branch choices do not change the committed
pins. Promote tested revisions through normal reviewed commits; commit library
changes before updating their parent-project gitlinks.
