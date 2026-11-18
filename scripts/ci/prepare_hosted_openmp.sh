#!/usr/bin/env bash
set -euo pipefail

if [[ ${RUNNER_ENVIRONMENT:-} != github-hosted || ${GITHUB_ACTIONS:-} != true ]]; then exit 0; fi
probe="$RUNNER_TEMP/timestepper_openmp"
if printf '#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n' | \
   clang++ -fopenmp -x c++ - -o "$probe" && "$probe"; then
    exit 0
fi
sudo apt-get -o Acquire::Retries=1 update
sudo apt-get -o Acquire::Retries=1 install -y --no-install-recommends libomp-dev
