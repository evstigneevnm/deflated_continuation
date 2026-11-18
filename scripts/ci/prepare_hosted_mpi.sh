#!/usr/bin/env bash
set -euo pipefail

work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

probe() {
    local compiler=${MPICXX:-mpicxx} launcher=${MPIEXEC:-mpiexec}
    command -v "$compiler" >/dev/null && command -v "$launcher" >/dev/null &&
        command -v make >/dev/null && command -v g++ >/dev/null || return 1
    "$compiler" -x c++ - -o "$work/probe" <<'CPP' || return 1
#include <mpi.h>
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int one = 1, total = 0;
    MPI_Allreduce(&one, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Finalize();
    return total == 2 ? 0 : 1;
}
CPP
    timeout -k 5s 30s "$launcher" --oversubscribe -n 2 "$work/probe"
}

if probe; then
    echo "Existing MPI compiler and two-rank runtime work; skipping apt."
    exit 0
fi
if [[ ${GITHUB_ACTIONS:-} != true || ${RUNNER_ENVIRONMENT:-} != github-hosted ]]; then
    echo "MPI is required; automatic installation is restricted to GitHub-hosted runners." >&2
    exit 1
fi

packages=(build-essential make openmpi-bin libopenmpi-dev)
options=(-o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20
         -o Acquire::Languages=none -o APT::Update::Error-Mode=any)
sources=()

# A read timeout does not bound a mirror that keeps delivering a few bytes.
# Apply wall-clock limits only to fetching, never interrupt an active dpkg transaction to retry.
fetch() {
    sudo -n timeout -k 5s "$1" apt-get "${options[@]}" "${sources[@]}" update &&
        sudo -n timeout -k 5s "$2" apt-get "${options[@]}" "${sources[@]}" --download-only \
            install -y --no-install-recommends "${packages[@]}"
}

echo "Fetching Open MPI from the configured runner mirrors."
if ! fetch 90s 150s; then
    echo "Runner mirror fetch did not finish; retrying through the signed Ubuntu Noble archives." >&2
    # Private to these apt invocations; do not rewrite the runner's configured mirrors.
    key=/usr/share/keyrings/ubuntu-archive-keyring.gpg
    printf 'deb [signed-by=%s] https://archive.ubuntu.com/ubuntu %s main universe\n' \
        "$key" noble "$key" noble-updates > "$work/noble.list"
    printf 'deb [signed-by=%s] https://security.ubuntu.com/ubuntu noble-security main universe\n' \
        "$key" >> "$work/noble.list"
    sources=(-o "Dir::Etc::sourcelist=$work/noble.list" -o Dir::Etc::sourceparts=-
             -o APT::Get::List-Cleanup=false)
    fetch 120s 300s
fi

sudo -n apt-get "${options[@]}" "${sources[@]}" --no-download install -y --no-install-recommends "${packages[@]}"
probe
