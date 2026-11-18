#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    echo "Expected at least one timestepper test name" >&2
    exit 2
fi
dir="${BUILD_DIR:-build/local_ci_operations}/implicit_sanitizers"
targets=()
for test in "$@"; do targets+=("test_${test}_cpu.bin"); done
make -j2 "${targets[@]}" CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc BUILD_DIR="$dir" \
    "G++FLAGS=-std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
for target in "${targets[@]}"; do
    OMP_NUM_THREADS=2 timeout 90s "$dir/$target"
done
