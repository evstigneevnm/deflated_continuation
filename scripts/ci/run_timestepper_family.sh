#!/usr/bin/env bash
set -euo pipefail

family=${1:?expected a timestepper group}
backend=${2:?expected CPU, CUDA or HIP}
options=("TIMESTEPPER_TEST_OPTIONS=--group $family")
target=test-timesteppers
if [[ $family == projection ]]; then target=test-projection; fi
case "$backend" in
    CPU)
        compiler=${3:?expected the matrix compiler}
        options+=("G++ = $compiler" "CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc"
                  "BUILD_DIR=$BUILD_DIR/$family/$compiler")
        if [[ $family != projection ]]; then make check_timestepper_headers_cpu "${options[@]}"; fi
        failed=0
        for platform in SERIAL OMP; do
            make "$target" PLATFORM="$platform" "${options[@]}" || failed=1
        done
        exit "$failed"
        ;;
    CUDA|HIP)
        source scripts/ci/hosted_gpu_policy.sh
        dir="$BUILD_DIR/${family}_${backend,,}"
        mkdir -p "$dir"
        if [[ $backend == CUDA ]]; then
            select_cuda_compiler "$dir/backend.json"
            options+=("NVCC=$NVCC")
        else
            select_hip_compiler "$dir"
            options+=("HIPCC=$HIPCC" "HIP_ROOT_PATH=$HIP_ROOT_PATH" "HIP_PATH=$HIP_PATH" "HIP_ARCH_LIST=$HIP_ARCH_LIST")
        fi
        make "$target" PLATFORM="$backend" CONFIG_FILE="$CONFIG_FILE" BUILD_DIR="$dir" "${options[@]}"
        ;;
    *) echo "Unknown timestepper CI backend: $backend" >&2; exit 2 ;;
esac
