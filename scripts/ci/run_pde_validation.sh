#!/usr/bin/env bash
set -euo pipefail

platform=${1:?expected CPU, CUDA or HIP}
case "$platform" in
    CPU) compiler=${2:?expected the matrix compiler}; dir="$BUILD_DIR/pde/$compiler"; runtime=cpu/openmp ;;
    CUDA|HIP) dir="$BUILD_DIR/pde_${platform,,}"; runtime=${platform,,} ;;
    *) echo "Unknown PDE platform: $platform" >&2; exit 2 ;;
esac
mkdir -p "$dir"
status() {
    printf '{"backend":"%s","executed":%s,"result":"%s"}\n' "$runtime" "$1" "$2" > "$dir/backend.json"
}
run_pde() {
    make test-pde "$@" "PDE_TEST_OPTIONS=--group pde"
}
status false failed
failed=0
if [[ $platform == CPU ]]; then
    "$compiler" --version > "$dir/compiler.txt"
    opts=("CONFIG_FILE=build_configs/config_github_ubuntu_cpu.inc" "G++ = $compiler")
    status true started
    for cpu in SERIAL OMP; do
        run_pde PLATFORM="$cpu" BUILD_DIR="$dir" "${opts[@]}" || failed=1
    done
    ASAN_OPTIONS=detect_leaks=1 run_pde PLATFORM=SERIAL BUILD_DIR="$dir/sanitizers" "${opts[@]}" \
        G++FLAGS="-std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" || failed=1
    chdir="$BUILD_DIR/ch_contracts/$compiler"
    for cpu in SERIAL OMP; do
        make test-ch PLATFORM="$cpu" BUILD_DIR="$chdir" "${opts[@]}" CH_TEST_OPTIONS="--group contracts" || failed=1
    done
    make test-ch PLATFORM=SERIAL BUILD_DIR="$chdir" "${opts[@]}" CH_TEST_OPTIONS="--group sanitizers" || failed=1
else
    source scripts/ci/hosted_gpu_policy.sh
    opts=()
    if [[ $platform == CUDA ]]; then
        select_cuda_compiler "$dir/backend.json" > >(tee "$dir/device.txt") 2>&1
        "$NVCC" --version > "$dir/compiler.txt"
        opts+=("NVCC=$NVCC")
    else
        select_hip_compiler "$dir" > >(tee "$dir/device.txt") 2>&1
        if grep -qi nvcc "$dir/compiler.txt"; then runtime=hip-on-nvidia; else runtime=hip-on-amd; fi
        opts+=("HIPCC=$HIPCC" "HIP_ROOT_PATH=$HIP_ROOT_PATH" "HIP_PATH=$HIP_PATH" "HIP_ARCH_LIST=$HIP_ARCH_LIST")
    fi
    opts+=("CONFIG_FILE=$CONFIG_FILE")
    status false started
    summary="$dir/$platform/pde/summary.json"
    mkdir -p "$(dirname "$summary")"
    # A Make/configuration failure before the runner starts must not reuse an old pass.
    printf '{"status":"not_started","passed":false,"results":{}}\n' > "$summary"
    if ! run_pde PLATFORM="$platform" BUILD_DIR="$dir" "${opts[@]}"; then
        failed=1
        # Do not start the CH suite if the selected device could not execute the probe.
        if ! python3 -c 'import json,sys; sys.exit(not bool(json.load(open(sys.argv[1]))["results"]))' \
                "$summary"; then
            status false failed
            exit 1
        fi
    fi
    make test-ch PLATFORM="$platform" BUILD_DIR="$dir/ch" "${opts[@]}" CH_TEST_OPTIONS="--group contracts" || failed=1
fi
if (( failed )); then status true failed; exit 1; fi
status true passed
