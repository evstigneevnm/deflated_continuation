# Source only in jobs that intentionally allow an absent GPU on GitHub-hosted runners.
# A local replay or a self-hosted job must not inherit the hosted skip.
gpu_skip_allowed() {
    [[ ${GPU_POLICY:-required} == optional-on-hosted &&
       ${RUNNER_ENVIRONMENT:-} == github-hosted && ${GITHUB_ACTIONS:-} == true ]]
}
gpu_unavailable() {
    local platform=$1 report=$2 reason=$3 result=failed code=1
    if gpu_skip_allowed; then
        result=skipped
        code=0
    fi
    printf '{"backend":"%s","executed":false,"result":"%s","reason":"%s"}\n' \
        "$platform" "$result" "$reason" | tee "$report"
    exit "$code"
}

# Export the selected compiler for Make; only the optional hosted lane may skip.
select_cuda_compiler() {
    local report=$1 compiler
    compiler=$(command -v "${NVCC:-nvcc}" || true)
    if [[ -z ${NVCC:-} && -z $compiler && -x /usr/local/cuda/bin/nvcc ]]; then
        compiler=/usr/local/cuda/bin/nvcc
    fi
    if [[ -z $compiler ]]; then gpu_unavailable cuda "$report" "compiler unavailable"; fi
    if gpu_skip_allowed && ! nvidia-smi -L; then gpu_unavailable cuda "$report" "device unavailable"; fi
    export NVCC="$compiler"
}

# HIP on NVIDIA and HIP on AMD use different architecture flags. A broken
# compiler is a failure even on an optional runner; only absence may skip.
select_hip_compiler() {
    local dir=$1 compiler root arch
    compiler=$(command -v "${HIPCC:-hipcc}" || true)
    if [[ -z $compiler ]]; then gpu_unavailable hip "$dir/backend.json" "compiler unavailable"; fi
    root="${ROCM_PATH:-$(dirname "$(dirname "$compiler")")}"
    "$compiler" --version > "$dir/compiler.txt" || return 1
    if grep -qi nvcc "$dir/compiler.txt"; then
        arch="${HIP_ARCH_LIST:-75}"
        if gpu_skip_allowed && ! nvidia-smi -L; then gpu_unavailable hip "$dir/backend.json" "device unavailable"; fi
    else
        arch="${HIP_ARCH_LIST:-native}"
        if gpu_skip_allowed && ! "$root/bin/rocm-smi"; then gpu_unavailable hip "$dir/backend.json" "device unavailable"; fi
    fi
    export HIPCC="$compiler" HIP_ROOT_PATH="$root" HIP_PATH="$root" HIP_ARCH_LIST="$arch"
}
