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
