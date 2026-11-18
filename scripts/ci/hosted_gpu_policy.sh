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
