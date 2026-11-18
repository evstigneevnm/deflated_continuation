#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
if [[ -z ${PYTHON:-} ]] && ! "$python" -c 'import importlib.util; assert all(importlib.util.find_spec(name) for name in ("numpy", "matplotlib"))' >/dev/null 2>&1; then
    fallback="$HOME/anaconda3/bin/python"
    if [[ -x $fallback ]]; then python="$fallback"; fi
fi
if [[ ${1:-} == --profile ]]; then
    [[ ${2:-} == ci ]] || { echo "Expected --profile ci" >&2; exit 2; }
    shift 2
    exec "$python" "$here/ch_ci.py" "$@"
fi
exec "$python" "$here/ch_benchmarks.py" "$@"
