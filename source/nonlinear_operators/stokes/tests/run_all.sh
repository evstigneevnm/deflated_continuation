#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
if [[ ${1:-} == --profile && ${2:-} == ci ]]; then
    shift 2
    exec "$python" "$here/projection_ci.py" "$@"
fi
if [[ -z ${PYTHON:-} ]] && ! "$python" -c 'import importlib.util; assert all(importlib.util.find_spec(name) for name in ("numpy", "matplotlib"))' >/dev/null 2>&1; then
    fallback="$HOME/anaconda3/bin/python"
    if [[ -x $fallback ]]; then python="$fallback"; fi
fi
exec "$python" "$here/stokes_benchmarks.py" "$@"
