#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
if [[ -z ${PYTHON:-} ]] && ! "$python" -c 'import importlib.util; assert all(importlib.util.find_spec(name) for name in ("numpy", "matplotlib"))' >/dev/null 2>&1; then
    fallback="$HOME/anaconda3/bin/python"
    if [[ -x $fallback ]]; then python="$fallback"; fi
fi
exec "$python" "$here/orbit_benchmarks.py" "$@"
