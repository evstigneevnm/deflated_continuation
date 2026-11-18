#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --profile && ${2:-} == ci ]]; then shift 2; fi
exec "${PYTHON:-python3}" "$here/pde_ci.py" "$@"
