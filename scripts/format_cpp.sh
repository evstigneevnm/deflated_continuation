#!/usr/bin/env bash
set -euo pipefail

root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$root"

options=(-i)
case "${1:-}" in
    "") ;;
    --check) options=(--dry-run --Werror) ;;
    *) printf 'Usage: bash scripts/format_cpp.sh [--check]\n' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    printf 'Usage: bash scripts/format_cpp.sh [--check]\n' >&2
    exit 2
fi

formatter="${CLANG_FORMAT:-clang-format-21}"
if ! command -v "$formatter" >/dev/null 2>&1; then
    printf '%s is unavailable; set CLANG_FORMAT to the formatter executable.\n' "$formatter" >&2
    exit 1
fi

files=()
while IFS= read -r -d '' path; do
    case "$path" in
        source/contrib/*|source/time_stepper/legacy/*|build/*|data/*|dat_files/*) continue ;;
    esac
    if [[ ! -f "$path" || -L "$path" ]]; then
        continue
    fi
    case "$path" in
        *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.hxx|*.cu|*.cuh|*.ipp|*.tpp) files+=("$path") ;;
    esac
done < <(git ls-files -z --cached --others --exclude-standard)

printf 'Formatting scope: %d C/C++ files\n' "${#files[@]}"
"$formatter" --version
if (( ${#files[@]} > 0 )); then
    mapfile -d '' crlf_files < <(LC_ALL=C grep -IlZ $'\r$' -- "${files[@]}")
    if (( ${#crlf_files[@]} > 0 )); then
        if [[ "${1:-}" == --check ]]; then
            printf 'CRLF source file: %s\n' "${crlf_files[@]}" >&2
            exit 1
        fi
        perl -pi -e 's/\r\n/\n/g' -- "${crlf_files[@]}"
    fi
    "$formatter" --style="file:$root/.clang-format" "${options[@]}" -- "${files[@]}"
fi
