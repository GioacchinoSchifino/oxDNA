#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"${CXX:-c++}" -std=c++14 -Wall -Wextra -Werror -pedantic \
 -I"$SRC/test/CUDA_RUNTIME_ERROR_CHECK" -I"$SRC/src/CUDA" \
 "$SRC/test/CUDA_RUNTIME_ERROR_CHECK/test_last_error.cpp" -o "$TMP/test_last_error"
"$TMP/test_last_error" success >"$TMP/success.log" 2>&1
set +e
"$TMP/test_last_error" failure >"$TMP/failure.log" 2>&1
rc=$?
set -e
[[ "$rc" -eq 1 ]]
grep -F 'mock failure' "$TMP/failure.log" >/dev/null
grep -F 'invalid value (mock)' "$TMP/failure.log" >/dev/null
printf 'CUDA_ERROR_MOCK_SUCCESS_AND_FAILURE=PASS\n'
