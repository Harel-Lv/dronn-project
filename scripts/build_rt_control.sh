#!/usr/bin/env bash
# Build dronn_rt_service and unit tests (Linux / WSL / macOS with POSIX sockets).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/rt_control/build"
cmake -S "${ROOT}/rt_control" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
"${BUILD}/dronn_rt_tests"
echo "OK: ${BUILD}/dronn_rt_service"
