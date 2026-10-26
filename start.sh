#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/run-test/env.sh"
exec "$RUN_TEST_PYTHON" "$ROOT/run-test/services/service.py" "${1:-start}"
