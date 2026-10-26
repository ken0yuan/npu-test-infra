#!/usr/bin/env bash
# The supervisor owns child PIDs and restarts only its own failed children.
set -euo pipefail
exec "${RUN_TEST_PYTHON:-/usr/bin/python3}" "$(dirname -- "${BASH_SOURCE[0]}")/service.py" _run
