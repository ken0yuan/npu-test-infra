#!/usr/bin/env bash
# Shared environment for project entry points; RUN_TEST_PYTHON can override it.
RUN_TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_TEST_PYTHON="${RUN_TEST_PYTHON:-$RUN_TEST_ROOT/venv/control/bin/python}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUN_TEST_ROOT/run-test/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES=""
unset PYTEST_PLUGINS PYTEST_ADDOPTS
if [ ! -x "$RUN_TEST_PYTHON" ]; then
  echo "Missing control Python: $RUN_TEST_PYTHON. Follow README.md to create venv/control." >&2
  return 1
fi
