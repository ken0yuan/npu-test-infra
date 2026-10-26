#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/run-test/env.sh"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false
mkdir -p data/validation
"$RUN_TEST_PYTHON" -m pytest -c run-test/pyproject.toml run-test/tests -q -p no:cacheprovider --junitxml=data/validation/tests.xml
"$RUN_TEST_PYTHON" run-test/tests/acceptance.py --output data/validation/acceptance
