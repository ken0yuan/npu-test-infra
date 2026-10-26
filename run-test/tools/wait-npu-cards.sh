#!/usr/bin/env bash
# 等待空闲 NPU 卡的独立入口（逻辑在 run_test.npu_cards）。
# 用法: wait-npu-cards.sh [--need N] [--timeout S] [--interval S]
#   成功: stdout 输出空闲卡 ID（逗号分隔），exit 0
#   超时: exit 1
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/run-test/env.sh"
exec "$RUN_TEST_PYTHON" -m run_test.npu_cards "$@"
