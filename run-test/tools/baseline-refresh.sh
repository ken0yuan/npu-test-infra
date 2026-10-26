#!/usr/bin/env bash
# 基线刷新一键化：给 PR 号，自动从本地镜像取提交时间补齐 merged_at，再调
# `run.sh baseline rollforward`；默认只保留最新一份 rollforward 输出目录
# （对比文件按设计用完即删，避免越堆越多）。
#
# 用法：
#   run-test/tools/baseline-refresh.sh 197470 201314        # 刷新这两条 PR 的基线
#   run-test/tools/baseline-refresh.sh --keep 197470        # 保留历史 run 目录
#   run-test/tools/baseline-refresh.sh --input prs.json     # 直接给 rollforward 输入
#
# merged_at 解析：pr_results.db 的 pytorch_commit → 本地镜像 `git log -1 %cI`。
# 三处（输入 / PR 库 / upstream_commits）都拿不到时本脚本用提交时间兜底，
# 仍拿不到就报错退出，不猜顺序。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/run-test/env.sh"
PY="$RUN_TEST_PYTHON"
KEEP=0
INPUT=""
PRS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --keep) KEEP=1; shift ;;
    --input) INPUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) PRS+=("$1"); shift ;;
  esac
done

if [ -z "$INPUT" ] && [ ${#PRS[@]} -eq 0 ]; then
  echo "用法: $0 [--keep] <PR 号>... | --input <prs.json>" >&2
  exit 2
fi

tmp_in=""
cleanup() { [ -n "$tmp_in" ] && rm -f "$tmp_in"; }
trap cleanup EXIT

if [ -z "$INPUT" ]; then
  tmp_in="$(mktemp)"
  ROOT="$ROOT" PY="$PY" PRS="${PRS[*]}" OUT="$tmp_in" "$PY" - <<'PY'
import json, os, sqlite3, subprocess, sys
from pathlib import Path

root = Path(os.environ["ROOT"])
cfg = {}
cfg_path = root / "settings.json"
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
data_root = Path(cfg.get("data_root") or (root / "data" / "runtime"))
db = data_root / "pr_results.db"
git_dir = data_root / "pytorch-main.git"
if not db.is_file():
    sys.exit(f"PR 库不存在: {db}（先跑出结果再刷新基线）")
if not git_dir.is_dir():
    sys.exit(f"本地镜像不存在: {git_dir}（bash run.sh mirror sync 后重试）")

conn = sqlite3.connect(db)
prs = []
errors = []
for pr_text in os.environ["PRS"].split():
    pr = int(pr_text)
    row = conn.execute(
        "SELECT pytorch_commit, merged_at FROM pr_results WHERE pr_number = ?", (pr,)
    ).fetchone()
    if row is None:
        errors.append(f"PR {pr}: PR 库里没有结果，先执行任务再刷新基线")
        continue
    commit, merged_at = row
    if not merged_at:
        merged_at = subprocess.run(
            ["git", "--git-dir", str(git_dir), "log", "-1", "--format=%cI", commit],
            capture_output=True, text=True,
        ).stdout.strip()
    if not merged_at:
        errors.append(f"PR {pr}: 拿不到 merged_at（commit {commit[:10]} 不在本地镜像）")
        continue
    prs.append({"pr_number": pr, "merged_at": merged_at})
    print(f"  {pr} -> merged_at={merged_at} (commit {commit[:10]})", file=sys.stderr)

if errors:
    sys.exit("拒绝刷新，先解决:\n  " + "\n  ".join(errors))
Path(os.environ["OUT"]).write_text(json.dumps({"prs": prs}), encoding="utf-8")
PY
  INPUT="$tmp_in"
fi

bash "$ROOT/run.sh" baseline rollforward --input "$INPUT"

if [ "$KEEP" = 0 ]; then
  ROOT="$ROOT" PY="$PY" "$PY" - <<'PY'
import json, os, shutil, sys
from pathlib import Path

root = Path(os.environ["ROOT"])
cfg = {}
cfg_path = root / "settings.json"
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
data_root = Path(cfg.get("data_root") or (root / "data" / "runtime"))
base = data_root / "rollforward"
runs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)
for old in runs[:-1]:
    shutil.rmtree(old, ignore_errors=True)
    print(f"已删除旧对比目录: {old}")
PY
fi
