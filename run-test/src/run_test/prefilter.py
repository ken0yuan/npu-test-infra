"""前置筛选对接：复用 pytorch-pr-analysis 的既有过滤规则。

对应《OBS监测与串行测试队列说明》第 4 节：

- 过滤脚本收归 infra 本地（用户 2026-09-17：全部功能放进 infra）：
  `<INFRA_ROOT>/run-test/tools/pr_exclusion_filter.py`（与 skill 仓库 main
  bf9b16e 逐字节同步）；规则版本固定，本模块不克隆 skill 仓库、
  不复制、不另写排除表（用户 2026-09-18：skill 筛选逻辑不会改动，
  检出工作区没有存在必要）；
- 判定脚本是单 PR diff 文件粒度的外部调用入口（pipeline-policies.md
  封闭列表的官方实现，权威来源
  https://gitcode.com/ken0yuan/pytorch-pr-analysis scripts/）；
- 输入从本地 main 镜像构造：`git diff <sha>^! --name-only`
  （blob:none 裸镜像只读 tree 元数据，不触网、不查 GitHub、不用 token）；
- 无法取得文件清单时不得排除（保守进入分析队列）；
- 规则版本（store.json skill.ref，已钉 bf9b16e）写入队列库 skill_state，
  保证规则可追溯。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .task_queue import QueueStore, STATE_PREFILTER_PENDING

PREFILTER_SCRIPT = "tools/pr_exclusion_filter.py"


def _infra_filter_script() -> Path | None:
    """infra 收归的过滤脚本；缺失时由调用方报错（无 SKILL_ROOT 回退）。"""
    p = Path(__file__).resolve().parents[2] / PREFILTER_SCRIPT
    return p if p.is_file() else None


class PrefilterError(RuntimeError):
    pass


def changed_files_from_mirror(mirror_dir: str | Path, sha: str) -> list[str]:
    """从本地 main 裸镜像取该 commit 相对父提交的改动文件清单。

    blob:none 镜像只有 commit/tree 元数据，`--name-only` 不读 blob。
    根提交（无父）或 diff 失败返回空列表——空清单按"无法判定"处理，
    不排除（交给分析阶段）。
    """
    cmd = ["git", "--git-dir", str(mirror_dir), "diff", f"{sha}^!", "--name-only"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise PrefilterError(f"git diff 执行失败 ({sha}): {e}") from e
    if proc.returncode != 0:
        # 常见：根提交无父。返回空清单，保守不排除。
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def write_simplified_diff(dest: Path, files: list[str]) -> Path:
    """生成 pr_exclusion_filter.py 支持的简化 diff 头（每行 `--- path +0 -0`）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "".join(f"--- {p} +0 -0\n" for p in files), encoding="utf-8"
    )
    return dest


def run_exclusion_filter(diff_file: Path, *, script: Path | None = None) -> bool:
    """调用 pr_exclusion_filter.py，返回是否可排除（JSON bool）。

    默认用 infra 收归脚本；`script` 供测试/调用方显式注入。
    """
    script = script or _infra_filter_script()
    if not script.is_file():
        raise PrefilterError(f"过滤脚本不存在: {script}")
    proc = subprocess.run(
        [sys.executable, str(script), str(diff_file)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    if proc.returncode != 0:
        raise PrefilterError(
            f"pr_exclusion_filter 退出码 {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}"
        )
    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        raise PrefilterError(f"pr_exclusion_filter 输出非 JSON: {proc.stdout[:200]!r}") from e
    if not isinstance(result, bool):
        raise PrefilterError(f"pr_exclusion_filter 输出非布尔: {result!r}")
    return result


def prefilter_pending_tasks(
    config: dict,
    *,
    queue: QueueStore | None = None,
    filter_script: Path | None = None,
    mirror_dir: str | Path | None = None,
) -> dict:
    """处理全部 prefilter_pending 任务，返回报告。

    单任务失败不中断批次：任务保持 prefilter_pending，原因记入报告。
    """
    from .config import store_paths

    ref = str((config.get("skill") or {}).get("ref") or "").strip()
    if not ref:
        raise PrefilterError(
            "skill.ref 未配置：规则版本必须显式钉定（当前收归脚本的版本 bf9b16e）"
        )

    paths = store_paths(config)
    queue = queue or QueueStore(config)
    queue.set_skill_commit(ref)
    mirror_dir = mirror_dir or paths.data_root / "pytorch-main.git"

    report = {
        "skill_commit": ref,
        "processed": 0, "filtered_out": 0, "kept": 0,
        "errors": [], "results": [],
    }
    for task in queue.list_tasks(STATE_PREFILTER_PENDING):
        report["processed"] += 1
        try:
            if not task.pytorch_commit:
                raise PrefilterError("任务缺少 pytorch_commit，无法构造 diff")
            files = changed_files_from_mirror(mirror_dir, task.pytorch_commit)
            diff_file = write_simplified_diff(
                paths.data_root / "prefilter" / f"{task.task_id}.diff", files
            )
            exclude = run_exclusion_filter(diff_file, script=filter_script)
            note = (
                f"pr_exclusion_filter(skill {ref[:12]}) 判定: "
                f"{'排除（docs/test·ci/设备专属封闭类）' if exclude else '保留（含核心代码路径或信息不全）'}"
                f"；改动文件 {len(files)} 个"
            )
            queue.prefilter_outcome(task.task_id, included=not exclude, note=note)
            report["filtered_out" if exclude else "kept"] += 1
            report["results"].append({"task_id": task.task_id, "exclude": exclude})
        except PrefilterError as e:
            report["errors"].append({"task_id": task.task_id, "error": str(e)})
    return report
