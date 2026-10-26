"""prefilter 对接：main 镜像 diff → skill 封闭规则脚本 → 队列落库。

不复制规则、不查 GitHub、不用 token（队列文档 4.1/4.2）；
无法取得文件清单时保守不排除（空清单 → 保留进分析）。
"""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import pytest

from run_test import prefilter as pf
from run_test.prefilter import (
    PrefilterError,
    changed_files_from_mirror,
    prefilter_pending_tasks,
    run_exclusion_filter,
    write_simplified_diff,
)
from run_test.task_queue import QueueStore, STATE_PREFILTER_PENDING

from conftest import COMMIT_A


def _git(args, cwd=None):
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


@pytest.fixture
def mirror(tmp_path):
    """真实 git：两提交的普通仓 + 裸镜像，验证 blob 不参与 --name-only diff。"""
    work = tmp_path / "work"
    work.mkdir()
    _git(["init", "-q"], cwd=work)
    _git(["config", "user.email", "t@t"], cwd=work)
    _git(["config", "user.name", "t"], cwd=work)
    (work / "torch" / "nn").mkdir(parents=True)
    (work / "torch" / "nn" / "a.py").write_text("x = 1\n")
    _git(["add", "."], cwd=work)
    _git(["commit", "-qm", "c1"], cwd=work)
    root_sha = _git(["rev-parse", "HEAD"], cwd=work)
    (work / "torch" / "nn" / "b.py").write_text("y = 2\n")
    (work / "docs" / "d.md").parent.mkdir(parents=True, exist_ok=True)
    (work / "docs" / "d.md").write_text("doc\n")
    _git(["add", "."], cwd=work)
    _git(["commit", "-qm", "c2"], cwd=work)
    head_sha = _git(["rev-parse", "HEAD"], cwd=work)
    bare = tmp_path / "pytorch-main.git"
    _git(["clone", "-q", "--bare", str(work), str(bare)])
    return {"work": work, "bare": bare, "root": root_sha, "head": head_sha}


def test_changed_files_from_mirror(mirror):
    files = changed_files_from_mirror(mirror["bare"], mirror["head"])
    assert set(files) == {"torch/nn/b.py", "docs/d.md"}


def test_root_commit_conservatively_kept(mirror):
    """根提交无父可 diff：返回空清单（= 无法判定，不排除）。"""
    assert changed_files_from_mirror(mirror["bare"], mirror["root"]) == []


def test_write_simplified_diff(tmp_path):
    dest = tmp_path / "nested" / "task.diff"
    out = write_simplified_diff(dest, ["a/b.py", "c.py"])
    assert out == dest
    assert dest.read_text(encoding="utf-8") == "--- a/b.py +0 -0\n--- c.py +0 -0\n"


def _make_filter_script(tmp_path, stdout: str, rc: int = 0) -> Path:
    """桩过滤脚本：输出 stdout 后以 rc 退出（替代真实 pr_exclusion_filter.py）。"""
    scripts = tmp_path / "stub-scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    body = (
        "import sys\n"
        f"print({stdout!r}, end='')\n"
        f"sys.exit({rc})\n"
    )
    p = scripts / "pr_exclusion_filter.py"
    p.write_text(body)
    return p


def test_run_exclusion_filter_parses_bool(tmp_path):
    diff = tmp_path / "d.diff"
    diff.write_text("--- docs/x.md +0 -0\n")
    stub = _make_filter_script(tmp_path, "true\n")
    assert run_exclusion_filter(diff, script=stub) is True
    stub2 = _make_filter_script(tmp_path, "false")
    assert run_exclusion_filter(diff, script=stub2) is False


def test_run_exclusion_filter_uses_infra_script(tmp_path):
    """收归约束：默认使用 infra 收归脚本（run-test/tools/pr_exclusion_filter.py）。"""
    diff = tmp_path / "d.diff"
    diff.write_text("--- docs/x.md +0 -0\n")
    assert run_exclusion_filter(diff) is True


def test_run_exclusion_filter_errors(tmp_path):
    diff = tmp_path / "d.diff"
    diff.write_text("--- docs/x.md +0 -0\n")
    mk = lambda out, rc=0: _make_filter_script(tmp_path, out, rc)
    with pytest.raises(PrefilterError, match="退出码"):
        run_exclusion_filter(diff, script=mk("boom", rc=3))
    with pytest.raises(PrefilterError, match="非 JSON"):
        run_exclusion_filter(diff, script=mk("not-json"))
    with pytest.raises(PrefilterError, match="非布尔"):
        run_exclusion_filter(diff, script=mk("123"))
    with pytest.raises(PrefilterError, match="不存在"):
        run_exclusion_filter(diff, script=tmp_path / "nope.py")


_ver = itertools.count(1)


def _mk_prefilter_pending_task(queue: QueueStore) -> str:
    """门禁通过 → 停在 prefilter_pending 的任务（每次唯一版本，避开去重）。"""
    ver = f"pf-v{next(_ver)}"
    task, _ = queue.register_candidate(
        f"obs://b/wheels-{COMMIT_A[:10]}.tar.gz",
        f"obs://b/source-{COMMIT_A[:10]}.tar.gz",
        obs_wheels_version=ver, obs_source_version=ver,
    )
    queue.register_identity_rule("rule-1", "规则原文", "测试 fixture")
    queue.verify_identity(
        task.task_id, "rule-1",
        {
            "pr_number": 42,
            "pytorch_commit": COMMIT_A,
            "torch_npu_commit": "1234567890abcdef1234567890abcdef12345678",
            "binding": "测试绑定",
        },
    )
    assert queue.get_task(task.task_id).state == STATE_PREFILTER_PENDING
    return task.task_id


def test_prefilter_pending_tasks_end_to_end(tmp_path, config, queue, monkeypatch):
    task_id = _mk_prefilter_pending_task(queue)
    # 桩过滤脚本：docs-only diff → 排除。
    stub = _make_filter_script(tmp_path, "true\n")
    config["skill"] = {"ref": "bf9b16e"}
    monkeypatch.setattr(
        pf, "changed_files_from_mirror", lambda m, sha: ["docs/x.md"]
    )
    report = prefilter_pending_tasks(config, queue=queue, filter_script=stub)
    assert report["processed"] == 1 and report["filtered_out"] == 1
    assert report["skill_commit"] == "bf9b16e"
    task = queue.get_task(task_id)
    assert task.state == "filtered_out"
    assert "pr_exclusion_filter" in task.history[-1]["reason"]
    # diff 文件已留档（可追溯判定依据）。
    diff_file = queue.paths.data_root / "prefilter" / f"{task_id}.diff"
    assert diff_file.is_file()


def test_prefilter_keeps_core_changes(tmp_path, config, queue, monkeypatch):
    task_id = _mk_prefilter_pending_task(queue)
    stub = _make_filter_script(tmp_path, "false\n")
    config["skill"] = {"ref": "bf9b16e"}
    monkeypatch.setattr(
        pf, "changed_files_from_mirror", lambda m, sha: ["torch/nn/functional.py"]
    )
    report = prefilter_pending_tasks(config, queue=queue, filter_script=stub)
    assert report["kept"] == 1
    assert queue.get_task(task_id).state == "pending_analysis"


def test_prefilter_requires_pinned_ref(config):
    config["skill"] = {"ref": None}
    with pytest.raises(PrefilterError, match="skill.ref 未配置"):
        prefilter_pending_tasks(config)


def test_prefilter_single_failure_does_not_abort_batch(
    tmp_path, config, queue, monkeypatch
):
    t1 = _mk_prefilter_pending_task(queue)
    t2 = _mk_prefilter_pending_task(queue)
    calls = {"n": 0}

    def flaky(m, sha):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PrefilterError("模拟镜像缺失")
        return ["docs/x.md"]

    monkeypatch.setattr(pf, "changed_files_from_mirror", flaky)
    stub = _make_filter_script(tmp_path, "true\n")
    config["skill"] = {"ref": "bf9b16e"}

    report = prefilter_pending_tasks(config, queue=queue, filter_script=stub)
    assert report["errors"] and len(report["errors"]) == 1
    assert report["errors"][0]["task_id"] == t1
    assert report["filtered_out"] == 1
    # 失败任务保持 prefilter_pending，可重试。
    assert queue.get_task(t1).state == STATE_PREFILTER_PENDING
    assert queue.get_task(t2).state == "filtered_out"
