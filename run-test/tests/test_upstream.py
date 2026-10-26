"""上游 main 镜像与 cycle 流程（2026-09-17 用户确认方案）。

覆盖：PR 解析（subject (#N)）、无 PR 跳过、对象定位入队、对象缺失保持
pending、7 天过期、幂等（中断重试不重复核实）、身份绑定（main 观察）。
全部使用本地 git 仓库与假 OBS，不触网。
"""

import subprocess
import time

import pytest

from run_test import task_queue as q
from run_test.task_queue import QueueStore
from run_test.upstream import (
    PyTorchMirror,
    UpstreamError,
    run_cycle,
)

from conftest import COMMIT_A, COMMIT_NPU

COMMIT_A2 = "b" * 40


def _git(path, *args, check=True):
    proc = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=str(path), capture_output=True, text=True, check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {args} 失败: {proc.stderr}")
    return proc.stdout.strip()


@pytest.fixture
def upstream_repo(tmp_path):
    """本地裸仓库模拟上游 pytorch main：三个 commit（两个带 PR 号，一个无）。"""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    (work / "f.txt").write_text("1\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "feat: add A (#101)")
    sha_pr1 = _git(work, "rev-parse", "HEAD")
    (work / "f.txt").write_text("2\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "direct push without PR")
    sha_nopr = _git(work, "rev-parse", "HEAD")
    (work / "f.txt").write_text("3\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "fix: add B (#102)")
    sha_pr2 = _git(work, "rev-parse", "HEAD")
    return work, {"pr1": sha_pr1, "nopr": sha_nopr, "pr2": sha_pr2}


class FakeObs:
    """模拟桶：list 返回预置快照；stat 供单点查询。"""

    def __init__(self, existing: dict):
        self.existing = existing

    def stat(self, url):
        return self.existing.get(url)

    def list(self, prefix_url):
        return {
            url: meta for url, meta in self.existing.items()
            if url.startswith(prefix_url)
        }


def _make_mirror(tmp_path, upstream_repo_path) -> PyTorchMirror:
    return PyTorchMirror(
        repo_url=str(upstream_repo_path),
        mirror_dir=tmp_path / "mirror.git",
        config=None,
    )


def _cycle(config, queue, mirror, existing_objects):
    return run_cycle(
        config, queue=queue, mirror=mirror,
        downloader=FakeObs(existing_objects),
    )


class TestMirror:
    def test_fetch_and_commits_between(self, tmp_path, upstream_repo):
        work, shas = upstream_repo
        mirror = _make_mirror(tmp_path, work)
        mirror.ensure()
        tip = mirror.fetch()
        assert tip == shas["pr2"]
        commits = mirror.commits_between(shas["pr1"], tip)
        assert [c.sha for c in commits] == [shas["nopr"], shas["pr2"]]
        assert commits[0].pr_number is None
        assert commits[1].pr_number == 102

    def test_origin_mismatch_rejected(self, tmp_path, upstream_repo, tmp_path_factory):
        work, _ = upstream_repo
        mirror = _make_mirror(tmp_path, work)
        mirror.ensure()
        # 把 origin 改成别的仓库。
        other = tmp_path_factory.mktemp("other")
        _git(other, "init", "-q", "-b", "main")
        _git(mirror.git_dir, "remote", "set-url", "origin", str(other))
        with pytest.raises(UpstreamError, match="不一致"):
            mirror.ensure()

    def test_proxy_prefixed_origin_normalized(self, tmp_path, upstream_repo):
        """经 insteadOf 代理克隆的镜像（origin 带代理前缀）被接受并改回直连。"""
        work, _ = upstream_repo
        mirror = _make_mirror(tmp_path, work)
        mirror.ensure()
        _git(mirror.git_dir, "remote", "set-url", "origin",
             f"https://ghfast.top/{work}")
        mirror.ensure()
        origin = _git(mirror.git_dir, "remote", "get-url", "origin")
        assert origin == str(work)

    def test_is_on_main(self, tmp_path, upstream_repo):
        work, shas = upstream_repo
        mirror = _make_mirror(tmp_path, work)
        mirror.ensure()
        assert mirror.is_on_main(shas["pr2"]) is True
        assert mirror.is_on_main("f" * 40) is False


class TestCycle:
    def test_qualified_bundle_and_ambiguous_build(self, setup):
        config, queue, mirror, shas = setup
        sha10 = shas["pr2"][:10]
        bucket = "obs://pytorch-ci-cache/7day-cache"
        name = f"{bucket}/wheels-{sha10}-npu-abcdef0123-py311-builder-aarch64.tar.gz"
        other = name.replace("builder", "cann-a3")
        objects = {name:{"etag":"e1"}, other:{"etag":"e2"},
                   f"{bucket}/pytorch-source-{sha10}.tar.gz":{"etag":"s1"}}
        report = _cycle(config, queue, mirror, objects)
        assert not report.enqueued and report.errors
        config["obs"]["wheel_name_contains"] = ["cann-a3"]
        report = _cycle(config, queue, mirror, objects)
        assert len(report.enqueued) == 1
        task = queue.get_task(report.enqueued[0]["task_id"])
        assert task.obs_wheels_url == other and task.pytorch_commit == shas["pr2"]
        assert not _cycle(config, queue, mirror, objects).enqueued

    @pytest.fixture
    def setup(self, tmp_path, upstream_repo, config):
        work, shas = upstream_repo
        mirror = _make_mirror(tmp_path, work)
        queue = QueueStore(config)
        return config, queue, mirror, shas

    def test_records_commits_and_enqueues_when_objects_exist(self, setup):
        config, queue, mirror, shas = setup
        sha10 = shas["pr2"][:10]
        objects = {
            f"obs://pytorch-ci-cache/7day-cache/wheels-{sha10}.tar.gz":
                {"last_modified": "2026-09-17T08:00:00Z", "etag": "e1", "size_kb": "1"},
            f"obs://pytorch-ci-cache/7day-cache/pytorch-source-{sha10}.tar.gz":
                {"last_modified": "2026-09-17T08:00:00Z", "etag": "e2", "size_kb": "1"},
        }
        report = _cycle(config, queue, mirror, objects)
        # 首轮回溯窗口覆盖全部三个 commit。
        assert report.recorded == 2  # #101、#102
        assert report.skipped_no_pr == 1  # 直接 push，完全跳过
        assert len(report.enqueued) == 1  # 只有 #102 的对象存在
        assert report.enqueued[0]["pr_number"] == 102

        # 任务：已过身份门禁，绑定 PR 与完整 commit；torch_npu 待执行端补。
        task = queue.get_task(report.enqueued[0]["task_id"])
        assert task.state == q.STATE_PREFILTER_PENDING
        assert task.pr_number == 102
        assert task.pytorch_commit == shas["pr2"]
        assert task.torch_npu_commit is None
        assert task.identity_rule_id == "upstream-main-fetch"
        assert task.obs_wheels_version == "e1"  # ETag 做对象版本标识

        # upstream_commits 表状态。
        rows = {r["sha"]: r for r in queue.list_upstream_commits()}
        assert rows[shas["pr2"]]["enqueue_state"] == "queued"
        assert rows[shas["pr1"]]["enqueue_state"] == "pending"  # 对象未出现
        assert rows[shas["nopr"]]["enqueue_state"] == "no_pr"

    def test_missing_objects_stay_pending_then_expire(self, setup):
        config, queue, mirror, shas = setup
        # 无任何对象：全部 pending。
        report = _cycle(config, queue, mirror, {})
        assert report.enqueued == []
        assert len(report.still_missing) == 2

        # 人为把发现时间拨到 8 天前：下轮标记 expired。
        conn = queue._connect()
        try:
            conn.execute(
                "UPDATE upstream_commits SET discovered_at = "
                "strftime('%Y-%m-%dT%H:%M:%S', 'now', '-8 days')")
        finally:
            conn.close()
        report = _cycle(config, queue, mirror, {})
        assert len(report.expired) == 2
        rows = queue.list_upstream_commits(state="expired")
        assert len(rows) == 2

    def test_idempotent_and_resume_after_interrupt(self, setup):
        """重复 cycle 不重复入队；登记后中断的任务不重复核实。"""
        config, queue, mirror, shas = setup
        sha10 = shas["pr2"][:10]
        objects = {
            f"obs://pytorch-ci-cache/7day-cache/wheels-{sha10}.tar.gz":
                {"last_modified": "2026-09-17T08:00:00Z", "etag": "e1", "size_kb": "1"},
            f"obs://pytorch-ci-cache/7day-cache/pytorch-source-{sha10}.tar.gz":
                {"last_modified": "2026-09-17T08:00:00Z", "etag": "e2", "size_kb": "1"},
        }
        _cycle(config, queue, mirror, objects)
        first_tasks = queue.list_tasks()
        assert len(first_tasks) == 1

        # 模拟中断：任务已核实到 prefilter_pending，但 commit 记录仍是
        # pending（例如核实与更新之间中断）。
        conn = queue._connect()
        try:
            conn.execute(
                "UPDATE upstream_commits SET enqueue_state = 'pending' "
                "WHERE sha = ?", (shas["pr2"],))
        finally:
            conn.close()
        report = _cycle(config, queue, mirror, objects)
        # 不重复核实、不新建任务。
        assert len(queue.list_tasks()) == 1
        assert queue.list_tasks()[0].state == q.STATE_PREFILTER_PENDING
        assert report.enqueued[0]["task_id"] == first_tasks[0].task_id

    def test_second_cycle_picks_up_new_objects(self, setup):
        """第一轮对象未齐、第二轮出现：正常入队。"""
        config, queue, mirror, shas = setup
        sha10 = shas["pr2"][:10]
        wheels = f"obs://pytorch-ci-cache/7day-cache/wheels-{sha10}.tar.gz"
        source = f"obs://pytorch-ci-cache/7day-cache/pytorch-source-{sha10}.tar.gz"
        _cycle(config, queue, mirror, {})
        report = _cycle(
            config, queue, mirror,
            {wheels: {"last_modified": "t", "etag": "e1", "size_kb": "1"},
             source: {"last_modified": "t", "etag": "e2", "size_kb": "1"}},
        )
        assert len(report.enqueued) == 1

    def test_no_downloader_reports_error(self, setup):
        config, queue, mirror, shas = setup
        report = run_cycle(config, queue=queue, mirror=mirror, downloader=None)
        assert report.errors and "未配置" in report.errors[0]
