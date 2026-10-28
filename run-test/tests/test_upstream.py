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
    def test_latest_complete_split_build_unblocks_pending_commit(self, setup):
        config, queue, mirror, shas = setup
        sha10 = shas['pr2'][:10]
        bucket = 'obs://pytorch-ci-cache/7day-cache'
        config['obs']['wheel_name_contains'] = ['-py311-', '-torch-npu-builder-', 'aarch64']
        def urls(image):
            suffix = f'py311-torch-npu-builder-aarch64-{image}-aarch64-v3.tar.gz'
            return (f'{bucket}/wheels-pytorch-{sha10}-{suffix}',
                    f'{bucket}/wheels-torchnpu-abcdef0123-{sha10}-{suffix}')
        old_pt, old_npu = urls('202610200103')
        new_pt, new_npu = urls('202610160225')
        objects = {
            old_pt: {'etag':'p1'}, old_npu: {'etag':'n1'},
            new_pt: {'etag':'p2'}, new_npu: {'etag':'n2'},
            f'{bucket}/pytorch-source-{sha10}.tar.gz': {'etag':'source'},
        }
        # A missing timestamp remains a visible pending commit, then a later
        # complete listing can admit it without manually rewriting its state.
        assert _cycle(config, queue, mirror, objects).errors
        for url in (old_pt, old_npu):
            objects[url]['last_modified'] = '2026-10-27T01:00:00Z'
        for url in (new_pt, new_npu):
            objects[url]['last_modified'] = '2026-10-27T02:00:00Z'
        report = _cycle(config, queue, mirror, objects)
        assert len(report.enqueued) == 1 and not report.errors
        task = queue.get_task(report.enqueued[0]['task_id'])
        assert (task.obs_wheels_url, task.obs_wheels_npu_url) == (new_pt, new_npu)
        assert (task.obs_wheels_version, task.obs_wheels_npu_version) == ('p2', 'n2')
        assert task.pytorch_commit == shas['pr2']
        assert not _cycle(config, queue, mirror, objects).enqueued

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


def test_detect_inversions_synthetic():
    from run_test.upstream import detect_inversions
    assert detect_inversions([("a", 100), ("b", 90)]) == []
    inv = detect_inversions([("a", 100), ("b", 150), ("c", 120)])
    assert [i["sha"] for i in inv] == ["b"]


# ---------------------------------------------------------------------------
# 合入序快照（评审阻塞项 1/4：成员绑定 + 原子持久化 + 位置比较）
# ---------------------------------------------------------------------------

import os as _os
import subprocess as _subprocess


def _make_git_fixture(tmp_path):
    """真实临时 Git 仓库：main 上三个 first-parent 提交（c1←c2←c3）。"""
    work = tmp_path / "fixture"
    work.mkdir()
    env = {**_os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    def _git(*args):
        _subprocess.run(["git", "--git-dir", str(work), *args],
                        check=True, capture_output=True, env=env)
    _subprocess.run(["git", "init", "--bare", "-q", str(work)], check=True,
                    capture_output=True, env=env)
    # 用临时工作区制造三个提交再推入裸仓库
    wt = tmp_path / "wt"
    wt.mkdir()
    def _commit(msg):
        (wt / "f.txt").write_text(msg, encoding="utf-8")
        for cmd in (["-C", str(wt), "add", "f.txt"],
                    ["-C", str(wt), "commit", "-q", "-m", msg]):
            _subprocess.run(["git", *cmd], check=True, capture_output=True, env=env)
    _subprocess.run(["git", "-C", str(wt), "init", "-q", "-b", "main"],
                    check=True, capture_output=True, env=env)
    _commit("c1"); c1 = _subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, env=env).stdout.strip()
    _commit("c2"); c2 = _subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, env=env).stdout.strip()
    _commit("c3"); c3 = _subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, env=env).stdout.strip()
    _subprocess.run(["git", "-C", str(wt), "push", "-q", str(work), "main"],
                    check=True, capture_output=True, env=env)
    return work, (c1, c2, c3)


class _FakeMirror:
    def __init__(self, git_dir):
        self.git_dir = str(git_dir)
        self.repo_url = "file:///fixture"


def test_validate_and_persist_order_snapshot(tmp_path, config):
    """fetch 时一次校验 + 原子持久化：快照含完整 SHA 序列/来源/schema。"""
    import json
    work, (c1, c2, c3) = _make_git_fixture(tmp_path)
    from run_test.upstream import validate_merge_order, persist_order_snapshot, load_order_snapshot
    mirror = type("M", (), {"git_dir": str(work), "repo_url": "file:///fixture"})()
    snapshot = validate_merge_order(mirror, limit=100)
    assert snapshot["order"] == [c3, c2, c1]  # 新→旧
    assert snapshot["schema"] == "upstream-order/1"
    persist_order_snapshot(config, snapshot)
    loaded = load_order_snapshot(config, expected_source="file:///fixture")
    assert loaded["order"] == [c3, c2, c1] and loaded["source"] == "file:///fixture"
    # 原子性：无临时残留
    assert not list((tmp_path / "data").glob("upstream-order.json.tmp*"))


def test_load_order_snapshot_corrupt_returns_none(config, tmp_path):
    from run_test.config import store_paths
    from run_test.upstream import load_order_snapshot
    paths = store_paths(config)
    target = paths.data_root / "upstream-order.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{corrupt", encoding="utf-8")
    assert load_order_snapshot(config) is None
    target.write_text('{"schema": "wrong"}', encoding="utf-8")
    assert load_order_snapshot(config) is None
    assert not target.exists() or load_order_snapshot(config) is None
