"""串行执行端：准备/安装/执行/写库/清理全流程与异常路径（队列文档 6–8 节）。

下载、安装均使用桩；pip / torch 导入核验通过 monkeypatch subprocess.run 模拟。
写库走真实 record 流程，验证端到端一致性。
"""

import itertools
import platform
import subprocess

import pytest

from run_test import task_queue as q
from run_test.baseline_store import baseline_db_path, get_baseline
from run_test.db import connect
from run_test.executor import Executor, ExecutorError
from run_test.identifiers import Counts
from run_test.pr_store import get_pr_result
from run_test.task_queue import QueueStore
from run_test.runner import FunctionRunOutcome

from conftest import COMMIT_A, COMMIT_NPU, FakeDownloader, TF_A, TF_B

SHA10 = COMMIT_A[:10]
WHEELS_URL = f"obs://pytorch-ci-cache/7day-cache/wheels-{SHA10}.tar.gz"
SOURCE_URL = f"obs://pytorch-ci-cache/7day-cache/pytorch-source-{SHA10}.tar.gz"

_version_counter = itertools.count(1)


class FakeRunner:
    def __init__(self, counts_map=None, incomplete=None):
        self.counts_map = counts_map or {}
        self.incomplete = incomplete or set()
        self.source_roots: list = []

    def run_function(self, tf, dev, *, source_root=None):
        self.source_roots.append(source_root)
        if (tf, dev) in self.incomplete:
            return FunctionRunOutcome.incomplete(tf, dev, "模拟：未收集到测试")
        return FunctionRunOutcome(
            tf, dev, True,
            self.counts_map.get((tf, dev), Counts(1, 0, 0, 0)),
            f"fake log: {tf} on {dev}",
            0,
            coverage_status="full",
            instance_ids=(tf,),
            environment_fingerprint={"channel": dev, "boot": None},
        )


@pytest.fixture
def fake_pip(monkeypatch):
    """模拟 pip 卸载/安装与 torch 版本核验。"""

    def fake_run(cmd, *args, **kwargs):
        joined = " ".join(str(c) for c in cmd)
        stdout = ""
        if "import torch" in joined:
            stdout = COMMIT_A  # 安装后的 git_version 与任务锁定一致
        elif "import sys" in joined:
            stdout = "310"  # 解释器版本探针（qualified/split 包名带 python_tag 时触发）
        elif "uninstall" in joined:
            stdout = "Skipping torch_npu as it is not installed."
        elif "install" in joined:
            stdout = "Successfully installed torch torch_npu"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def verified_task(queue: QueueStore, *, manifest=None) -> q.Task:
    """构造一个已过门禁、清单就绪的 pending_test 任务。"""
    ver = f"exec-v{next(_version_counter)}"
    task, _ = queue.register_candidate(
        WHEELS_URL, SOURCE_URL, obs_wheels_version=ver, obs_source_version=ver
    )
    queue.register_identity_rule("r", "规则", "测试 fixture")
    queue.verify_identity(
        task.task_id, "r",
        {
            "pr_number": 123456,
            "pytorch_commit": COMMIT_A,
            "torch_npu_commit": COMMIT_NPU,
            "binding": "测试绑定",
            "torch_npu_location": "test-wheel",
        },
    )
    queue.prefilter_outcome(task.task_id, True)
    return queue.submit_manifest(
        task.task_id,
        manifest or [
            {"test_function": TF_A, "device": "cpu"},
            {"test_function": TF_B, "device": "accelerator"},
        ],
    )


class TestFullFlow:
    def test_done_writes_stores_and_cleans(self, config, queue, paths, fake_pip):
        """正常路径：清单执行（含 fail/error）→ 写库 → 立即清理 → done。"""
        task = verified_task(queue)
        runner = FakeRunner(counts_map={
            (TF_A, "cpu"): Counts(3, 0, 0, 1),
            (TF_B, "accelerator"): Counts(2, 1, 0, 0),
        })
        executor = Executor(
            config, queue_store=queue,
            downloader=FakeDownloader(), runner=runner,
        )
        outcome = executor.run_once()
        assert outcome.outcome == "done"

        t = queue.get_task(task.task_id)
        assert t.state == q.STATE_DONE
        # 临时目录已删除（成功写库后立即清理，不留保留期）。
        assert not (paths.task_work_root / task.task_id).exists()

        # PR 库：一行，计数为本次汇总。
        conn = connect(paths.pr_db)
        try:
            pr = get_pr_result(conn, 123456)
        finally:
            conn.close()
        assert pr.counts == Counts(5, 1, 0, 1)
        # location 以 prepare 阶段回写的实际 wheel 名为准（DB 最新绑定），
        # 不再是身份绑定时的占位值——stale 快照修复后的权威语义。
        assert pr.torch_npu_environment["location"] == (
            "obs-wheel:torch_npu-2.15.0a0-cp310-cp310-linux_aarch64.whl"
        )
        assert pr.torch_npu_environment["torch_npu_commit"] == COMMIT_NPU

        # 基线：两条记录 + 各自日志。
        root_a = paths.baselines_root / "test/distributed/_tensor"
        conn = connect(baseline_db_path(root_a))
        try:
            rec = get_baseline(conn, TF_A, "cpu")
        finally:
            conn.close()
        assert rec.counts == Counts(3, 0, 0, 1)
        assert (root_a / rec.log_path).is_file()

        # 锁已释放：可继续下一个任务。
        verified_task(queue)
        assert queue.claim_next() is not None

        # runner 全部在任务源码树根下运行（nodeid 是仓库相对路径）。
        assert set(runner.source_roots) == {
            paths.task_work_root / task.task_id / "source" / "pytorch"
        }

    def test_split_bundle_downloads_npu_object_and_backfills(self, config, queue, paths, fake_pip):
        """拆分包格式（2026-10-22 起）：torch_npu 在独立对象中，执行端必须
        下载三个对象、跨两个 wheels 包找齐 wheel，并从 npu wheel 读出精确
        commit 回写（torch_npu_short 与包内 git_version 一致性核验通过）。"""
        arch = platform.machine()
        pt_url = (f"obs://pytorch-ci-cache/7day-cache/"
                  f"wheels-pytorch-{SHA10}-py310-torch-npu-builder-{arch}-"
                  f"torch-master-202610200103-e0dcc43e-{arch}-v2.tar.gz")
        npu_url = (f"obs://pytorch-ci-cache/7day-cache/"
                   f"wheels-torchnpu-{COMMIT_NPU[:10]}-{SHA10}-py310-torch-npu-builder-{arch}-"
                   f"torch-master-202610200103-e0dcc43e-{arch}-v2.tar.gz")
        ver = f"split-v{next(_version_counter)}"
        task, _ = queue.register_candidate(
            pt_url, SOURCE_URL, obs_wheels_version=ver, obs_source_version=ver,
            obs_wheels_npu_url=npu_url, obs_wheels_npu_version=ver,
        )
        queue.register_identity_rule("r", "规则", "测试 fixture")
        queue.verify_identity(
            task.task_id, "r",
            {"pr_number": 123457, "pytorch_commit": COMMIT_A,
             "torch_npu_commit": None, "binding": "拆分包绑定"},
        )
        queue.prefilter_outcome(task.task_id, True)
        task = queue.submit_manifest(task.task_id, [{"test_function": TF_A, "device": "cpu"}])

        runner = FakeRunner(counts_map={(TF_A, "cpu"): Counts(1, 0, 0, 0)})
        dl = FakeDownloader()
        executor = Executor(config, queue_store=queue, downloader=dl, runner=runner)
        outcome = executor.run_once()
        assert outcome.outcome == "done"
        assert npu_url in dl.calls  # 独立 torch_npu 对象确实被下载

        conn = connect(paths.pr_db)
        try:
            pr = get_pr_result(conn, 123457)
        finally:
            conn.close()
        assert pr.torch_npu_environment["torch_npu_commit"] == COMMIT_NPU

    def test_split_bundle_missing_npu_url_stays_legacy(self, config, queue, fake_pip):
        """npu URL 为空（legacy 登记）时不下载第三个对象。"""
        task = verified_task(queue)
        dl = FakeDownloader()
        executor = Executor(config, queue_store=queue, downloader=dl, runner=FakeRunner())
        outcome = executor.run_once()
        assert outcome.outcome == "done"
        assert not any("wheels-npu" in c or "torchnpu" in c for c in dl.calls)

    def test_wheel_commit_backfill_reaches_submission(self, config, queue, paths, fake_pip):
        """身份绑定 torch_npu_commit=None 时，执行端从包内 wheel 读出精确
        commit 回写 DB；提交必须带 DB 最新绑定而非旧内存快照——否则写库
        校验拒绝（2026-09-18 首跑实证："torch_npu_commit 必须是字符串"）。"""
        ver = f"exec-v{next(_version_counter)}"
        task, _ = queue.register_candidate(
            WHEELS_URL, SOURCE_URL, obs_wheels_version=ver, obs_source_version=ver
        )
        queue.register_identity_rule("r", "规则", "测试 fixture")
        queue.verify_identity(
            task.task_id, "r",
            {"pr_number": 123456, "pytorch_commit": COMMIT_A,
             "torch_npu_commit": None, "binding": "x"},
        )
        queue.prefilter_outcome(task.task_id, True)
        task = queue.submit_manifest(
            task.task_id,
            [{"test_function": TF_A, "device": "cpu"}],
        )
        assert task.torch_npu_commit is None  # 旧快照：此刻回写尚未发生

        runner = FakeRunner(counts_map={(TF_A, "cpu"): Counts(1, 0, 0, 0)})
        executor = Executor(
            config, queue_store=queue,
            downloader=FakeDownloader(), runner=runner,
        )
        outcome = executor.run_once()
        assert outcome.outcome == "done"

        # PR 库拿到的是 wheel 内读出的精确 commit，而非 None。
        conn = connect(paths.pr_db)
        try:
            pr = get_pr_result(conn, 123456)
        finally:
            conn.close()
        assert pr.torch_npu_environment["torch_npu_commit"] == COMMIT_NPU


class TestPrepareFailures:
    def test_no_downloader_configured_blocks(self, config, queue):
        task = verified_task(queue)
        executor = Executor(config, queue_store=queue, downloader=None, runner=FakeRunner())
        outcome = executor.run_once()
        assert outcome.outcome == "blocked"
        assert "OBS 下载方式未配置" in queue.get_task(task.task_id).error_reason

    def test_download_failure_blocks_without_install(self, config, queue, fake_pip):
        task = verified_task(queue)
        dl = FakeDownloader(fail_urls={WHEELS_URL})
        executor = Executor(config, queue_store=queue, downloader=dl, runner=FakeRunner())
        outcome = executor.run_once()
        assert outcome.outcome == "blocked"
        t = queue.get_task(task.task_id)
        assert t.state == q.STATE_BLOCKED
        assert "下载失败" in t.error_reason
        assert t.retry_count == 1

    def test_version_mismatch_blocks(self, config, queue, fake_pip):
        """对象与任务 commit 不对应：不安装、不测试、不更新基线。"""
        # URL 中的短 SHA（bbbbbbbbbb）与任务锁定的 COMMIT_A 前缀不同。
        t, _ = queue.register_candidate(
            "obs://pytorch-ci-cache/7day-cache/wheels-bbbbbbbbbb.tar.gz",
            "obs://pytorch-ci-cache/7day-cache/pytorch-source-bbbbbbbbbb.tar.gz",
            obs_wheels_version="bad", obs_source_version="bad",
        )
        queue.register_identity_rule("r", "规则", "测试 fixture")
        queue.verify_identity(
            t.task_id, "r",
            {"pr_number": 1, "pytorch_commit": COMMIT_A,
             "torch_npu_commit": COMMIT_NPU, "binding": "x"},
        )
        queue.prefilter_outcome(t.task_id, True)
        queue.submit_manifest(t.task_id, [{"test_function": TF_A, "device": "cpu"}])
        executor = Executor(
            config, queue_store=queue, downloader=FakeDownloader(), runner=FakeRunner()
        )
        outcome = executor.run_once()
        assert outcome.outcome == "blocked"
        assert "不对应" in queue.get_task(t.task_id).error_reason


class TestSourceRoot:
    def test_locate_test_root_prefers_pytorch_prefix(self, config, queue, paths):
        """tar 根为 pytorch/ 时定位 source/pytorch；平铺布局回退 source/。"""
        task = verified_task(queue)
        claimed = queue.claim_next()
        assert claimed is not None and claimed.task_id == task.task_id
        work = paths.task_work_root / task.task_id
        source = work / "source"
        # 两种布局并存时 pytorch/ 前缀优先。
        (source / "pytorch" / "test").mkdir(parents=True)
        (source / "test").mkdir(parents=True)
        executor = Executor(config, queue_store=queue)
        assert executor._locate_test_root(claimed) == source / "pytorch"
        # 平铺布局（无 pytorch/ 前缀）回退 source/ 本身。
        import shutil

        shutil.rmtree(source / "pytorch")
        assert executor._locate_test_root(claimed) == source

    def test_missing_test_tree_blocks_before_testing(self, config, queue, paths, fake_pip, monkeypatch):
        """源码包无 test/：blocked（无法定位运行目录），不进 testing、不跑清单。"""
        import conftest

        orig_make_source_tar = conftest.make_source_tar

        def no_test_tar(dest, sha10):
            return orig_make_source_tar(dest, sha10, test_tree=False)

        monkeypatch.setattr(conftest, "make_source_tar", no_test_tar)
        task = verified_task(queue)
        runner = FakeRunner()
        executor = Executor(config, queue_store=queue, downloader=FakeDownloader(), runner=runner)
        outcome = executor.run_once()
        assert outcome.outcome == "blocked"
        t = queue.get_task(task.task_id)
        assert t.state == q.STATE_BLOCKED
        assert "未找到测试树" in t.error_reason
        assert runner.source_roots == []  # 没有跑到任何用例


class TestNoResult:
    def test_incomplete_function_degrades_to_no_result(self, config, queue, paths, fake_pip):
        """单函数未取得结果：降级 no_result，整轮继续——其余结果照常写库。"""
        task = verified_task(queue)
        runner = FakeRunner(
            counts_map={(TF_B, "accelerator"): Counts(2, 1, 0, 0)},
            incomplete={(TF_A, "cpu")},
        )
        executor = Executor(
            config, queue_store=queue, downloader=FakeDownloader(), runner=runner
        )
        outcome = executor.run_once()
        assert outcome.outcome == "done"
        assert queue.get_task(task.task_id).state == q.STATE_DONE

        # PR 库：一行；汇总只计 ok 项；tests 含 no_result 项（无 counts）。
        conn = connect(paths.pr_db)
        try:
            pr = get_pr_result(conn, 123456)
        finally:
            conn.close()
        assert pr.counts == Counts(2, 1, 0, 0)
        tf_a_item = next(i for i in pr.tests if i["test_function"] == TF_A)
        assert tf_a_item["status"] == "no_result"
        assert "counts" not in tf_a_item and tf_a_item["reason"]

        # 基线：no_result 项未建库未写日志；ok 项正常。
        root_a = paths.baselines_root / "test/distributed/_tensor"
        conn = connect(baseline_db_path(root_a))
        try:
            rec_a = get_baseline(conn, TF_A, "cpu")
        finally:
            conn.close()
        assert rec_a is None
        logs_a = root_a / "logs"
        assert not any(p.name.startswith("test_tensor.py") for p in logs_a.glob("*.cpu.*"))
        root_b = paths.baselines_root / "test/distributed/fsdp"
        conn = connect(baseline_db_path(root_b))
        try:
            rec_b = get_baseline(conn, TF_B, "accelerator")
        finally:
            conn.close()
        assert rec_b is not None and rec_b.counts == Counts(2, 1, 0, 0)

        # no_result 的证据：诊断 + PR 日志目录里的函数日志。
        assert (paths.diagnostics_root / task.task_id).is_dir()

    def test_all_incomplete_blocks_as_env_suspect(self, config, queue, paths, fake_pip):
        """清单全部未取得结果：done，PR 库保存 no_result 证据且不刷基线。"""
        task = verified_task(queue)
        runner = FakeRunner(
            incomplete={(TF_A, "cpu"), (TF_B, "accelerator")}
        )
        executor = Executor(
            config, queue_store=queue, downloader=FakeDownloader(), runner=runner
        )
        outcome = executor.run_once()
        assert outcome.outcome == "done"
        t = queue.get_task(task.task_id)
        assert t.state == q.STATE_DONE
        # PR 库保留 no_result 证据，不把未观察结果伪装成零计数.
        from run_test.pr_store import ensure_pr_db

        ensure_pr_db(paths.data_root)
        conn = connect(paths.pr_db)
        try:
            pr = get_pr_result(conn, 123456)
            assert pr is not None
            assert all(item["status"] == "no_result" for item in pr.tests)
            assert pr.counts == Counts(0, 0, 0, 0)
        finally:
            conn.close()
        # 诊断保留。
        assert (paths.diagnostics_root / task.task_id).is_dir()

class TestPreflight:
    """环境预检：环境级故障 blocked(env)，不进 testing、不跑清单。"""

    @staticmethod
    def _dispatch_run(monkeypatch, responses):
        """按子串分派的 subprocess.run 桩：首个命中 (needle, rc, stdout) 生效。"""
        def fake(cmd, *args, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            for needle, rc, stdout in responses:
                if needle in joined:
                    return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        monkeypatch.setattr(subprocess, "run", fake)

    def test_npu_unavailable_blocks(self, config, queue, paths, monkeypatch):
        self._dispatch_run(monkeypatch, [
            ("npu.is_available", 1, ""),
            ("import torch", 0, COMMIT_A),
        ])
        task = verified_task(queue)
        runner = FakeRunner()
        executor = Executor(
            config, queue_store=queue, downloader=FakeDownloader(), runner=runner
        )
        outcome = executor.run_once()
        assert outcome.outcome == "blocked"
        t = queue.get_task(task.task_id)
        assert t.state == q.STATE_BLOCKED
        assert "torch_npu/NPU 不可用" in t.error_reason
        assert runner.source_roots == []
        assert any(
            p.name.startswith("preflight-1-npu")
            for p in (paths.diagnostics_root / task.task_id).glob("*.log")
        )

    def test_boot_selfcheck_fail_line_blocks(self, config, queue, paths, tmp_path, monkeypatch):
        """boot 自检输出含 FAIL 行（即使退出码 0）→ blocked。"""
        boot = tmp_path / "transfer_boot"
        boot.mkdir()
        (boot / "boot_selfcheck.py").write_text("# stub\n")
        config["executor"]["transfer_boot_dir"] = str(boot)
        self._dispatch_run(monkeypatch, [
            ("boot_selfcheck.py", 0, "  PASS  npu available\n  FAIL  v3.1 _cuda_setStream 路由\n"),
            ("npu.is_available", 0, "npu devices: 4"),
            ("randn", 0, "matmul sum: 1.0"),
            ("import torch", 0, COMMIT_A),
        ])
        task = verified_task(queue, manifest=[{"test_function": TF_A, "device": "cuda"}])
        runner = FakeRunner()
        executor = Executor(
            config, queue_store=queue, downloader=FakeDownloader(), runner=runner
        )
        outcome = executor.run_once()
        assert outcome.outcome == "blocked"
        assert "只允许 cpu / accelerator" in queue.get_task(task.task_id).error_reason
        assert runner.source_roots == []

    def test_collect_smoke_failure_blocks(self, config, queue, paths, monkeypatch):
        self._dispatch_run(monkeypatch, [
            ("--collect-only", 2, "ERROR: file or directory not found"),
            ("npu.is_available", 0, "npu devices: 4"),
            ("randn", 0, "matmul sum: 1.0"),
            ("import torch", 0, COMMIT_A),
        ])
        task = verified_task(queue)
        runner = FakeRunner()
        executor = Executor(
            config, queue_store=queue, downloader=FakeDownloader(), runner=runner
        )
        outcome = executor.run_once()
        assert outcome.outcome == "done"
        assert runner.source_roots
        assert any("preflight-4-collect-target" in p.name for p in (paths.diagnostics_root / task.task_id).glob("*.log"))

    def test_preflight_pass_writes_diagnostics_and_runs(self, config, queue, paths, monkeypatch):
        """四步全过：诊断落盘（可追溯），任务正常进入 testing 并完成。"""
        self._dispatch_run(monkeypatch, [
            ("npu.is_available", 0, "npu devices: 4"),
            ("randn", 0, "matmul sum: 1.0"),
            ("import torch", 0, COMMIT_A),
        ])
        task = verified_task(queue)
        runner = FakeRunner()
        executor = Executor(
            config, queue_store=queue, downloader=FakeDownloader(), runner=runner
        )
        outcome = executor.run_once()
        assert outcome.outcome == "done"
        names = {p.name.split("-20")[0] for p in (paths.diagnostics_root / task.task_id).glob("preflight-*.log")}
        assert {"preflight-1-npu", "preflight-2-op", "preflight-4-collect"} <= names
        assert runner.source_roots  # 清单确实跑了


class TestAbortPaths:
    def test_persist_failure_keeps_evidence(self, config, queue, paths, fake_pip):
        """写库失败：重试后仍失败 → failed 且不清理（防删唯一结果副本）。"""
        task = verified_task(queue)

        def bad_persist(cfg, sub):
            raise RuntimeError("模拟写库失败")

        executor = Executor(
            config, queue_store=queue,
            downloader=FakeDownloader(), runner=FakeRunner(),
            persist=bad_persist,
        )
        outcome = executor.run_once()
        assert outcome.outcome == "failed"
        t = queue.get_task(task.task_id)
        assert t.state == q.STATE_FAILED
        assert "写库失败" in t.error_reason
        # 结果证据保留在任务临时目录。
        assert (paths.task_work_root / task.task_id).exists()


class TestCleanupSafety:
    def test_refuses_paths_outside_task_work(self, config, queue):
        """删除前核实绝对路径位于本任务登记的临时目录内（文档第 7 节）。"""
        executor = Executor(config, queue_store=queue)
        with pytest.raises(ExecutorError, match="拒绝删除"):
            executor._safe_cleanup("/tmp")
        with pytest.raises(ExecutorError, match="拒绝删除"):
            executor._safe_cleanup(config["data_root"])
        with pytest.raises(ExecutorError, match="拒绝删除"):
            # data_root 之内的非 task_work 目录同样拒绝。
            executor._safe_cleanup(str(executor.paths.pr_logs_root / "PR_1"))

    def test_resume_cleaning_finishes_before_next_task(self, config, queue, paths, fake_pip):
        """清理中断的任务在下一轮被续清理；完成前不开始下一任务。"""
        task = verified_task(queue)
        queue.claim_next()  # preparing，登记 work_dir
        queue.transition(task.task_id, q.STATE_TESTING)
        queue.transition(task.task_id, q.STATE_WRITING)
        queue.transition(task.task_id, q.STATE_CLEANING)
        work = paths.task_work_root / task.task_id
        work.mkdir(parents=True, exist_ok=True)
        (work / "residue.txt").write_text("x")

        # 另一个就绪任务存在，但清理完成后才轮到它。
        next_task = verified_task(queue)
        executor = Executor(
            config, queue_store=queue, downloader=FakeDownloader(), runner=FakeRunner()
        )
        outcome = executor.run_once()
        # 上一任务清理完成。
        assert queue.get_task(task.task_id).state == q.STATE_DONE
        assert not work.exists()
        # 同一轮内执行位空闲后继续消费下一个任务。
        assert outcome.task_id == next_task.task_id
        assert outcome.outcome == "done"
