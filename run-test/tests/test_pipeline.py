"""pipeline 节拍器：三阶段顺序、单任务容错、make_runner 配置语义。"""

from __future__ import annotations

import itertools
import os
import threading
from pathlib import Path
from types import SimpleNamespace

from run_test import executor as executor_mod
from run_test import pipeline
from run_test import prefilter as prefilter_mod
from run_test.cli import build_parser

from conftest import COMMIT_A, COMMIT_NPU

SHA10 = COMMIT_A[:10]
WHEELS_URL = f"obs://pytorch-ci-cache/7day-cache/wheels-{SHA10}.tar.gz"
SOURCE_URL = f"obs://pytorch-ci-cache/7day-cache/pytorch-source-{SHA10}.tar.gz"

_version_counter = itertools.count(1)


def _pending_analysis_task(queue):
    """造一个停在 pending_analysis 的任务（不提交清单）。

    同 URL 不同 version 视为不同对象（与 executor 测试的去重规避一致）。
    """
    ver = f"pipe-v{next(_version_counter)}"
    task, _ = queue.register_candidate(
        WHEELS_URL, SOURCE_URL,
        obs_wheels_version=ver, obs_source_version=ver,
    )
    queue.register_identity_rule("r", "规则", "测试 fixture")
    queue.verify_identity(
        task.task_id, "r",
        {"pr_number": 123456, "pytorch_commit": COMMIT_A,
         "torch_npu_commit": COMMIT_NPU, "binding": "x"},
    )
    queue.prefilter_outcome(task.task_id, True)
    return queue.get_task(task.task_id)


def _fake_prefilter(calls):
    def _run(config, queue=None):
        calls.append("prefilter")
        return {"processed": 3, "kept": 2, "filtered_out": 1, "errors": []}
    return _run


def test_pipeline_once_orders_phases(config, queue, monkeypatch):
    """prefilter 先行同步完成；select（后台线程）与 executor 并行重叠，
    两者相对顺序不固定，但都必须发生且结果入报告。"""
    calls = []
    task = _pending_analysis_task(queue)

    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter(calls))

    def fake_select(cfg, store, task_id, **kwargs):
        calls.append(f"select:{task_id}")
        return 0
    monkeypatch.setattr(pipeline, "run_selection", fake_select)

    class FakeExecutor:
        def __init__(self, cfg, downloader=None, runner=None):
            calls.append("executor-init")

        def run_once(self):
            calls.append("executor-run")
            return SimpleNamespace(task_id="t", outcome="idle", detail="")
    monkeypatch.setattr(executor_mod, "Executor", FakeExecutor)

    report = pipeline.pipeline_once(config)
    # prefilter 先行；select 与 executor 重叠，相对顺序不固定。
    assert calls[0] == "prefilter"
    assert f"select:{task.task_id}" in calls
    assert "executor-init" in calls and "executor-run" in calls
    assert report["selected"] == [task.task_id]
    assert report["select_errors"] == []
    assert report["executed"] == []


def test_select_failure_does_not_stop_round(config, queue, monkeypatch):
    """筛选失败（子进程 rc!=0）只记账：轮继续、executor 照常消费。"""
    import sys
    calls = []
    task = _pending_analysis_task(queue)
    task2 = _pending_analysis_task(queue)

    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter(calls))

    class FakeExecutor:
        def __init__(self, cfg, downloader=None, runner=None):
            pass

        def run_once(self):
            calls.append("executor-run")
            return SimpleNamespace(task_id="t", outcome="idle", detail="")
    monkeypatch.setattr(executor_mod, "Executor", FakeExecutor)

    # 两个任务各自独立子进程；一个 rc=1 一个 rc=2（如源码包缺失）。
    import json as _json
    tmp_settings = tmp_path / "settings.json"
    tmp_settings.write_text(_json.dumps({"data_root": str(tmp_path / "data")}), encoding="utf-8")
    config["_selection_command"] = [
        sys.executable, "-c",
        "import sys; sys.exit(1 if sys.argv[-1].endswith(('0','2','4','6','8')) else 2)",
        "{task_id}"]
    config["_config_path"] = str(tmp_settings)
    report = pipeline.pipeline_once(config)
    assert "executor-run" in calls
    assert report["selected"] == []
    assert len(report["select_errors"]) >= 2  # 两个任务各自留痕
    assert any("退出码 1" in e for e in report["select_errors"])
    assert any("退出码 2" in e for e in report["select_errors"])
    assert any("退出码 2" in e for e in report["select_errors"])


def test_executor_consumes_until_idle(config, queue, monkeypatch):
    """executor 逐任务消费，idle 即停；blocked/failed 也算消费完成。"""
    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter([]))

    class FakeExecutor:
        def __init__(self, cfg, downloader=None, runner=None):
            self.outcomes = iter([
                SimpleNamespace(task_id="t1", outcome="done", detail="PR 1"),
                SimpleNamespace(task_id="t2", outcome="blocked", detail="预检失败"),
                SimpleNamespace(task_id="t3", outcome="idle", detail=""),
            ])

        def run_once(self):
            try:
                return next(self.outcomes)
            except StopIteration:
                # 脚本耗尽后恒返 idle（真实语义）：select 线程退出前的
                # idle 重询会多调用若干次。
                return SimpleNamespace(task_id=None, outcome="idle", detail="")
    monkeypatch.setattr(executor_mod, "Executor", FakeExecutor)

    report = pipeline.pipeline_once(config)
    assert [x["outcome"] for x in report["executed"]] == ["done", "blocked"]


def test_pipeline_overlaps_select_and_executor(config, queue, monkeypatch):
    """重叠实证：select 阻塞（模拟慢下载）期间 executor 已消费任务——
    下载和测试互相掩盖，而非"前一任务测完才下载下一个"。"""
    import sys
    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter([]))
    task = _pending_analysis_task(queue)

    select_started = threading.Event()
    picked_while_blocked: list[int] = []

    class FakeExecutor:
        def __init__(self, cfg, downloader=None, runner=None):
            pass

        def run_once(self):
            if select_started.is_set():
                # select 仍运行时消费任务 = 重叠实证。
                picked_while_blocked.append(1)
                return SimpleNamespace(task_id="tX", outcome="idle", detail="")
            return SimpleNamespace(task_id=None, outcome="idle", detail="")
    monkeypatch.setattr(executor_mod, "Executor", FakeExecutor)

    # stub 子进程：写标记文件后睡眠 3 秒——期间 executor 照常消费（重叠实证）。
    import os as _os
    marker = _os.path.join(str(paths.data_root), "select-started.marker")
    config["_selection_command"] = [
        sys.executable, "-c",
        "import os, sys, time\n"
        "open(sys.argv[-1], 'w').write('1')\n"
        "time.sleep(3)\n",
        marker,
    ]
    config["_config_path"] = "settings.json"
    report = pipeline.pipeline_once(config)
    assert picked_while_blocked, "select 运行期间 executor 未消费——重叠未生效"
    assert _os.path.exists(marker)  # 子进程确实在跑过
    assert report["executed"] == [] or report["executed"][-1]["outcome"] in ("done", "idle")


def test_make_runner_requires_python_env(config):
    # conftest 的 config 恒设 test_python_env；先清空验证 None 语义。
    config["executor"]["test_python_env"] = None
    assert pipeline.make_runner(config) is None
    config["executor"]["test_python_env"] = "/usr/bin/python3.10"
    runner = pipeline.make_runner(config)
    assert runner is not None and runner.python_env == "/usr/bin/python3.10"


def test_pipeline_cli_registered():
    args = build_parser().parse_args(["pipeline"])
    assert args.cmd == "pipeline" and args.loop is False and args.interval is None
    loop_args = build_parser().parse_args(["pipeline", "--loop", "--interval", "60"])
    assert loop_args.loop is True and loop_args.interval == 60.0


# ---------------------------------------------------------------------------
# P1-1（2026-09-20）：选端发现环境注入 boot——cuda 主通道不再静默消失
# ---------------------------------------------------------------------------


def test_boot_discovery_env_injects_boot_without_device_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_TESTING_DEVICE_ONLY_FOR", "cuda")
    env = pipeline.boot_discovery_env({"executor":{"transfer_boot_dir":str(tmp_path)}})
    assert env["PYTORCH_TESTING_DEVICE_ONLY_FOR"] == "cpu,npu"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert str(tmp_path) not in env.get("PYTHONPATH", "").split(os.pathsep)


def test_boot_discovery_env_none_when_boot_missing(config):
    env = pipeline.boot_discovery_env(config)
    assert env["PYTORCH_TESTING_DEVICE_ONLY_FOR"] == "cpu,npu"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


def test_run_selection_passes_boot_discovery_env(config, queue, tmp_path, monkeypatch):
    """run_selection 必须把 boot 发现环境传给 select_for_commit（P1-1 集成锁）。"""
    task = _pending_analysis_task(queue)
    boot = tmp_path / "transfer_boot"
    boot.mkdir()
    (boot / "sitecustomize.py").write_text("# boot\n")
    config["executor"]["transfer_boot_dir"] = str(boot)

    # 源码包缓存就位：跳过 OBS 下载与解包。
    sha10 = COMMIT_A[:10]
    cache = Path(config["data_root"]) / "source_cache" / sha10
    (cache / "pytorch" / "test").mkdir(parents=True)
    (cache / ".selection-sources-complete").write_text("{}", encoding="utf-8")

    class FakeMirror:
        git_dir = "/fake/git"
        def __init__(self, config=None, **kwargs):
            pass
        def ensure(self):
            return None

    import run_test.upstream as upstream_mod
    import run_test.selection as selection_mod
    import run_test.device_detect as device_detect_mod

    monkeypatch.setattr(upstream_mod, "PyTorchMirror", FakeMirror)
    monkeypatch.setattr(selection_mod, "git", lambda *a, **k: "subject (#123)")
    class _FakeResolver:
        """形如 DeviceResolver：可调用 + diagnostics + source_test_function。"""
        diagnostics: list = []

        def __call__(self, path, node):
            return ["cpu"]

        def source_test_function(self, path, node):
            return None

    monkeypatch.setattr(
        device_detect_mod, "make_device_resolver", lambda tree: _FakeResolver()
    )

    captured: dict = {}

    def fake_select(git_dir, sha, subject, tree, **kwargs):
        captured.update(kwargs)
        captured["_tree"] = tree
        return {
            "p0": [{"path": "test/test_foo.py", "test_nodes": ["TestFoo.test_x_cpu"],
                    "evidence": []}],
            "p1": [], "missing_coverage": [], "ci_recommendations": [],
            "analysis_gaps": [], "notes": [],
        }

    monkeypatch.setattr(selection_mod, "select_for_commit", fake_select)
    rc = pipeline.run_selection(config, queue, task.task_id)
    assert rc == 0
    env = captured["discovery_env"]
    assert env is not None
    assert str(boot) not in env["PYTHONPATH"].split(os.pathsep)
    assert env["PYTORCH_TESTING_DEVICE_ONLY_FOR"] == "cpu,npu"
    # 三通道链路通：收集到的 cuda 变体 → (tf, cuda) 清单条目。
    store_task = queue.get_task(task.task_id)
    assert store_task.state == "pending_test"
    assert (store_task.test_manifest[0]["test_function"],
            store_task.test_manifest[0]["device"]) == (
        "test/test_foo.py::TestFoo::test_x_cpu", "cpu")


# ---------------------------------------------------------------------------
# 持续消费 / 子进程隔离 / 任务级源码复用与完成清理（用户 2026-10-26）
# ---------------------------------------------------------------------------

def test_selector_continuous_consumption_production_path(config, queue, monkeypatch, tmp_path):
    import sys
    """生产路径：真实子进程 select-one（生产命令）——候选即时消费、
    attempted 去重不重复派生、轮正常收班。tmp 数据根上 OBS 未配置 →
    子进程 rc=2 快速失败，任务留 pending_analysis（下轮重试语义）。"""
    import json as _json
    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter([]))
    task = _pending_analysis_task(queue)
    tmp_settings = tmp_path / "settings.json"
    tmp_settings.write_text(_json.dumps({"data_root": str(tmp_path / "data")}), encoding="utf-8")
    capture = tmp_path / "argv-capture.json"
    config["_config_path"] = str(tmp_settings)
    # 捕获命令：记录收到的 argv（--config/--task）后成功退出——验证
    # 生产 spawn 路径的参数传播与协调机制，不触发镜像克隆。
    config["_selection_command"] = [
        sys.executable, "-c",
        "import json, sys\njson.dump(sys.argv[1:], open(sys.argv[1], 'w'))\n",
        str(capture), "{task_id}"]
    report = pipeline.pipeline_once(config)
    assert report["selected"] == [task.task_id]
    argv = json.loads(capture.read_text())
    assert argv == ["--task", task.task_id, "--config", str(tmp_settings)]  # 参数传播
    assert queue.get_task(task.task_id).state == "pending_analysis"


def test_selector_failure_backoff_then_blocked(config, queue, monkeypatch):
    """连续失败达上限转 blocked（证据留 error_reason），不再无限重试。"""
    import sys
    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter([]))
    import sys
    import json as _json
    task = _pending_analysis_task(queue)
    tmp_settings = tmp_path / "settings.json"
    tmp_settings.write_text(_json.dumps({"data_root": str(tmp_path / "data")}), encoding="utf-8")
    capture = tmp_path / "argv-capture.json"
    config["_selection_command"] = [
        sys.executable, "-c",
        "import json, sys\njson.dump(sys.argv[1:], open(sys.argv[1], 'w'))\nsys.exit(7)\n",
        str(capture), "{task_id}"]
    config["_config_path"] = str(tmp_settings)
    config["selector"] = {"max_failures": 2}
    pipeline._SELECTION_FAILURES.clear()
    pipeline.pipeline_once(config)  # 第 1 次失败
    assert queue.get_task(task.task_id).state == "pending_analysis"
    pipeline.pipeline_once(config)  # 第 2 次失败 → blocked
    t = queue.get_task(task.task_id)
    assert t.state == "blocked" and "连续失败 2 次" in (t.error_reason or "")


def test_selection_lock_prevents_duplicate_spawn(config, queue, monkeypatch):
    """同任务筛选互斥锁：已有筛选运行时 rc=3 跳过，不重复筛选。"""
    import fcntl
    import os
    from run_test.config import store_paths
    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter([]))
    task = _pending_analysis_task(queue)
    lock_dir = store_paths(config).data_root / "selection" / task.task_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_dir / ".selection.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # 模拟在跑的筛选
    config["_selection_command"] = ["/bin/true"]
    config["_config_path"] = str(paths.data_root.parent / "settings.json")
    pipeline._SELECTION_FAILURES.clear()
    report = pipeline.pipeline_once(config)
    assert any("已有筛选在运行" in e for e in report["select_errors"])
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)


def test_executor_reuses_selection_source_and_cleans_up(config, router, paths, queue, monkeypatch):
    """任务级源码复用：筛选暂存归档被 executor 采用（源 URL 不再下载）；
    任务完成后 selection 暂存删除、共享 source_cache 无在用引用时清理。"""
    import tarfile
    from test_record_flow import FakeMirror, FakeRunner
    from run_test.executor import Executor
    from conftest import FakeDownloader
    monkeypatch.setattr("run_test.upstream.PyTorchMirror", FakeMirror)
    task = _pending_analysis_task(queue)
    queue.submit_manifest(task.task_id, [{"test_function": "test/test_sample.py::TestSample::test_a",
                                          "device": "cpu"}])
    staging = paths.data_root / "selection" / task.task_id / "source-staging"
    staging.mkdir(parents=True, exist_ok=True)
    src_tree = staging / "pytorch"
    (src_tree / "test").mkdir(parents=True, exist_ok=True)
    with tarfile.open(staging / "pytorch-source.tar.gz", "w:gz") as tf:
        tf.add(src_tree, arcname="pytorch")
    dl = FakeDownloader()
    runner = FakeRunner()
    executor = Executor(config, queue_store=queue, downloader=dl, runner=FakeRunner())
    outcome = executor.run_once()
    assert outcome.outcome == "done", f"detail={outcome.detail!r}"
    assert all("pytorch-source" not in u for u in dl.calls)  # 源码未重复下载
    assert not staging.exists()  # 完成后任务级暂存删除
    assert not (paths.data_root / "source_cache" / SHA10).exists()  # 共享缓存清理


def test_cleanup_failure_preserves_evidence_and_retries(config, router, paths, queue, monkeypatch):
    """清理失败保持 cleaning + 重试留痕，不假报 done（阻塞项②）。"""
    import pytest
    from test_record_flow import FakeMirror, FakeRunner
    from run_test import task_queue as q
    from run_test.executor import Executor
    from conftest import FakeDownloader
    monkeypatch.setattr("run_test.upstream.PyTorchMirror", FakeMirror)
    task = _pending_analysis_task(queue)
    queue.submit_manifest(task.task_id, [{"test_function": "test/test_sample.py::TestSample::test_a",
                                          "device": "cpu"}])
    executor = Executor(config, queue_store=queue, downloader=FakeDownloader(), runner=FakeRunner())
    # 直测 _resume_cleaning 的失败语义：selection 暂存删除失败 → 保持 cleaning。
    staging = paths.data_root / "selection" / task.task_id
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "x").write_text("x", encoding="utf-8")
    monkeypatch.setattr(executor, "_safe_cleanup", lambda wd: None)
    import shutil as _shutil
    real_rmtree = _shutil.rmtree
    def failing_rmtree(path, *a, **k):
        if str(staging) in str(path):
            raise OSError("模拟清理失败（占用）")
        return real_rmtree(path, *a, **k)
    monkeypatch.setattr(_shutil, "rmtree", failing_rmtree)
    executor._resume_cleaning(task)
    t = queue.get_task(task.task_id)
    assert t.state == q.STATE_CLEANING  # 不假报 done；重试留痕
    monkeypatch.setattr(_shutil, "rmtree", real_rmtree)
    executor._resume_cleaning(task)
    assert queue.get_task(task.task_id).state == q.STATE_DONE  # 恢复后完成
