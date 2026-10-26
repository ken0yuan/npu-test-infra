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
    """单任务筛选失败（异常或非零退出码）只记账；executor 阶段照常执行。"""
    calls = []
    task = _pending_analysis_task(queue)
    task2 = _pending_analysis_task(queue)

    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter(calls))

    def fake_select(cfg, store, task_id, **kwargs):
        calls.append(f"select:{task_id}")
        if task_id == task.task_id:
            raise RuntimeError("镜像取 subject 失败")
        return 2  # 非零退出码（如源码包缺失）
    monkeypatch.setattr(pipeline, "run_selection", fake_select)

    class FakeExecutor:
        def __init__(self, cfg, downloader=None, runner=None):
            pass

        def run_once(self):
            calls.append("executor-run")
            return SimpleNamespace(task_id="t", outcome="idle", detail="")
    monkeypatch.setattr(executor_mod, "Executor", FakeExecutor)

    report = pipeline.pipeline_once(config)
    # 两个任务的筛选都尝试了，executor 也执行了。
    assert f"select:{task.task_id}" in calls and f"select:{task2.task_id}" in calls
    assert "executor-run" in calls
    assert report["selected"] == []
    assert len(report["select_errors"]) == 2
    assert any("镜像取 subject 失败" in e for e in report["select_errors"])
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
    monkeypatch.setattr(prefilter_mod, "prefilter_pending_tasks", _fake_prefilter([]))
    task = _pending_analysis_task(queue)

    select_started = threading.Event()
    release_select = threading.Event()
    picked_while_blocked: list[int] = []

    def fake_select(cfg, store, task_id, **kwargs):
        select_started.set()
        assert release_select.wait(timeout=10), "select 未被放行（死锁）"
        return 0
    monkeypatch.setattr(pipeline, "run_selection", fake_select)

    class FakeExecutor:
        def __init__(self, cfg, downloader=None, runner=None):
            pass

        def run_once(self):
            if select_started.is_set() and not release_select.is_set():
                # select 仍阻塞时消费任务 = 重叠实证；随后放行 select。
                picked_while_blocked.append(1)
                release_select.set()
                return SimpleNamespace(task_id="tX", outcome="done", detail="PR 1")
            return SimpleNamespace(task_id=None, outcome="idle", detail="")
    monkeypatch.setattr(executor_mod, "Executor", FakeExecutor)

    report = pipeline.pipeline_once(config)
    assert picked_while_blocked, "select 阻塞期间 executor 未消费——重叠未生效"
    assert report["executed"] == [
        {"task_id": "tX", "outcome": "done", "detail": "PR 1"},
    ]
    assert report["selected"] == [task.task_id]


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
