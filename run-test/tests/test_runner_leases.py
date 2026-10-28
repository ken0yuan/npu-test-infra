"""Card leases belong to each subprocess call, including runtime discovery."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from run_test.device_policy import execution_env, pytest_command
from run_test.runner import PytestRunner
from run_test.process_lock import lease_handle


TF = "test/test_sample.py::TestSample::test_value"
NODE = "test_sample.py::TestSample::test_value"


def source_tree(tmp_path):
    root = tmp_path / "source"
    (root / "test").mkdir(parents=True)
    (root / "test/test_sample.py").write_text(
        "class TestSample:\n    def test_value(self):\n        pass\n", encoding="utf-8")
    return root


@pytest.mark.skipif(os.name == "nt", reason="POSIX inherited card locks")
def test_discovery_inherits_executor_and_current_card_lease(tmp_path, monkeypatch):
    calls = []
    def collect(cmd, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0, NODE + "\n", "")
    monkeypatch.setattr(subprocess, "run", collect)
    runner = PytestRunner(sys.executable)
    runner.process_lease_fds = (11,)
    assert runner.discover_runtime_nodes(
        TF, "accelerator", source_root=source_tree(tmp_path),
        extra_env={"ASCEND_RT_VISIBLE_DEVICES": "4"}, lease_fds=(22, 11)) == [TF]
    assert set(calls[0]["pass_fds"]) == {11, 22}
    assert calls[0]["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4"
    assert runner.process_lease_fds == (11,)


@pytest.mark.skipif(os.name == "nt", reason="POSIX inherited card locks")
def test_live_collection_keeps_both_locks_after_parent_closes_them(tmp_path, monkeypatch):
    root = source_tree(tmp_path)
    ready, release = tmp_path / "ready", tmp_path / "release"
    (root / "test/test_sample.py").write_text(
        "import os, pathlib, time\n"
        "pathlib.Path(os.environ['LEASE_READY']).write_text('ready')\n"
        "deadline = time.monotonic() + 20\n"
        "while not pathlib.Path(os.environ['LEASE_RELEASE']).exists():\n"
        "    if time.monotonic() > deadline: raise RuntimeError('release timeout')\n"
        "    time.sleep(.02)\n"
        "class TestSample:\n    def test_value(self): pass\n", encoding="utf-8")
    monkeypatch.setenv("LEASE_READY", str(ready))
    monkeypatch.setenv("LEASE_RELEASE", str(release))
    executor_lock, card_lock = tmp_path / "executor.lock", tmp_path / "npu-4.lock"
    runner = PytestRunner(sys.executable, timeout_seconds=30)
    pool = ThreadPoolExecutor(1)
    try:
        with lease_handle(executor_lock) as executor_fd, lease_handle(card_lock) as card_fd:
            runner.process_lease_fds = (executor_fd,)
            future = pool.submit(runner.discover_runtime_nodes, TF, "cpu",
                                 source_root=root, lease_fds=(card_fd,))
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists(), "Collection child never reached its module import"
        # Collection still owns the inherited open file descriptions, even
        # after the executor's local handles have closed.
        with lease_handle(executor_lock) as fd:
            assert fd is None
        with lease_handle(card_lock) as fd:
            assert fd is None
        release.touch()
        assert future.result(timeout=10) == [TF]
    finally:
        release.touch()
        pool.shutdown(wait=True)
    with lease_handle(executor_lock) as fd:
        assert fd is not None
    with lease_handle(card_lock) as fd:
        assert fd is not None


def test_parallel_discovery_uses_call_local_lease_fds(tmp_path, monkeypatch):
    root = source_tree(tmp_path)
    runner = PytestRunner(sys.executable)
    runner.begin_task()
    runner.process_lease_fds = (11,)
    # Capture the per-call argument on every platform; the POSIX test above
    # separately verifies the real pass_fds merge.
    monkeypatch.setattr(runner, "_lease_options", lambda fds=(): {"lease_arguments": tuple(fds)})
    barrier = threading.Barrier(2)
    seen = {}
    def collect(cmd, **kwargs):
        barrier.wait(timeout=5)
        seen[kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"]] = kwargs["lease_arguments"]
        return subprocess.CompletedProcess(cmd, 0, NODE + "\n", "")
    monkeypatch.setattr(subprocess, "run", collect)
    def discover(card):
        return runner.discover_runtime_nodes(
            TF, "accelerator", source_root=root,
            extra_env={"ASCEND_RT_VISIBLE_DEVICES": str(card)}, lease_fds=(card + 100,))
    with ThreadPoolExecutor(2) as executor:
        assert list(executor.map(discover, (4, 6))) == [[TF], [TF]]
    assert seen == {"4": (104,), "6": (106,)}
    assert runner.process_lease_fds == (11,)


def test_receipt_identity_uses_device_mask_not_ephemeral_fd(tmp_path, monkeypatch):
    root = source_tree(tmp_path)
    runner = PytestRunner(sys.executable)
    runner.begin_task()
    calls = []
    monkeypatch.setattr(runner, "_lease_options", lambda fds=(): {"lease_arguments": tuple(fds)})
    def collect(cmd, **kwargs):
        calls.append((kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], kwargs["lease_arguments"]))
        return subprocess.CompletedProcess(cmd, 0, NODE + "\n", "")
    monkeypatch.setattr(subprocess, "run", collect)
    for mask, fd in (("4", 104), ("4", 204), ("6", 106)):
        assert runner.discover_runtime_nodes(
            TF, "accelerator", source_root=root,
            extra_env={"ASCEND_RT_VISIBLE_DEVICES": mask}, lease_fds=(fd,)) == [TF]
    assert calls == [("4", (104,)), ("6", (106,))]
    assert len(runner._collection_cache) == 2


def test_execution_recollection_and_child_keep_same_card_lease(tmp_path, monkeypatch):
    root = source_tree(tmp_path)
    runner = PytestRunner(sys.executable)
    runner.begin_task()
    calls = []
    monkeypatch.setattr(runner, "_lease_options", lambda fds=(): {"lease_arguments": tuple(fds)})
    def collect(cmd, **kwargs):
        calls.append(("collect", kwargs))
        return subprocess.CompletedProcess(cmd, 0, NODE + "\n", "")
    collect.__module__ = "subprocess"
    monkeypatch.setattr(subprocess, "run", collect)
    class Child:
        returncode = 0
        pid = 42
        def __init__(self, cmd, **kwargs):
            calls.append(("execute", kwargs))
            path = Path(next(x.split("=", 1)[1] for x in cmd if x.startswith("--junitxml=")))
            path.write_text('<testsuite><testcase classname="TestSample" name="test_value"/></testsuite>', encoding="utf-8")
        def communicate(self, timeout=None):
            return "1 passed", ""
    monkeypatch.setattr(subprocess, "Popen", Child)
    outcome = runner.run_function(
        TF, "accelerator", source_root=root, source_test_function=TF,
        extra_env={"ASCEND_RT_VISIBLE_DEVICES": "6"}, lease_fds=(106,))
    assert outcome.completed
    assert [(phase, kw["lease_arguments"]) for phase, kw in calls] == [
        ("collect", (106,)), ("execute", (106,))]
    assert all(kw["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "6" for _, kw in calls)


@pytest.mark.parametrize("device", ["cpu", "accelerator"])
def test_channel_explicitly_controls_backend_registration(monkeypatch, device):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    env = execution_env(device)
    assert env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert env["PYTORCH_TESTING_DEVICE_ONLY_FOR"] == ("cpu" if device == "cpu" else "npu")
    command = pytest_command(sys.executable, device)
    if device == "cpu":
        assert command == [sys.executable, "-m", "pytest"]
    else:
        assert "import torch, torch_npu" in command[2]
