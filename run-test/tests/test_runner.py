"""runner boot 注入：device=cuda 注入 transfer boot，cpu/accelerator 不注入。

用户 2026-09-17 决策：cuda 经 boot v3（sitecustomize）transfer 到 NPU；
cpu / accelerator 直接用 test 源码跑。PYTHONPATH 必须先剥空项
（尾冒号陷阱：空路径项 = 当前目录，炸 import torch）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from run_test.config import ConfigPendingError
from run_test.runner import PytestRunner, _merge_pythonpath

TF_CUDA = "test/test_boot.py::test_cuda_case"
BOOT_SUFFIX = "test_cuda_case"


def _make_junit_for(cmd: list[str]) -> None:
    """fake_run：找到 --junitxml 参数并写一份含 1 个 testcase 的报告。"""
    xml_arg = next(c for c in cmd if c.startswith("--junitxml="))
    xml_path = Path(xml_arg.split("=", 1)[1])
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(ET.fromstring(
        '<testsuite name="s">'
        f'<testcase classname="TestBoot" name="{BOOT_SUFFIX}"/>'
        "</testsuite>"
    )).write(xml_path)


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append({"cmd": cmd, "env": kwargs.get("env"), "cwd": kwargs.get("cwd")})
        _make_junit_for(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def make_runner(tmp_path, boot_dir=None, **kwargs) -> PytestRunner:
    return PytestRunner(
        sys.executable, transfer_boot_dir=boot_dir, **kwargs
    )


def test_cuda_injects_boot_env(tmp_path, captured):
    from run_test.config import ConfigError
    runner = make_runner(tmp_path, boot_dir=tmp_path)
    with pytest.raises(ConfigError, match="只允许 cpu / accelerator"):
        runner.run_function(TF_CUDA, "cuda")
    assert captured == []


def test_cpu_and_accelerator_do_not_inject(tmp_path, captured):
    boot = tmp_path / "transfer_boot"
    boot.mkdir()
    runner = make_runner(tmp_path, boot_dir=boot)
    for dev in ("cpu", "accelerator"):
        outcome = runner.run_function(TF_CUDA, dev)
        assert outcome.completed
        assert captured[-1]["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert "transfer boot 注入" not in outcome.log_text


def test_cuda_without_boot_dir_runs_plain(tmp_path, captured):
    from run_test.config import ConfigError
    runner = make_runner(tmp_path, boot_dir=tmp_path)
    with pytest.raises(ConfigError, match="只允许 cpu / accelerator"):
        runner.run_function(TF_CUDA, "cuda")
    assert captured == []


def test_cuda_with_blank_boot_dir_ignored(tmp_path, captured):
    from run_test.config import ConfigError
    runner = make_runner(tmp_path, boot_dir=tmp_path)
    with pytest.raises(ConfigError, match="只允许 cpu / accelerator"):
        runner.run_function(TF_CUDA, "cuda")
    assert captured == []


def test_source_root_passed_as_cwd(tmp_path, captured):
    """source_root → 在 <root>/test 下运行 + nodeid 剥掉前导 test/。

    `python -m pytest` 把 cwd 前插 sys.path：从仓库根跑会让源码树里的
    torch/ 遮蔽已安装 wheel（实证调用形态见 boot USAGE §3.1），因此
    必须进 test/ 目录、用相对 test/ 的 nodeid。
    """
    runner = make_runner(tmp_path)
    tree = tmp_path / "pytorch-tree"
    outcome = runner.run_function(TF_CUDA, "cpu", source_root=tree)
    assert outcome.completed
    assert captured[0]["cwd"] == str(tree / "test")
    assert captured[0]["cmd"][3] == "test_boot.py::test_cuda_case"
    assert f"[cwd: {tree / 'test'}]" in outcome.log_text


def test_without_source_root_inherits_cwd(tmp_path, captured):
    """不传 source_root：cwd=None（继承执行器进程目录），nodeid 原样。"""
    runner = make_runner(tmp_path)
    outcome = runner.run_function(TF_CUDA, "cpu")
    assert outcome.completed
    assert captured[0]["cwd"] is None
    assert captured[0]["cmd"][3] == TF_CUDA
    assert "[cwd:" not in outcome.log_text


def test_merge_pythonpath_strips_empty_items(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(("/a", "/b", "")))
    merged = _merge_pythonpath("/boot").split(os.pathsep)
    assert merged == ["/boot", "/a", "/b"]
    monkeypatch.setenv("PYTHONPATH", "")
    assert _merge_pythonpath("/boot") == "/boot"
    monkeypatch.delenv("PYTHONPATH")
    assert _merge_pythonpath("/boot") == "/boot"


def test_failure_texts_classified_into_triage(tmp_path, monkeypatch):
    """junit 失败子用例文本 → triage 已知签名并集；未知失败不产生签名。"""
    from run_test.identifiers import Counts
    from run_test.runner import parse_junit_failure_texts

    xml = tmp_path / "report.xml"
    xml.write_text(
        '<testsuite name="s">'
        '<testcase classname="T" name="test_a">'
        '<failure message="AttributeError: module &#x27;torch._C&#x27; has no attribute &#x27;_cuda_setStream&#x27;">trace</failure>'
        "</testcase>"
        '<testcase classname="T" name="test_b">'
        '<error message="RuntimeError: No backend type associated with device type npu">trace</error>'
        "</testcase>"
        '<testcase classname="T" name="test_c">'
        '<failure message="AssertionError: some real bug">trace</failure>'
        "</testcase>"
        '<testcase classname="T" name="test_d"/>'
        "</testsuite>",
        encoding="utf-8",
    )
    texts = parse_junit_failure_texts(xml)
    assert len(texts) == 3

    def fake_run(cmd, *args, **kwargs):
        xml_arg = next(c for c in cmd if c.startswith("--junitxml="))
        Path(xml_arg.split("=", 1)[1]).write_text(xml.read_text(), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1, stdout="2 failed", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = make_runner(tmp_path)
    outcome = runner.run_function(TF_CUDA, "cpu")
    assert outcome.completed
    assert outcome.counts == Counts(1, 2, 1, 0)  # pass/fail/error/skip
    assert outcome.triage == (
        "C01-torchC-private-cuda-api",
        "C03-pg-no-npu-backend",
    )  # 未知失败（test_c）不产生签名


def test_missing_python_env_rejected(tmp_path):
    with pytest.raises(ConfigPendingError):
        PytestRunner(None, transfer_boot_dir=tmp_path)
    with pytest.raises(ConfigPendingError):
        PytestRunner("  ")


def test_nonzero_interrupt_with_junit_is_incomplete_and_artifact(tmp_path, monkeypatch):
    """A partial junit report with pytest exit 2 must never become a full result."""
    def fake_run(cmd, *args, **kwargs):
        xml_arg = next(c for c in cmd if c.startswith("--junitxml="))
        Path(xml_arg.split("=", 1)[1]).write_text(
            '<testsuite><testcase classname="T" name="test_a"/></testsuite>', encoding="utf-8"
        )
        return subprocess.CompletedProcess(cmd, 2, stdout="interrupted", stderr="trace on stderr")

    monkeypatch.setattr(subprocess, "run", fake_run)
    artifact_root = tmp_path / "artifacts"
    outcome = PytestRunner(sys.executable, artifact_root=artifact_root).run_function(
        TF_CUDA, "cpu"
    )
    assert not outcome.completed
    assert outcome.execution_status == "interrupted"
    assert outcome.counts is None
    artifact = Path(outcome.artifact_dir)
    assert (artifact / "junit.xml").is_file()
    assert "trace on stderr" in (artifact / "stderr.log").read_text()
    assert (artifact / "result.json").is_file()


# ---------------------------------------------------------------------------
# discover_runtime_nodes v2（2026-09-20：干净环境/通道过滤/事件插件/空结果抛错）
# ---------------------------------------------------------------------------


class _DiscoveryFake:
    """伪造 discovery 子进程：写事件 JSON，记录 env，返回给定 rc。"""

    def __init__(self, nodeids, rc=0, stdout="", write_event=True):
        self.nodeids = nodeids
        self.rc = rc
        self.stdout = stdout
        self.write_event = write_event
        self.envs: list[dict] = []
        self.cmds: list[list] = []

    def __call__(self, cmd, *args, **kwargs):
        self.cmds.append(cmd)
        self.envs.append(kwargs.get("env"))
        output = "\n".join(self.nodeids) + "\n" + self.stdout
        return subprocess.CompletedProcess(cmd, self.rc, stdout=output, stderr="")


def _disc_source(tmp_path):
    root = tmp_path / "source"
    (root / "test").mkdir(parents=True, exist_ok=True)
    return root


def test_discovery_cpu_channel_clean_env_and_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/ascend:")  # 尾冒号：必须被剥掉
    monkeypatch.delenv("PYTORCH_TESTING_DEVICE_ONLY_FOR", raising=False)
    fake = _DiscoveryFake(["test_foo.py::TestFoo::test_x_cpu"])
    monkeypatch.setattr(subprocess, "run", fake)
    boot = tmp_path / "boot"; boot.mkdir()
    runner = make_runner(tmp_path, boot_dir=boot)
    nodes = runner.discover_runtime_nodes(
        "test/test_foo.py::TestFoo::test_x", "cpu", source_root=_disc_source(tmp_path)
    )
    assert nodes == ["test/test_foo.py::TestFoo::test_x_cpu"]
    env = fake.envs[0]
    # 干净环境：绝不继承裸 env=None；尾冒号空项被剥；cpu 通道过滤 = cpu。
    assert env is not None
    assert "" not in env["PYTHONPATH"].split(os.pathsep)
    assert env["PYTORCH_TESTING_DEVICE_ONLY_FOR"] == "cpu"
    # cpu 通道不注入 boot（用户 2026-09-17 决策），boot 不在路径上。
    assert str(boot) not in env["PYTHONPATH"].split(os.pathsep)


def test_discovery_channel_only_for_matrix(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    for dev, only_for in (("cpu", "cpu"), ("accelerator", "npu")):
        fake = _DiscoveryFake(["test_foo.py::TestFoo::test_x_" + ("cuda" if dev == "cuda" else "cpu" if dev == "cpu" else "npu")])
        monkeypatch.setattr(subprocess, "run", fake)
        runner = make_runner(tmp_path, boot_dir=None)
        runner.discover_runtime_nodes(
            f"test/test_foo.py::TestFoo::test_x", dev, source_root=_disc_source(tmp_path)
        )
        assert fake.envs[0]["PYTORCH_TESTING_DEVICE_ONLY_FOR"] == only_for, dev


def test_discovery_cuda_channel_injects_boot(tmp_path, monkeypatch):
    from run_test.config import ConfigError
    fake = _DiscoveryFake([])
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(ConfigError, match="只允许 cpu / accelerator"):
        make_runner(tmp_path).discover_runtime_nodes(TF_CUDA, "cuda", source_root=tmp_path)
    assert not fake.cmds


def test_discovery_rejects_prefix_collisions(tmp_path, monkeypatch):
    """P1-2 反例回归锁：TestFooBar / test_sub_scalar 不得混入 TestFoo.test_sub。"""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    fake = _DiscoveryFake([
        "test_foo.py::TestFooBar::test_sub_cpu",       # 同前缀类，非变体
        "test_foo.py::TestFoo::test_sub_scalar",       # 同前缀方法，非变体
        "test_foo.py::TestFoo::test_sub_cpu_float32",  # 真变体：必须命中
    ])
    monkeypatch.setattr(subprocess, "run", fake)
    runner = make_runner(tmp_path, boot_dir=None)
    nodes = runner.discover_runtime_nodes(
        "test/test_foo.py::TestFoo::test_sub", "cpu", source_root=_disc_source(tmp_path)
    )
    assert nodes == ["test/test_foo.py::TestFoo::test_sub_cpu_float32"]


def test_discovery_empty_match_raises_for_diagnostics(tmp_path, monkeypatch):
    """收集到节点但家族无一命中：必须抛错（executor 落 discover-* 诊断），
    不再静默回退烧一次注定 not found 的运行。"""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    fake = _DiscoveryFake(["test_foo.py::TestOther::test_y_cpu"])
    monkeypatch.setattr(subprocess, "run", fake)
    runner = make_runner(tmp_path, boot_dir=None)
    with pytest.raises(RuntimeError, match="found no node"):
        runner.discover_runtime_nodes(
            "test/test_foo.py::TestFoo::test_x", "cpu", source_root=_disc_source(tmp_path)
        )


def test_discovery_stdout_fallback_when_event_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    fake = _DiscoveryFake(
        [], rc=0, stdout="test_foo.py::TestFoo::test_x_cpu\n", write_event=False
    )
    monkeypatch.setattr(subprocess, "run", fake)
    runner = make_runner(tmp_path, boot_dir=None)
    nodes = runner.discover_runtime_nodes(
        "test/test_foo.py::TestFoo::test_x", "cpu", source_root=_disc_source(tmp_path)
    )
    assert nodes == ["test/test_foo.py::TestFoo::test_x_cpu"]


def test_discovery_bad_rc_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    fake = _DiscoveryFake([], rc=2)
    monkeypatch.setattr(subprocess, "run", fake)
    runner = make_runner(tmp_path, boot_dir=None)
    with pytest.raises(RuntimeError, match="exited 2"):
        runner.discover_runtime_nodes(
            "test/test_foo.py::TestFoo::test_x", "cpu", source_root=_disc_source(tmp_path)
        )


def test_discovery_accepts_rootdir_relative_nodeids(tmp_path, monkeypatch):
    """收集输出的 nodeid 以 pytest rootdir 为基准（抽取树缺 pyproject.toml
    时落到 INFRA 根）：文件段 data/…/test/<rel_file> 必须命中——43-PR 实跑
    60 例 "not found" 的真正根因（2026-09-20 实证修复）。"""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    fake = _DiscoveryFake([
        "data/source_cache/abc123/pytorch/test/test_foo.py::TestFoo::test_x_cpu",
        "data/source_cache/abc123/pytorch/test/test_foo.py::TestFooBar::test_y_cpu",
    ])
    monkeypatch.setattr(subprocess, "run", fake)
    runner = make_runner(tmp_path, boot_dir=None)
    nodes = runner.discover_runtime_nodes(
        "test/test_foo.py::TestFoo::test_x", "cpu", source_root=_disc_source(tmp_path)
    )
    # rootdir 相对文件段被还原成 tf 同源的标准形态。
    assert nodes == ["test/test_foo.py::TestFoo::test_x_cpu"]
