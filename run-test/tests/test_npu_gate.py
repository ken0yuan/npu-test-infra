"""等卡门禁（用户 2026-10-26 要求）：accelerator 任务执行前确认空闲 NPU 卡，
够才运行，不够轮询等待。探测不 shell out，npu-smi 输出用 2026-10-26 在
8×910B4 实证的忙/闲/健康样本固化。

探针契约（与 run_test.npu_cards.wait_free_cards 一致）：内部自行轮询等待，
一次调用返回最终结果 (空闲卡, 原因)；超时/不可满足返回 ([], 原因)。
本轮为串行等卡门禁：不钉卡、不做设备租约（遗留项，见修复日志）。
"""

import itertools

import pytest

from run_test import task_queue as q
from run_test.executor import Executor
from run_test.identifiers import Counts
from run_test.npu_cards import card_free, npu_ids

from conftest import COMMIT_A, COMMIT_NPU, FakeDownloader, TF_A, TF_B

WHEELS_URL = "obs://pytorch-ci-cache/7day-cache/wheels-a1b2c3d4e5.tar.gz"
SOURCE_URL = "obs://pytorch-ci-cache/7day-cache/pytorch-source-a1b2c3d4e5.tar.gz"
_version_counter = itertools.count(1)

# npu-smi info -m 样本（Ascend 芯片行 + Mcu 辅助行）。
INFO_M = """
NPU ID        Chip ID    Chip Logic ID    Chip Name
0             0          0                Mcu
0             1          0                Ascend 910B4
1             0          1                Mcu
1             1          1                Ascend 910B4
"""

# proc-mem / health 两种状态输出（2026-10-26 实证）。
PROC_FREE = "\tNo process in device.\n\tChip ID                        : 0\n"
PROC_BUSY = "\tProcess id:42450   Process name:python   Process memory(MB):342\n"
HEALTH_OK = "\tHealth                         : OK\n"
HEALTH_BAD = "\tHealth                         : Warning\n"


@pytest.fixture
def fake_pip(monkeypatch):
    """模拟 pip 卸载/安装与 torch 版本核验（同 test_executor 口径）。"""
    import subprocess

    def fake_run(cmd, *args, **kwargs):
        joined = " ".join(str(c) for c in cmd)
        stdout = ""
        if "import torch" in joined:
            stdout = COMMIT_A
        elif "import sys" in joined:
            stdout = "310"
        elif "uninstall" in joined:
            stdout = "Skipping torch_npu as it is not installed."
        elif "install" in joined:
            stdout = "Successfully installed torch torch_npu"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_npu_ids_parses_ascend_rows_only(monkeypatch):
    import run_test.npu_cards as m

    monkeypatch.setattr(m, "_run", lambda cmd, **kw: INFO_M)
    assert npu_ids() == [0, 1]  # Mcu 辅助行不计


def test_card_free_needs_no_process_and_health_ok(monkeypatch):
    import run_test.npu_cards as m

    def smi(cmd, **kw):
        if "proc-mem" in cmd:
            return PROC_FREE
        if "health" in cmd:
            return HEALTH_OK
        return INFO_M

    monkeypatch.setattr(m, "_run", smi)
    assert card_free(0) is True

    def busy(cmd, **kw):
        return PROC_BUSY if "proc-mem" in cmd else HEALTH_OK

    monkeypatch.setattr(m, "_run", busy)
    assert card_free(0) is False

    def unhealthy(cmd, **kw):
        return PROC_FREE if "proc-mem" in cmd else HEALTH_BAD

    monkeypatch.setattr(m, "_run", unhealthy)
    assert card_free(0) is False  # 健康异常不算空闲

    def dead(cmd, **kw):
        return None  # 查询失败按占用处理

    monkeypatch.setattr(m, "_run", dead)
    assert card_free(0) is False


def test_wait_ignores_probe_failures_until_free(monkeypatch):
    import run_test.npu_cards as m

    state = {"n": 0}

    def fake_free(smi=None):
        state["n"] += 1
        return [4, 5, 6, 7] if state["n"] >= 3 else []

    monkeypatch.setattr(m, "free_npu_cards", fake_free)
    monkeypatch.setattr(m, "npu_ids", lambda smi=None: list(range(8)))
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    cards, why = m.wait_free_cards(4, log=lambda *_: None)
    assert cards == [4, 5, 6, 7] and why == ""


def test_wait_timeout_and_over_capacity(monkeypatch):
    import run_test.npu_cards as m

    monkeypatch.setattr(m, "free_npu_cards", lambda smi=None: [])
    monkeypatch.setattr(m, "npu_ids", lambda smi=None: list(range(8)))
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    cards, why = m.wait_free_cards(2, timeout=0.05, interval=1, log=lambda *_: None)
    assert cards == [] and "超时" in why

    # 可见卡总数不足：立即诊断，不进入等待。
    monkeypatch.setattr(m, "npu_ids", lambda smi=None: list(range(4)))
    cards, why = m.wait_free_cards(8, timeout=60, interval=1, log=lambda *_: None)
    assert cards == [] and "仅可见 4" in why and "等待不会使其增加" in why


def _verified_task(queue, *, pr=1, manifest=None):
    ver = f"gate-v{next(_version_counter)}"
    task, _ = queue.register_candidate(WHEELS_URL, SOURCE_URL,
                                       obs_wheels_version=ver, obs_source_version=ver)
    queue.register_identity_rule("r", "规则", "测试 fixture")
    queue.verify_identity(task.task_id, "r", {
        "pr_number": pr, "pytorch_commit": COMMIT_A,
        "torch_npu_commit": COMMIT_NPU, "binding": "x"})
    queue.prefilter_outcome(task.task_id, True)
    return queue.submit_manifest(task.task_id, manifest or [
        {"test_function": TF_A, "device": "cpu"},
        {"test_function": TF_B, "device": "accelerator"},
    ])


class TestExecutorGate:
    def _executor(self, config, queue, probe, runner, **cfg):
        return Executor({**config, "executor": {**(config.get("executor") or {}), **cfg}},
                        queue_store=queue, downloader=FakeDownloader(),
                        runner=runner, npu_probe=probe)

    def test_gate_waits_then_runs(self, config, queue, paths, fake_pip):
        """卡不足时门禁在探针内部等待，释放后进入 testing。"""
        task = _verified_task(queue)
        probe = FakeProbe(rounds_busy=1, free_cards=[2, 3, 6, 7])
        runner = FakeRunner(counts_map={
            (TF_A, "cpu"): Counts(1, 0, 0, 0),
            (TF_B, "accelerator"): Counts(1, 0, 0, 0),
        })
        outcome = self._executor(config, queue, probe, runner,
                                 wait_free_npu_cards=2,
                                 wait_free_npu_interval_seconds=1).run_once()
        assert outcome.outcome == "done", f"detail={outcome.detail!r}"
        assert probe.polls == 2  # 第一轮无卡，第二轮放行
        # 本轮不钉卡：运行环境不带 ASCEND_RT_VISIBLE_DEVICES。
        assert all(not (e or {}).get("ASCEND_RT_VISIBLE_DEVICES")
                   for e in runner.extra_envs)

    def test_gate_timeout_blocks(self, config, queue, fake_pip):
        task = _verified_task(queue)
        probe = FakeProbe(rounds_busy=None, free_cards=[], timeout_reason="空闲卡一直不足")
        executor = self._executor(config, queue, probe, FakeRunner(),
                                  wait_free_npu_cards=1,
                                  wait_free_npu_timeout_seconds=5)
        outcome = executor.run_once()
        assert outcome.outcome == "blocked"
        t = queue.get_task(task.task_id)
        assert t.state == q.STATE_BLOCKED
        assert "空闲卡" in (t.error_reason or "")

    def test_gate_empty_reason_gets_fallback(self, config, queue, fake_pip):
        """探针返回空原因时不得以空原因 mark_blocked（防御）。"""
        task = _verified_task(queue)
        probe = FakeProbe(rounds_busy=None, free_cards=[], timeout_reason="")
        executor = self._executor(config, queue, probe, FakeRunner(),
                                  wait_free_npu_cards=1,
                                  wait_free_npu_timeout_seconds=5)
        outcome = executor.run_once()
        assert outcome.outcome == "blocked"
        assert (queue.get_task(task.task_id).error_reason or "") != ""

    def test_cpu_only_task_skips_gate(self, config, queue, fake_pip):
        task = _verified_task(queue, pr=2, manifest=[{"test_function": TF_A, "device": "cpu"}])

        def probe(*a):
            raise AssertionError("纯 cpu 任务不应触发等卡探测")

        executor = self._executor(config, queue, probe, FakeRunner(),
                                  wait_free_npu_cards=1)
        assert executor.run_once().outcome == "done"

    def test_gate_disabled_without_config(self, config, queue, fake_pip):
        """未配置 wait_free_npu_cards 不启用门禁（不猜默认值约定）。"""
        _verified_task(queue)
        calls = {"n": 0}

        def probe(*a):
            calls["n"] += 1
            return list(range(8)), ""

        runner = FakeRunner()
        outcome = self._executor(config, queue, probe, runner).run_once()
        assert outcome.outcome == "done", f"detail={outcome.detail!r}"
        assert calls["n"] == 0


class FakeProbe:
    """符合探针契约的假探针：内部轮询 rounds_busy 轮无卡后放行；
    rounds_busy=None 表示永不放行，按 timeout_reason 超时返回。"""

    def __init__(self, rounds_busy, free_cards, timeout_reason=""):
        self.rounds_busy = rounds_busy
        self.free_cards = free_cards
        self.timeout_reason = timeout_reason
        self.polls = 0

    def __call__(self, need, timeout, interval):
        while True:
            self.polls += 1
            if self.rounds_busy is None:
                return [], self.timeout_reason
            if self.polls > self.rounds_busy:
                return self.free_cards, ""


# FakeRunner 同 test_executor（记录每次调用的 extra_env，供钉卡断言用）。
class FakeRunner:
    def __init__(self, counts_map=None, incomplete=None):
        self.counts_map = counts_map or {}
        self.incomplete = incomplete or set()
        self.source_roots: list = []
        self.extra_envs: list = []

    def run_function(self, tf, dev, *, source_root=None, extra_env=None):
        from run_test.runner import FunctionRunOutcome

        self.source_roots.append(source_root)
        self.extra_envs.append(extra_env)
        if (tf, dev) in self.incomplete:
            return FunctionRunOutcome.incomplete(tf, dev, "模拟：未收集到测试")
        return FunctionRunOutcome(
            tf, dev, True, self.counts_map.get((tf, dev), Counts(1, 0, 0, 0)),
            f"fake log: {tf} on {dev}", 0,
            coverage_status="full", instance_ids=(tf,),
            environment_fingerprint={"channel": dev, "boot": None},
        )
