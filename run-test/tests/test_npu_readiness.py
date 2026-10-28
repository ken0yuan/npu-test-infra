import json
import os
import subprocess
from pathlib import Path

import pytest

from run_test.npu_readiness import NpuReadiness, RESULT_PREFIX, ProbeResult
from run_test.test_scheduler import CardLease


def result(cmd, *, ok=True, checked=None, failed=None, error="", rc=None):
    checked = json.loads(cmd[3]) if checked is None else checked
    return subprocess.CompletedProcess(cmd, (0 if ok else 73) if rc is None else rc,
        RESULT_PREFIX + json.dumps(dict(ok=ok, checked=checked, failed_logical_device=failed, error=error)), error)


def test_probe_uses_visible_logical_indices_and_inherits_both_leases(tmp_path):
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs)); return result(cmd)
    probe = NpuReadiness("test-python", tmp_path, executor_fds=(11,), run=run)
    lease = CardLease((4, 7), fds=(21, 22))
    assert probe.ensure(lease, {"ASCEND_RT_VISIBLE_DEVICES": "4,7"}).ok
    cmd, kwargs = calls[0]
    assert json.loads(cmd[3]) == [0, 1] and cmd[4] == "2"
    assert kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4,7"
    if os.name != "nt":
        assert set(kwargs["pass_fds"]) == {11, 21, 22}
    # The readiness cache uses physical identities, not their current index.
    assert probe.ensure(CardLease((7,), fds=(23,)), {"ASCEND_RT_VISIBLE_DEVICES": "7"}).ok
    assert len(calls) == 1


def test_known_initialization_failure_retries_in_fresh_process_and_keeps_both_attempts(tmp_path):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return result(cmd, ok=False, checked=[], failed=0, error="TsdOpen failed 507033") if len(calls) == 1 else result(cmd)
    probe = NpuReadiness("python", tmp_path, run=run, retry_delay=0)
    outcome = probe.ensure(CardLease((6,), fds=(12,)), {"ASCEND_RT_VISIBLE_DEVICES": "6"})
    assert outcome.ok and len(calls) == 2
    receipts = [json.loads(p.read_text()) for p in tmp_path.glob("*/result.json")]
    assert sorted(p["attempt"] for p in receipts) == [1, 2]
    assert sum(p["completed"] for p in receipts) == 1


def test_failure_maps_only_the_identified_card_and_preserves_successful_cards(tmp_path):
    def run(cmd, **kwargs):
        return result(cmd, ok=False, checked=[0], failed=1, error="HDC recv failed 507033")
    probe = NpuReadiness("python", tmp_path, run=run, retry_delay=0)
    failure = probe.ensure(CardLease((2, 4, 6), fds=(10, 11, 12)), {"ASCEND_RT_VISIBLE_DEVICES": "2,4,6"})
    assert not failure.ok and failure.failed_cards == (4,) and failure.execution_status == "env_init_error"
    assert probe.ensure(CardLease((2,), fds=(10,)), {"ASCEND_RT_VISIBLE_DEVICES": "2"}).ok
    assert len(list(tmp_path.glob("*/result.json"))) == 2


def test_unclassified_operation_failure_is_not_retried_or_called_an_init_failure(tmp_path):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd); return result(cmd, ok=False, checked=[], failed=0, error="matmul output mismatch")
    failure = NpuReadiness("python", tmp_path, run=run).ensure(
        CardLease((4,), fds=(12,)), {"ASCEND_RT_VISIBLE_DEVICES": "4"})
    assert len(calls) == 1 and failure.execution_status == "npu_probe_error"


@pytest.mark.parametrize("out", ["", "ok", RESULT_PREFIX+'{"ok":true,"checked":[]}'])
def test_missing_or_incomplete_completion_receipt_never_admits_work(tmp_path, out):
    probe = NpuReadiness("python", tmp_path, run=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, out, ""))
    assert not probe.ensure(CardLease((4,), fds=(12,)), {"ASCEND_RT_VISIBLE_DEVICES": "4"}).ok


def test_no_hardware_probe_without_matching_nonempty_lease(tmp_path):
    def forbidden(*args, **kwargs):pytest.fail("No child should be started")
    probe = NpuReadiness("python", tmp_path, run=forbidden)
    with pytest.raises(ValueError):probe.ensure(CardLease(()), {})
    with pytest.raises(ValueError):probe.ensure(CardLease((4,)), {"ASCEND_RT_VISIBLE_DEVICES": "0"})


def test_native_abort_retains_last_entered_device_without_swallowing_signal(tmp_path):
    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, -6, "NPU_READINESS_DEVICE:1\n", "fatal native error")
    failure = NpuReadiness("python", tmp_path, run=run).ensure(
        CardLease((2, 6), fds=(11, 12)), {"ASCEND_RT_VISIBLE_DEVICES": "2,6"})
    assert failure.failed_cards == (6,) and not failure.ok
    assert json.loads(next(tmp_path.glob("*/result.json")).read_text())["returncode"] == -6
