"""持久化队列：状态机、去重、身份核实门禁、串行锁、恢复（队列文档第 5 节）。"""

import itertools

import pytest

from run_test import task_queue as q
from run_test.identifiers import Counts

from conftest import COMMIT_A, COMMIT_NPU, TF_A, TF_B

WHEELS_URL = "obs://pytorch-ci-cache/7day-cache/wheels-aaaaaaaaaa.tar.gz"
SOURCE_URL = "obs://pytorch-ci-cache/7day-cache/pytorch-source-aaaaaaaaaa.tar.gz"

_version_counter = itertools.count(1)


def register_verified_task(
    queue: q.QueueStore, *, pr=123456, included=True, submit=True
) -> q.Task:
    """走完 门禁→过滤（→清单） 的合法路径。

    每次调用使用新的对象版本，避免触发“同一远端对象版本不重复排队”的去重。
    submit=False 时停在 pending_analysis（用于测试清单校验本身）。
    """
    ver = f"v{next(_version_counter)}"
    task, _ = queue.register_candidate(
        WHEELS_URL, SOURCE_URL, obs_wheels_version=ver, obs_source_version=ver
    )
    queue.register_identity_rule("rule-1", "规则原文（测试桩）", "测试 fixture")
    queue.verify_identity(
        task.task_id, "rule-1",
        {
            "pr_number": pr,
            "pytorch_commit": COMMIT_A,
            "torch_npu_commit": COMMIT_NPU,
            "binding": "测试绑定说明",
        },
    )
    queue.prefilter_outcome(task.task_id, included)
    if not included or not submit:
        return queue.get_task(task.task_id)
    queue.submit_manifest(
        task.task_id,
        [{"test_function": TF_A, "device": "cpu"}, {"test_function": TF_B, "device": "accelerator"}],
    )
    return queue.get_task(task.task_id)


class TestCandidateRegistration:
    def test_register_creates_identity_pending(self, queue):
        task, created = queue.register_candidate(WHEELS_URL, SOURCE_URL)
        assert created is True
        assert task.state == q.STATE_IDENTITY_PENDING
        assert task.pr_number is None  # 绑定信息在核实前不存在
        assert task.error_reason is None

    def test_same_object_version_not_requeued(self, queue):
        """同一远端对象版本反复监测不重复排队（验收 9-3）。"""
        t1, c1 = queue.register_candidate(WHEELS_URL, SOURCE_URL, obs_wheels_version="v1")
        t2, c2 = queue.register_candidate(WHEELS_URL, SOURCE_URL, obs_wheels_version="v1")
        assert (c1, c2) == (True, False)
        assert t1.task_id == t2.task_id
        assert len(queue.list_tasks()) == 1
    def test_object_update_creates_new_candidate(self, queue):
        """同名对象更新不能当原对象使用，必须重新核实绑定关系。"""
        _, _ = queue.register_candidate(WHEELS_URL, SOURCE_URL, obs_wheels_version="v1")
        t2, c2 = queue.register_candidate(WHEELS_URL, SOURCE_URL, obs_wheels_version="v2")
        assert c2 is True
        assert len(queue.list_tasks()) == 2
        assert t2.state == q.STATE_IDENTITY_PENDING  # 新对象回到待核实


class TestIdentityGate:
    def test_verify_without_rule_refuses(self, queue):
        """规则未确认时强制拒绝，不放行、不默认 true（验收 9-1/9-2）。"""
        task, _ = queue.register_candidate(WHEELS_URL, SOURCE_URL)
        with pytest.raises(q.IdentityGateError, match="判定规则"):
            queue.verify_identity(
                task.task_id, "any-rule",
                {"pr_number": 1, "pytorch_commit": COMMIT_A,
                 "torch_npu_commit": COMMIT_NPU, "binding": "x"},
            )
        # 状态保持 identity_pending：候选未被放行。
        assert queue.get_task(task.task_id).state == q.STATE_IDENTITY_PENDING

    def test_verify_with_registered_rule_binds_pr_and_commits(self, queue):
        task, _ = queue.register_candidate(WHEELS_URL, SOURCE_URL)
        queue.register_identity_rule("r", "规则原文", "OBS 负责人 2026-09-17 确认")
        verified = queue.verify_identity(
            task.task_id, "r",
            {
                "pr_number": 42,
                "pytorch_commit": COMMIT_A,
                "torch_npu_commit": COMMIT_NPU,
                "binding": "按规则 X 核对",
            },
        )
        assert verified.state == q.STATE_PREFILTER_PENDING
        assert verified.pr_number == 42
        assert verified.pytorch_commit == COMMIT_A
        assert verified.torch_npu_commit == COMMIT_NPU
        assert verified.identity_evidence["rule_source"].startswith("OBS 负责人")

    def test_verify_rejects_incomplete_evidence(self, queue):
        task, _ = queue.register_candidate(WHEELS_URL, SOURCE_URL)
        queue.register_identity_rule("r", "规则", "来源")
        for bad in [
            {"pytorch_commit": COMMIT_A, "torch_npu_commit": COMMIT_NPU, "binding": "x"},
            {"pr_number": 1, "pytorch_commit": "main", "torch_npu_commit": COMMIT_NPU, "binding": "x"},
            {"pr_number": 1, "pytorch_commit": COMMIT_A, "torch_npu_commit": "latest", "binding": "x"},
            {"pr_number": 1, "pytorch_commit": COMMIT_A, "torch_npu_commit": COMMIT_NPU},
        ]:
            with pytest.raises(q.QueueError):
                queue.verify_identity(task.task_id, "r", bad)
        assert queue.get_task(task.task_id).state == q.STATE_IDENTITY_PENDING

    def test_only_identity_pending_can_be_verified(self, queue):
        task = register_verified_task(queue)
        with pytest.raises(q.QueueError, match="只有 identity_pending"):
            queue.verify_identity(
                task.task_id, "rule-1",
                {"pr_number": 1, "pytorch_commit": COMMIT_A,
                 "torch_npu_commit": COMMIT_NPU, "binding": "x"},
            )


class TestFilterAndManifest:
    def test_filtered_out_is_terminal(self, queue):
        task = register_verified_task(queue, included=False)
        assert task.state == q.STATE_FILTERED_OUT
        with pytest.raises(q.InvalidTransitionError):
            queue.transition(task.task_id, q.STATE_PENDING_ANALYSIS)

    def test_manifest_required_before_pending_test(self, queue):
        task, _ = queue.register_candidate(WHEELS_URL, SOURCE_URL)
        queue.register_identity_rule("r", "规则", "来源")
        queue.verify_identity(
            task.task_id, "r",
            {"pr_number": 1, "pytorch_commit": COMMIT_A,
             "torch_npu_commit": COMMIT_NPU, "binding": "x"},
        )
        queue.prefilter_outcome(task.task_id, True)
        # 非 pending_analysis 状态（如 identity_pending）不可提交清单。
        t_other, _ = queue.register_candidate(
            WHEELS_URL, SOURCE_URL, obs_wheels_version="man-v1"
        )
        with pytest.raises(q.QueueError, match="pending_analysis"):
            queue.submit_manifest(t_other.task_id, [])
        with pytest.raises(q.QueueError, match="非空"):
            queue.submit_manifest(task.task_id, [])
        task = queue.submit_manifest(
            task.task_id, [{"test_function": TF_A, "device": "cpu"}]
        )
        assert task.state == q.STATE_PENDING_TEST
        assert task.test_manifest == [{"test_function": TF_A, "device": "cpu"}]

    def test_manifest_validates_entries(self, queue):
        task = register_verified_task(queue, submit=False)
        assert task.state == q.STATE_PENDING_ANALYSIS
        with pytest.raises(q.QueueError):
            queue.submit_manifest(task.task_id, [{"test_function": TF_A}])
        with pytest.raises(q.QueueError):
            queue.submit_manifest(task.task_id, [{"test_function": TF_A, "device": "npu"}])
        with pytest.raises(q.QueueError, match="重复"):
            queue.submit_manifest(
                task.task_id,
                [{"test_function": TF_A, "device": "cpu"},
                 {"test_function": TF_A, "device": "cpu"}],
            )


class TestSerialExecution:
    def test_claim_next_takes_oldest_pending_test(self, queue):
        t1 = register_verified_task(queue, pr=1)
        t2 = register_verified_task(queue, pr=2)  # 第二个对象版本不同即可
        # 两个任务都 pending_test；claim 取最早入队的。
        claimed = queue.claim_next()
        assert claimed.task_id == t1.task_id
        assert claimed.state == q.STATE_PREPARING
        assert claimed.work_dir is not None

    def test_only_one_active_task_at_a_time(self, queue):
        """同一时间只允许一个任务处于 准备/测试/写库/清理（验收 9-10）。"""
        t1 = register_verified_task(queue, pr=1)
        register_verified_task(queue, pr=2)
        assert queue.claim_next() is not None
        # 第一个任务仍在 preparing：第二个不能开始。
        assert queue.claim_next() is None
        # 推进到终态后锁释放，才可取下一个。
        queue.transition(t1.task_id, q.STATE_TESTING)
        queue.transition(t1.task_id, q.STATE_WRITING)
        queue.transition(t1.task_id, q.STATE_CLEANING)
        queue.transition(t1.task_id, q.STATE_DONE)
        assert queue.claim_next() is not None

    def test_fifo_order(self, queue):
        t1 = register_verified_task(queue, pr=1)
        t2 = register_verified_task(queue, pr=2)
        first = queue.claim_next()
        assert first.task_id == t1.task_id
        queue.transition(first.task_id, q.STATE_TESTING)
        queue.transition(first.task_id, q.STATE_WRITING)
        queue.transition(first.task_id, q.STATE_CLEANING)
        queue.transition(first.task_id, q.STATE_DONE)
        second = queue.claim_next()
        assert second.task_id == t2.task_id


class TestStateMachine:
    def test_invalid_transition_rejected(self, queue):
        task = register_verified_task(queue)
        with pytest.raises(q.InvalidTransitionError):
            queue.transition(task.task_id, q.STATE_DONE)  # 跳过执行阶段

    def test_terminal_states_immutable(self, queue):
        task = register_verified_task(queue, included=False)
        assert task.state == q.STATE_FILTERED_OUT
        with pytest.raises(q.InvalidTransitionError, match="终态"):
            queue.transition(task.task_id, q.STATE_PENDING_ANALYSIS)

    def test_blocked_then_requeue(self, queue):
        task = register_verified_task(queue)
        task = queue.claim_next()
        task = queue.mark_blocked(task.task_id, "资源不齐")
        assert task.state == q.STATE_BLOCKED
        assert task.error_reason == "资源不齐"
        task = queue.requeue(task.task_id, q.STATE_PENDING_TEST, "人工确认资源已齐")
        assert task.state == q.STATE_PENDING_TEST

    def test_requeue_only_from_blocked(self, queue):
        task = register_verified_task(queue)  # pending_test
        with pytest.raises(q.InvalidTransitionError):
            queue.requeue(task.task_id, q.STATE_PENDING_TEST)


class TestRecovery:
    def test_interrupted_active_tasks_blocked(self, queue):
        """进程重启后发现未完成任务：先核实，不重复执行（验收 9 / 文档第 8 节）。"""
        task = register_verified_task(queue)
        queue.claim_next()  # → preparing
        queue.transition(task.task_id, q.STATE_TESTING)
        recovered = queue.recover_interrupted()
        assert [t.task_id for t in recovered] == [task.task_id]
        t = queue.get_task(task.task_id)
        assert t.state == q.STATE_BLOCKED
        assert "重启" in t.error_reason
        # 恢复后锁已清，队列可继续服务其他就绪任务。
        register_verified_task(queue, pr=7)
        assert queue.claim_next() is not None

    def test_interrupted_cleaning_stays_cleaning(self, queue):
        """清理未完成不开始下一任务。"""
        task = register_verified_task(queue)
        queue.claim_next()
        queue.transition(task.task_id, q.STATE_TESTING)
        queue.transition(task.task_id, q.STATE_WRITING)
        queue.transition(task.task_id, q.STATE_CLEANING)
        queue.recover_interrupted()
        assert queue.get_task(task.task_id).state == q.STATE_CLEANING
        # cleaning 任务占着执行位：新任务不能 claim。
        register_verified_task(queue, pr=8)
        assert queue.claim_next() is None


class TestTaskInfo:
    def test_task_carries_required_fields(self, queue):
        """任务至少携带的排队信息（文档第 5 节表格）。"""
        task, _ = queue.register_candidate(
            WHEELS_URL, SOURCE_URL,
            obs_wheels_version="etag-1", obs_remote_time="2026-09-17T00:00:00Z",
        )
        assert task.obs_wheels_url == WHEELS_URL
        assert task.obs_source_url == SOURCE_URL
        assert task.obs_remote_time == "2026-09-17T00:00:00Z"
        assert task.enqueued_at  # 入队时间
        assert task.history  # 状态迁移历史（中断恢复用）


class TestSplitBundleRegistration:
    """拆分包格式（2026-10-22 起）：独立 torch_npu 对象随任务登记。"""

    PT_URL = ("obs://pytorch-ci-cache/7day-cache/"
              "wheels-pytorch-bbbbbbbbbb-py311-torch-npu-builder-aarch64-"
              "torch-master-202610200103-e0dcc43e-aarch64-v2.tar.gz")
    NPU_URL = ("obs://pytorch-ci-cache/7day-cache/"
               "wheels-torchnpu-abcdef0123-bbbbbbbbbb-py311-torch-npu-builder-aarch64-"
               "torch-master-202610200103-e0dcc43e-aarch64-v2.tar.gz")

    def test_register_split_bundle_keeps_npu_object(self, queue):
        task, created = queue.register_candidate(
            self.PT_URL, SOURCE_URL,
            obs_wheels_version="v2", obs_source_version="s1",
            obs_wheels_npu_url=self.NPU_URL, obs_wheels_npu_version="v2",
        )
        assert created is True
        stored = queue.get_task(task.task_id)
        assert stored.obs_wheels_npu_url == self.NPU_URL
        assert stored.obs_wheels_npu_version == "v2"

    def test_split_bundle_npu_variant_change_is_new_candidate(self, queue):
        """同一主体包配不同 torch_npu 构建，是不同对象组合，必须重新排队。"""
        queue.register_candidate(self.PT_URL, SOURCE_URL, obs_wheels_npu_url=self.NPU_URL)
        other = self.NPU_URL.replace("e0dcc43e", "8a315d17")
        task2, created = queue.register_candidate(self.PT_URL, SOURCE_URL, obs_wheels_npu_url=other)
        assert created is True
        assert len(queue.list_tasks()) == 2

    def test_same_split_combination_dedup(self, queue):
        queue.register_candidate(
            self.PT_URL, SOURCE_URL,
            obs_wheels_npu_url=self.NPU_URL, obs_wheels_npu_version="v2",
        )
        _, created = queue.register_candidate(
            self.PT_URL, SOURCE_URL,
            obs_wheels_npu_url=self.NPU_URL, obs_wheels_npu_version="v2",
        )
        assert created is False

    def test_legacy_registration_unaffected(self, queue):
        """legacy 组合去重键不变（不因拆分包字段引入而重复排队）。"""
        queue.register_candidate(WHEELS_URL, SOURCE_URL, obs_wheels_version="v1")
        _, created = queue.register_candidate(WHEELS_URL, SOURCE_URL, obs_wheels_version="v1")
        assert created is False
