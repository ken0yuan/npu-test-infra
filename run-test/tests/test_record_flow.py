"""完整提交的六步刷新流程（存储文档第 6、7 节验收项）。

第 7 节示例场景：
  PR 123456 提交 A(cpu 3,0,0,1) / A(accelerator 2,1,0,1) / B(accelerator 0,0,1,0)
  → PR 库一行 counts(5,1,1,2)，基线三条，各自日志；
  → 已有 (A, cuda) 记录与日志不变（本次未运行）；
  → 重复提交同一份完整结果：仍一行、计数不翻倍；
  → 新提交只测 A：B 的基线保留原样，不清零、不删除。
"""

import json

import pytest

from run_test.baseline_store import (
    baseline_db_path,
    get_baseline,
    resolve_baseline_log,
)
from run_test.config import store_paths
from run_test.db import connect
from run_test.identifiers import Counts
from run_test.pr_store import get_pr_result, list_pr_results, pr_db_path
from run_test.record import (
    FunctionResult,
    RecordError,
    Submission,
    SubmissionValidationError,
    submit_complete_result,
)

from pathlib import Path

from conftest import (
    COMMIT_A,
    COMMIT_B,
    COMMIT_NPU,
    TF_A,
    TF_A_CLASS,
    TF_B,
    make_pr_log_dir,
    make_submission,
)


def _baseline(router, tf, dev):
    root = router.route(tf).root
    conn = connect(baseline_db_path(root))
    try:
        return get_baseline(conn, tf, dev)
    finally:
        conn.close()


def test_section7_example_full_flow(config, router, paths):
    sub = make_submission()  # A cpu / A acc / B acc
    report = submit_complete_result(config, sub, router)

    # PR 库：只有一行，计数为本次汇总 (5,1,1,2)，测试列表含三个组合。
    conn = connect(paths.pr_db)
    try:
        rows = list_pr_results(conn)
        assert len(rows) == 1
        pr = get_pr_result(conn, 123456)
    finally:
        conn.close()
    assert pr.counts == Counts(5, 1, 1, 2)
    assert pr.tests == [
        {"test_function": TF_A, "device": "cpu",
         "counts": Counts(3, 0, 0, 1).as_dict(), "instance_ids": [TF_A],
         "environment_fingerprint": {"channel": "cpu", "boot": None}},
        {"test_function": TF_A, "device": "accelerator",
         "counts": Counts(2, 1, 0, 1).as_dict(), "instance_ids": [TF_A],
         "environment_fingerprint": {"channel": "accelerator", "boot": None}},
        {"test_function": TF_B, "device": "accelerator",
         "counts": Counts(0, 0, 1, 0).as_dict(), "instance_ids": [TF_B],
         "environment_fingerprint": {"channel": "accelerator", "boot": None}},
    ]
    assert pr.pytorch_commit == COMMIT_A
    assert pr.torch_npu_environment["torch_npu_commit"] == COMMIT_NPU

    # 基线：三条记录各自保存对应日志（即使含 fail/error 也完整刷新）。
    for tf, dev, expect in [
        (TF_A, "cpu", Counts(3, 0, 0, 1)),
        (TF_A, "accelerator", Counts(2, 1, 0, 1)),
        (TF_B, "accelerator", Counts(0, 0, 1, 0)),
    ]:
        rec = _baseline(router, tf, dev)
        assert rec is not None and rec.counts == expect
        assert resolve_baseline_log(router.route(tf).root, rec).is_file()

    # 核对信息全部通过。
    assert all(report.checks.values()) or report.checks["pr_row_count"] == 1
    assert report.checks["pr_counts_match_sum"] is True
    assert report.checks["baselines_match_submission"] is True


def test_unrun_function_and_device_preserved(config, router):
    """未运行的函数和 device 的原记录不变（验收 8-10）。"""
    # 首次提交建立 B(cpu) 基线。
    submit_complete_result(
        config,
        make_submission(items=[(TF_B, "cpu", Counts(4, 0, 0, 0))]),
        router,
    )
    old = _baseline(router, TF_B, "cpu")
    assert old is not None

    # 第二次提交只含 A：B 的基线保留原值，PR 记录测试列表只含 A。
    submit_complete_result(
        config,
        make_submission(items=[(TF_A, "cpu", Counts(1, 0, 0, 0))]),
        router,
    )
    assert _baseline(router, TF_B, "cpu").counts == Counts(4, 0, 0, 0)
    conn = connect(paths_db(config))
    try:
        pr = get_pr_result(conn, 123456)
    finally:
        conn.close()
    assert pr.tests == [{
        "test_function": TF_A, "device": "cpu",
        "counts": Counts(1, 0, 0, 0).as_dict(),
        "instance_ids": [TF_A],
        "environment_fingerprint": {"channel": "cpu", "boot": None},
    }]


def paths_db(config):
    from run_test.config import store_paths

    return store_paths(config).pr_db


def test_existing_cuda_record_untouched(config, router):
    """示例：原有 (A, cuda) 记录及日志不变，因为本次没有运行它。"""
    # 先建立 A cuda 基线。
    submit_complete_result(
        config,
        make_submission(items=[(TF_A, "cuda", Counts(7, 0, 0, 0))]),
        router,
    )
    cuda_rec = _baseline(router, TF_A, "cuda")
    cuda_log = resolve_baseline_log(router.route(TF_A).root, cuda_rec)
    content_before = cuda_log.read_text()

    # 再提交不含 cuda 的完整结果。
    submit_complete_result(config, make_submission(), router)
    after = _baseline(router, TF_A, "cuda")
    assert after.counts == Counts(7, 0, 0, 0)
    assert after.pytorch_commit == cuda_rec.pytorch_commit
    assert cuda_log.read_text() == content_before


def test_resubmission_idempotent_no_doubling(config, router):
    """重复提交同一份完整结果：仍一行、计数与基线内容相同（验收 8-11）。"""
    sub = make_submission()
    r1 = submit_complete_result(config, sub, router)
    r2 = submit_complete_result(config, make_submission(), router)
    assert r1.totals == r2.totals == Counts(5, 1, 1, 2)

    conn = connect(paths_db(config))
    try:
        assert len(list_pr_results(conn)) == 1
    finally:
        conn.close()
    # 各函数计数与首次提交完全一致（未翻倍）。
    assert _baseline(router, TF_A, "cpu").counts == Counts(3, 0, 0, 1)
    assert _baseline(router, TF_A, "accelerator").counts == Counts(2, 1, 0, 1)
    assert _baseline(router, TF_B, "accelerator").counts == Counts(0, 0, 1, 0)


def test_new_submission_replaces_counts_not_accumulates(config, router):
    submit_complete_result(config, make_submission(), router)
    submit_complete_result(
        config,
        make_submission(items=[(TF_A, "cpu", Counts(0, 2, 0, 0))]),
        router,
    )
    conn = connect(paths_db(config))
    try:
        pr = get_pr_result(conn, 123456)
    finally:
        conn.close()
    assert pr.counts == Counts(0, 2, 0, 0)
    assert _baseline(router, TF_A, "cpu").counts == Counts(0, 2, 0, 0)


def test_log_update_replaces_old_log(config, router):
    """每次更新对应基线记录时，必须替换其对应日志（验收 8-8）。"""
    submit_complete_result(
        config, make_submission(items=[(TF_A, "cpu", Counts(1, 0, 0, 0))]), router
    )
    rec = _baseline(router, TF_A, "cpu")
    log = resolve_baseline_log(router.route(TF_A).root, rec)
    assert "counts={'pass': 1" in log.read_text()

    submit_complete_result(
        config,
        make_submission(
            items=[(TF_A, "cpu", Counts(0, 1, 0, 0))],
        ),
        router,
    )
    rec2 = _baseline(router, TF_A, "cpu")
    assert rec2.log_path == rec.log_path  # 同一 (tf, device) 日志名稳定
    assert "counts={'pass': 0" in log.read_text()


def test_pr_log_dir_managed_and_superseded(config, router, paths):
    """PR 日志按 PR 分目录；重复提交时旧日志可分辨、不混读。"""
    submit_complete_result(config, make_submission(), router)
    pr_dir = paths.pr_logs_root / "PR_123456"
    assert pr_dir.is_dir()
    assert (pr_dir / "summary.txt").is_file()

    # 新一次提交：旧内容移入 .superseded，当前目录只有新内容。
    submit_complete_result(
        config, make_submission(items=[(TF_A, "cpu", Counts(1, 0, 0, 0))]), router
    )
    assert (pr_dir / "summary.txt").is_file()
    superseded = paths.pr_logs_root / ".superseded" / "PR_123456"
    assert superseded.is_dir() and any(superseded.iterdir())
    # 当前目录与 superseded 不混。
    assert not (pr_dir / ".superseded").exists()


def test_incomplete_submissions_rejected(config, router):
    """执行中断/缺失结果/日志缺失不是完整提交，拒绝写入（第 6 节）。"""
    good = make_submission()

    # 缺 PR 日志目录。
    bad = Submission(
        pr_number=good.pr_number,
        pytorch_commit=good.pytorch_commit,
        torch_npu_environment=good.torch_npu_environment,
        function_results=good.function_results,
        pr_log_source_dir=None,
    )
    with pytest.raises(SubmissionValidationError, match="pr_log_source_dir"):
        submit_complete_result(config, bad, router)

    # 缺函数结果。
    bad2 = Submission(
        pr_number=good.pr_number,
        pytorch_commit=good.pytorch_commit,
        torch_npu_environment=good.torch_npu_environment,
        function_results=[],
        pr_log_source_dir=make_pr_log_dir(),
    )
    with pytest.raises(SubmissionValidationError, match="至少一个"):
        submit_complete_result(config, bad2, router)

    # 函数日志缺失。
    bad3 = Submission(
        pr_number=good.pr_number,
        pytorch_commit=good.pytorch_commit,
        torch_npu_environment=good.torch_npu_environment,
        function_results=[
            FunctionResult(TF_A, "cpu", Counts(1, 0, 0, 0), log_content=None)
        ],
        pr_log_source_dir=make_pr_log_dir(),
    )
    with pytest.raises(SubmissionValidationError, match="缺少日志"):
        submit_complete_result(config, bad3, router)

    # 日志源文件不存在。
    bad4 = Submission(
        pr_number=good.pr_number,
        pytorch_commit=good.pytorch_commit,
        torch_npu_environment=good.torch_npu_environment,
        function_results=[
            FunctionResult(
                TF_A, "cpu", Counts(1, 0, 0, 0), log_source_path="/nonexistent/a.log"
            )
        ],
        pr_log_source_dir=make_pr_log_dir(),
    )
    with pytest.raises(SubmissionValidationError, match="不存在"):
        submit_complete_result(config, bad4, router)


def test_duplicate_function_device_in_submission_rejected(config, router):
    dup = Submission(
        pr_number=123456,
        pytorch_commit=COMMIT_A,
        torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
        function_results=[
            FunctionResult(TF_A, "cpu", Counts(1, 0, 0, 0), log_content="x"),
            FunctionResult(TF_A, "cpu", Counts(1, 0, 0, 0), log_content="y"),
        ],
        pr_log_source_dir=make_pr_log_dir(),
    )
    with pytest.raises(SubmissionValidationError, match="出现多次"):
        submit_complete_result(config, dup, router)


def test_function_without_baseline_dir_rejected(config, router):
    sub = Submission(
        pr_number=123456,
        pytorch_commit=COMMIT_A,
        torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
        function_results=[
            FunctionResult(
                "test/unknown_area/test_x.py::test_f", "cpu",
                Counts(1, 0, 0, 0), log_content="x",
            )
        ],
        pr_log_source_dir=make_pr_log_dir(),
    )
    with pytest.raises(SubmissionValidationError, match="未归属"):
        submit_complete_result(config, sub, router)


def test_failed_log_preparation_leaves_old_state_intact(config, router):
    """新日志准备失败时不破坏旧基线日志、不写任何记录（第 6 节步骤 3）。"""
    submit_complete_result(
        config, make_submission(items=[(TF_A, "cpu", Counts(1, 0, 0, 0))]), router
    )
    old_rec = _baseline(router, TF_A, "cpu")
    old_log = resolve_baseline_log(router.route(TF_A).root, old_rec)
    old_content = old_log.read_text()

    # 一份两个函数、第二个日志源缺失的提交：整体拒绝。
    bad = Submission(
        pr_number=123456,
        pytorch_commit=COMMIT_A,
        torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
        function_results=[
            FunctionResult(TF_A, "cpu", Counts(2, 0, 0, 0), log_content="new log"),
            FunctionResult(
                TF_A, "accelerator", Counts(0, 0, 0, 1),
                log_source_path="/nonexistent/b.log",
            ),
        ],
        pr_log_source_dir=make_pr_log_dir(tag="bad"),
    )
    with pytest.raises(SubmissionValidationError):
        submit_complete_result(config, bad, router)

    # 旧日志未变、旧记录未变、未产生 tmp 残留。
    assert old_log.read_text() == old_content
    assert _baseline(router, TF_A, "cpu").counts == Counts(1, 0, 0, 0)
    assert _baseline(router, TF_A, "accelerator") is None
    logs_dir = router.route(TF_A).root / "logs"
    assert not list(logs_dir.glob("*.tmp.*"))


def test_class_variant_and_cross_file_same_name_logs(config, router):
    """不同文件或类的同名函数不共享日志（验收 8-9）。"""
    from conftest import TF_B_SAME_NAME

    submit_complete_result(
        config,
        make_submission(
            items=[
                (TF_A, "cpu", Counts(1, 0, 0, 0)),
                (TF_A_CLASS, "cpu", Counts(1, 0, 0, 0)),
                (TF_B, "cpu", Counts(1, 0, 0, 0)),
                (TF_B_SAME_NAME, "cpu", Counts(1, 0, 0, 0)),
            ]
        ),
        router,
    )
    roots = {tf: router.route(tf).root for tf in (TF_A, TF_A_CLASS, TF_B, TF_B_SAME_NAME)}
    paths = {
        resolve_baseline_log(roots[tf], _baseline(router, tf, "cpu"))
        for tf in roots
    }
    assert len(paths) == 4  # 四个不同日志文件


def test_counts_derived_from_function_results(config, router):
    """PR 四计数由函数结果汇总，不单独提供（第 7 节写入效果）。"""
    report = submit_complete_result(config, make_submission(), router)
    assert report.totals == Counts(5, 1, 1, 2)
    conn = connect(paths_db(config))
    try:
        pr = get_pr_result(conn, 123456)
    finally:
        conn.close()
    assert pr.counts == report.totals


# ---------------------------------------------------------------------------
# no_result：单函数未取得结果的显式无信号通道（不计数、不刷基线）
# ---------------------------------------------------------------------------


def _mixed_submission(**kwargs) -> Submission:
    """A(cpu) 取得结果 + A(accelerator) no_result 的混合提交。"""
    return Submission(
        pr_number=123456,
        pytorch_commit=COMMIT_A,
        torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
        function_results=[
            FunctionResult(TF_A, "cpu", Counts(3, 0, 0, 1), log_content="ok log",
                           coverage_status="full", instance_ids=(TF_A,),
                           environment_fingerprint={"channel": "cpu", "boot": None}),
            FunctionResult(
                TF_A, "accelerator", counts=None, status="no_result",
                reason="未收集到任何测试", log_content="no result log",
            ),
        ],
        pr_log_source_dir=make_pr_log_dir(tag="mixed"),
        **kwargs,
    )


def test_no_result_submission_flow(config, router, paths):
    """混合提交：ok 项正常入库，no_result 项不计数、不建基线、不写基线日志。"""
    report = submit_complete_result(config, _mixed_submission(), router)

    # 汇总只计 ok 项。
    assert report.totals == Counts(3, 0, 0, 1)
    conn = connect(paths_db(config))
    try:
        pr = get_pr_result(conn, 123456)
    finally:
        conn.close()
    assert pr.counts == Counts(3, 0, 0, 1)
    assert pr.tests == [
        {"test_function": TF_A, "device": "cpu",
         "counts": Counts(3, 0, 0, 1).as_dict(), "instance_ids": [TF_A],
         "environment_fingerprint": {"channel": "cpu", "boot": None}},
        {"test_function": TF_A, "device": "accelerator",
         "status": "no_result", "reason": "未收集到任何测试",
         "coverage_status": "unknown"},
    ]

    # 基线：cpu 建立，accelerator 未建立、无日志。
    assert _baseline(router, TF_A, "cpu") is not None
    assert _baseline(router, TF_A, "accelerator") is None
    logs_dir = router.route(TF_A).root / "logs"
    assert not list(logs_dir.glob("*accelerator*"))

    # 对比档案：no_result 条目如实记录。
    archive = json.loads(
        (paths.pr_logs_root / "PR_123456" / "baseline-diff.json").read_text()
    )
    nr = next(
        c for c in archive["comparisons"] if c["device"] == "accelerator"
    )
    assert nr["baseline_action"] == "no_result"
    assert nr["after"]["counts"] is None

    # diff 摘要标无信号；核对全过。
    nr_diff = next(d for d in report.diffs if d.device == "accelerator")
    assert "无信号" in nr_diff.summary()
    assert report.checks["baselines_match_submission"] is True


def test_no_result_preserves_existing_baseline(config, router):
    """已有基线的函数本次 no_result：旧记录与旧日志逐字段不变。"""
    submit_complete_result(
        config,
        make_submission(items=[(TF_A, "accelerator", Counts(9, 2, 0, 0))]),
        router,
    )
    old_rec = _baseline(router, TF_A, "accelerator")
    old_log = resolve_baseline_log(router.route(TF_A).root, old_rec)
    old_content = old_log.read_text()

    report = submit_complete_result(config, _mixed_submission(), router)

    after = _baseline(router, TF_A, "accelerator")
    assert after == old_rec  # 逐字段一致（计数、commit、环境、日志路径）
    assert old_log.read_text() == old_content  # 日志未被覆盖
    assert report.checks["baselines_match_submission"] is True


def test_no_result_validation_rules(config, router):
    """no_result 契约：counts 必须 None、reason 必填、全 no_result 拒绝。"""
    base = dict(
        pr_number=123456,
        pytorch_commit=COMMIT_A,
        torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
        pr_log_source_dir=make_pr_log_dir(tag="bad"),
    )
    # 声明 no_result 却带计数。
    with pytest.raises(SubmissionValidationError, match="无结果不得携带计数"):
        submit_complete_result(
            config,
            Submission(
                function_results=[
                    FunctionResult(TF_A, "cpu", Counts(1, 0, 0, 0),
                                   status="no_result", reason="x",
                                   log_content="log"),
                ],
                **base,
            ),
            router,
        )
    # no_result 缺 reason。
    with pytest.raises(SubmissionValidationError, match="reason"):
        submit_complete_result(
            config,
            Submission(
                function_results=[
                    FunctionResult(TF_A, "cpu", None, status="no_result",
                                   log_content="log"),
                ],
                **base,
            ),
            router,
        )
    # 全部 no_result：仍然提交 PR 证据，但不产生计数或基线。
    report = submit_complete_result(
        config,
        Submission(
            function_results=[
                FunctionResult(TF_A, "cpu", None, status="no_result",
                               reason="x", log_content="log"),
                FunctionResult(TF_B, "cpu", None, status="no_result",
                               reason="y", log_content="log"),
            ],
            **base,
        ),
        router,
    )
    assert report.totals == Counts(0, 0, 0, 0)
    assert all(d.new_counts is None for d in report.diffs)


# ---------------------------------------------------------------------------
# D：签名级基线对比——环境噪声 vs 回归候选
# ---------------------------------------------------------------------------

SIG_C01_TEXT = (
    "FAILED test_a [AttributeError]: module 'torch._C' has no attribute "
    "'_cuda_setStream'"
)


def _seed_baseline_with_signature(config, router, counts):
    """预置一条基线，其日志含 C01 签名（模拟基线期已知环境失败）。"""
    submit_complete_result(
        config,
        Submission(
            pr_number=999001,
            pytorch_commit=COMMIT_B,
            torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
            function_results=[
                FunctionResult(TF_A, "cpu", counts, log_content=SIG_C01_TEXT,
                               coverage_status="full", instance_ids=(TF_A,),
                               environment_fingerprint={"channel": "cpu", "boot": None}),
            ],
            pr_log_source_dir=make_pr_log_dir(tag="seed"),
        ),
        router,
    )


def _triaged_submission(counts, triage):
    return Submission(
        pr_number=123456,
        pytorch_commit=COMMIT_A,
        torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
        function_results=[
            FunctionResult(
                TF_A, "cpu", counts, log_content="pr run log", triage=triage,
                coverage_status="full", instance_ids=(TF_A,),
                environment_fingerprint={"channel": "cpu", "boot": None},
            ),
        ],
        pr_log_source_dir=make_pr_log_dir(tag="triaged"),
    )


def test_same_signature_same_counts_is_known_env_noise(config, router):
    """计数不变 + 签名无新增 = 基线已知环境失败，无新信号。"""
    _seed_baseline_with_signature(config, router, Counts(0, 3, 0, 0))
    report = submit_complete_result(
        config, _triaged_submission(Counts(0, 3, 0, 0), ["C01-torchC-private-cuda-api"]), router,
    )
    d = report.diffs[0]
    assert d.old_triage == ["C01-torchC-private-cuda-api"]
    assert "失败类别与基线相同" in d.summary()
    assert "无法判断是否同一失败" in d.summary()


def test_new_signature_flagged_even_with_same_counts(config, router):
    """计数不变但出现新签名：刷新 + 新增签名标注（潜在新问题）。"""
    _seed_baseline_with_signature(config, router, Counts(0, 3, 0, 0))
    report = submit_complete_result(
        config,
        _triaged_submission(
            Counts(0, 3, 0, 0),
            ["C01-torchC-private-cuda-api", "C03-pg-no-npu-backend"],
        ),
        router,
    )
    summary = report.diffs[0].summary()
    assert "基线刷新" in summary
    assert "新增签名: C03-pg-no-npu-backend" in summary


def test_counts_change_is_refresh_with_signature_note(config, router, monkeypatch):
    """计数变化：基线刷新 + 签名标注（回归候选，不判噪声）。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    FakeMirror.relations = {(COMMIT_B, COMMIT_A): True}  # 提交晚于基线，照常刷新
    _seed_baseline_with_signature(config, router, Counts(0, 3, 0, 0))
    report = submit_complete_result(
        config, _triaged_submission(Counts(1, 2, 0, 0), ["C01-torchC-private-cuda-api"]), router,
    )
    summary = report.diffs[0].summary()
    assert "基线刷新" in summary and "{'pass': 0, 'fail': 3" in summary
    assert "含已知环境签名: C01-torchC-private-cuda-api" in summary

    # PR 库 tests 项与对比档案都带签名。
    conn = connect(paths_db(config))
    try:
        pr = get_pr_result(conn, 123456)
    finally:
        conn.close()
    assert pr.tests[0]["triage"] == ["C01-torchC-private-cuda-api"]
    archive = json.loads(
        (store_paths(config).pr_logs_root / "PR_123456" / "baseline-diff.json")
        .read_text(encoding="utf-8")
    )
    cmp_entry = archive["comparisons"][0]
    assert cmp_entry["after"]["triage"] == ["C01-torchC-private-cuda-api"]
    assert cmp_entry["before"]["triage"] == ["C01-torchC-private-cuda-api"]


def test_partial_coverage_is_saved_without_overwriting_baseline(config, router):
    first = make_submission(items=[(TF_A, "cpu", Counts(4, 0, 0, 0))])
    submit_complete_result(config, first, router)
    old = _baseline(router, TF_A, "cpu")
    second = Submission(
        pr_number=123457,
        pytorch_commit=COMMIT_B,
        torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
        function_results=[FunctionResult(
            TF_A, "cpu", Counts(1, 0, 0, 0), log_content="partial",
            coverage_status="partial", instance_ids=("test_basic[0]",),
        )],
        pr_log_source_dir=make_pr_log_dir(tag="partial"),
    )
    report = submit_complete_result(config, second, router)
    assert _baseline(router, TF_A, "cpu") == old
    assert report.diffs[0].comparable is False
    assert "不可比" in report.diffs[0].summary()


def test_no_result_count_visible_in_pr_row_and_report(config, router, paths):
    """no_result 可见化（2026-09-20）：PR 行独立成列，报告独立计数。

    0 pass + 0 fail + no_result_count>0 = "没测到"而非"全过"——
    起因 PR 197081 全 no_result 记 done、计数全 0 无法辨别。
    """
    report = submit_complete_result(config, _mixed_submission(), router)
    assert report.no_result_count == 1
    conn = connect(paths_db(config))
    try:
        pr = get_pr_result(conn, 123456)
    finally:
        conn.close()
    assert pr.no_result_count == 1
    assert pr.counts == Counts(3, 0, 0, 1)


def test_all_no_result_submission_persists_with_zero_counts(config, router, paths):
    """全部 no_result：仍持久化（与 executor 行为对齐的文档定稿），
    四计数全零 + no_result_count=N，不拒绝、不刷基线。"""
    sub = Submission(
        pr_number=197081,
        pytorch_commit=COMMIT_A,
        torch_npu_environment={"location": "e", "torch_npu_commit": COMMIT_NPU},
        function_results=[
            FunctionResult(
                TF_A, dev, counts=None, status="no_result",
                reason="未收集到任何测试", log_content="no result log",
            )
            for dev in ("cpu", "accelerator")
        ],
        pr_log_source_dir=make_pr_log_dir(tag="allnr"),
    )
    report = submit_complete_result(config, sub, router)
    assert report.totals == Counts(0, 0, 0, 0)
    assert report.no_result_count == 2
    conn = connect(paths_db(config))
    try:
        pr = get_pr_result(conn, 197081)
    finally:
        conn.close()
    assert pr.counts == Counts(0, 0, 0, 0)
    assert pr.no_result_count == 2
    # 全 no_result：一条基线都不建。
    assert _baseline(router, TF_A, "cpu") is None
    assert _baseline(router, TF_A, "accelerator") is None


class FakeMirror:
    """按预设祖先关系应答的镜像替身；未登记关系一律 unknown（保守保留现有基线）。"""
    relations: dict = {}
    commit_times: dict = {}

    def __init__(self, config=None):
        pass

    def commit_timestamp(self, sha):
        return self.commit_times.get(sha)

    def is_ancestor(self, older, newer):
        return bool(self.relations.get((older, newer)))

    def merge_relation(self, a, b):
        if a == b:
            return 'equal'
        if self.relations.get((a, b)):
            return 'a_earlier'
        if self.relations.get((b, a)):
            return 'a_later'
        return 'unknown'


def test_merge_order_guard_skips_older_result(config, router, monkeypatch):
    FakeMirror.relations = {}  # 防类属性跨用例泄漏
    """合入序防护（用户 2026-10-26 要求）：基线已持有更晚合入 PR 的结果时，
    更早 PR 的提交不得把基线回退到旧值，且跳过决策留痕于 report.checks。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    later = make_submission(pr_number=2, pytorch_commit=COMMIT_B,
                            items=[(TF_A, "cpu", Counts(5, 0, 0, 0))])
    submit_complete_result(config, later, router)
    FakeMirror.relations = {(COMMIT_A, COMMIT_B): 'a_earlier'}  # A 合入早于 B
    earlier = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                              items=[(TF_A, "cpu", Counts(1, 1, 1, 1))])
    report = submit_complete_result(config, earlier, router)
    rec = _baseline(router, TF_A, "cpu")
    assert rec.pytorch_commit == COMMIT_B  # 基线未被回退
    assert report.checks["baseline_skipped_merge_order"] == [
        {"test_function": TF_A, "device": "cpu",
         "baseline_commit": COMMIT_B, "submission_commit": COMMIT_A,
         "relation": "a_earlier", "reason": "baseline_newer"}]


def test_merge_order_guard_unknown_ancestry_keeps_baseline(config, router, monkeypatch):
    """未知祖先（关系不可判）必须保守：基线行与日志均不动，跳过原因
    relation_unknown 留痕，不得静默覆盖较新的基线。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    first = make_submission(pr_number=1, pytorch_commit=COMMIT_B,
                            items=[(TF_A, "cpu", Counts(3, 0, 0, 0))])
    submit_complete_result(config, first, router)
    FakeMirror.relations = {}  # 提交 commit 与基线 commit 关系不可判
    unknown = make_submission(pr_number=9, pytorch_commit=COMMIT_A,
                              items=[(TF_A, "cpu", Counts(9, 9, 9, 9))])
    report = submit_complete_result(config, unknown, router)
    rec = _baseline(router, TF_A, "cpu")
    assert rec.pytorch_commit == COMMIT_B and rec.counts == Counts(3, 0, 0, 0)
    assert report.checks["baseline_skipped_merge_order"] == [
        {"test_function": TF_A, "device": "cpu",
         "baseline_commit": COMMIT_B, "submission_commit": COMMIT_A,
         "relation": "unknown", "reason": "relation_unknown"}]
    # T4：对比档案把该行记为预期跳过，而不是"已覆盖"。
    archive = json.loads((Path(config['data_root']) / report.diff_archive).read_text())
    row = next(c for c in archive['comparisons']
               if c['test_function'] == TF_A and c['device'] == 'cpu')
    assert row['baseline_action'] == 'merge_order_skipped'
    assert row['after']['reason'] == 'relation_unknown'


def test_merge_order_guard_allows_newer_result(config, router, monkeypatch):
    FakeMirror.relations = {}  # 防类属性跨用例泄漏
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    first = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                            items=[(TF_A, "cpu", Counts(1, 0, 0, 0))])
    submit_complete_result(config, first, router)
    FakeMirror.relations = {(COMMIT_A, COMMIT_B): 'a_later'}  # B 合入晚于 A
    second = make_submission(pr_number=2, pytorch_commit=COMMIT_B,
                             items=[(TF_A, "cpu", Counts(2, 0, 0, 0))])
    report = submit_complete_result(config, second, router)
    rec = _baseline(router, TF_A, "cpu")
    assert rec.pytorch_commit == COMMIT_B  # 更晚合入的结果照常刷新
    assert report.checks["baseline_skipped_merge_order"] == []


def test_same_commit_rerun_overrides_regardless_of_guard(config, router, monkeypatch):
    FakeMirror.relations = {}  # 防类属性跨用例泄漏
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    first = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                            items=[(TF_A, "cpu", Counts(1, 0, 0, 0))])
    submit_complete_result(config, first, router)
    FakeMirror.relations = {(COMMIT_A, COMMIT_A): True}
    rerun = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                            items=[(TF_A, "cpu", Counts(2, 2, 2, 2))])
    submit_complete_result(config, rerun, router)
    rec = _baseline(router, TF_A, "cpu")
    assert rec.pytorch_commit == COMMIT_A and rec.counts == Counts(2, 2, 2, 2)


def test_merge_order_guard_preserves_db_and_log_when_newer_first(config, router, paths, monkeypatch):
    FakeMirror.relations = {}  # 防类属性跨用例泄漏
    """用户要求的验证用例：先提交新结果、再提交旧结果——基线 DB 行与
    日志文件都必须保持新结果版本，不被旧结果回退。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    newer = make_submission(pr_number=2, pytorch_commit=COMMIT_B,
                            items=[(TF_A, "cpu", Counts(5, 0, 0, 0))])
    submit_complete_result(config, newer, router)
    rec_before = _baseline(router, TF_A, "cpu")
    log_path = resolve_baseline_log(router.route(TF_A).root, rec_before)
    log_before = log_path.read_text(encoding='utf-8')

    FakeMirror.relations = {(COMMIT_A, COMMIT_B): 'a_earlier'}
    older = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                            items=[(TF_A, "cpu", Counts(1, 1, 1, 1))])
    report = submit_complete_result(config, older, router)

    rec_after = _baseline(router, TF_A, "cpu")
    assert rec_after == rec_before  # DB 行保持新结果版本
    assert log_path.read_text(encoding='utf-8') == log_before  # 日志同样未被覆盖
    assert rec_after.pytorch_commit == COMMIT_B
    skipped = report.checks["baseline_skipped_merge_order"]
    assert skipped and skipped[0]["reason"] == "baseline_newer"
    # 旧 PR 结果本身照常保存到 PR 结果库（仅基线受保护）。
    conn = connect(paths.pr_db)
    try:
        old_saved = get_pr_result(conn, 1)
    finally:
        conn.close()
    assert old_saved is not None and old_saved.pytorch_commit == COMMIT_A


def test_merge_order_unknown_relation_preserves_baseline(config, router, paths, monkeypatch):
    FakeMirror.relations = {}  # 防类属性跨用例泄漏
    """关系未知（commit 不在镜像历史等）→ 保守保留现有基线并留痕。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    first = make_submission(pr_number=1, pytorch_commit=COMMIT_B,
                            items=[(TF_A, "cpu", Counts(5, 0, 0, 0))])
    submit_complete_result(config, first, router)
    rec_before = _baseline(router, TF_A, "cpu")
    log_path = resolve_baseline_log(router.route(TF_A).root, rec_before)
    log_before = log_path.read_text(encoding='utf-8')

    other = make_submission(pr_number=2, pytorch_commit=COMMIT_A,  # 未登记关系
                            items=[(TF_A, "cpu", Counts(1, 1, 1, 1))])
    report = submit_complete_result(config, other, router)

    assert _baseline(router, TF_A, "cpu") == rec_before
    assert log_path.read_text(encoding='utf-8') == log_before
    skipped = report.checks["baseline_skipped_merge_order"]
    assert skipped and skipped[0]["reason"] == "relation_unknown"


# ---------------------------------------------------------------------------
# merged_at 合入序方案（用户 2026-10-26 七条要求之测试矩阵）
# ---------------------------------------------------------------------------

def _verified_order(config, commits):
    import run_test.upstream as up
    snapshot = {'schema': up.ORDER_SNAPSHOT_SCHEMA, 'source': up.PYTORCH_UPSTREAM_URL,
                'tip': commits[0], 'checked_tip': commits[0], 'checked': len(commits),
                'order': commits, 'times': [1] * len(commits), 'inversions': []}
    snapshot['sha256'] = up._order_digest(snapshot)
    up.persist_order_snapshot(config, snapshot)


def test_merged_at_timezone_equivalence_is_ambiguous(config, router, monkeypatch):
    """时区写法不同但同刻（+08:00 与 Z）：不同 commit 视为歧义，保守保留。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)

    monkeypatch.setattr(up.PyTorchMirror, 'merge_relation', lambda *a: 'unknown')
    first = make_submission(pr_number=1, pytorch_commit=COMMIT_B,
                            merged_at="2026-10-26T12:00:00+08:00",
                            merged_at_provenance="upstream-main-commit-time",
                            items=[(TF_A, "cpu", Counts(3, 0, 0, 0))])
    submit_complete_result(config, first, router)
    same_instant = make_submission(pr_number=2, pytorch_commit=COMMIT_A,
                                   merged_at="2026-10-26T04:00:00Z",
                                   merged_at_provenance="upstream-main-commit-time",
                                   items=[(TF_A, "cpu", Counts(9, 0, 0, 0))])
    report = submit_complete_result(config, same_instant, router)
    rec = _baseline(router, TF_A, "cpu")
    assert rec.pytorch_commit == COMMIT_B and rec.counts == Counts(3, 0, 0, 0)
    assert report.checks["baseline_skipped_merge_order"][0]["reason"] == "relation_unknown"


def test_merged_at_newer_applies_without_git_query(config, router, monkeypatch):
    """双方精确 commit 在有效快照中：按位置比较，不触发 Git 查询。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    _verified_order(config, [COMMIT_B, COMMIT_A])

    def forbidden(self, a, b):
        raise AssertionError("时间戳路径不应做 Git 祖先查询")
    monkeypatch.setattr(up.PyTorchMirror, 'merge_relation', forbidden)
    first = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                            merged_at="2026-10-26T01:00:00Z",
                            merged_at_provenance="upstream-main-commit-time",
                            items=[(TF_A, "cpu", Counts(1, 0, 0, 0))])
    submit_complete_result(config, first, router)
    second = make_submission(pr_number=2, pytorch_commit=COMMIT_B,
                             merged_at="2026-10-26T02:00:00Z",
                             merged_at_provenance="upstream-main-commit-time",
                             items=[(TF_A, "cpu", Counts(2, 0, 0, 0))])
    report = submit_complete_result(config, second, router)
    assert _baseline(router, TF_A, "cpu").pytorch_commit == COMMIT_B
    assert report.checks["baseline_skipped_merge_order"] == []


def test_merged_at_invalid_falls_back_then_preserves(config, router, monkeypatch):
    """提交侧 merged_at 无效：回退 Git 祖先；关系不可判 → 保留基线并留痕。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    first = make_submission(pr_number=1, pytorch_commit=COMMIT_B,
                            merged_at="2026-10-26T02:00:00Z",
                            items=[(TF_A, "cpu", Counts(3, 0, 0, 0))])
    submit_complete_result(config, first, router)
    FakeMirror.relations = {}  # Git 关系不可判
    bad = make_submission(pr_number=2, pytorch_commit=COMMIT_A,
                          merged_at="not-a-timestamp",
                          items=[(TF_A, "cpu", Counts(9, 9, 9, 9))])
    report = submit_complete_result(config, bad, router)
    rec = _baseline(router, TF_A, "cpu")
    assert rec.pytorch_commit == COMMIT_B and rec.counts == Counts(3, 0, 0, 0)
    assert report.checks["baseline_skipped_merge_order"][0]["reason"] == "relation_unknown"


def test_merged_at_older_submission_skipped(config, router, monkeypatch):
    """提交在有效快照中位于基线之前：行与日志都不回退。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    _verified_order(config, [COMMIT_B, COMMIT_A])
    monkeypatch.setattr(up.PyTorchMirror, 'merge_relation',
                        lambda self, a, b: (_ for _ in ()).throw(
                            AssertionError("时间戳可判时不查 Git")))
    first = make_submission(pr_number=2, pytorch_commit=COMMIT_B,
                            merged_at="2026-10-26T05:00:00Z",
                            merged_at_provenance="upstream-main-commit-time",
                            items=[(TF_A, "cpu", Counts(5, 0, 0, 0))])
    submit_complete_result(config, first, router)
    earlier = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                              merged_at="2026-10-26T01:00:00Z",
                              merged_at_provenance="upstream-main-commit-time",
                              items=[(TF_A, "cpu", Counts(1, 1, 1, 1))])
    report = submit_complete_result(config, earlier, router)
    rec = _baseline(router, TF_A, "cpu")
    assert rec.pytorch_commit == COMMIT_B and rec.counts == Counts(5, 0, 0, 0)
    assert report.checks["baseline_skipped_merge_order"][0]["reason"] == "verified_first_parent_order"


def test_rerun_without_merged_at_preserves_known_value(config, router, monkeypatch):
    """同 commit 重跑且提交未带 merged_at：沿用基线已知值，不因 None 丢失。"""
    import run_test.upstream as up
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    first = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                            merged_at="2026-10-26T01:00:00Z",
                            merged_at_provenance="upstream-main-commit-time",
                            items=[(TF_A, "cpu", Counts(1, 0, 0, 0))])
    submit_complete_result(config, first, router)
    rerun = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                            items=[(TF_A, "cpu", Counts(2, 2, 2, 2))])
    submit_complete_result(config, rerun, router)
    rec = _baseline(router, TF_A, "cpu")
    assert rec.merged_at == "2026-10-26T01:00:00Z"
    assert rec.merged_at_provenance == "upstream-main-commit-time"


def test_backfill_merged_at_idempotent_and_kept(config, router, monkeypatch, paths):
    """回填：只填 NULL、幂等；已有值不一致记 conflict 不覆盖。"""
    import run_test.upstream as up
    from run_test.record import backfill_merged_at
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    sub = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                          items=[(TF_A, "cpu", Counts(1, 0, 0, 0))])
    submit_complete_result(config, sub, router)
    FakeMirror.commit_times = {COMMIT_A: "2026-10-26T01:00:00+00:00"}
    stats = backfill_merged_at(config)
    assert stats["pr_backfilled"] == 1 and stats["baseline_backfilled"] == 1
    again = backfill_merged_at(config)
    assert again["pr_backfilled"] == 0 and again["pr_kept"] == 1  # 幂等
    # 已有值与解析值不一致：保留原值并记 conflict。
    FakeMirror.commit_times[COMMIT_A] = "2026-10-26T09:00:00+00:00"
    conflicted = backfill_merged_at(config)
    assert conflicted["pr_backfilled"] == 0
    assert conflicted["conflicts"] and conflicted["conflicts"][0]["scope"] == "pr_results"
    conn = connect(pr_db_path(paths.data_root))
    try:
        row = get_pr_result(conn, 1)
    finally:
        conn.close()
    assert row.merged_at == "2026-10-26T01:00:00+00:00"  # 未被覆盖


def test_backfill_lease_released_when_backup_raises(config, router, monkeypatch, paths):
    """安全探针 1：备份抛异常时执行位必须随 with 释放，不得泄漏。"""
    import run_test.upstream as up
    import run_test.record as rec
    from run_test.record import backfill_merged_at
    from run_test.process_lock import executor_lease
    from run_test.config import store_paths
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    sub = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                          items=[(TF_A, "cpu", Counts(1, 0, 0, 0))])
    submit_complete_result(config, sub, router)
    monkeypatch.setattr(rec, '_backup_database',
                        lambda src, root: (_ for _ in ()).throw(RuntimeError("备份盘满")))
    lock_path = store_paths(config).queue_db.with_suffix(".executor.lock")
    with pytest.raises(RuntimeError):
        backfill_merged_at(config)
    with executor_lease(lock_path) as acquired:
        assert acquired is True  # 异常路径后租约已释放，可再次获取


def test_backfill_migrates_preupgrade_baseline_schema(config, router, monkeypatch, paths):
    """安全探针 2：老 schema（无合入元数据列）先备份原样、再幂等迁移，
    回填不再抛 no such column。"""
    import run_test.upstream as up
    from run_test.record import backfill_merged_at
    from run_test.baseline_store import baseline_db_path, BaselineRouter
    from run_test.db import connect
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    sub = make_submission(pr_number=1, pytorch_commit=COMMIT_A,
                          items=[(TF_A, "cpu", Counts(1, 0, 0, 0))])
    submit_complete_result(config, sub, router)
    # 构造升级前 schema：删掉三列（SQLite 重建表模拟）。
    router_obj = BaselineRouter(config.get("baseline_dirs", []), store_paths(config).baselines_root)
    for d in router_obj.dirs:
        conn = connect(baseline_db_path(router_obj.managed_root(d)))
        try:
            conn.executescript("""
                CREATE TABLE baseline_old AS SELECT
                    test_function, device, pytorch_commit, torch_npu_environment,
                    pass_count, fail_count, error_count, skip_count, log_path,
                    coverage_status, instance_ids, environment_fingerprint
                FROM baseline;
                DROP TABLE baseline;
                ALTER TABLE baseline_old RENAME TO baseline;
            """)
            conn.commit()
        finally:
            conn.close()
    FakeMirror.commit_times = {COMMIT_A: "2026-10-26T01:00:00+00:00"}
    stats = backfill_merged_at(config)  # 不应抛 OperationalError
    assert stats["baseline_backfilled"] == 1
    conn = connect(baseline_db_path(router_obj.managed_root(router_obj.dirs[0])))
    try:
        row = conn.execute("SELECT merged_at FROM baseline").fetchone()
        assert row["merged_at"] == "2026-10-26T01:00:00+00:00"
    finally:
        conn.close()
