"""基线前滚（baseline_rollforward）：时序回放、前后对比文件、基线刷新。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_test.baseline_rollforward import (
    DELETION_NOTICE,
    RollforwardError,
    rollforward_baselines,
)
from run_test.baseline_store import (
    BaselineRecord,
    BaselineRouter,
    baseline_db_path,
    baseline_log_filename,
    connect,
    ensure_baseline_db,
    get_baseline,
    upsert_baseline,
)
from run_test.config import store_paths
from run_test.identifiers import Counts
from run_test.pr_store import (
    PrResultRecord,
    ensure_pr_db,
    pr_db_path,
    set_merged_at,
    upsert_pr_result,
)
from run_test.record import (
    FunctionResult,
    Submission,
    submit_complete_result,
)

from conftest import BASELINE_DIRS, COMMIT_A, COMMIT_B, COMMIT_NPU, TF_A, TF_B

ENV = {"location": "/usr/local/lib/python3.10/site-packages",
       "torch_npu_commit": COMMIT_NPU}


def _counts(p, f=0, e=0, s=0):
    return Counts(pass_count=p, fail_count=f, error_count=e, skip_count=s)


def _seed_baseline(config, paths, tf, device, counts, commit=COMMIT_A):
    router = BaselineRouter(config["baseline_dirs"], paths.baselines_root)
    route = router.route(tf)
    ensure_baseline_db(route.root)
    logs = route.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / baseline_log_filename(tf, device)).write_text("PASSED x\n", encoding="utf-8")
    conn = connect(baseline_db_path(route.root))
    try:
        upsert_baseline(conn, BaselineRecord(
            test_function=tf, device=device, pytorch_commit=commit,
            torch_npu_environment=ENV, counts=counts,
            log_path=f"logs/{baseline_log_filename(tf, device)}",
            coverage_status="full", instance_ids=("legacy",),
            environment_fingerprint={"channel": device, "boot": None},
        ))
        conn.commit()
    finally:
        conn.close()


def _seed_pr(config, pr_number, tests_counts, merged_at=None, with_archive=False,
             commit=COMMIT_A):
    """直接构造 PR 库行（含每函数计数）；可选带日志目录/对比档案。"""
    paths = store_paths(config)
    ensure_pr_db(paths.data_root)
    tests = [
        {"test_function": tf, "device": dev, "counts": c.as_dict(),
         "coverage_status": "full", "instance_ids": ["legacy"],
         "environment_fingerprint": {"channel": dev, "boot": None}}
        for (tf, dev, c) in tests_counts
    ]
    conn = connect(pr_db_path(paths.data_root))
    try:
        upsert_pr_result(conn, PrResultRecord(
            pr_number=pr_number, pytorch_commit=commit, torch_npu_environment=ENV,
            tests=tests,
            counts=Counts(sum(c.pass_count for _, _, c in tests_counts),
                          sum(c.fail_count for _, _, c in tests_counts),
                          sum(c.error_count for _, _, c in tests_counts),
                          sum(c.skip_count for _, _, c in tests_counts)),
            log_path=f"pr_logs/PR_{pr_number}", merged_at=merged_at,
        ))
        conn.commit()
    finally:
        conn.close()
    log_dir = paths.data_root / f"pr_logs/PR_{pr_number}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run.log").write_text("x\n", encoding="utf-8")
    for tf, dev, _ in tests_counts:
        (log_dir / baseline_log_filename(tf, dev)).write_text(f"PR {pr_number}\n", encoding="utf-8")
    if with_archive:
        (log_dir / "baseline-diff.json").write_text(json.dumps({
            "pr_number": pr_number, "generated_at": "2026-09-18T00:00:00+00:00",
            "comparisons": [
                {
                    "test_function": tf, "device": dev, "managed_dir": "test",
                    "baseline_action": "overwritten" if baseline_hit(config, tf, dev) else "created",
                    "before": None,
                    "after": {"counts": c.as_dict(), "coverage_status": "full",
                              "instance_ids": ["legacy"],
                              "environment_fingerprint": {"channel": dev, "boot": None},
                              "pytorch_commit": commit, "torch_npu_environment": ENV},
                }
                for (tf, dev, c) in tests_counts
            ],
        }, ensure_ascii=False), encoding="utf-8")
    return log_dir


def baseline_hit(config, tf, device):
    paths = store_paths(config)
    router = BaselineRouter(config["baseline_dirs"], paths.baselines_root)
    route = router.route(tf)
    ensure_baseline_db(route.root)
    conn = connect(baseline_db_path(route.root))
    try:
        return get_baseline(conn, tf, device) is not None
    finally:
        conn.close()


def read_baseline(config, tf, device):
    paths = store_paths(config)
    router = BaselineRouter(config["baseline_dirs"], paths.baselines_root)
    route = router.route(tf)
    ensure_baseline_db(route.root)
    conn = connect(baseline_db_path(route.root))
    try:
        return get_baseline(conn, tf, device)
    finally:
        conn.close()


@pytest.fixture
def rf_config(config):
    config["skill"] = {"ref": "bf9b16e"}
    return config


def test_input_parsing_errors(rf_config):
    with pytest.raises(RollforwardError, match="prs"):
        rollforward_baselines(rf_config, {"nope": []})
    with pytest.raises(RollforwardError, match="非空"):
        rollforward_baselines(rf_config, {"prs": []})
    with pytest.raises(RollforwardError, match="非法"):
        rollforward_baselines(rf_config, {"prs": ["123"]})


def test_unknown_pr_and_missing_merge_time_are_errors(rf_config):
    report = rollforward_baselines(rf_config, {"prs": [424242]})
    assert report["prs"] == []
    assert len(report["errors"]) == 1
    assert "无该 PR 的测试结果" in report["errors"][0]["error"]


def test_order_and_files_with_archive(rf_config, tmp_path):
    """两个 PR 同键先后覆盖：文件按 merged_at 升序产出、基线终值取时间在后者。"""
    config = rf_config
    paths = store_paths(config)
    # 旧基线：TF_A pass=1（改动前状态）。
    _seed_baseline(config, paths, TF_A, "cuda", _counts(1))
    # PR 101（早）改 TF_A → pass=2；PR 102（晚）改 TF_A → pass=3。均带档案。
    _seed_pr(config, 101, [(TF_A, "cuda", _counts(2))], merged_at="2026-09-18T01:00:00+00:00",
             with_archive=True)
    _seed_pr(config, 102, [(TF_A, "cuda", _counts(3))], merged_at="2026-09-18T02:00:00+00:00",
             with_archive=True)

    report = rollforward_baselines(config, {"prs": [102, 101]})  # 乱序传入
    assert report["errors"] == []
    assert [p["pr_number"] for p in report["prs"]] == [101, 102]  # 时间序
    # 两个文件都存在且带提醒与前后对比。
    for entry in report["prs"]:
        assert Path(entry["file"]).is_file()
        data = json.loads(Path(entry["file"]).read_text(encoding="utf-8"))
        assert data["notice"] == DELETION_NOTICE
        assert data["source"] == "archived"
        assert data["comparisons"][0]["after"]["counts"]["pass"] == (
            2 if entry["pr_number"] == 101 else 3
        )
    # index.json 也有提醒与顺序。
    index = json.loads(Path(report["index_file"]).read_text(encoding="utf-8"))
    assert index["notice"] == DELETION_NOTICE
    assert [p["pr_number"] for p in index["prs"]] == [101, 102]
    # 基线终值 = 时间在后的 PR 102。
    assert read_baseline(config, TF_A, "cuda").counts.pass_count == 3


def test_new_key_creates_baseline_and_single_result_file(rf_config):
    """基线查不到（新测试/新 device）→ 文件只有 after，基线动作 = created。"""
    config = rf_config
    paths = store_paths(config)
    _seed_pr(config, 201, [(TF_A, "accelerator", _counts(5))],
             merged_at="2026-09-18T03:00:00+00:00", with_archive=True)
    # TF_A/accelerator 无基线日志文件 → archived 重放要求日志在位——
    # 真实流程中 PR 提交时基线日志由六步流程写入；这里手工补日志模拟提交后状态。
    route = BaselineRouter(config["baseline_dirs"], paths.baselines_root).route(TF_A)
    (route.root / "logs").mkdir(parents=True, exist_ok=True)
    (route.root / "logs" / baseline_log_filename(TF_A, "accelerator")).write_text(
        "PASSED test\n", encoding="utf-8")

    report = rollforward_baselines(config, {"prs": [201]})
    assert report["errors"] == []
    data = json.loads(Path(report["prs"][0]["file"]).read_text(encoding="utf-8"))
    cmp = data["comparisons"][0]
    assert cmp["baseline_action"] == "created"
    assert cmp["before"] is None                       # 只有 PR 测试结果
    assert cmp["after"]["counts"]["pass"] == 5
    assert read_baseline(config, TF_A, "accelerator").counts.pass_count == 5


def test_reconstructed_path_via_six_step_submit(rf_config):
    """无档案时回退重建：PR 库每函数计数 → 完整六步提交 → before 为真实旧基线。"""
    config = rf_config
    paths = store_paths(config)
    _seed_baseline(config, paths, TF_B, "cuda", _counts(1, 0, 0, 1))
    # PR 301 改 TF_B；PR 库行有每函数计数，但不带 baseline-diff.json 档案。
    _seed_pr(config, 301, [(TF_B, "cuda", _counts(2, 1))],
             merged_at="2026-09-18T04:00:00+00:00", with_archive=False)

    report = rollforward_baselines(config, {"prs": [301]})
    assert report["errors"] == []
    entry = report["prs"][0]
    assert entry["source"] == "reconstructed"
    data = json.loads(Path(entry["file"]).read_text(encoding="utf-8"))
    cmp = data["comparisons"][0]
    assert cmp["before"]["counts"] == _counts(1, 0, 0, 1).as_dict()
    assert cmp["after"]["counts"] == _counts(2, 1).as_dict()
    assert cmp["baseline_action"] == "overwritten"
    assert read_baseline(config, TF_B, "cuda").counts.pass_count == 2


def test_merged_at_from_upstream_commits_backfill(rf_config):
    """PR 库无 merged_at 时从队列库 upstream_commits 取 commit_time 并回填。"""
    config = rf_config
    paths = store_paths(config)
    from run_test.task_queue import QueueStore

    QueueStore(config)  # 建队列库 schema
    _seed_pr(config, 401, [(TF_A, "cuda", _counts(1))], merged_at=None,
             with_archive=True)
    qconn = connect(paths.queue_db)
    try:
        qconn.execute(
            "INSERT INTO upstream_commits (sha, pr_number, commit_time, subject, "
            "discovered_at, discovered_via, enqueue_state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (COMMIT_B, 401, "2026-09-18T05:00:00+00:00", "(#401) x",
             "2026-09-18T05:00:00+00:00", "hourly-fetch", "enqueued"),
        )
        qconn.commit()
    finally:
        qconn.close()

    report = rollforward_baselines(config, {"prs": [401]})
    assert report["errors"] == []
    assert report["prs"][0]["merged_at"] == "2026-09-18T05:00:00+00:00"
    # 回填已写回 PR 库。
    conn = connect(pr_db_path(paths.data_root))
    try:
        row = conn.execute("SELECT merged_at FROM pr_results WHERE pr_number=401").fetchone()
        assert row["merged_at"] == "2026-09-18T05:00:00+00:00"
    finally:
        conn.close()


def test_explicit_merged_at_in_input_wins(rf_config):
    config = rf_config
    _seed_pr(config, 501, [(TF_A, "cuda", _counts(1))], merged_at="2026-09-18T06:00:00+00:00",
             with_archive=True)
    report = rollforward_baselines(
        config, {"prs": [{"pr_number": 501, "merged_at": "2026-09-18T07:30:00+00:00"}]}
    )
    assert report["errors"] == []
    assert report["prs"][0]["merged_at"] == "2026-09-18T07:30:00+00:00"


def test_submit_materializes_diff_archive(rf_config):
    """submit_complete_result 现在固化 baseline-diff.json（档案数据源）。"""
    from run_test.record import DIFF_ARCHIVE_NAME

    config = rf_config
    paths = store_paths(config)
    _seed_baseline(config, paths, TF_A, "cuda", _counts(9))
    log_dir = paths.data_root / "pr_logs" / "src_502"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "f.log").write_text("log\n", encoding="utf-8")
    sub = Submission(
        pr_number=502, pytorch_commit=COMMIT_B, torch_npu_environment=ENV,
        function_results=[FunctionResult(
            TF_A, "cuda", _counts(1, 2), log_content="FAILED x\n",
            coverage_status="full", instance_ids=("legacy",),
            environment_fingerprint={"channel": "cuda", "boot": None},
        )],
        pr_log_source_dir=log_dir,
    )
    report = submit_complete_result(config, sub)
    assert report.diff_archive == f"pr_logs/PR_502/{DIFF_ARCHIVE_NAME}"
    data = json.loads((paths.data_root / report.diff_archive).read_text(encoding="utf-8"))
    assert data["comparisons"][0]["before"]["counts"]["pass"] == 9
    assert data["comparisons"][0]["after"]["counts"]["fail"] == 2
    assert data["comparisons"][0]["baseline_action"] == "overwritten"
