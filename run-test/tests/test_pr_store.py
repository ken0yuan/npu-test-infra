"""PR 测试结果数据库（存储文档第 3 节验收项）。"""

import json

import pytest

from run_test.db import connect
from run_test.identifiers import Counts
from run_test.pr_store import (
    PrResultRecord,
    PrStoreError,
    ensure_pr_db,
    get_pr_result,
    list_pr_results,
    pr_db_path,
    upsert_pr_result,
)

from conftest import COMMIT_A, COMMIT_NPU, TF_A, TF_B


@pytest.fixture
def conn(paths):
    ensure_pr_db(paths.data_root)
    return connect(pr_db_path(paths.data_root))


def make_record(pr=123456, counts=Counts(5, 1, 1, 2), log_path="pr_logs/PR_123456"):
    return PrResultRecord(
        pr_number=pr,
        pytorch_commit=COMMIT_A,
        torch_npu_environment={"location": "env", "torch_npu_commit": COMMIT_NPU},
        tests=[
            {"test_function": TF_A, "device": "cpu"},
            {"test_function": TF_A, "device": "accelerator"},
            {"test_function": TF_B, "device": "accelerator"},
        ],
        counts=counts,
        log_path=log_path,
    )


def test_upsert_and_read_back(conn):
    upsert_pr_result(conn, make_record())
    got = get_pr_result(conn, 123456)
    assert got is not None
    assert got.pr_number == 123456
    assert got.counts == Counts(5, 1, 1, 2)
    assert [t["device"] for t in got.tests] == ["cpu", "accelerator", "accelerator"]
    assert got.torch_npu_environment["torch_npu_commit"] == COMMIT_NPU


def test_pr_number_is_unique_key_no_second_row(conn):
    """同 PR 重复提交不增加第二行，计数替换不累加（验收 3.1/8-1）。"""
    upsert_pr_result(conn, make_record(counts=Counts(5, 1, 1, 2)))
    upsert_pr_result(conn, make_record(counts=Counts(9, 0, 0, 3)))
    rows = list_pr_results(conn)
    assert len(rows) == 1
    assert rows[0].counts == Counts(9, 0, 0, 3)


def test_different_prs_are_separate_rows(conn):
    upsert_pr_result(conn, make_record(pr=1))
    upsert_pr_result(conn, make_record(pr=2))
    assert len(list_pr_results(conn)) == 2


def test_tests_requires_device(conn):
    rec = make_record()
    rec = PrResultRecord(
        **{**rec.__dict__, "tests": [{"test_function": TF_A}]}
    )
    with pytest.raises(PrStoreError, match="device"):
        upsert_pr_result(conn, rec)


def test_tests_rejects_duplicate_function_device(conn):
    rec = make_record()
    rec = PrResultRecord(
        **{
            **rec.__dict__,
            "tests": [
                {"test_function": TF_A, "device": "cpu"},
                {"test_function": TF_A, "device": "cpu"},
            ],
        }
    )
    with pytest.raises(PrStoreError, match="只能出现一次"):
        upsert_pr_result(conn, rec)


def test_tests_must_be_nonempty(conn):
    rec = make_record()
    rec = PrResultRecord(**{**rec.__dict__, "tests": []})
    with pytest.raises(PrStoreError, match="非空列表"):
        upsert_pr_result(conn, rec)


def test_rejects_imprecise_commit(conn):
    rec = make_record()
    rec = PrResultRecord(**{**rec.__dict__, "pytorch_commit": "main"})
    with pytest.raises(PrStoreError):
        upsert_pr_result(conn, rec)


def test_rejects_bad_device(conn):
    rec = make_record()
    rec = PrResultRecord(
        **{
            **rec.__dict__,
            "tests": [{"test_function": TF_A, "device": "npu"}],
        }
    )
    with pytest.raises(PrStoreError):
        upsert_pr_result(conn, rec)


def test_negative_counts_blocked_by_db_constraint(conn):
    """应用层在 Counts 构造时拒绝负数（见 test_identifiers）；数据库 CHECK 兜底。"""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            # 10 列（含 2026-09-20 增设的 no_result_count，位于 skip 之后）。
            "INSERT INTO pr_results VALUES (999, ?, ?, ?, -1, 0, 0, 0, 0, ?, NULL)",
            (COMMIT_A, "{}", "[]", "x"),
        )


def test_json_roundtrip(conn):
    """tests 与环境字段 JSON 序列化可还原。"""
    upsert_pr_result(conn, make_record())
    raw = conn.execute(
        "SELECT tests, torch_npu_environment FROM pr_results WHERE pr_number=123456"
    ).fetchone()
    assert json.loads(raw["tests"])[0] == {"test_function": TF_A, "device": "cpu"}
    assert json.loads(raw["torch_npu_environment"])["location"] == "env"


# ---------------------------------------------------------------------------
# no_result_count 列（2026-09-20）：零计数行的"没测到"与"全过"必须可区分
# ---------------------------------------------------------------------------


def test_no_result_count_roundtrip(conn):
    record = make_record(pr=197081, counts=Counts(0, 0, 0, 0))
    record = PrResultRecord(
        pr_number=197081,
        pytorch_commit=record.pytorch_commit,
        torch_npu_environment=record.torch_npu_environment,
        tests=[
            {"test_function": TF_A, "device": "cpu", "status": "no_result",
             "reason": "未收集到任何测试"},
            {"test_function": TF_A, "device": "accelerator", "status": "no_result",
             "reason": "未收集到任何测试"},
        ],
        counts=Counts(0, 0, 0, 0),
        no_result_count=2,
        log_path="pr_logs/PR_197081",
    )
    upsert_pr_result(conn, record)
    got = get_pr_result(conn, 197081)
    assert got.no_result_count == 2
    assert got.counts == Counts(0, 0, 0, 0)


def test_no_result_count_must_match_tests_declaration(conn):
    """no_result_count 与 tests 中 status=no_result 条数不一致 = 伪造，拒绝。"""
    record = make_record(pr=197082)
    record = PrResultRecord(
        pr_number=197082,
        pytorch_commit=record.pytorch_commit,
        torch_npu_environment=record.torch_npu_environment,
        tests=[{"test_function": TF_A, "device": "cpu", "status": "no_result",
                "reason": "x"}],
        counts=Counts(0, 0, 0, 0),
        no_result_count=3,  # 声明了 3，实际 1
        log_path="pr_logs/PR_197082",
    )
    with pytest.raises(PrStoreError, match="no_result_count"):
        upsert_pr_result(conn, record)


def test_migration_adds_no_result_column(paths):
    """旧库（无该列）打开即迁移补列，旧行默认 0，不伪造。"""
    import sqlite3

    db_path = pr_db_path(paths.data_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """CREATE TABLE pr_results (
            pr_number INTEGER PRIMARY KEY, pytorch_commit TEXT NOT NULL,
            torch_npu_environment TEXT NOT NULL, tests TEXT NOT NULL,
            pass_count INTEGER NOT NULL, fail_count INTEGER NOT NULL,
            error_count INTEGER NOT NULL, skip_count INTEGER NOT NULL,
            log_path TEXT NOT NULL)"""
    )
    legacy.execute(
        "INSERT INTO pr_results VALUES (1, ?, ?, ?, 1, 0, 0, 0, 'x')",
        (COMMIT_A, json.dumps({"location": "e", "torch_npu_commit": COMMIT_NPU}),
         json.dumps([{"test_function": TF_A, "device": "cpu"}])),
    )
    legacy.commit()
    legacy.close()
    ensure_pr_db(paths.data_root)  # 迁移入口
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT no_result_count FROM pr_results WHERE pr_number = 1"
        ).fetchone()
        got = get_pr_result(conn, 1)
    finally:
        conn.close()
    assert row["no_result_count"] == 0
    assert got.no_result_count == 0
