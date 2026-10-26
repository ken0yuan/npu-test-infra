"""PR 测试结果数据库（集中维护，PR 号唯一主键）。

契约见《本地测试结果与基线存储说明》第 3 节：
- 一条记录对应一个 PR 当前保存的一次完整测试提交；
- 唯一主键 pr_number：同 PR 再次提交完整结果时替换这一行，
  不追加第二行，不把运行 ID、device、环境或 commit 加入主键；
- tests 列表每项必须同时含 test_function 与 device，
  同一 (test_function, device) 在一份提交内只出现一次；
- 四个计数是本次所有函数、所有 device 的汇总，不累加历次结果；
- no_result_count 是 tests 中 status=no_result 的 (函数, device) 数：
  **未取得结果的函数不产生任何四计数**，零计数行 + 高 no_result_count
  = "没测到"而非"没失败"，二者必须在汇总层面可区分（2026-09-20 增列，
  起因 PR 197081 全 no_result 记 done、计数全 0 无法辨别）；
- 不建立运行历史表，不增加运行 ID 字段。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import PR_DB_NAME
from .db import connect
from .identifiers import (
    Counts,
    IdentifierError,
    validate_commit,
    validate_device,
    validate_test_function,
    validate_torch_npu_environment,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pr_results (
    pr_number             INTEGER PRIMARY KEY CHECK (pr_number > 0),
    pytorch_commit        TEXT    NOT NULL,
    torch_npu_environment TEXT    NOT NULL,
    tests                 TEXT    NOT NULL,
    pass_count            INTEGER NOT NULL CHECK (pass_count  >= 0),
    fail_count            INTEGER NOT NULL CHECK (fail_count  >= 0),
    error_count           INTEGER NOT NULL CHECK (error_count >= 0),
    skip_count            INTEGER NOT NULL CHECK (skip_count  >= 0),
    no_result_count       INTEGER NOT NULL DEFAULT 0 CHECK (no_result_count >= 0),
    log_path              TEXT    NOT NULL,
    merged_at             TEXT
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """已存在库的增量迁移：补 merged_at 与 no_result_count 列。

    no_result_count 语义见模块 docstring；旧行迁移后为 0，真实值由
    tests JSON 回填（存量行）或新提交写入。
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(pr_results)")}
    if "merged_at" not in cols:
        conn.execute("ALTER TABLE pr_results ADD COLUMN merged_at TEXT")
    if "no_result_count" not in cols:
        conn.execute(
            "ALTER TABLE pr_results ADD COLUMN no_result_count "
            "INTEGER NOT NULL DEFAULT 0"
        )


@dataclass(frozen=True)
class PrResultRecord:
    pr_number: int
    pytorch_commit: str
    torch_npu_environment: dict
    # 每项 {"test_function", "device", "counts": {pass/fail/error/skip}}；
    # counts 供基线前滚按函数复现结果，缺省（旧记录）为无 counts。
    tests: list[dict] = field(default_factory=list)
    counts: Counts = Counts.zero()
    # tests 中 status=no_result 的条数：未取得结果的函数不进四计数。
    # 0 pass + 0 fail + no_result_count>0 = "没测到"，不是"全过"。
    no_result_count: int = 0
    # 相对 data_root 的 PR 日志目录（如 pr_logs/PR_123456）。
    log_path: str = ""
    # PR 合入时间（ISO8601）；用于基线前滚的时序标注，可为 None。
    merged_at: str | None = None


class PrStoreError(ValueError):
    pass


def pr_db_path(data_root: Path) -> Path:
    return Path(data_root) / PR_DB_NAME


def ensure_pr_db(data_root: Path) -> Path:
    path = pr_db_path(data_root)
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()
    return path


def validate_pr_record(record: PrResultRecord) -> None:
    """按字段契约校验，非法即抛 PrStoreError。"""
    if isinstance(record.pr_number, bool) or not isinstance(record.pr_number, int) \
            or record.pr_number <= 0:
        raise PrStoreError(f"pr_number 必须是正整数，收到 {record.pr_number!r}")
    try:
        validate_commit(record.pytorch_commit, "pytorch_commit")
        validate_torch_npu_environment(record.torch_npu_environment)
    except IdentifierError as e:
        raise PrStoreError(str(e)) from e
    if not isinstance(record.tests, list) or not record.tests:
        raise PrStoreError("tests 必须是非空列表（一次完整提交至少一个函数/device 组合）")
    seen: set[tuple[str, str]] = set()
    for item in record.tests:
        if not isinstance(item, dict):
            raise PrStoreError(f"tests 每项必须是对象，收到 {item!r}")
        try:
            tf = validate_test_function(item.get("test_function"))
            dev = validate_device(item.get("device"))
        except IdentifierError as e:
            raise PrStoreError(f"tests 项非法（{item!r}）: {e}") from e
        # counts 可选（旧记录无）；给了就必须是合法四计数，供基线前滚复现。
        if "counts" in item:
            try:
                Counts.from_dict(item["counts"])
            except (IdentifierError, ValueError, TypeError, KeyError) as e:
                raise PrStoreError(f"tests 项 ({tf}, {dev}) counts 非法: {e}") from e
        key = (tf, dev)
        if key in seen:
            raise PrStoreError(
                f"同一份完整提交内 ({tf}, {dev}) 只能出现一次，不得重复累加"
            )
        seen.add(key)
    if not isinstance(record.counts, Counts):
        raise PrStoreError(
            f"counts 必须是 Counts（非负整数四计数），收到 {record.counts!r}"
        )
    # Counts 构造时已保证四个计数均为非负整数，这里再显式断言一次字段契约。
    if min(
        record.counts.pass_count, record.counts.fail_count,
        record.counts.error_count, record.counts.skip_count,
    ) < 0:
        raise PrStoreError("四个计数必须为非负整数")
    if isinstance(record.no_result_count, bool) or not isinstance(record.no_result_count, int) \
            or record.no_result_count < 0:
        raise PrStoreError(
            f"no_result_count 必须为非负整数，收到 {record.no_result_count!r}"
        )
    # no_result_count 必须与 tests 内容一致：声明了 no_result 的条数就是它。
    declared = sum(1 for item in record.tests if item.get("status") == "no_result")
    if record.no_result_count != declared:
        raise PrStoreError(
            f"no_result_count={record.no_result_count} 与 tests 中 "
            f"status=no_result 条数 {declared} 不一致，不得伪造"
        )
    if not isinstance(record.log_path, str) or not record.log_path.strip():
        raise PrStoreError("log_path 不能为空，必须能找到对应 PR 日志目录")


def upsert_pr_result(conn: sqlite3.Connection, record: PrResultRecord) -> None:
    """按 PR 号插入或替换当前记录（计数为本次汇总值，不累加旧值）。

    merged_at 语义：显式给出时覆盖；为 None 时保留库中已有值（重新提交
    结果不应抹掉合入时间）。
    """
    validate_pr_record(record)
    conn.execute(
        """
        INSERT INTO pr_results
            (pr_number, pytorch_commit, torch_npu_environment, tests,
             pass_count, fail_count, error_count, skip_count,
             no_result_count, log_path, merged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pr_number) DO UPDATE SET
            pytorch_commit        = excluded.pytorch_commit,
            torch_npu_environment = excluded.torch_npu_environment,
            tests                 = excluded.tests,
            pass_count            = excluded.pass_count,
            fail_count            = excluded.fail_count,
            error_count           = excluded.error_count,
            skip_count            = excluded.skip_count,
            no_result_count       = excluded.no_result_count,
            log_path              = excluded.log_path,
            merged_at             = CASE
                WHEN excluded.merged_at IS NOT NULL THEN excluded.merged_at
                WHEN pr_results.pytorch_commit = excluded.pytorch_commit
                    THEN pr_results.merged_at
                ELSE NULL END
        """,
        (
            record.pr_number,
            record.pytorch_commit,
            json.dumps(record.torch_npu_environment, ensure_ascii=False),
            json.dumps(record.tests, ensure_ascii=False),
            record.counts.pass_count,
            record.counts.fail_count,
            record.counts.error_count,
            record.counts.skip_count,
            record.no_result_count,
            record.log_path,
            record.merged_at,
        ),
    )


def set_merged_at(conn: sqlite3.Connection, pr_number: int, merged_at: str) -> None:
    """写入/更新 PR 合入时间（ISO8601 文本，必须是可解析的时间戳）。"""
    if isinstance(merged_at, bool) or not isinstance(merged_at, str) \
            or not merged_at.strip():
        raise PrStoreError(f"merged_at 必须是非空时间戳文本，收到 {merged_at!r}")
    try:
        datetime.fromisoformat(merged_at.strip())
    except ValueError as e:
        raise PrStoreError(f"merged_at 不是合法 ISO8601 时间戳: {merged_at!r}") from e
    cur = conn.execute(
        "UPDATE pr_results SET merged_at = ? WHERE pr_number = ?",
        (merged_at.strip(), pr_number),
    )
    if cur.rowcount == 0:
        raise PrStoreError(f"PR {pr_number} 不存在，无法设置合入时间")


def row_to_record(row: sqlite3.Row) -> PrResultRecord:
    keys = row.keys()
    # 旧快照库可能缺 no_result_count（_migrate 之前的镜像）；缺失按 0 读出，
    # 不伪造，由调用方决定是否需要回填。
    no_result = row["no_result_count"] if "no_result_count" in keys else 0
    return PrResultRecord(
        pr_number=row["pr_number"],
        pytorch_commit=row["pytorch_commit"],
        torch_npu_environment=json.loads(row["torch_npu_environment"]),
        tests=json.loads(row["tests"]),
        counts=Counts(
            pass_count=row["pass_count"],
            fail_count=row["fail_count"],
            error_count=row["error_count"],
            skip_count=row["skip_count"],
        ),
        no_result_count=no_result,
        log_path=row["log_path"],
        merged_at=row["merged_at"] if "merged_at" in keys else None,
    )


def get_pr_result(conn: sqlite3.Connection, pr_number: int) -> PrResultRecord | None:
    row = conn.execute(
        "SELECT * FROM pr_results WHERE pr_number = ?", (pr_number,)
    ).fetchone()
    return row_to_record(row) if row else None


def list_pr_results(conn: sqlite3.Connection) -> list[PrResultRecord]:
    rows = conn.execute(
        "SELECT * FROM pr_results ORDER BY pr_number"
    ).fetchall()
    return [row_to_record(r) for r in rows]
