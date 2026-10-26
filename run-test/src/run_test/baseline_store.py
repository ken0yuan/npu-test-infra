"""按最小测试管理目录拆分的基线数据库。

契约见《本地测试结果与基线存储说明》第 5 节：
- 每个选定的最小测试管理目录对应一个 baseline.db，其下有 logs/ 目录；
- 每个测试文件明确归属唯一一个基线库，不得同时写入父子两个库；
- 联合主键严格为 (test_function, device)，不含 PR 号、环境、commit、参数；
- 每条记录的日志位于本库所在目录内，更新记录时替换对应日志；
- “最小目录”清单由配置给定，实现不自行扫描仓库批量建库。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import BASELINES_DIR_NAME
from .db import connect
from .identifiers import (
    Counts,
    IdentifierError,
    baseline_log_filename,
    parse_test_function,
    validate_commit,
    validate_device,
    validate_test_function,
    validate_torch_npu_environment,
)

BASELINE_DB_NAME = "baseline.db"
BASELINE_LOGS_DIR_NAME = "logs"

SCHEMA = """
CREATE TABLE IF NOT EXISTS baseline (
    test_function         TEXT    NOT NULL,
    device                TEXT    NOT NULL CHECK (device IN ('cpu','accelerator','cuda')),
    pytorch_commit        TEXT    NOT NULL,
    torch_npu_environment TEXT    NOT NULL,
    pass_count            INTEGER NOT NULL CHECK (pass_count  >= 0),
    fail_count            INTEGER NOT NULL CHECK (fail_count  >= 0),
    error_count           INTEGER NOT NULL CHECK (error_count >= 0),
    skip_count            INTEGER NOT NULL CHECK (skip_count  >= 0),
    log_path              TEXT    NOT NULL,
    -- 合入序元数据（用户 2026-10-26 merged_at 方案）：基线来源 PR 与
    -- 其 main 合入时刻（provenance 记录来源类别，如 upstream-main-commit-time）。
    merged_at             TEXT,
    merged_at_provenance  TEXT,
    source_pr_number      INTEGER,
    PRIMARY KEY (test_function, device)
);
"""


class BaselineStoreError(ValueError):
    pass


class NoBaselineDirectoryError(BaselineStoreError):
    """函数文件未归属任何已配置的最小管理目录。"""


@dataclass(frozen=True)
class BaselineRoute:
    managed_dir: str      # 清单内的目录（相对测试树），如 "test/distributed/_tensor"
    root: Path            # 该库所在磁盘目录


@dataclass(frozen=True)
class BaselineRecord:
    test_function: str
    device: str
    pytorch_commit: str
    torch_npu_environment: dict
    counts: Counts = Counts.zero()
    # 相对本库所在目录的日志路径，如 "logs/xxx.cpu.3fa9….log"。
    log_path: str = ""
    coverage_status: str = "unknown"
    instance_ids: tuple[str, ...] = ()
    environment_fingerprint: dict | None = None
    merged_at: str | None = None
    merged_at_provenance: str | None = None
    source_pr_number: int | None = None


class BaselineRouter:
    """把函数文件路径路由到唯一的最小管理目录（最长段级前缀匹配）。"""

    def __init__(self, baseline_dirs: Sequence[str], baselines_root: Path):
        self.baselines_root = Path(baselines_root)
        normalized: list[str] = []
        for d in baseline_dirs or []:
            if not isinstance(d, str) or not d.strip():
                raise BaselineStoreError(f"baseline_dirs 项必须是非空字符串: {d!r}")
            dd = d.strip().strip("/")
            if dd.startswith("/") or ".." in dd.split("/"):
                raise BaselineStoreError(f"baseline_dirs 项必须是相对目录且不含越段: {d!r}")
            if dd not in normalized:
                normalized.append(dd)
        # 允许父子目录共存（如 test/ 兜底 + test/distributed/ 子库）：
        # route() 按段数降序做最长段级匹配，每个文件仍唯一归属一个库，
        # 满足"每个测试文件明确归属一个基线数据库，不得同时写入父子两库"。
        # 长目录在前，保证最长匹配优先。
        self._dirs = sorted(normalized, key=lambda d: (-len(d.split("/")), d))
        self._dir_set = list(normalized)

    @property
    def dirs(self) -> list[str]:
        return list(self._dir_set)

    def managed_root(self, managed_dir: str) -> Path:
        return self.baselines_root / managed_dir

    def route(self, test_function: str) -> BaselineRoute:
        """返回函数归属的最小管理目录；无归属时抛 NoBaselineDirectoryError。"""
        tf = validate_test_function(test_function)
        file_path = parse_test_function(tf).file_path
        parts = file_path.split("/")
        for d in self._dirs:
            segs = d.split("/")
            if parts[: len(segs)] == segs:
                return BaselineRoute(managed_dir=d, root=self.managed_root(d))
        raise NoBaselineDirectoryError(
            f"{tf} 的文件路径未归属任何已配置的最小管理目录（当前清单: {self._dir_set}）；"
            "请先通过配置/CLI 登记 baseline_dirs"
        )

    def ensure_all(self) -> list[Path]:
        """创建全部已配置目录的 baseline.db 与 logs/。"""
        made = []
        for d in self._dirs:
            root = self.managed_root(d)
            made.append(ensure_baseline_db(root))
        return made


# ---------------------------------------------------------------------------
# 单库操作
# ---------------------------------------------------------------------------

def baseline_db_path(managed_root: Path) -> Path:
    return Path(managed_root) / BASELINE_DB_NAME


def baseline_logs_dir(managed_root: Path) -> Path:
    return Path(managed_root) / BASELINE_LOGS_DIR_NAME


def ensure_baseline_db(managed_root: Path) -> Path:
    root = Path(managed_root)
    root.mkdir(parents=True, exist_ok=True)
    baseline_logs_dir(root).mkdir(parents=True, exist_ok=True)
    path = baseline_db_path(root)
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(baseline)")}
        migrations = {
            "coverage_status": "ALTER TABLE baseline ADD COLUMN coverage_status TEXT NOT NULL DEFAULT 'unknown'",
            "instance_ids": "ALTER TABLE baseline ADD COLUMN instance_ids TEXT NOT NULL DEFAULT '[]'",
            "environment_fingerprint": "ALTER TABLE baseline ADD COLUMN environment_fingerprint TEXT",
            "merged_at": "ALTER TABLE baseline ADD COLUMN merged_at TEXT",
            "merged_at_provenance": "ALTER TABLE baseline ADD COLUMN merged_at_provenance TEXT",
            "source_pr_number": "ALTER TABLE baseline ADD COLUMN source_pr_number INTEGER",
        }
        for name, sql in migrations.items():
            if name not in cols:
                conn.execute(sql)
        conn.commit()
    finally:
        conn.close()
    return path


def validate_baseline_record(record: BaselineRecord) -> None:
    try:
        tf = validate_test_function(record.test_function)
        dev = validate_device(record.device)
        validate_commit(record.pytorch_commit, "pytorch_commit")
        validate_torch_npu_environment(record.torch_npu_environment)
    except IdentifierError as e:
        raise BaselineStoreError(str(e)) from e
    Counts(
        pass_count=record.counts.pass_count,
        fail_count=record.counts.fail_count,
        error_count=record.counts.error_count,
        skip_count=record.counts.skip_count,
    )
    if not isinstance(record.log_path, str) or not record.log_path.strip():
        raise BaselineStoreError(f"({tf}, {dev}) 的 log_path 不能为空")
    if record.coverage_status not in {"full", "partial", "unknown", "not_comparable"}:
        raise BaselineStoreError(f"({tf}, {dev}) coverage_status 非法: {record.coverage_status!r}")
    if not isinstance(record.instance_ids, (tuple, list)) or not all(isinstance(x, str) and x for x in record.instance_ids):
        raise BaselineStoreError(f"({tf}, {dev}) instance_ids 非法")
    if record.environment_fingerprint is not None and not isinstance(record.environment_fingerprint, dict):
        raise BaselineStoreError(f"({tf}, {dev}) environment_fingerprint 必须是对象")


def check_log_filename_free(
    conn: sqlite3.Connection, test_function: str, device: str, log_path: str
) -> None:
    """确保日志文件名未被其他 (test_function, device) 占用（严格无冲突）。"""
    row = conn.execute(
        """
        SELECT test_function, device FROM baseline
        WHERE log_path = ? AND NOT (test_function = ? AND device = ?)
        """,
        (log_path, test_function, device),
    ).fetchone()
    if row:
        raise BaselineStoreError(
            f"日志文件名 {log_path!r} 已被 ({row['test_function']}, {row['device']}) 占用；"
            "不同函数不得共享日志"
        )


def upsert_baseline(conn: sqlite3.Connection, record: BaselineRecord) -> None:
    """按 (test_function, device) 插入或替换，计数为本次真实值。"""
    validate_baseline_record(record)
    check_log_filename_free(conn, record.test_function, record.device, record.log_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO baseline
            (test_function, device, pytorch_commit, torch_npu_environment,
             pass_count, fail_count, error_count, skip_count, log_path,
             coverage_status, instance_ids, environment_fingerprint,
             merged_at, merged_at_provenance, source_pr_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.test_function,
            record.device,
            record.pytorch_commit,
            json.dumps(record.torch_npu_environment, ensure_ascii=False),
            record.counts.pass_count,
            record.counts.fail_count,
            record.counts.error_count,
            record.counts.skip_count,
            record.log_path,
            record.coverage_status,
            json.dumps(list(record.instance_ids), ensure_ascii=False),
            json.dumps(record.environment_fingerprint, ensure_ascii=False) if record.environment_fingerprint is not None else None,
            record.merged_at,
            record.merged_at_provenance,
            record.source_pr_number,
        ),
    )


def row_to_record(row: sqlite3.Row) -> BaselineRecord:
    return BaselineRecord(
        test_function=row["test_function"],
        device=row["device"],
        pytorch_commit=row["pytorch_commit"],
        torch_npu_environment=json.loads(row["torch_npu_environment"]),
        counts=Counts(
            pass_count=row["pass_count"],
            fail_count=row["fail_count"],
            error_count=row["error_count"],
            skip_count=row["skip_count"],
        ),
        log_path=row["log_path"],
        coverage_status=row["coverage_status"] if "coverage_status" in row.keys() else "unknown",
        instance_ids=tuple(json.loads(row["instance_ids"] or "[]")) if "instance_ids" in row.keys() else (),
        environment_fingerprint=(json.loads(row["environment_fingerprint"]) if "environment_fingerprint" in row.keys() and row["environment_fingerprint"] else None),
        merged_at=(row["merged_at"] if "merged_at" in row.keys() else None),
        merged_at_provenance=(row["merged_at_provenance"] if "merged_at_provenance" in row.keys() else None),
        source_pr_number=(row["source_pr_number"] if "source_pr_number" in row.keys() else None),
    )


def get_baseline(
    conn: sqlite3.Connection, test_function: str, device: str
) -> BaselineRecord | None:
    row = conn.execute(
        "SELECT * FROM baseline WHERE test_function = ? AND device = ?",
        (test_function, device),
    ).fetchone()
    return row_to_record(row) if row else None


def list_baselines(conn: sqlite3.Connection) -> list[BaselineRecord]:
    rows = conn.execute(
        "SELECT * FROM baseline ORDER BY test_function, device"
    ).fetchall()
    return [row_to_record(r) for r in rows]


def resolve_baseline_log(managed_root: Path, record: BaselineRecord) -> Path:
    return Path(managed_root) / record.log_path
