"""SQLite 连接基础：WAL、外键、手动事务。"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: str | Path) -> sqlite3.Connection:
    """打开（必要时创建）一个 SQLite 数据库连接。

    - WAL 模式 + busy_timeout：多进程读写下减少锁冲突；
    - isolation_level=None：由调用方显式 BEGIN/COMMIT，便于把
      “先落日志、后提交事务”的一致性顺序写清楚。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def begin_immediate(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def commit(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("COMMIT")


def rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")
