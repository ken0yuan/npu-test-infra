"""轻量持久化任务队列（SQLite 实现）。

对应《OBS监测与串行测试队列说明》第 5 节：

    发现对象并记录候选
        ↓
    按用户确认的规则核实包身份（当前阻塞项，见 verify_identity）
        ↓
    既有脚本过滤（prefilter_pending，对接点待输入来源确认）
        ↓
    待分析 → 待测试 → 准备中 → 测试中 → 写库中 → 清理中 → 完成

关键约束：
- 候选不是已获准的测试任务：身份未核实/规则未确认的对象永远停在
  identity_pending，不进入分析与测试（验收要求：不默认放行、不恒 true）；
- 同一远端对象版本反复监测不重复排队（obs_objects 去重）；
- 同名对象更新视为新对象，必须重新核实绑定关系；
- 同一时间只允许一个任务处于 准备/测试/写库/清理（executor_lock + 事务）；
- 待测试任务按入队先后处理；队列标识不改变 PR 结果库主键。
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import QUEUE_DB_NAME, store_paths
from .db import connect
from .identifiers import (
    IdentifierError,
    validate_commit,
    validate_device,
    validate_test_function,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id               TEXT PRIMARY KEY,
    state                 TEXT NOT NULL,
    -- 候选登记信息（OBS 对象）
    obs_wheels_url        TEXT NOT NULL,
    obs_source_url        TEXT NOT NULL,
    -- 拆分包格式（2026-10-22 起构建方拆分）：独立 torch_npu 包对象；
    -- legacy/qualified 格式 torch_npu 在主体包内，此列为 NULL。
    obs_wheels_npu_url    TEXT,
    obs_wheels_npu_version TEXT,
    obs_wheels_version    TEXT,
    obs_source_version    TEXT,
    obs_remote_time       TEXT,
    expiry_rule           TEXT,
    expires_at            TEXT,
    -- 身份核实后填充的绑定信息（待负责人确认规则后由证据提供）
    pr_number             INTEGER,
    pytorch_commit        TEXT,
    torch_npu_commit      TEXT,
    torch_npu_location    TEXT,
    identity_rule_id      TEXT,
    identity_evidence     TEXT,
    identity_verified_at  TEXT,
    -- 分析产物与测试清单
    analysis_output       TEXT,
    test_manifest         TEXT,
    -- 执行
    work_dir              TEXT,
    -- 记录
    enqueued_at           TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    state_changed_at      TEXT NOT NULL,
    error_reason          TEXT,
    retry_count           INTEGER NOT NULL DEFAULT 0,
    cleanup_attempts      INTEGER NOT NULL DEFAULT 0,
    history               TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS obs_objects (
    -- 去重键：URL + 远端对象版本标识。版本标识的具体形式待负责人确认；
    -- 缺失时退化为 URL 本身，但绝不因此放行身份。
    object_key            TEXT PRIMARY KEY,
    first_seen_at         TEXT NOT NULL,
    last_seen_at          TEXT NOT NULL,
    remote_time           TEXT,
    task_id               TEXT
);

CREATE TABLE IF NOT EXISTS identity_rules (
    -- 用户确认的“OBS 包对应已合入 commit”判定规则原文与来源。
    -- 未注册任何规则时 verify_identity 一律拒绝（强制门禁）。
    rule_id               TEXT PRIMARY KEY,
    rule_text             TEXT NOT NULL,
    source                TEXT NOT NULL,
    confirmed_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_state (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    skill_commit          TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS executor_lock (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    task_id               TEXT NOT NULL,
    acquired_at           TEXT NOT NULL
);

-- 上游 main commit 记录（2026-09-17 用户确认的“GitHub main 驱动”方案）：
-- 每轮定时 fetch 后记录新 commit 及其 PR 号；包在桶中出现时入队。
CREATE TABLE IF NOT EXISTS upstream_commits (
    sha                   TEXT PRIMARY KEY,
    pr_number             INTEGER,
    commit_time           TEXT NOT NULL,
    subject               TEXT NOT NULL,
    discovered_at         TEXT NOT NULL,
    discovered_via        TEXT NOT NULL,
    enqueue_state         TEXT NOT NULL,
    task_id               TEXT,
    last_checked_at       TEXT,
    note                  TEXT
);

CREATE TABLE IF NOT EXISTS mirror_state (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    repo_url              TEXT NOT NULL,
    last_tip              TEXT NOT NULL,
    last_fetch_at         TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------

STATE_CANDIDATE = "candidate"
STATE_IDENTITY_PENDING = "identity_pending"
STATE_PREFILTER_PENDING = "prefilter_pending"
STATE_PENDING_ANALYSIS = "pending_analysis"
STATE_PENDING_TEST = "pending_test"
STATE_PREPARING = "preparing"
STATE_TESTING = "testing"
STATE_WRITING = "writing"
STATE_CLEANING = "cleaning"
STATE_DONE = "done"
STATE_FILTERED_OUT = "filtered_out"
STATE_BLOCKED = "blocked"
STATE_EXPIRED = "expired"
STATE_FAILED = "failed"

ALL_STATES = frozenset({
    STATE_CANDIDATE, STATE_IDENTITY_PENDING, STATE_PREFILTER_PENDING,
    STATE_PENDING_ANALYSIS, STATE_PENDING_TEST, STATE_PREPARING,
    STATE_TESTING, STATE_WRITING, STATE_CLEANING, STATE_DONE,
    STATE_FILTERED_OUT, STATE_BLOCKED, STATE_EXPIRED, STATE_FAILED,
})

# 终态：不可再迁移（blocked 的恢复走 requeue）。
TERMINAL_STATES = frozenset({STATE_DONE, STATE_FILTERED_OUT, STATE_EXPIRED, STATE_FAILED})

# 这四个状态同一时间全局互斥（文档第 6 节：同一时间只能执行一个测试任务）。
ACTIVE_STATES = frozenset({STATE_PREPARING, STATE_TESTING, STATE_WRITING, STATE_CLEANING})

TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_CANDIDATE: frozenset({STATE_IDENTITY_PENDING}),
    # 核实通过 → 待过滤；核实不过/对象不齐 → blocked 等待；资源过期 → expired。
    STATE_IDENTITY_PENDING: frozenset({
        STATE_PREFILTER_PENDING, STATE_FILTERED_OUT, STATE_BLOCKED, STATE_EXPIRED,
    }),
    # 过滤结果：进入分析队列，或被既有规则排除（终态）。
    STATE_PREFILTER_PENDING: frozenset({
        STATE_PENDING_ANALYSIS, STATE_FILTERED_OUT, STATE_BLOCKED,
    }),
    # AI 形成明确函数/device 清单后才可进入待测试；没有清单不得回退全量。
    STATE_PENDING_ANALYSIS: frozenset({
        STATE_PENDING_TEST, STATE_FILTERED_OUT, STATE_BLOCKED,
    }),
    STATE_PENDING_TEST: frozenset({STATE_PREPARING, STATE_BLOCKED, STATE_EXPIRED}),
    STATE_PREPARING: frozenset({STATE_TESTING, STATE_BLOCKED, STATE_CLEANING}),
    STATE_TESTING: frozenset({STATE_WRITING, STATE_CLEANING, STATE_BLOCKED}),
    # 写库失败保留证据重试；重试耗尽 → failed（不清理，防删唯一结果副本）。
    # 进程重启导致写库中断 → blocked（人工核实后 requeue）。
    STATE_WRITING: frozenset({STATE_CLEANING, STATE_FAILED, STATE_BLOCKED}),
    STATE_CLEANING: frozenset({STATE_DONE, STATE_FAILED}),
    # blocked 恢复目标由人工按原因指定（requeue）。
    STATE_BLOCKED: frozenset({
        STATE_IDENTITY_PENDING, STATE_PENDING_ANALYSIS, STATE_PENDING_TEST,
        STATE_EXPIRED, STATE_FAILED,
    }),
}

REQUEUE_TARGETS = frozenset({
    STATE_IDENTITY_PENDING, STATE_PENDING_ANALYSIS, STATE_PENDING_TEST,
})


class QueueError(ValueError):
    pass


class InvalidTransitionError(QueueError):
    pass


class IdentityGateError(QueueError):
    """身份核实门禁拒绝：判定规则未确认或证据不全。

    对应文档第 1 节：OBS 负责人未确认“包对应已合入 commit”的判定规则前，
    任何候选不得放行；不得以默认放行、恒为 true、仅看文件名等方式绕过。
    """


# ---------------------------------------------------------------------------
# Task 数据对象
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    task_id: str
    state: str
    obs_wheels_url: str
    obs_source_url: str
    obs_wheels_npu_url: str | None = None
    obs_wheels_npu_version: str | None = None
    obs_wheels_version: str | None = None
    obs_source_version: str | None = None
    obs_remote_time: str | None = None
    expiry_rule: str | None = None
    expires_at: str | None = None
    pr_number: int | None = None
    pytorch_commit: str | None = None
    torch_npu_commit: str | None = None
    torch_npu_location: str | None = None
    identity_rule_id: str | None = None
    identity_evidence: dict | None = None
    identity_verified_at: str | None = None
    merged_at: str | None = None
    merged_at_provenance: str | None = None
    analysis_output: str | None = None
    test_manifest: list[dict] | None = None
    work_dir: str | None = None
    enqueued_at: str = ""
    updated_at: str = ""
    state_changed_at: str = ""
    error_reason: str | None = None
    retry_count: int = 0
    cleanup_attempts: int = 0
    history: list[dict] = field(default_factory=list)

    def binding_summary(self) -> str:
        if self.pr_number is None:
            return "未核实（无 PR 绑定）"
        return (
            f"PR {self.pr_number} @ pytorch {self.pytorch_commit} / "
            f"torch-npu {self.torch_npu_commit}"
        )


def _row_to_task(row: sqlite3.Row) -> Task:
    evidence = row["identity_evidence"]
    manifest = row["test_manifest"]
    history = row["history"]
    return Task(
        task_id=row["task_id"],
        state=row["state"],
        obs_wheels_url=row["obs_wheels_url"],
        obs_source_url=row["obs_source_url"],
        obs_wheels_npu_url=row["obs_wheels_npu_url"],
        obs_wheels_npu_version=row["obs_wheels_npu_version"],
        obs_wheels_version=row["obs_wheels_version"],
        obs_source_version=row["obs_source_version"],
        obs_remote_time=row["obs_remote_time"],
        expiry_rule=row["expiry_rule"],
        expires_at=row["expires_at"],
        pr_number=row["pr_number"],
        pytorch_commit=row["pytorch_commit"],
        torch_npu_commit=row["torch_npu_commit"],
        torch_npu_location=row["torch_npu_location"],
        identity_rule_id=row["identity_rule_id"],
        identity_evidence=json.loads(evidence) if evidence else None,
        identity_verified_at=row["identity_verified_at"],
        merged_at=(row["merged_at"] if "merged_at" in row.keys() else None),
        merged_at_provenance=(row["merged_at_provenance"] if "merged_at_provenance" in row.keys() else None),
        analysis_output=row["analysis_output"],
        test_manifest=json.loads(manifest) if manifest else None,
        work_dir=row["work_dir"],
        enqueued_at=row["enqueued_at"],
        updated_at=row["updated_at"],
        state_changed_at=row["state_changed_at"],
        error_reason=row["error_reason"],
        retry_count=row["retry_count"],
        cleanup_attempts=row["cleanup_attempts"],
        history=json.loads(history) if history else [],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_task_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"t-{stamp}-{secrets.token_hex(3)}"


def _object_key(
    wheels_url: str, wheels_version: str | None,
    source_url: str, source_version: str | None,
    npu_url: str | None = None, npu_version: str | None = None,
) -> str:
    key = f"{wheels_url}|{wheels_version or ''}|{source_url}|{source_version or ''}"
    if npu_url or npu_version:
        key += f"|{npu_url or ''}|{npu_version or ''}"
    return key


def validate_manifest(manifest: Any) -> list[dict]:
    """校验函数/device 清单：非空、格式合法、组合唯一。"""
    if not isinstance(manifest, list) or not manifest:
        raise QueueError(
            "测试清单必须是非空列表（没有清单不得回退全量测试）"
        )
    seen: set[tuple[str, str]] = set()
    for item in manifest:
        if not isinstance(item, dict):
            raise QueueError(f"清单项必须是对象: {item!r}")
        try:
            tf = validate_test_function(item.get("test_function"))
            dev = validate_device(item.get("device"))
        except IdentifierError as e:
            raise QueueError(f"清单项非法（{item!r}）: {e}") from e
        if (tf, dev) in seen:
            raise QueueError(f"清单内 ({tf}, {dev}) 重复")
        seen.add((tf, dev))
    normalized = []
    for item in manifest:
        row = {"test_function": item["test_function"], "device": item["device"]}
        for key in ("source_test_function", "runtime_nodeid", "runtime_nodeids", "execution_mode", "device_basis"):
            if key in item:
                if key == "runtime_nodeids":
                    if (not isinstance(item[key], list) or not item[key]
                            or not all(isinstance(v, str) and v.strip() for v in item[key])):
                        raise QueueError(f"清单字段 {key} 必须是非空字符串列表")
                    row[key] = list(dict.fromkeys(item[key]))
                else:
                    if not isinstance(item[key], str) or not item[key].strip():
                        raise QueueError(f"清单字段 {key} 必须是非空字符串")
                    row[key] = item[key]
        normalized.append(row)
    return normalized


# ---------------------------------------------------------------------------
# QueueStore
# ---------------------------------------------------------------------------

class QueueStore:
    def __init__(self, config: dict):
        self.config = config
        self.paths = store_paths(config)
        self.paths.queue_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.paths.queue_db
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            self._migrate(conn)
        finally:
            conn.close()

    # 拆分包格式新增列；CREATE TABLE IF NOT EXISTS 不会改已有库，逐列补齐。
    _MIGRATE_COLUMNS = {
        "tasks": {
            "obs_wheels_npu_url": "TEXT",
            "obs_wheels_npu_version": "TEXT",
            "merged_at": "TEXT",
            "merged_at_provenance": "TEXT",
        },
    }

    def _migrate(self, conn: sqlite3.Connection) -> None:
        for table, columns in self._MIGRATE_COLUMNS.items():
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def _connect(self) -> sqlite3.Connection:
        return connect(self.db_path)

    # -- 候选登记与去重 ---------------------------------------------------

    def register_candidate(
        self,
        obs_wheels_url: str,
        obs_source_url: str,
        *,
        obs_wheels_version: str | None = None,
        obs_source_version: str | None = None,
        obs_wheels_npu_url: str | None = None,
        obs_wheels_npu_version: str | None = None,
        obs_remote_time: str | None = None,
        expiry_rule: str | None = None,
    ) -> tuple[Task, bool]:
        """登记一个 OBS 对象组合为候选任务。

        拆分包格式下 obs_wheels_npu_url 为独立 torch_npu 包（必填时缺省
        由调用方负责——对象不齐时调用方应等待，同 obs_source_url 约定）。
        同一 (URL, 版本标识) 组合重复登记不产生新任务（只刷新 last_seen）。
        登记后立即处于 identity_pending：候选不是已获准的测试任务。
        返回 (task, created)。
        """
        if not isinstance(obs_wheels_url, str) or not obs_wheels_url.strip():
            raise QueueError("obs_wheels_url 不能为空")
        if not isinstance(obs_source_url, str) or not obs_source_url.strip():
            raise QueueError(
                "obs_source_url 不能为空；对象不齐时调用方应等待两个对象都出现后再登记"
            )
        key = _object_key(obs_wheels_url, obs_wheels_version,
                          obs_source_url, obs_source_version,
                          obs_wheels_npu_url, obs_wheels_npu_version)
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT task_id FROM obs_objects WHERE object_key = ?", (key,)
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE obs_objects SET last_seen_at = ?, remote_time = ? "
                    "WHERE object_key = ?",
                    (now, obs_remote_time, key),
                )
                conn.execute("COMMIT")
                task = self.get_task(row["task_id"])
                assert task is not None
                return task, False
            task_id = _new_task_id()
            history = [
                {"from": None, "to": STATE_CANDIDATE, "at": now, "reason": "对象登记"},
                {"from": STATE_CANDIDATE, "to": STATE_IDENTITY_PENDING, "at": now,
                 "reason": "等待按已确认规则核实包身份"},
            ]
            conn.execute(
                """
                INSERT INTO tasks
                    (task_id, state, obs_wheels_url, obs_source_url,
                     obs_wheels_npu_url, obs_wheels_npu_version,
                     obs_wheels_version, obs_source_version, obs_remote_time,
                     expiry_rule, expires_at, enqueued_at, updated_at,
                     state_changed_at, history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
                """,
                (
                    task_id, STATE_IDENTITY_PENDING,
                    obs_wheels_url, obs_source_url,
                    obs_wheels_npu_url, obs_wheels_npu_version,
                    obs_wheels_version, obs_source_version, obs_remote_time,
                    expiry_rule, now, now, now,
                    json.dumps(history, ensure_ascii=False),
                ),
            )
            conn.execute(
                """
                INSERT INTO obs_objects
                    (object_key, first_seen_at, last_seen_at, remote_time, task_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, now, now, obs_remote_time, task_id),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        task = self.get_task(task_id)
        assert task is not None
        return task, True

    # -- 身份核实门禁（文档第 1 节） ---------------------------------------

    def register_identity_rule(self, rule_id: str, rule_text: str, source: str) -> None:
        """登记用户/OBS 负责人确认的判定规则原文与来源。"""
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise QueueError("rule_id 不能为空")
        if not isinstance(rule_text, str) or not rule_text.strip():
            raise QueueError("rule_text 不能为空（规则原文）")
        if not isinstance(source, str) or not source.strip():
            raise QueueError("source 不能为空（规则来源：谁在何时确认）")
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO identity_rules
                    (rule_id, rule_text, source, confirmed_at)
                VALUES (?, ?, ?, ?)
                """,
                (rule_id.strip(), rule_text.strip(), source.strip(), _now()),
            )
        finally:
            conn.close()

    def list_identity_rules(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM identity_rules ORDER BY confirmed_at"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def verify_identity(
        self, task_id: str, rule_id: str, evidence: dict
    ) -> Task:
        """用已确认的规则核实候选身份，写入绑定信息。

        - 规则未注册 → IdentityGateError（强制门禁，绝不默认放行）；
        - evidence 必须含 pr_number、完整 pytorch_commit、完整 torch_npu_commit
          与绑定依据说明——即“包与 PR、commit 的对应信息”。
        """
        conn = self._connect()
        try:
            rule = conn.execute(
                "SELECT * FROM identity_rules WHERE rule_id = ?", (rule_id,)
            ).fetchone()
            if rule is None:
                raise IdentityGateError(
                    f"判定规则 {rule_id!r} 未登记：OBS 负责人尚未确认“包对应已合入 "
                    "PyTorch main 的 commit”的判定规则。按文档第 1 节，本次不放行；"
                    "请先由用户提供规则原文与来源后 register_identity_rule。"
                )
            task = self._get_task_conn(conn, task_id)
            if task.state != STATE_IDENTITY_PENDING:
                raise QueueError(
                    f"任务 {task_id} 处于 {task.state}，只有 identity_pending 状态可核实身份"
                )
            if not isinstance(evidence, dict):
                raise QueueError("evidence 必须是对象")
            pr_number = evidence.get("pr_number")
            if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
                raise QueueError("evidence.pr_number 必须是正整数")
            try:
                pytorch_commit = validate_commit(
                    evidence.get("pytorch_commit"), "evidence.pytorch_commit"
                )
            except IdentifierError as e:
                raise QueueError(str(e)) from e
            # torch_npu_commit 允许缺省：git-main 驱动方案下，torch-npu 随
            # wheels 包体自带，精确 commit 由执行端下载解析后回写任务
            #（见 set_task_torch_npu）；若显式提供则仍严格校验。
            torch_npu_commit = evidence.get("torch_npu_commit")
            if torch_npu_commit is not None:
                try:
                    torch_npu_commit = validate_commit(
                        torch_npu_commit, "evidence.torch_npu_commit"
                    )
                except IdentifierError as e:
                    raise QueueError(str(e)) from e
            binding = evidence.get("binding")
            if not isinstance(binding, (str, dict)) or (
                isinstance(binding, str) and not binding.strip()
            ) or (isinstance(binding, dict) and not binding):
                raise QueueError(
                    "evidence.binding 必须说明对象↔PR↔commit 的绑定依据"
                )
            now = _now()
            ev = dict(evidence)
            ev["pytorch_commit"] = pytorch_commit
            ev["torch_npu_commit"] = torch_npu_commit
            ev["rule_id"] = rule_id
            ev["rule_source"] = rule["source"]
            self._transition_conn(
                conn, task, STATE_PREFILTER_PENDING,
                reason=f"身份已按规则 {rule_id} 核实",
                extra_set={
                    "pr_number": pr_number,
                    "pytorch_commit": pytorch_commit,
                    "torch_npu_commit": torch_npu_commit,
                    "torch_npu_location": evidence.get("torch_npu_location"),
                    "identity_rule_id": rule_id,
                    "identity_evidence": json.dumps(ev, ensure_ascii=False),
                    "identity_verified_at": now,
                },
            )
        finally:
            conn.close()
        task = self.get_task(task_id)
        assert task is not None
        return task

    # -- 过滤与分析 ---------------------------------------------------------

    def prefilter_outcome(self, task_id: str, included: bool, note: str | None = None) -> Task:
        """记录既有脚本过滤结果。

        注意：过滤执行端与 batch_input 的对接待确认（文档 4.2：输入来源未定，
        依赖 GitHub token 的脚本不得直接运行）。本方法仅供输入来源确认后的
        分析端按既有规则调用，不包含任何过滤逻辑本身。
        """
        target = STATE_PENDING_ANALYSIS if included else STATE_FILTERED_OUT
        conn = self._connect()
        try:
            task = self._get_task_conn(conn, task_id)
            if task.state != STATE_PREFILTER_PENDING:
                raise QueueError(
                    f"任务 {task_id} 处于 {task.state}，只有 prefilter_pending 可记录过滤结果"
                )
            self._transition_conn(
                conn, task, target,
                reason=note or ("进入分析队列" if included else "被既有前置规则排除"),
            )
        finally:
            conn.close()
        task = self.get_task(task_id)
        assert task is not None
        return task

    def submit_manifest(
        self, task_id: str, manifest: list[dict], analysis_output: str | None = None
    ) -> Task:
        """AI 分析产物：明确的函数/device 清单。清单就绪才可进入待测试。"""
        conn = self._connect()
        try:
            task = self._get_task_conn(conn, task_id)
            if task.state != STATE_PENDING_ANALYSIS:
                raise QueueError(
                    f"任务 {task_id} 处于 {task.state}，只有 pending_analysis 可提交清单"
                )
            manifest = validate_manifest(manifest)
            self._transition_conn(
                conn, task, STATE_PENDING_TEST,
                reason="函数/device 清单就绪",
                extra_set={
                    "test_manifest": json.dumps(manifest, ensure_ascii=False),
                    "analysis_output": analysis_output,
                },
            )
        finally:
            conn.close()
        task = self.get_task(task_id)
        assert task is not None
        return task

    # -- 串行执行 -----------------------------------------------------------

    def claim_next(self) -> Task | None:
        """取出下一个待测试任务并锁定（同一时间只允许一个活动任务）。

        事务内检查 executor_lock 与活动状态任务，保证多进程并发下
        也只有一个任务进入 准备/测试/写库/清理。按入队先后取最旧任务。
        """
        now = _now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            lock = conn.execute(
                "SELECT task_id FROM executor_lock WHERE id = 1"
            ).fetchone()
            active = conn.execute(
                f"SELECT task_id FROM tasks WHERE state IN "
                f"({','.join('?' * len(ACTIVE_STATES))})",
                tuple(ACTIVE_STATES),
            ).fetchall()
            if lock is not None or active:
                conn.execute("ROLLBACK")
                return None
            row = conn.execute(
                "SELECT * FROM tasks WHERE state = ? "
                "ORDER BY enqueued_at, task_id LIMIT 1",
                (STATE_PENDING_TEST,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            task = _row_to_task(row)
            work_dir = str(self.paths.task_work_root / task.task_id)
            self._transition_conn(
                conn, task, STATE_PREPARING,
                reason="轮到本任务，锁定版本与 OBS 对象",
                extra_set={"work_dir": work_dir},
            )
            conn.execute(
                "INSERT OR REPLACE INTO executor_lock (id, task_id, acquired_at) "
                "VALUES (1, ?, ?)",
                (task.task_id, now),
            )
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        task = self.get_task(task.task_id)
        assert task is not None
        return task

    def transition(
        self, task_id: str, new_state: str, reason: str | None = None,
        extra_set: dict[str, Any] | None = None,
    ) -> Task:
        """按合法迁移表变更任务状态（终态任务不可再迁移）。"""
        conn = self._connect()
        try:
            task = self._get_task_conn(conn, task_id)
            self._transition_conn(conn, task, new_state, reason, extra_set)
            if new_state in (STATE_DONE, STATE_FAILED):
                conn.execute("DELETE FROM executor_lock WHERE id = 1")
        finally:
            conn.close()
        task = self.get_task(task_id)
        assert task is not None
        return task

    def mark_blocked(self, task_id: str, reason: str) -> Task:
        if not isinstance(reason, str) or not reason.strip():
            raise QueueError("blocked 原因不能为空")
        return self.transition(task_id, STATE_BLOCKED, reason)

    def mark_expired(self, task_id: str, reason: str) -> Task:
        return self.transition(task_id, STATE_EXPIRED, reason)

    def bump_retry(self, task_id: str) -> Task:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE tasks SET retry_count = retry_count + 1, updated_at = ? "
                "WHERE task_id = ?",
                (_now(), task_id),
            )
        finally:
            conn.close()
        task = self.get_task(task_id)
        assert task is not None
        return task

    def bump_cleanup_attempt(self, task_id: str) -> int:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE tasks SET cleanup_attempts = cleanup_attempts + 1, updated_at = ? "
                "WHERE task_id = ?",
                (_now(), task_id),
            )
            row = conn.execute(
                "SELECT cleanup_attempts FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return int(row["cleanup_attempts"]) if row else 0
        finally:
            conn.close()

    def requeue(self, task_id: str, target_state: str, reason: str | None = None) -> Task:
        """人工确认后把 blocked 任务放回指定等待阶段。"""
        if target_state not in REQUEUE_TARGETS:
            raise QueueError(
                f"requeue 目标必须是 {sorted(REQUEUE_TARGETS)}，收到 {target_state!r}"
            )
        return self.transition(task_id, target_state, reason)

    def recover_interrupted(self, *, _lease_owned: bool = False) -> list[Task]:
        from .process_lock import executor_lease
        if _lease_owned:
            return self._recover_interrupted_owned()
        with executor_lease(self.paths.queue_db.with_suffix(".executor.lock")) as acquired:
            return self._recover_interrupted_owned() if acquired else []

    def _recover_interrupted_owned(self) -> list[Task]:
        """进程重启后的状态核实（文档第 8 节）。

        - preparing/testing/writing 中断 → blocked（原因注明中断，人工核实后
          requeue），避免重复安装/双重执行/重复累计结果；
        - cleaning 中断 → 保持 cleaning（清理完成前不开始下一任务），
          释放锁行，由执行器在下一轮先续清理；
        - 锁行一并清理（持有进程已不存在）。
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM executor_lock WHERE id = 1")
            recovered: list[Task] = []
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE state IN "
                f"({','.join('?' * len(ACTIVE_STATES))})",
                tuple(ACTIVE_STATES),
            ).fetchall()
            for row in rows:
                task = _row_to_task(row)
                if task.state == STATE_CLEANING:
                    # 保持 cleaning：run_once 会先继续清理。
                    continue
                self._transition_conn(
                    conn, task, STATE_BLOCKED,
                    reason=(
                        f"进程重启时任务处于 {task.state}；已中断，需人工核实残留进程与"
                        f"状态后 requeue，避免重复安装或双重执行"
                    ),
                )
                recovered.append(self._get_task_conn(conn, task.task_id))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return recovered

    # -- 查询 ---------------------------------------------------------------

    def get_task(self, task_id: str) -> Task | None:
        conn = self._connect()
        try:
            return self._get_task_conn(conn, task_id)
        finally:
            conn.close()

    def _get_task_conn(self, conn: sqlite3.Connection, task_id: str) -> Task | None:
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise QueueError(f"任务不存在: {task_id}")
        return _row_to_task(row)

    def list_tasks(self, state: str | None = None) -> list[Task]:
        conn = self._connect()
        try:
            if state is not None:
                if state not in ALL_STATES:
                    raise QueueError(f"未知状态 {state!r}")
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE state = ? ORDER BY enqueued_at, task_id",
                    (state,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY enqueued_at, task_id"
                ).fetchall()
            return [_row_to_task(r) for r in rows]
        finally:
            conn.close()

    def list_active_tasks(self) -> list[Task]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE state IN "
                f"({','.join('?' * len(ACTIVE_STATES))})",
                tuple(ACTIVE_STATES),
            ).fetchall()
            return [_row_to_task(r) for r in rows]
        finally:
            conn.close()

    # -- 上游 commit 记录与镜像状态（2026-09-17 方案） --------------------

    def record_upstream_commits(self, commits: list[dict]) -> tuple[int, int]:
        """记录一轮 fetch 发现的新 commit。

        commits 项：{sha, pr_number|None, commit_time, subject, discovered_via}。
        已存在的 sha 跳过（幂等）。返回 (记录数, 无 PR 跳过数)。
        无 PR 号的 commit 按用户确认完全跳过：不入队，仅记一条 no_pr 状态。
        """
        now = _now()
        recorded = no_pr = 0
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            for c in commits:
                exists = conn.execute(
                    "SELECT 1 FROM upstream_commits WHERE sha = ?", (c["sha"],)
                ).fetchone()
                if exists:
                    continue
                if c.get("pr_number") is None:
                    conn.execute(
                        """
                        INSERT INTO upstream_commits
                            (sha, pr_number, commit_time, subject,
                             discovered_at, discovered_via, enqueue_state, note)
                        VALUES (?, NULL, ?, ?, ?, ?, 'no_pr', ?)
                        """,
                        (c["sha"], c["commit_time"], c["subject"], now,
                         c.get("discovered_via", "hourly-fetch"),
                         "直接 push、无 PR 信息，按约定跳过"),
                    )
                    no_pr += 1
                else:
                    conn.execute(
                        """
                        INSERT INTO upstream_commits
                            (sha, pr_number, commit_time, subject,
                             discovered_at, discovered_via, enqueue_state)
                        VALUES (?, ?, ?, ?, ?, ?, 'pending')
                        """,
                        (c["sha"], c["pr_number"], c["commit_time"], c["subject"],
                         now, c.get("discovered_via", "hourly-fetch")),
                    )
                    recorded += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        return recorded, no_pr

    def get_upstream_commit(self, sha: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM upstream_commits WHERE sha = ?", (sha,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def pending_upstream_commits(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM upstream_commits WHERE enqueue_state = 'pending' "
                "ORDER BY commit_time, sha"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def list_upstream_commits(self, state: str | None = None) -> list[dict]:
        conn = self._connect()
        try:
            if state:
                rows = conn.execute(
                    "SELECT * FROM upstream_commits WHERE enqueue_state = ? "
                    "ORDER BY commit_time DESC", (state,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM upstream_commits ORDER BY commit_time DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_upstream_commit(self, sha: str, *, state: str,
                               task_id: str | None = None, note: str | None = None) -> None:
        now = _now()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE upstream_commits SET enqueue_state = ?, task_id = ?, "
                "note = COALESCE(?, note), last_checked_at = ? WHERE sha = ?",
                (state, task_id, note, now, sha),
            )
        finally:
            conn.close()

    def get_mirror_tip(self) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT last_tip FROM mirror_state WHERE id = 1"
            ).fetchone()
            return row["last_tip"] if row else None
        finally:
            conn.close()

    def set_mirror_tip(self, repo_url: str, tip: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO mirror_state (id, repo_url, last_tip, last_fetch_at) "
                "VALUES (1, ?, ?, ?)",
                (repo_url, tip, _now()),
            )
        finally:
            conn.close()

    def set_task_merged_at(self, task_id: str, merged_at: str | None,
                           provenance: str | None = None) -> None:
        """登记任务的 main 合入时刻（upstream-main-commit-time 等来源）。

        merged_at 供提交侧基线合入序比较使用；重测/重试沿用已登记值，
        不随执行过程改写。"""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE tasks SET merged_at = ?, merged_at_provenance = ?, "
                "updated_at = ? WHERE task_id = ?",
                (merged_at, provenance, _now(), task_id),
            )
        finally:
            conn.close()

    def set_task_torch_npu(self, task_id: str, commit: str | None,
                           location: str | None = None) -> Task:
        """执行端从包内 torch_npu wheel 解析出精确版本后回写任务。"""
        if commit is not None:
            try:
                commit = validate_commit(commit, "torch_npu_commit")
            except IdentifierError as e:
                raise QueueError(str(e)) from e
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE tasks SET torch_npu_commit = ?, torch_npu_location = ?, "
                "updated_at = ? WHERE task_id = ?",
                (commit, location, _now(), task_id),
            )
        finally:
            conn.close()
        task = self.get_task(task_id)
        assert task is not None
        return task

    # -- skill 检出记录（文档 4.1：任务记录应保存本次使用的 skill commit） ----

    def set_skill_commit(self, commit: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO skill_state (id, skill_commit, updated_at) "
                "VALUES (1, ?, ?)",
                (commit, _now()),
            )
        finally:
            conn.close()

    def get_skill_commit(self) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT skill_commit FROM skill_state WHERE id = 1"
            ).fetchone()
            return row["skill_commit"] if row else None
        finally:
            conn.close()

    # -- 内部 ----------------------------------------------------------------

    def _transition_conn(
        self,
        conn: sqlite3.Connection,
        task: Task,
        new_state: str,
        reason: str | None = None,
        extra_set: dict[str, Any] | None = None,
    ) -> None:
        if new_state not in ALL_STATES:
            raise QueueError(f"未知状态 {new_state!r}")
        if task.state in TERMINAL_STATES:
            raise InvalidTransitionError(
                f"任务 {task.task_id} 已处于终态 {task.state}"
            )
        allowed = TRANSITIONS.get(task.state, frozenset())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"任务 {task.task_id} 不允许 {task.state} → {new_state} "
                f"（合法目标: {sorted(allowed)}）"
            )
        now = _now()
        history = list(task.history)
        history.append(
            {"from": task.state, "to": new_state, "at": now, "reason": reason}
        )
        sets = [
            "state = ?", "updated_at = ?", "state_changed_at = ?",
            "history = ?",
        ]
        params: list[Any] = [new_state, now, now, json.dumps(history, ensure_ascii=False)]
        if reason is not None:
            sets.append("error_reason = ?")
            params.append(reason)
        if extra_set:
            for key, value in extra_set.items():
                sets.append(f"{key} = ?")
                params.append(value)
        params.append(task.task_id)
        conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE task_id = ?", params
        )
