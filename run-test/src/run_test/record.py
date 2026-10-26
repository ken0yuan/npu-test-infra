"""收到一份完整测试结果后的记录与刷新流程。

严格对应《本地测试结果与基线存储说明》第 6 节的六步顺序：

1. 校验输入；
2. 读取旧基线并对比（先读、对比，再覆盖；没有记录即首次建立基线）；
3. 准备日志（新日志全部准备成功之前，不破坏任何旧日志）；
4. 更新 PR 记录（按 PR 号插入或替换，计数为本次汇总，不累加）；
5. 逐项刷新基线（按函数与 device 插入或替换，版本/环境/日志一并更新）；
6. 核对完成情况（PR 汇总等于本次函数结果之和；未运行的函数与其他
   device 的原记录不改变）。

一致性策略（满足第 6 节末段约束）：
- 基线日志先写到同目录 ``*.tmp.<时间戳>``，全部写成功后统一替换正式名，
  任一失败即清理本次 tmp 并中止——旧日志不可能被未准备成功的新日志破坏；
- 日志全部就位后才提交数据库事务，因此不会出现“新数据库记录配旧日志”；
  中断只会留下“新日志配旧记录”，重放同一份完整提交即可幂等修复
  （重复提交得到相同计数与基线内容，不翻倍）；
- 执行中断、缺失函数结果、日志缺失不是完整提交，直接拒绝，不伪造计数；
  no_result 是这一原则下的显式例外通道：status="no_result" + counts=None
  + 必附 reason 与日志，如实声明"该函数本次未取得结果"。它不产生任何
  计数、不刷新基线、不覆盖基线日志，仅在 PR 库与对比档案里记为无信号
  （"无结果"≠"四个零"）。

  **全部结果均为 no_result 的提交**（2026-09-20 定稿，与 executor 行为
  对齐——此前本 docstring 写"应拒绝、执行端应 blocked"，与
  executor.py 的实际实现相反，属文档漂移）：仍然按正常六步流程持久化。
  理由（executor.py all-no-result 注释同源）：PR 记录和原始 artifacts
  是排障证据，不能因全部未取得就丢弃真实错误线索；提交不刷新任何基线
  （no_result 行天然不刷），也不产生任何四计数。可区分性由
  ``no_result_count`` 列保障：0 pass + 0 fail + no_result_count>0 =
  "没测到"而非"全过"。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline_store import (
    BaselineRecord,
    BaselineRouter,
    NoBaselineDirectoryError,
    baseline_db_path,
    baseline_log_filename,
    baseline_logs_dir,
    ensure_baseline_db,
    get_baseline,
    resolve_baseline_log,
    upsert_baseline,
)
from .config import PR_LOGS_DIR_NAME, store_paths
from .db import connect
from .identifiers import (
    Counts,
    validate_commit,
    validate_device,
    validate_test_function,
    validate_torch_npu_environment,
)
from .pr_store import (
    PrResultRecord,
    ensure_pr_db,
    get_pr_result,
    pr_db_path,
    upsert_pr_result,
)

SUPERSEDED_DIR_NAME = ".superseded"
STAGING_PREFIX = ".staging."


class RecordError(ValueError):
    pass


class SubmissionValidationError(RecordError):
    """提交不完整或不符合字段契约，拒绝写入。"""


@dataclass
class FunctionResult:
    """一个函数在一个 device 下的本次结果及其日志。

    status="ok"：counts 为本次四计数，参与 PR 汇总并刷新基线；
    status="no_result"：本次未取得结果（收集失败/执行中断/超时），
    counts 必须为 None 且必附 reason 与日志——不计数、不刷新基线、
    不覆盖基线日志，仅作无信号记录（不伪造四个零）。
    """

    test_function: str
    device: str
    counts: Counts | None
    # 日志内容二选一：直接文本，或指向已生成日志文件的路径。
    log_content: str | None = None
    log_source_path: str | Path | None = None
    status: str = "ok"
    reason: str | None = None
    # 已知环境失败签名（triage.py）函数级并集；空/None = 无已知签名。
    triage: list[str] | None = None
    execution_status: str = "completed"
    # Coverage is unknown unless the runner supplies the complete instance set
    # and comparable execution evidence. A caller-provided count is never
    # sufficient to infer source-function coverage.
    coverage_status: str = "unknown"
    instance_ids: tuple[str, ...] = ()
    environment_fingerprint: dict | None = None
    observed_counts: Counts | None = None
    artifact_ref: str | None = None
    artifact_refs: tuple[str, ...] = ()
    event_reports: list[dict] | None = None
    # Build identity observed by the child process. This is provenance only;
    # it is deliberately excluded from the stable environment comparison key.
    provenance: dict | None = None

    @property
    def is_no_result(self) -> bool:
        return self.status == "no_result"

    @property
    def known_signatures(self) -> list[str]:
        return list(self.triage or [])

    def log_text(self) -> str:
        if self.log_content is not None:
            if not self.log_content.strip():
                raise SubmissionValidationError(
                    f"({self.test_function}, {self.device}) 的日志内容为空"
                )
            return self.log_content
        if self.log_source_path is not None:
            p = Path(self.log_source_path)
            if not p.is_file():
                raise SubmissionValidationError(
                    f"({self.test_function}, {self.device}) 的日志源文件不存在: {p}"
                )
            text = p.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                raise SubmissionValidationError(
                    f"({self.test_function}, {self.device}) 的日志源文件为空: {p}"
                )
            return text
        raise SubmissionValidationError(
            f"({self.test_function}, {self.device}) 缺少日志（log_content 或 log_source_path）"
        )


@dataclass
class Submission:
    """一份完整测试提交：PR 号、精确版本、torch-npu 环境与全部函数结果。

    PR 库的四个计数不单独提供，由 function_results 汇总计算。
    """

    pr_number: int
    pytorch_commit: str
    torch_npu_environment: dict
    function_results: list[FunctionResult] = field(default_factory=list)
    # 本次结果对应的 PR 日志目录来源（整目录复制到 pr_logs/PR_<n>/）。
    pr_log_source_dir: str | Path | None = None


def _normalized_instance_ids(ids) -> tuple[str, ...]:
    return tuple(sorted(set(ids or ())))

def baseline_metadata_comparable(old_record: BaselineRecord | None, *, counts: Counts | None,
                                 coverage_status: str, instance_ids,
                                 environment_fingerprint: dict | None) -> bool:
    if counts is None or coverage_status != "full":
        return False
    new_ids = _normalized_instance_ids(instance_ids)
    if not new_ids or environment_fingerprint is None:
        return False
    if old_record is None:
        return True
    if old_record.coverage_status != "full":
        return False
    old_ids = _normalized_instance_ids(old_record.instance_ids)
    if not old_ids or old_record.environment_fingerprint is None:
        return False
    if old_ids != new_ids:
        return False
    old_env = old_record.environment_fingerprint
    return old_env == environment_fingerprint

def eligible_baseline_keys(diffs: list["BaselineDiff"]) -> set[tuple[str, str]]:
    return {(d.test_function, d.device) for d in diffs if d.comparable}


@dataclass
class BaselineDiff:
    """一条 (test_function, device) 的旧基线与新计数对比。

    new_counts=None 表示本次 no_result：无信号，不刷新基线。
    new_triage：本次结果命中的已知环境签名（triage.py）；old_triage 为
    旧基线日志命中的签名（D 阶段对比：同签名 = 基线已知环境失败）。
    """

    test_function: str
    device: str
    old_record: BaselineRecord | None
    new_counts: Counts | None
    managed_dir: str
    reason: str | None = None
    new_triage: list[str] = field(default_factory=list)
    old_triage: list[str] = field(default_factory=list)
    new_coverage_status: str = "unknown"
    new_instance_ids: tuple[str, ...] = ()
    new_environment_fingerprint: dict | None = None

    @property
    def is_first_baseline(self) -> bool:
        return self.old_record is None

    @property
    def comparable(self) -> bool:
        return baseline_metadata_comparable(
            self.old_record, counts=self.new_counts,
            coverage_status=self.new_coverage_status,
            instance_ids=self.new_instance_ids,
            environment_fingerprint=self.new_environment_fingerprint,
        )

    def summary(self) -> str:
        head = f"({self.test_function}, {self.device})"
        if self.new_counts is None:
            return f"{head} 无信号（本次未取得结果: {self.reason}；基线不动）"
        if self.old_record is None:
            suffix = (
                f"（含已知环境签名: {', '.join(self.new_triage)}）"
                if self.new_triage else ""
            )
            action = "首次建立基线" if self.comparable else "结果已保存但不可建立可比基线"
            return f"{head} {action} {self.new_counts.as_dict()}{suffix}"
        old, new = self.old_record.counts, self.new_counts
        if not self.comparable:
            return f"{head} 结果已保存但不可比，基线不动（覆盖={self.new_coverage_status}）"
        # A category match is only a coarse label. Without per-case identity
        # and traceback evidence it must never be promoted to "no new signal".
        if old == new and self.new_triage and set(self.new_triage) <= set(self.old_triage):
            return (
                f"{head} 失败类别与基线相同（{', '.join(self.new_triage)}），"
                "但缺少逐用例证据，无法判断是否同一失败"
            )
        note = ""
        if self.new_triage:
            new_sigs = set(self.new_triage) - set(self.old_triage)
            if old == new and new_sigs:
                note = f"（新增签名: {', '.join(sorted(new_sigs))}）"
            else:
                note = f"（含已知环境签名: {', '.join(self.new_triage)}）"
        return f"{head} 基线刷新 {old.as_dict()} -> {new.as_dict()}{note}"


@dataclass
class SubmitReport:
    pr_number: int
    pr_log_dir: str
    totals: Counts
    # no_result 条数（不产生四计数，独立可区分"没测到"vs"全过"）。
    no_result_count: int = 0
    diffs: list[BaselineDiff] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    # 基线前后对比档案（pr_logs/PR_<n>/baseline-diff.json）相对路径。
    diff_archive: str = ""


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")


# ---------------------------------------------------------------------------
# 步骤 1：校验输入
# ---------------------------------------------------------------------------

def validate_submission(sub: Submission, router: BaselineRouter) -> None:
    """完整性与字段契约校验；任一问题抛 SubmissionValidationError。"""
    if isinstance(sub.pr_number, bool) or not isinstance(sub.pr_number, int) \
            or sub.pr_number <= 0:
        raise SubmissionValidationError(f"pr_number 必须是正整数，收到 {sub.pr_number!r}")
    try:
        validate_commit(sub.pytorch_commit, "pytorch_commit")
        validate_torch_npu_environment(sub.torch_npu_environment)
    except ValueError as e:
        raise SubmissionValidationError(str(e)) from e

    if not sub.function_results:
        raise SubmissionValidationError("一份完整提交必须包含至少一个函数结果")

    seen: set[tuple[str, str]] = set()
    for fr in sub.function_results:
        if not isinstance(fr, FunctionResult):
            raise SubmissionValidationError(f"function_results 项类型非法: {fr!r}")
        try:
            tf = validate_test_function(fr.test_function)
            dev = validate_device(fr.device)
        except ValueError as e:
            raise SubmissionValidationError(str(e)) from e
        if fr.is_no_result:
            # 无结果：counts 必须为 None（带了计数 = 冒充完整结果），
            # reason 必填（无结果的原因不得缺失）。
            if fr.counts is not None:
                raise SubmissionValidationError(
                    f"({tf}, {dev}) 声明 no_result 但 counts 非 None，"
                    "无结果不得携带计数"
                )
            if not (fr.reason or "").strip():
                raise SubmissionValidationError(
                    f"({tf}, {dev}) no_result 必须附 reason（未取得结果的原因）"
                )
        elif not isinstance(fr.counts, Counts):
            raise SubmissionValidationError(
                f"({tf}, {dev}) 的 counts 必须是 Counts（非负整数四计数）"
            )
        key = (tf, dev)
        if key in seen:
            raise SubmissionValidationError(
                f"同一份完整提交内 ({tf}, {dev}) 出现多次，不得重复累加"
            )
        seen.add(key)
        # 日志可用性：no_result 也必须附日志（证据）；内容为空/缺失即拒绝。
        fr.log_text()
        try:
            router.route(tf)
        except NoBaselineDirectoryError as e:
            raise SubmissionValidationError(str(e)) from e

    if sub.pr_log_source_dir is None:
        raise SubmissionValidationError("缺少 PR 日志目录来源（pr_log_source_dir）")
    src = Path(sub.pr_log_source_dir)
    if not src.is_dir():
        raise SubmissionValidationError(f"PR 日志来源不是目录: {src}")
    if not any(src.iterdir()):
        raise SubmissionValidationError(f"PR 日志来源目录为空: {src}")


# ---------------------------------------------------------------------------
# 步骤 2：读取旧基线并对比
# ---------------------------------------------------------------------------

def read_old_baselines(
    router: BaselineRouter, sub: Submission
) -> list[BaselineDiff]:
    """按每个 (test_function, device) 读取原记录并生成对比（只读，不写）。

    有旧记录时同时分类旧基线日志（triage 签名）——D 阶段信号分级：
    计数不变且签名集合未新增 = 基线已知环境失败（无新信号）。日志按
    整体 + 逐行双粒度分类（C05 类复合签名可能跨行）。
    """
    from .triage import classify_texts

    diffs: list[BaselineDiff] = []
    for fr in sub.function_results:
        route = router.route(fr.test_function)
        conn = connect(baseline_db_path(route.root))
        try:
            old = get_baseline(conn, fr.test_function, fr.device)
        finally:
            conn.close()
        old_triage: list[str] = []
        if old is not None and old.log_path:
            try:
                log_text = resolve_baseline_log(route.root, old).read_text(
                    encoding="utf-8", errors="replace"
                )
                old_triage = classify_texts([log_text, *log_text.splitlines()])
            except OSError:
                old_triage = []
        diffs.append(
            BaselineDiff(
                test_function=fr.test_function,
                device=fr.device,
                old_record=old,
                new_counts=fr.counts,
                managed_dir=route.managed_dir,
                reason=fr.reason,
                new_triage=fr.known_signatures,
                old_triage=old_triage,
                new_coverage_status=fr.coverage_status,
                new_instance_ids=tuple(fr.instance_ids),
                new_environment_fingerprint=fr.environment_fingerprint,
            )
        )
    return diffs


# ---------------------------------------------------------------------------
# 步骤 3：准备日志
# ---------------------------------------------------------------------------

def _prepare_pr_logs(paths, sub: Submission) -> str:
    """把本次 PR 日志复制到 pr_logs/PR_<n>/，返回相对 data_root 的路径。

    顺序保证：先完整复制到 staging；成功后才移走旧目录（保留策略为
    supersede_keep，旧内容移入 .superseded/，不删除、不混读），最后原子
    改名。任何失败都不会破坏旧日志。
    """
    pr_logs_root = paths.pr_logs_root
    pr_logs_root.mkdir(parents=True, exist_ok=True)
    final_name = f"PR_{sub.pr_number}"
    final = pr_logs_root / final_name
    staging = pr_logs_root / f"{STAGING_PREFIX}{final_name}.{_ts()}"

    shutil.copytree(Path(sub.pr_log_source_dir), staging)
    superseded_to: Path | None = None
    try:
        if final.exists():
            superseded_to = (
                pr_logs_root / SUPERSEDED_DIR_NAME / final_name / _ts()
            )
            superseded_to.parent.mkdir(parents=True, exist_ok=True)
            final.rename(superseded_to)
        staging.rename(final)
    except Exception:
        # 回滚：把旧目录放回原位，清理 staging。
        if superseded_to is not None and superseded_to.exists() and not final.exists():
            superseded_to.rename(final)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return f"{PR_LOGS_DIR_NAME}/{final_name}"


def _prepare_baseline_logs(
    router: BaselineRouter, sub: Submission,
    eligible_keys: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], tuple[str, str]]:
    """为每个取得结果的 (test_function, device) 写新基线日志。

    no_result 不写基线日志——本次未取得结果，旧基线日志（有效证据）
    不被环境失败日志覆盖；其日志只留在 PR 日志目录。
    全部先写成 ``logs/<名字>.tmp.<时间戳>``，全部成功后统一替换正式名；
    任一失败清理本次全部 tmp 并抛错——旧基线日志不会被破坏。
    返回 {(tf, device): (managed_dir, 相对库目录的 log_path)}。
    """
    ts = _ts()
    staged: list[tuple[Path, Path]] = []  # (tmp, final)
    result: dict[tuple[str, str], tuple[str, str]] = {}
    try:
        for fr in sub.function_results:
            if fr.is_no_result or fr.coverage_status != "full":
                continue
            if eligible_keys is not None and (fr.test_function, fr.device) not in eligible_keys:
                continue
            route = router.route(fr.test_function)
            logs_dir = baseline_logs_dir(route.root)
            logs_dir.mkdir(parents=True, exist_ok=True)
            filename = baseline_log_filename(fr.test_function, fr.device)
            final = logs_dir / filename
            tmp = logs_dir / f"{filename}.tmp.{ts}"
            tmp.write_text(fr.log_text(), encoding="utf-8")
            staged.append((tmp, final))
            result[(fr.test_function, fr.device)] = (route.managed_dir, filename)
        # 全部 tmp 写成功，统一替换正式名（覆盖旧版本日志）。
        for tmp, final in staged:
            os.replace(tmp, final)
    except Exception:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise
    return result


# ---------------------------------------------------------------------------
# 步骤 4 + 5：更新 PR 记录、逐项刷新基线
# ---------------------------------------------------------------------------

def _metadata_fields(fr: FunctionResult) -> dict:
    """Serialize execution evidence consistently for complete and no_result rows."""
    item = {}
    if fr.execution_status != "completed":
        item["execution_status"] = fr.execution_status
    if fr.coverage_status != "full":
        item["coverage_status"] = fr.coverage_status
    if fr.instance_ids:
        item["instance_ids"] = list(_normalized_instance_ids(fr.instance_ids))
    if fr.environment_fingerprint is not None:
        item["environment_fingerprint"] = fr.environment_fingerprint
    if fr.observed_counts is not None:
        item["observed_counts"] = fr.observed_counts.as_dict()
    refs = tuple(fr.artifact_refs) or ((fr.artifact_ref,) if fr.artifact_ref else ())
    if refs:
        item["artifact_refs"] = list(refs)
    if fr.event_reports:
        item["event_reports"] = fr.event_reports
    if fr.provenance is not None:
        item["provenance"] = fr.provenance
    return item


def _pr_tests_items(function_results: list[FunctionResult]) -> list[dict]:
    """PR 库 tests 列表，完整结果和 no_result 都保留执行证据。"""
    items: list[dict] = []
    for fr in function_results:
        item = {"test_function": fr.test_function, "device": fr.device}
        if fr.is_no_result:
            item["status"] = "no_result"
            item["reason"] = fr.reason
        else:
            item["counts"] = fr.counts.as_dict()
            if fr.known_signatures:
                item["triage"] = fr.known_signatures
        item.update(_metadata_fields(fr))
        items.append(item)
    return items


def _write_databases(
    paths, router: BaselineRouter, sub: Submission, pr_log_rel: str,
    baseline_logs: dict[tuple[str, str], tuple[str, str]],
    diffs: list[BaselineDiff] | None = None,
) -> None:
    # PR 汇总只计取得结果的函数；no_result 不产生任何计数。
    totals = Counts.sum(
        fr.counts for fr in sub.function_results if not fr.is_no_result
    )
    # no_result 条数独立成列：0 pass + 0 fail + no_result=N = "没测到"
    # 而非"全过"（2026-09-20 增列，起因 PR 197081 全 no_result 记 done、
    # 计数全 0 无法辨别）。
    no_result_count = sum(1 for fr in sub.function_results if fr.is_no_result)
    # 每函数计数随 tests 一并入 PR 库：基线前滚脚本靠它按 PR 复现结果。
    tests = _pr_tests_items(sub.function_results)

    # 步骤 4：更新 PR 记录（先于基线，顺序与文档一致）。
    pr_conn = connect(pr_db_path(paths.data_root))
    try:
        pr_conn.execute("BEGIN IMMEDIATE")
        upsert_pr_result(
            pr_conn,
            PrResultRecord(
                pr_number=sub.pr_number,
                pytorch_commit=sub.pytorch_commit,
                torch_npu_environment=sub.torch_npu_environment,
                tests=tests,
                counts=totals,
                no_result_count=no_result_count,
                log_path=pr_log_rel,
            ),
        )
        pr_conn.execute("COMMIT")
    except Exception:
        pr_conn.execute("ROLLBACK")
        raise
    finally:
        pr_conn.close()

    # 步骤 5：逐项刷新基线（no_result 不刷新——本次无结果，旧基线保留），
    # 按库分组各自一个事务。
    by_root: dict[Path, list[FunctionResult]] = {}
    comparable_keys = eligible_baseline_keys(diffs or [])
    for fr in sub.function_results:
        if fr.is_no_result or fr.coverage_status != "full":
            continue
        if diffs is not None and (fr.test_function, fr.device) not in comparable_keys:
            continue
        route = router.route(fr.test_function)
        by_root.setdefault(route.root, []).append(fr)
    for root, frs in by_root.items():
        conn = connect(baseline_db_path(root))
        try:
            conn.execute("BEGIN IMMEDIATE")
            for fr in frs:
                _, filename = baseline_logs[(fr.test_function, fr.device)]
                upsert_baseline(
                    conn,
                    BaselineRecord(
                        test_function=fr.test_function,
                        device=fr.device,
                        pytorch_commit=sub.pytorch_commit,
                        torch_npu_environment=sub.torch_npu_environment,
                        counts=fr.counts,
                        log_path=f"logs/{filename}",
                        coverage_status=fr.coverage_status,
                        instance_ids=fr.instance_ids,
                        environment_fingerprint=fr.environment_fingerprint,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 提交时物化基线前后对比（基线前滚脚本的档案数据源）
# ---------------------------------------------------------------------------

DIFF_ARCHIVE_NAME = "baseline-diff.json"


def _write_baseline_diff_archive(
    paths, pr_log_rel: str, sub: Submission, diffs: list[BaselineDiff]
) -> str:
    """把步骤 2 的旧基线对比固化为 pr_logs/PR_<n>/baseline-diff.json。

    这是该 PR 真实的"改动前基线 vs 本次结果"档案：基线前滚脚本优先用它
    生成前后对比文件（重建模式只作回退，before 可能已被后续 PR 刷新）。
    """
    comparisons = []
    for d in diffs:
        if d.old_record is None:
            before = None
        else:
            before = {
                "counts": d.old_record.counts.as_dict(),
                "pytorch_commit": d.old_record.pytorch_commit,
                "torch_npu_environment": d.old_record.torch_npu_environment,
                "coverage_status": d.old_record.coverage_status,
                "instance_ids": list(d.old_record.instance_ids),
                "environment_fingerprint": d.old_record.environment_fingerprint,
            }
            if d.old_triage:
                before["triage"] = list(d.old_triage)
        if d.new_counts is None:
            # no_result：基线不动（未创建也未覆盖），档案里如实记无信号。
            fr_meta = next((fr for fr in sub.function_results if fr.test_function == d.test_function and fr.device == d.device), None)
            after = {
                "counts": None,
                "status": "no_result",
                "reason": d.reason,
                "execution_status": fr_meta.execution_status if fr_meta else "runner_error",
                "pytorch_commit": sub.pytorch_commit,
                "torch_npu_environment": sub.torch_npu_environment,
            }
            if fr_meta is not None:
                after.update(_metadata_fields(fr_meta))
            comparisons.append(
                {
                    "test_function": d.test_function,
                    "device": d.device,
                    "managed_dir": d.managed_dir,
                    "baseline_action": "no_result",
                    "before": before,
                    "after": after,
                }
            )
            continue
        fr_meta = next((fr for fr in sub.function_results if fr.test_function == d.test_function and fr.device == d.device), None)
        action = ("created" if before is None else "overwritten") if d.comparable else "incomparable"
        after = {
            "counts": d.new_counts.as_dict(),
            "coverage_status": fr_meta.coverage_status if fr_meta else "unknown",
            "instance_ids": list(fr_meta.instance_ids) if fr_meta else [],
            "environment_fingerprint": fr_meta.environment_fingerprint if fr_meta else None,
            "pytorch_commit": sub.pytorch_commit,
            "torch_npu_environment": sub.torch_npu_environment,
        }
        if d.new_triage:
            after["triage"] = list(d.new_triage)
        comparisons.append(
            {
                "test_function": d.test_function,
                "device": d.device,
                "managed_dir": d.managed_dir,
                "baseline_action": action,
                "before": before,
                "after": after,
            }
        )
    payload = {
        "pr_number": sub.pr_number,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
    }
    diff_file = paths.data_root / pr_log_rel / DIFF_ARCHIVE_NAME
    diff_file.parent.mkdir(parents=True, exist_ok=True)
    diff_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return f"{pr_log_rel}/{DIFF_ARCHIVE_NAME}"


# ---------------------------------------------------------------------------
# 步骤 6：核对完成情况
# ---------------------------------------------------------------------------

def _verify(
    paths, router: BaselineRouter, sub: Submission, pr_log_rel: str,
    diffs: list[BaselineDiff],
) -> dict[str, Any]:
    """六步收尾核对。no_result 行核对"基线未被触碰"（与步骤 2 读值一致）。"""
    checks: dict[str, Any] = {}
    totals = Counts.sum(
        fr.counts for fr in sub.function_results if not fr.is_no_result
    )

    pr_conn = connect(pr_db_path(paths.data_root))
    try:
        saved = get_pr_result(pr_conn, sub.pr_number)
    finally:
        pr_conn.close()
    checks["pr_row_count"] = 1 if saved else 0
    checks["pr_counts_match_sum"] = bool(saved and saved.counts == totals)
    checks["pr_tests_match"] = bool(
        saved and saved.tests == _pr_tests_items(sub.function_results)
    )
    checks["pr_log_dir_present"] = (paths.data_root / pr_log_rel).is_dir()

    old_by_key = {
        (d.test_function, d.device): d.old_record for d in diffs
    }
    baseline_ok = True
    for fr in sub.function_results:
        route = router.route(fr.test_function)
        conn = connect(baseline_db_path(route.root))
        try:
            rec = get_baseline(conn, fr.test_function, fr.device)
        finally:
            conn.close()
        diff = next(d for d in diffs if (d.test_function, d.device) == (fr.test_function, fr.device))
        if not diff.comparable:
            # no_result, partial coverage, unknown metadata or changed environment:
            # preserve the old baseline exactly and report the run separately.
            old = old_by_key.get((fr.test_function, fr.device))
            ok = (rec is None and old is None) or (
                rec is not None and old is not None and rec == old
            )
        else:
            ok = bool(
                rec
                and rec.counts == fr.counts
                and rec.coverage_status == fr.coverage_status
                and _normalized_instance_ids(rec.instance_ids) == _normalized_instance_ids(fr.instance_ids)
                and rec.environment_fingerprint == fr.environment_fingerprint
                and rec.pytorch_commit == sub.pytorch_commit
                and rec.torch_npu_environment == sub.torch_npu_environment
                and resolve_baseline_log(route.root, rec).is_file()
            )
        if not ok:
            baseline_ok = False
    checks["baselines_match_submission"] = baseline_ok
    return checks


# ---------------------------------------------------------------------------
# 完整流程入口
# ---------------------------------------------------------------------------

def submit_complete_result(
    config: dict, sub: Submission, router: BaselineRouter | None = None
) -> SubmitReport:
    """按第 6 节六步顺序处理一份完整提交；幂等：重复提交结果不变、不翻倍。"""
    paths = store_paths(config)
    if router is None:
        router = BaselineRouter(
            config.get("baseline_dirs", []), paths.baselines_root
        )

    # 1. 校验输入。
    validate_submission(sub, router)

    # 库就位（幂等；PR 库与各最小目录基线库）。
    ensure_pr_db(paths.data_root)
    for d in router.dirs:
        ensure_baseline_db(router.managed_root(d))

    # 2. 读取旧基线并对比。
    diffs = read_old_baselines(router, sub)

    # 3. 准备日志（PR 日志 + 每函数/device 基线日志；失败不破坏旧日志）。
    pr_log_rel = _prepare_pr_logs(paths, sub)
    try:
        eligible_keys = {(d.test_function, d.device) for d in diffs if d.comparable}
        baseline_logs = _prepare_baseline_logs(router, sub, eligible_keys)
    except Exception:
        # 基线日志准备失败：PR 日志已复制，但不写任何数据库记录。
        raise

    # 4 + 5. 更新 PR 记录、逐项刷新基线。
    _write_databases(paths, router, sub, pr_log_rel, baseline_logs, diffs)

    # 4.5 物化基线前后对比档案（步骤 2 的 diffs；供基线前滚脚本按 PR 取用）。
    diff_archive = _write_baseline_diff_archive(paths, pr_log_rel, sub, diffs)

    # 6. 核对完成情况（含 no_result 行的"基线未触碰"核对）。
    checks = _verify(paths, router, sub, pr_log_rel, diffs)
    report = SubmitReport(
        pr_number=sub.pr_number,
        pr_log_dir=pr_log_rel,
        totals=Counts.sum(
            fr.counts for fr in sub.function_results if not fr.is_no_result
        ),
        no_result_count=sum(1 for fr in sub.function_results if fr.is_no_result),
        diffs=diffs,
        checks=checks,
        diff_archive=diff_archive,
    )
    if not all(
        [
            checks["pr_row_count"] == 1,
            checks["pr_counts_match_sum"],
            checks["pr_tests_match"],
            checks["pr_log_dir_present"],
            checks["baselines_match_submission"],
        ]
    ):
        raise RecordError(
            f"写库后核对未通过（pr={sub.pr_number}）: {checks}；"
            "请重放同一份完整提交以修复中断状态"
        )
    return report
