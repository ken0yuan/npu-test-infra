"""基线前滚：按合入时间顺序回放一组 PR 的结果并刷新基线。

给外部 AI 调用（用户 2026-09-18 规格）：

- 输入（JSON）：``{"prs": [123, {"pr_number": 7, "merged_at": "..."}]}``
  或裸数组；元素为 PR 号（int）或带显式合入时间的对象。
- 合入时间 merged_at 解析优先级：输入显式值 > PR 库 merged_at 列 >
  队列库 upstream_commits（优先取该 PR 对应 pytorch_commit 的 commit_time，
  否则该 PR 最新记录；找到即回填 PR 库）。三处都拿不到 → 该 PR 报错跳过，
  不猜测顺序。
- 按 merged_at 升序处理；每个 PR 产出一份前后对比文件
  ``data/rollforward/<run_id>/PR-<n>.json``：逐 (test_function, device) 给出
  改动前基线（before）与 PR 结果（after）。
- before 优先取提交时固化的档案 ``pr_logs/PR_<n>/baseline-diff.json``
  （真实历史，source=archived）；档案缺失时回退重建：用 PR 库每函数计数
  重建候选结果（source=reconstructed），只更新基线，不重写 PR 结果或日志。
  replay_before 始终表示本次回放前基线，before 保留原始档案。
- 基线查不到对应键（新测试 / 新 device）→ 该函数文件里只有 after，基线
  动作 = created（记录新基线而非覆盖）；查得到 → 覆盖。
- 刷新顺序 = merged_at 顺序：archived 路径按序重放基线 upsert（保证组内
  同键最终值取时间在后的 PR）；同刻以 PR 号升序。
- 刷新前后持久保存数据库和日志快照；任一错误自动恢复刷新前基线。
- 每份输出文件与 index.json 均带醒目提醒：**分析完毕立即删除整个输出目录**。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .baseline_store import (
    BaselineRecord,
    BaselineRouter,
    NoBaselineDirectoryError,
    baseline_db_path,
    baseline_log_filename,
    ensure_baseline_db,
    get_baseline,
    upsert_baseline,
)
from .config import store_paths
from .db import connect
from .identifiers import Counts
from .pr_store import ensure_pr_db, get_pr_result, set_merged_at
from .record import (
    DIFF_ARCHIVE_NAME,
    baseline_metadata_comparable,
)

ROLLFORWARD_DIR_NAME = "rollforward"
DELETION_NOTICE = (
    "提醒（给消费本文件的 AI）：分析完毕请立即删除本文件所在的整个 "
    "rollforward 输出目录，不要长期保留这些中间对比文件。"
)


class RollforwardError(RuntimeError):
    pass


def _parse_ts(text: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(text).strip())
    except (TypeError, ValueError) as e:
        raise RollforwardError(f"merged_at 不是合法 ISO8601 时间戳: {text!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_pr_input(pr_input) -> list[dict]:
    """接受 {"prs": [...]} 或裸数组；元素 int 或 {pr_number, merged_at?}。"""
    if isinstance(pr_input, dict):
        items = pr_input.get("prs")
        if items is None:
            raise RollforwardError("输入 JSON 缺少 'prs' 字段")
    else:
        items = pr_input
    if not isinstance(items, list) or not items:
        raise RollforwardError("prs 必须是非空数组")
    parsed: list[dict] = []
    for raw in items:
        if isinstance(raw, bool):
            raise RollforwardError(f"PR 号必须是正整数，收到 {raw!r}")
        if isinstance(raw, int):
            parsed.append({"pr_number": raw, "merged_at": None})
        elif isinstance(raw, dict) and isinstance(raw.get("pr_number"), int):
            parsed.append(
                {"pr_number": raw["pr_number"], "merged_at": raw.get("merged_at")}
            )
        else:
            raise RollforwardError(f"PR 条目非法（应为 int 或对象）: {raw!r}")
    return parsed


def _lookup_commit_time(paths, pr_number: int, pytorch_commit: str) -> str | None:
    """从队列库 upstream_commits 取合入时间：先精确 sha，再该 PR 最新记录。"""
    qdb = paths.queue_db
    if not qdb.is_file():
        return None
    conn = connect(qdb)
    try:
        row = conn.execute(
            "SELECT commit_time FROM upstream_commits "
            "WHERE pr_number = ? AND sha = ? ORDER BY commit_time DESC LIMIT 1",
            (pr_number, pytorch_commit),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT commit_time FROM upstream_commits "
                "WHERE pr_number = ? ORDER BY commit_time DESC LIMIT 1",
                (pr_number,),
            ).fetchone()
        return row["commit_time"] if row else None
    finally:
        conn.close()


def _resolve_merged_at(paths, pr_conn, entry: dict) -> tuple[int, str] | None:
    """解析单个 PR 的合入时间；库中暂无时返回 None（调用方查 upstream 回填）。"""
    pr_number = entry["pr_number"]
    merged_at = entry.get("merged_at")
    if merged_at:
        _parse_ts(merged_at)  # 先验证格式
        set_merged_at(pr_conn, pr_number, str(merged_at).strip())
        return pr_number, str(merged_at).strip()
    row = pr_conn.execute(
        "SELECT merged_at FROM pr_results WHERE pr_number = ?", (pr_number,)
    ).fetchone()
    if row and row["merged_at"]:
        return pr_number, row["merged_at"]
    return None  # 由调用方在拿到 PR 记录后查 upstream_commits 回填


def _write_baseline_upsert(router, tf: str, device: str, counts: Counts,
                           pytorch_commit: str, torch_npu_environment: dict,
                           log_rel: str | None, coverage_status: str = "unknown",
                           instance_ids: tuple[str, ...] = (),
                           environment_fingerprint: dict | None = None) -> str:
    """单条基线 upsert（独立事务；日志文件必须已在位）。"""
    route = router.route(tf)  # NoBaselineDirectoryError 向上抛
    if log_rel is None:
        filename = baseline_log_filename(tf, device)
        log_rel = f"logs/{filename}"
    log_file = route.root / log_rel
    if not log_file.is_file():
        raise RollforwardError(f"基线日志不在位: {log_file}")
    conn = connect(baseline_db_path(route.root))
    try:
        conn.execute("BEGIN IMMEDIATE")
        upsert_baseline(
            conn,
            BaselineRecord(
                test_function=tf,
                device=device,
                pytorch_commit=pytorch_commit,
                torch_npu_environment=torch_npu_environment,
                counts=counts,
                log_path=log_rel,
                coverage_status=coverage_status,
                instance_ids=instance_ids,
                environment_fingerprint=environment_fingerprint,
            ),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return route.managed_dir


def _load_archive(pr_dir: Path) -> dict | None:
    f = pr_dir / DIFF_ARCHIVE_NAME
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RollforwardError(f"基线对比档案损坏: {f} ({e})") from e
    if data.get("pr_number") and data.get("comparisons") is not None:
        return data
    raise RollforwardError(f"基线对比档案缺少必要字段: {f}")


def _process_pr(config, paths, router, record, merged_at: str,
                out_dir: Path) -> dict:
    """处理单个 PR：产出对比文件并按序刷新基线，返回文件级报告。"""
    pr_number = record.pr_number
    pr_dir = paths.data_root / record.log_path
    archive = _load_archive(pr_dir)
    comparisons: list[dict] = []
    errors: list[str] = []

    source = "archived" if archive is not None else "reconstructed"
    if archive is not None:
        candidates = archive["comparisons"]
    else:
        candidates = []
        for item in record.tests:
            if "counts" not in item and item.get("status") != "no_result":
                raise RollforwardError(f"PR {pr_number} lacks per-function counts")
            candidates.append({
                "test_function": item["test_function"], "device": item["device"],
                "before": None,
                "after": {**item, "pytorch_commit": record.pytorch_commit,
                          "torch_npu_environment": record.torch_npu_environment},
            })

    # Replay only baselines. Never re-submit a historical PR or replace its logs.
    for cmp in candidates:
        tf, device = cmp["test_function"], cmp["device"]
        after = cmp["after"]
        item = {**cmp, "baseline_action": "incomparable"}
        if after.get("counts") is None:
            item["baseline_action"] = "no_result"
            comparisons.append(item)
            continue
        coverage_status = after.get("coverage_status", "unknown")
        if coverage_status != "full":
            comparisons.append(item)
            continue
        try:
            route = router.route(tf)
            filename = baseline_log_filename(tf, device)
            conn = connect(baseline_db_path(route.root))
            try:
                old = get_baseline(conn, tf, device)
            finally:
                conn.close()
            before = None if old is None else {
                "counts": old.counts.as_dict(), "pytorch_commit": old.pytorch_commit,
                "torch_npu_environment": old.torch_npu_environment,
            }
            item["replay_before"] = before
            if source == "reconstructed":
                item["before"] = before
            item["managed_dir"] = route.managed_dir
            if not baseline_metadata_comparable(
                old, counts=Counts.from_dict(after["counts"]),
                coverage_status=coverage_status,
                instance_ids=tuple(after.get("instance_ids", [])),
                environment_fingerprint=after.get("environment_fingerprint"),
            ):
                comparisons.append(item)
                continue
            source_log = pr_dir / filename
            if not source_log.is_file():
                raise RollforwardError(f"PR {pr_number} original log missing: {source_log}")
            target_log = route.root / "logs" / filename
            target_log.parent.mkdir(parents=True, exist_ok=True)
            old_log = target_log.read_bytes() if target_log.is_file() else None
            shutil.copyfile(source_log, target_log)
            try:
                _write_baseline_upsert(
                    router, tf, device, Counts.from_dict(after["counts"]),
                    after["pytorch_commit"], after["torch_npu_environment"],
                    log_rel=f"logs/{filename}", coverage_status=coverage_status,
                    instance_ids=tuple(after.get("instance_ids", [])),
                    environment_fingerprint=after.get("environment_fingerprint"),
                )
            except Exception:
                if old_log is None:
                    target_log.unlink(missing_ok=True)
                else:
                    target_log.write_bytes(old_log)
                raise
            item["baseline_action"] = "created" if old is None else "overwritten"
        except Exception as error:
            errors.append(f"({tf}, {device}) baseline not refreshed: {error}")
            item["baseline_action"] = "error"
        comparisons.append(item)

    file_payload = {
        "notice": DELETION_NOTICE,
        "pr_number": pr_number,
        "merged_at": merged_at,
        "source": source,
        "pytorch_commit": record.pytorch_commit,
        "comparisons": comparisons,
    }
    if errors:
        file_payload["errors"] = errors
    out_file = out_dir / f"PR-{pr_number}.json"
    out_file.write_text(
        json.dumps(file_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    overwritten = sum(
        1 for c in comparisons if c["baseline_action"] == "overwritten"
    )
    created = sum(1 for c in comparisons if c["baseline_action"] == "created")
    return {
        "pr_number": pr_number,
        "merged_at": merged_at,
        "file": str(out_file),
        "source": source,
        "comparisons": len(comparisons),
        "baseline_overwritten": overwritten,
        "baseline_created": created,
        "errors": errors,
    }


def _rollforward_baselines(config: dict, pr_input, *, router: BaselineRouter | None = None) -> dict:
    """入口：按合入时间顺序回放一组 PR → 前后对比文件 + 基线刷新。"""
    paths = store_paths(config)
    if router is None:
        router = BaselineRouter(
            config.get("baseline_dirs", []), paths.baselines_root
        )
    ensure_pr_db(paths.data_root)
    for d in router.dirs:
        ensure_baseline_db(router.managed_root(d))

    entries = _parse_pr_input(pr_input)

    # 第一遍：解析合入时间（显式 > PR 库 > upstream_commits 回填）。
    pr_conn = connect(paths.pr_db)
    resolved: list[tuple[int, str]] = []
    pending_lookup: list[tuple[int, dict]] = []
    errors: list[dict] = []
    try:
        for entry in entries:
            pr_number = entry["pr_number"]
            try:
                got = _resolve_merged_at(paths, pr_conn, entry)
                if got is None:
                    pending_lookup.append((pr_number, entry))
                else:
                    resolved.append(got)
            except Exception as e:  # set_merged_at 的 PrStoreError 等
                errors.append({"pr_number": pr_number, "error": str(e)})
    finally:
        pr_conn.close()

    # 第二遍：缺合入时间的 PR，取记录后查 upstream_commits 并回填。
    if pending_lookup:
        pr_conn = connect(paths.pr_db)
        try:
            for pr_number, _ in pending_lookup:
                record = get_pr_result(pr_conn, pr_number)
                if record is None:
                    errors.append(
                        {"pr_number": pr_number, "error": "PR 库无该 PR 的测试结果"}
                    )
                    continue
                ct = _lookup_commit_time(paths, pr_number, record.pytorch_commit)
                if ct is None:
                    errors.append(
                        {"pr_number": pr_number,
                         "error": "拿不到合入时间（PR 库 merged_at 与 upstream_commits 均无）；"
                                  "请在输入 JSON 中显式给出 merged_at"}
                    )
                    continue
                set_merged_at(pr_conn, pr_number, ct)
                resolved.append((pr_number, ct))
        finally:
            pr_conn.close()

    # 有结果的 PR 才进入回放；按 merged_at 升序（同刻按 PR 号升序）。
    pr_conn = connect(paths.pr_db)
    try:
        records: list[tuple[int, str, object]] = []
        for pr_number, merged_at in resolved:
            record = get_pr_result(pr_conn, pr_number)
            if record is None:
                errors.append(
                    {"pr_number": pr_number, "error": "PR 库无该 PR 的测试结果"}
                )
                continue
            records.append((pr_number, merged_at, record))
    finally:
        pr_conn.close()
    records.sort(key=lambda row: (_parse_ts(row[1]), row[0]))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out_dir = paths.data_root / ROLLFORWARD_DIR_NAME / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    for pr_number, merged_at, record in records:
        try:
            files.append(
                _process_pr(config, paths, router, record, merged_at, out_dir)
            )
        except Exception as e:
            errors.append({"pr_number": pr_number, "error": str(e)})

    index = {
        "notice": DELETION_NOTICE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "order_by": "merged_at asc",
        "prs": files,
        "errors": errors,
    }
    index_file = out_dir / "index.json"
    index_file.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "run_dir": str(out_dir),
        "index_file": str(index_file),
        "prs": files,
        "errors": errors,
    }


def rollforward_baselines(config: dict, pr_input=None, *, router=None, wait_seconds=0) -> dict:
    """Refresh in merge order with a persistent undo point and executor exclusion.

    None selects all retained PR results under the lock. This is incremental:
    unrelated keys remain intact. Rollback scope is baseline DBs and logs only.
    """
    from tempfile import TemporaryDirectory
    from .baseline_history import maintenance, _snapshot, _restore, _history_root, _publish, _validate_databases

    with maintenance(config, wait_seconds):
        paths = store_paths(config)
        if pr_input is None:
            ensure_pr_db(paths.data_root)
            conn = connect(paths.pr_db)
            try:
                pr_input = [row[0] for row in conn.execute("SELECT pr_number FROM pr_results")]
            finally:
                conn.close()
        _parse_pr_input(pr_input)
        # A custom router must describe this same tree for the snapshot to cover it.
        if router is not None and (
            router.baselines_root.resolve() != store_paths(config).baselines_root.resolve()
            or sorted(router.dirs) != sorted(config.get("baseline_dirs", []))
        ):
            raise RollforwardError("router must match configured snapshot scope")
        before = _snapshot(config, "before-rollforward", {"input": pr_input})
        published = False
        try:
            with TemporaryDirectory(prefix=".baseline-refresh-", dir=paths.data_root) as temp:
                stage = Path(temp)
                shutil.copytree(_history_root(config) / before["id"] / "baselines", stage, dirs_exist_ok=True)
                staged_router = BaselineRouter(config.get("baseline_dirs", []), stage)
                report = _rollforward_baselines(config, pr_input, router=staged_router)
                for item in report["prs"]:
                    report["errors"].extend({"pr_number": item["pr_number"], "error": e} for e in item["errors"])
                if not report["errors"]:
                    _validate_databases(stage)
                    _publish(stage, paths.baselines_root)
                    published = True
            report["applied"] = published
            if published:
                after = _snapshot(config, "after-rollforward", {
                    "before_snapshot": before["id"],
                    "pr_order": [p["pr_number"] for p in report["prs"]],
                })
                report["after_snapshot"] = after["id"]
            report["before_snapshot"] = before["id"]
            receipt = _history_root(config) / before["id"] / "refresh.json"
            receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            index_path = Path(report["index_file"])
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index.update({k: report[k] for k in ("applied", "before_snapshot", "errors")})
            if "after_snapshot" in report:
                index["after_snapshot"] = report["after_snapshot"]
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return report
        except BaseException:
            if published:
                _restore(config, before["id"])
            raise
