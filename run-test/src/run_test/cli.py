"""命令行入口。

用法：python -m run_test.cli <子命令> …

默认配置文件为项目根目录的 settings.json。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import task_queue as q
from .baseline_store import BaselineRouter
from .config import (
    CONFIG_FILE_NAME,
    default_config,
    load_config,
    save_config,
    store_paths,
)
from .environment import collect_environment, validate_environment, write_environment_file
from .executor import Executor
from .identifiers import Counts, validate_commit
from .obs import make_downloader
from .pr_store import ensure_pr_db
from .task_queue import QueueStore
from .record import FunctionResult, Submission, submit_complete_result


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / CONFIG_FILE_NAME


def _load_config(path: str | None) -> dict:
    return load_config(Path(path) if path else _default_config_path())


def cmd_obs_scan(args):
    from .artifacts import inventory
    config = _load_config(args.config)
    downloader = make_downloader(config.get("obs"))
    if downloader is None:
        raise ValueError("obs.downloader 未配置")
    bucket = f"obs://{config['obs']['bucket']}/7day-cache"
    report = inventory(downloader.list(bucket + "/"), bucket, config)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"{report['objects']} 个对象，{len(report['unique_pairs'])} 对唯一匹配，{len(report['ambiguous'])} 项待选构建；已写入 {target}")
    else:
        print(text)
    return 0


def cmd_results(args):
    import dataclasses
    from .db import connect
    from .pr_store import get_pr_result, list_pr_results
    conn = connect(store_paths(_load_config(args.config)).pr_db)
    try:
        rows = [get_pr_result(conn, args.pr)] if args.pr else list_pr_results(conn)
        print(json.dumps([dataclasses.asdict(row) for row in rows if row is not None],
                         ensure_ascii=False, indent=2, default=str))
    finally:
        conn.close()
    return 0


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config) if args.config else _default_config_path()
    if cfg_path.is_file():
        config = load_config(cfg_path)
        print(f"已存在配置: {cfg_path}")
    else:
        config = default_config()
        cfg_dir = Path(__file__).resolve().parents[3]
        config["data_root"] = str(cfg_path.resolve().parent / "data" / "runtime")
        cfg_path = save_config(config, cfg_path)
        print(f"已写入默认配置: {cfg_path}")

    paths = store_paths(config)
    paths.data_root.mkdir(parents=True, exist_ok=True)
    paths.pr_logs_root.mkdir(parents=True, exist_ok=True)
    paths.baselines_root.mkdir(parents=True, exist_ok=True)
    paths.task_work_root.mkdir(parents=True, exist_ok=True)
    ensure_pr_db(paths.data_root)
    router = BaselineRouter(config.get("baseline_dirs", []), paths.baselines_root)
    made = router.ensure_all()
    QueueStore(config)  # 建队列库
    print(f"数据根目录: {paths.data_root}")
    print(f"PR 结果库: {paths.pr_db}")
    print(f"队列库: {paths.queue_db}")
    if made:
        print(f"基线库: {len(made)} 个最小管理目录")
    else:
        print("基线库: 尚未配置最小管理目录（用 baseline add 登记，不自动扫描仓库）")
    return 0


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------

def cmd_env(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    paths = store_paths(config)
    if args.write:
        path = write_environment_file(paths.data_root)
        env = json.loads(path.read_text(encoding="utf-8"))
    else:
        env = collect_environment()
    print(json.dumps(env, ensure_ascii=False, indent=2))
    problems = validate_environment(env)
    if problems:
        print("\n[注意] 以下字段未采集到（保持 null，禁止猜测值）:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    if args.write:
        print(f"\n已写入: {paths.env_file}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

def cmd_baseline(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config) if args.config else _default_config_path()
    config = load_config(cfg_path)
    paths = store_paths(config)

    if args.baseline_cmd == "list":
        dirs = config.get("baseline_dirs", [])
        if not dirs:
            print("（空）尚未登记最小管理目录")
        for d in dirs:
            db = paths.baselines_root / d / "baseline.db"
            print(f"{d}  -> {db}{' [已建库]' if db.is_file() else ' [未建库]'}")
        return 0

    if args.baseline_cmd == "add":
        new_dir = args.dir.strip().strip("/")
        dirs = list(config.get("baseline_dirs", []))
        if new_dir in dirs:
            print(f"已存在: {new_dir}")
            return 0
        # 先验证不与前缀重叠，再落盘。
        BaselineRouter(dirs + [new_dir], paths.baselines_root)
        dirs.append(new_dir)
        config["baseline_dirs"] = dirs
        save_config(config, cfg_path)
        router = BaselineRouter(dirs, paths.baselines_root)
        db = router.ensure_all()
        print(f"已登记 {new_dir}，基线库: {db[-1] if isinstance(db, list) else db}")
        return 0

    if args.baseline_cmd == "remove":
        dirs = list(config.get("baseline_dirs", []))
        if args.dir in dirs:
            dirs.remove(args.dir)
            config["baseline_dirs"] = dirs
            save_config(config, cfg_path)
            print(f"已从清单移除 {args.dir}（已存在的数据库文件不动）")
        else:
            print(f"清单中不存在: {args.dir}")
        return 0

    if args.baseline_cmd == "import":
        from .baseline_import import import_baseline

        if not config.get("baseline_dirs"):
            print("拒绝导入：baseline_dirs 为空，函数将无归属（先 baseline add 登记）",
                  file=sys.stderr)
            return 1
        npu_commit = validate_commit(args.npu_commit, "torch_npu_commit")
        report = import_baseline(
            config, args.json, args.src, npu_commit,
            router=BaselineRouter(config["baseline_dirs"], paths.baselines_root),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(
            f"\n导入完成: 函数级 {report['files_imported']}/{report['files_total']} 文件，"
            f"文件级标记 {report['files_file_status']} 个，函数记录 {report['functions']} 条"
            f"（device=cuda，重复导入按本次值覆盖）"
        )
        return 0
    if args.baseline_cmd in {"history", "snapshot", "rollback"}:
        from .baseline_history import list_snapshots, snapshot_baselines, rollback_baselines
        if args.baseline_cmd == "history":
            report = list_snapshots(config)
        elif args.baseline_cmd == "snapshot":
            report = snapshot_baselines(config, args.reason, wait_seconds=args.wait_seconds)
        else:
            report = rollback_baselines(config, args.snapshot_id, wait_seconds=args.wait_seconds)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.baseline_cmd == "rollforward":
        from .baseline_rollforward import DELETION_NOTICE, rollforward_baselines

        pr_input = json.loads(Path(args.input).read_text(encoding="utf-8")) if args.input else None
        report = rollforward_baselines(
            config, pr_input, wait_seconds=args.wait_seconds,
            router=BaselineRouter(config["baseline_dirs"], paths.baselines_root),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        print(f"\n{DELETION_NOTICE}", file=sys.stderr)
        return 1 if report["errors"] else 0
    return 1


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------

def _queue_cmd_register(store: QueueStore, args) -> int:
    task, created = store.register_candidate(
        args.wheels_url, args.source_url,
        obs_wheels_version=args.wheels_version,
        obs_source_version=args.source_version,
        obs_remote_time=args.remote_time,
    )
    print(f"task_id={task.task_id} state={task.state} created={created}")
    if created:
        print(
            "注意：候选不是已获准的测试任务。OBS 判定规则未确认前，"
            "任务停留在 identity_pending，不进入分析与测试。"
        )
    return 0


def _queue_cmd_list(store: QueueStore, args) -> int:
    tasks = store.list_tasks(args.state)
    if not tasks:
        print("（无任务）")
        return 0
    for t in tasks:
        print(
            f"{t.task_id}  {t.state:<20} {t.binding_summary()}  "
            f"enqueued={t.enqueued_at}{('  err=' + t.error_reason[:80]) if t.error_reason else ''}"
        )
    return 0


def _queue_cmd_show(store: QueueStore, args) -> int:
    task = store.get_task(args.task_id)
    if task is None:
        print(f"任务不存在: {args.task_id}", file=sys.stderr)
        return 1
    print(json.dumps(task.__dict__, ensure_ascii=False, indent=2, default=str))
    return 0


def _queue_cmd_add_rule(store: QueueStore, args) -> int:
    store.register_identity_rule(args.id, args.text, args.source)
    print(f"已登记判定规则 {args.id}（来源: {args.source}）")
    return 0


def _queue_cmd_verify(store: QueueStore, args) -> int:
    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    try:
        task = store.verify_identity(args.task_id, args.rule, evidence)
    except q.IdentityGateError as e:
        print(f"[拒绝] {e}", file=sys.stderr)
        return 2
    print(f"已核实: {task.task_id} -> {task.state} ({task.binding_summary()})")
    return 0


def _queue_cmd_prefilter(store: QueueStore, args) -> int:
    task = store.prefilter_outcome(args.task_id, args.include, args.note)
    print(f"{task.task_id} -> {task.state}")
    return 0


def _queue_cmd_prefilter_run(store: QueueStore, args) -> int:
    from .prefilter import prefilter_pending_tasks

    config = _load_config(args.config)
    report = prefilter_pending_tasks(
        config, queue=store, mirror_dir=args.mirror_dir or None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report["errors"]:
        print(f"\n{len(report['errors'])} 个任务处理失败，保持 prefilter_pending", file=sys.stderr)
        return 1
    return 0


def _queue_cmd_manifest(store: QueueStore, args) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    task = store.submit_manifest(args.task_id, manifest, args.analysis_output)
    print(f"{task.task_id} -> {task.state}（清单 {len(task.test_manifest or [])} 项）")
    return 0


def _queue_cmd_select(store: QueueStore, args) -> int:
    from .pipeline import run_selection

    return run_selection(
        _load_config(args.config), store, args.task_id,
        max_tests=args.max_tests,
        device=args.device,
        python_env=args.python_env,
        collect_timeout=args.collect_timeout,
        no_submit=args.no_submit,
    )


def _queue_cmd_requeue(store: QueueStore, args) -> int:
    task = store.requeue(args.task_id, args.to, args.reason)
    print(f"{task.task_id} -> {task.state}")
    return 0


def _queue_cmd_recover(store: QueueStore, args) -> int:
    recovered = store.recover_interrupted()
    if not recovered:
        print("无中断任务")
    for t in recovered:
        print(f"{t.task_id}: {t.state} ({t.error_reason})")
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    store = QueueStore(config)
    handlers = {
        "register-candidate": _queue_cmd_register,
        "list": _queue_cmd_list,
        "show": _queue_cmd_show,
        "add-identity-rule": _queue_cmd_add_rule,
        "verify-identity": _queue_cmd_verify,
        "prefilter": _queue_cmd_prefilter,
        "prefilter-run": _queue_cmd_prefilter_run,
        "submit-manifest": _queue_cmd_manifest,
        "select": _queue_cmd_select,
        "requeue": _queue_cmd_requeue,
        "recover": _queue_cmd_recover,
    }
    return handlers[args.queue_cmd](store, args)


# ---------------------------------------------------------------------------
# skill / executor / submit-result / config
# ---------------------------------------------------------------------------

def cmd_executor(args: argparse.Namespace) -> int:
    from .pipeline import make_runner

    config = _load_config(args.config)
    executor = Executor(
        config,
        downloader=make_downloader(config.get("obs")),
        runner=make_runner(config),
    )
    outcome = executor.run_once()
    print(f"task={outcome.task_id} outcome={outcome.outcome} {outcome.detail}")
    return 0 if outcome.outcome in ("done", "idle") else 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    import time as _time
    from datetime import datetime

    from .pipeline import pipeline_once

    config = _load_config(args.config)

    def _once() -> int:
        report = pipeline_once(config)
        pre = report["prefilter"]
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"prefilter 处理={pre.get('processed', 0)} "
            f"保留={pre.get('kept', 0)} 排除={pre.get('filtered_out', 0)} "
            f"筛选成功={len(report['selected'])} 执行={len(report['executed'])}",
            flush=True,
        )
        for e in report["select_errors"]:
            print(f"  [错误] {e}", file=sys.stderr, flush=True)
        for x in report["executed"]:
            print(f"  [执行] {x['task_id']} {x['outcome']} {x['detail']}", flush=True)
        return 1 if report["select_errors"] else 0

    if not args.loop:
        return _once()
    interval = float(
        args.interval
        or (config.get("upstream") or {}).get("cycle_interval_seconds")
        or 3600.0
    )
    print(f"常驻模式：每 {interval}s 一轮（Ctrl-C 退出）", flush=True)
    while True:
        try:
            _once()
        except KeyboardInterrupt:
            print("收到退出信号", flush=True)
            return 0
        except Exception as e:  # 单轮失败不终止常驻循环
            print(f"[本轮失败] {e}", file=sys.stderr, flush=True)
        try:
            _time.sleep(interval)
        except KeyboardInterrupt:
            print("收到退出信号", flush=True)
            return 0


def cmd_submit_result(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    sub = submission_from_payload(payload)
    report = submit_complete_result(config, sub)
    print(f"PR {report.pr_number}: counts={report.totals.as_dict()} no_result={report.no_result_count} log={report.pr_log_dir}")
    for d in report.diffs:
        print(f"  [基线] {d.summary()}")
    print(f"  [核对] {report.checks}")
    return 0


def submission_from_payload(payload: dict) -> Submission:
    """把 JSON 载荷转为 Submission（字段校验由 submit_complete_result 完成）。

    no_result 项不带 counts，带 status/reason（与 PR 库 tests 列表同形）。
    """
    frs = []
    for item in payload.get("function_results", []):
        frs.append(
            FunctionResult(
                test_function=item["test_function"],
                device=item["device"],
                counts=(
                    Counts.from_dict(item["counts"])
                    if "counts" in item else None
                ),
                log_content=item.get("log_content"),
                log_source_path=item.get("log_source_path"),
                status=item.get("status", "ok"),
                reason=item.get("reason"),
                triage=item.get("triage"),
                execution_status=item.get("execution_status", "completed"),
                coverage_status=item.get("coverage_status", "unknown"),
                instance_ids=tuple(item.get("instance_ids", [])),
                environment_fingerprint=item.get("environment_fingerprint"),
            )
        )
    return Submission(
        pr_number=payload["pr_number"],
        pytorch_commit=payload["pytorch_commit"],
        torch_npu_environment=payload["torch_npu_environment"],
        function_results=frs,
        pr_log_source_dir=payload.get("pr_log_source_dir"),
    )


def cmd_mirror(args: argparse.Namespace) -> int:
    from .upstream import PyTorchMirror

    config = _load_config(args.config)
    mirror = PyTorchMirror(config=config)
    if args.mirror_cmd == "sync":
        tip = mirror.fetch()
        print(f"tip: {tip}")
        return 0
    if args.mirror_cmd == "check":
        sha = args.sha
        mirror.ensure()
        full = mirror.resolve_full_sha(sha)
        on_main = mirror.is_on_main(sha) if full else False
        print(f"sha={sha} full={full} on_main={on_main}")
        return 0 if on_main else 1
    return 1


def cmd_cycle(args: argparse.Namespace) -> int:
    import time as _time

    from .upstream import run_cycle

    config = _load_config(args.config)
    if args.backfill_hours is None:
        backfill = (config.get("upstream") or {}).get("first_run_backfill_hours", 2.0)
    else:
        backfill = args.backfill_hours

    def _once() -> int:
        report = run_cycle(config, first_run_backfill_hours=backfill)
        print(f"[{__import__('datetime').datetime.now().isoformat(timespec='seconds')}] "
              f"tip={report.fetched_tip[:10]} "
              f"新commit={report.new_commits}(记{report.recorded}/无PR{report.skipped_no_pr}) "
              f"入队={len(report.enqueued)} 等待={len(report.still_missing)} "
              f"过期={len(report.expired)}", flush=True)
        for e in report.enqueued:
            print(f"  [入队] {e['task_id']} PR#{e['pr_number']} {e['sha10']} "
                  f"{e['subject'][:70]}", flush=True)
        for err in report.errors:
            print(f"  [错误] {err}", file=sys.stderr, flush=True)
        return 0

    if not args.loop:
        return _once()
    interval = float(
        (config.get("upstream") or {}).get("cycle_interval_seconds") or 3600.0
    )
    print(f"常驻模式：每 {interval}s 一轮（Ctrl-C 退出）", flush=True)
    while True:
        try:
            _once()
        except KeyboardInterrupt:
            print("收到退出信号", flush=True)
            return 0
        except Exception as e:  # 单轮失败不终止常驻循环
            print(f"[本轮失败] {e}", file=sys.stderr, flush=True)
        try:
            _time.sleep(interval)
        except KeyboardInterrupt:
            print("收到退出信号", flush=True)
            return 0


def cmd_upstream(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    store = QueueStore(config)
    rows = store.list_upstream_commits(args.state)
    if not rows:
        print("（无记录）")
        return 0
    for r in rows:
        pr = f"#{r['pr_number']}" if r["pr_number"] else "no-pr"
        task = r["task_id"] or "-"
        print(
            f"{r['sha'][:10]}  {pr:<8} {r['enqueue_state']:<16} "
            f"{r['commit_time'][:19]}  {task}  {r['subject'][:60]}"
        )
    return 0


def cmd_merged_at_backfill(args: argparse.Namespace) -> int:
    """merged_at 回填：先做 DB 备份，再按精确 commit 幂等回填（只填 NULL）。"""
    import sqlite3
    from datetime import datetime as _dt, timezone as _tz

    from .baseline_store import baseline_db_path
    from .config import store_paths
    from .db import connect as _connect
    from .record import backfill_merged_at

    config = _load_config(args.config)
    paths = store_paths(config)
    stamp = _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = paths.data_root / "merged-at-backups" / stamp
    backup_root.mkdir(parents=True, exist_ok=True)
    router = BaselineRouter(config.get("baseline_dirs", []), paths.baselines_root)
    targets = [paths.queue_db, paths.pr_db]
    targets += [baseline_db_path(router.managed_root(d)) for d in router.dirs]
    for src in targets:
        if Path(src).is_file():
            src_conn = _connect(src)
            try:
                # 每个来源唯一子目录：不同 managed 目录的 baseline.db 不互覆。
                slot = hashlib.sha256(str(src).encode()).hexdigest()[:10]
                dst = backup_root / slot / Path(src).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst_conn = sqlite3.connect(dst)
                try:
                    src_conn.backup(dst_conn)
                finally:
                    dst_conn.close()
            finally:
                src_conn.close()
    report = backfill_merged_at(config, refresh_reports=args.refresh_reports)
    report["backup"] = str(backup_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args):
    from .result_docs import generate, scan
    config = _load_config(args.config)
    if args.report_cmd in ('publish', 'publish-scan'):
        from .feishu_publish import publish, scan as publish_scan
        result = publish(config, args.pr) if args.report_cmd == 'publish' else publish_scan(config)
    else:
        result = generate(config, args.pr) if args.report_cmd == 'render' else scan(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get('errors') else 0


def cmd_ledger(args):
    from .issue_ledger import Ledger, scan
    config = _load_config(args.config)
    if args.ledger_cmd == 'scan':
        result = scan(config)
    else:
        ledger = Ledger(config)
        try:
            if args.ledger_cmd == 'resolve':
                ledger.resolve(args.identifier, args.note, args.evidence)
                ledger.render()
            result = ledger.summary()
        finally:
            ledger.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_test.cli",
        description="NPU PR 测试结果与基线存储 / OBS 串行测试队列",
    )
    p.add_argument("--config", help="配置文件路径（默认: 项目根/settings.json）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="初始化数据根目录与数据库")
    scan = sub.add_parser("obs-scan", help="只读列出真实 OBS 配套包；不下载、不入队")
    scan.add_argument("--output", help="JSON 清单保存路径")
    results = sub.add_parser("results", help="查询 PR 结果、失败计数及 no_result 数")
    results.add_argument("--pr", type=int, help="限定 PR 编号")

    report = sub.add_parser('report', help='从真实落库结果生成飞书报告内容和离线预览')
    rsub = report.add_subparsers(dest='report_cmd', required=True)
    render = rsub.add_parser('render', help='生成指定 PR；不上传飞书')
    render.add_argument('--pr', type=int, required=True)
    rsub.add_parser('scan', help='增量生成全部当前 PR 报告；保留历史版本')
    publish = rsub.add_parser('publish', help='发布指定 PR 到飞书测试目录，完整校验后归档旧版')
    publish.add_argument('--pr', type=int, required=True)
    rsub.add_parser('publish-scan', help='幂等发布全部当前 PR 报告到飞书')

    ledger = sub.add_parser('ledger', help='真实问题台账：证据快照、U/S 归类、修复归档')
    lsub = ledger.add_subparsers(dest='ledger_cmd', required=True)
    lsub.add_parser('scan', help='扫描真实队列、选例和结果并幂等记账')
    lsub.add_parser('status', help='查询台账统计')
    repair = lsub.add_parser('resolve', help='附验证证据关闭问题，保留原始事件')
    repair.add_argument('identifier')
    repair.add_argument('--note', required=True)
    repair.add_argument('--evidence', required=True, help='data_root 下真实验证日志路径')

    env = sub.add_parser("env", help="通用环境说明")
    env.add_argument("--write", action="store_true", help="采集并写入 common_environment.json")

    base = sub.add_parser("baseline", help="最小管理目录清单")
    bsub = base.add_subparsers(dest="baseline_cmd", required=True)
    bsub.add_parser("list", help="列出已登记目录")
    b_add = bsub.add_parser("add", help="登记一个最小管理目录")
    b_add.add_argument("dir")
    b_rm = bsub.add_parser("remove", help="从清单移除")
    b_rm.add_argument("dir")
    b_imp = bsub.add_parser(
        "import",
        help="把 transfer-CUDA 基线快照（baseline.json + junit/）导入基线库",
    )
    b_imp.add_argument("--json", required=True, help="baseline.json 路径")
    b_rf = bsub.add_parser(
        "rollforward",
        help="基线前滚：按合入时间顺序回放一组 PR → 前后对比文件 + 刷新基线"
             "（供外部 AI 调用；输出文件分析后应立即删除）",
    )
    rf_input = b_rf.add_mutually_exclusive_group(required=True)
    rf_input.add_argument("--all", action="store_true", help="回放当前结果库全部 PR")
    rf_input.add_argument("--input",
                      help="PR 清单 JSON：{\"prs\": [123, {\"pr_number\": 7, "
                           "\"merged_at\": \"…\"}]} 或裸数组")
    b_rf.add_argument("--wait-seconds", type=float, default=0, help="等待执行位的最长秒数；不终止测试")
    bsub.add_parser("history", help="列出持久基线快照（不会随临时对比报告删除）")
    b_snap = bsub.add_parser("snapshot", help="备份全部基线 DB 与日志")
    b_snap.add_argument("--reason", default="manual")
    b_snap.add_argument("--wait-seconds", type=float, default=0)
    b_back = bsub.add_parser("rollback", help="恢复指定快照；先保存当前状态以便撤销")
    b_back.add_argument("snapshot_id")
    b_back.add_argument("--wait-seconds", type=float, default=0)
    b_imp.add_argument("--src", required=True, help="test-source 根（含 test/ 的仓库 worktree）")
    b_imp.add_argument("--npu-commit", required=True,
                       help="torch_npu 完整 40/64 位 commit（快照 meta 只有版本短 sha）")

    qu = sub.add_parser("queue", help="任务队列")
    qsub = qu.add_subparsers(dest="queue_cmd", required=True)
    q_reg = qsub.add_parser("register-candidate", help="登记 OBS 对象候选（会停在 identity_pending）")
    q_reg.add_argument("--wheels-url", required=True)
    q_reg.add_argument("--source-url", required=True)
    q_reg.add_argument("--wheels-version")
    q_reg.add_argument("--source-version")
    q_reg.add_argument("--remote-time")
    q_list = qsub.add_parser("list", help="列出任务")
    q_list.add_argument("--state")
    q_show = qsub.add_parser("show", help="查看任务详情")
    q_show.add_argument("task_id")
    q_rule = qsub.add_parser("add-identity-rule", help="登记已确认的 OBS 包判定规则")
    q_rule.add_argument("--id", required=True)
    q_rule.add_argument("--text", required=True, help="规则原文")
    q_rule.add_argument("--source", required=True, help="规则来源（谁在何时确认）")
    q_ver = qsub.add_parser("verify-identity", help="按已确认规则核实候选身份")
    q_ver.add_argument("task_id")
    q_ver.add_argument("--rule", required=True)
    q_ver.add_argument("--evidence", required=True, help="证据 JSON 文件路径")
    q_pre = qsub.add_parser("prefilter", help="记录既有脚本过滤结果（输入来源确认后由分析端调用）")
    q_pre.add_argument("task_id")
    q_pre.add_argument("--include", required=True, choices=["yes", "no"])
    q_pre.add_argument("--note")
    q_pre_run = qsub.add_parser(
        "prefilter-run",
        help="批处理 prefilter_pending：main 镜像 diff → skill 封闭规则过滤 → 落库",
    )
    q_pre_run.add_argument("--mirror-dir", help="main 裸镜像路径（默认 data_root/pytorch-main.git）")
    q_man = qsub.add_parser("submit-manifest", help="提交 AI 分析产出的函数/device 清单")
    q_man.add_argument("task_id")
    q_man.add_argument("--manifest", required=True, help="清单 JSON 文件路径")
    q_man.add_argument("--analysis-output")
    q_sel = qsub.add_parser(
        "select",
        help="运行用例筛选器（移植 pytorch-commit-pre-check）：P0/P1 + 全局 15 预算 → 提交清单",
    )
    q_sel.add_argument("task_id")
    q_sel.add_argument("--max-tests", type=int, default=15, help="全局具体用例预算（默认 15）")
    q_sel.add_argument("--device", default="auto", choices=["auto", "cpu", "accelerator"],
                       help="清单 device：auto=逐用例 AST 语义判定（默认），"
                            "或固定 accelerator/cpu")
    q_sel.add_argument("--python-env", help="Python 家族展开用解释器；默认取 executor.test_python_env")
    q_sel.add_argument("--collect-timeout", type=int, default=120, help="单个发现进程超时秒数")
    q_sel.add_argument("--no-submit", action="store_true", help="只产出计划与诊断，不提交清单")
    q_rq = qsub.add_parser("requeue", help="人工确认后把 blocked 任务放回等待阶段")
    q_rq.add_argument("task_id")
    q_rq.add_argument("--to", required=True,
                      choices=["identity_pending", "pending_analysis", "pending_test"])
    q_rq.add_argument("--reason")
    qsub.add_parser("recover", help="进程重启后核实中断任务状态")

    mir = sub.add_parser("mirror", help="上游 PyTorch main 镜像")
    msub = mir.add_subparsers(dest="mirror_cmd", required=True)
    msub.add_parser("sync", help="克隆/拉取镜像（只做 git 同步）")
    m_chk = msub.add_parser("check", help="检查 sha 是否可解析且在 main 历史中")
    m_chk.add_argument("sha")

    cyc = sub.add_parser("cycle", help="一轮：fetch → 记录新 commit → 定位桶中对象 → 入队")
    cyc.add_argument("--backfill-hours", type=float, default=None,
                     help="首轮回溯窗口（小时，仅首轮生效；默认取 upstream.first_run_backfill_hours）")
    cyc.add_argument("--loop", action="store_true",
                     help="常驻模式：按 upstream.cycle_interval_seconds 循环（无 crond 环境用）")

    up = sub.add_parser("upstream", help="上游 commit 记录")
    up.add_argument("--state", help="pending/queued/no_pr/expired")

    ex = sub.add_parser("executor", help="串行执行一个任务")
    ex.add_argument("unused", nargs="?", help="（占位）执行 run-once 语义")

    pl = sub.add_parser(
        "pipeline",
        help="一轮节拍：prefilter 批放行 → 逐任务筛选 → executor 消费到 idle",
    )
    pl.add_argument("--loop", action="store_true",
                    help="常驻模式：按 upstream.cycle_interval_seconds 循环（无 crond 环境用）")
    pl.add_argument("--interval", type=float, default=None,
                    help="常驻轮间隔秒（默认取 upstream.cycle_interval_seconds）")

    sr = sub.add_parser("submit-result", help="按六步流程写入一份完整测试结果")
    sr.add_argument("--input", required=True, help="提交 JSON 文件路径")

    sub.add_parser("config", help="显示当前配置")
    mab = sub.add_parser("merged-at-backfill",
                         help="存量 PR/基线行 merged_at 回填（先租约+唯一备份；只填 NULL）")
    mab.add_argument("--refresh-reports", action="store_true",
                     help="对已有报告的 PR 受控回填并立即再生成报告（默认跳过并登记 needs_refresh）")
    mab.set_defaults(func=cmd_merged_at_backfill)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    table = {
        "init": cmd_init,
        "obs-scan": cmd_obs_scan,
        "results": cmd_results,
        "env": cmd_env,
        "baseline": cmd_baseline,
        "queue": cmd_queue,
        "mirror": cmd_mirror,
        "cycle": cmd_cycle,
        "pipeline": cmd_pipeline,
        "upstream": cmd_upstream,
        "executor": cmd_executor,
        "submit-result": cmd_submit_result,
        "config": cmd_config_show,
        "ledger": cmd_ledger,
        "report": cmd_report,
    }
    return table[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
