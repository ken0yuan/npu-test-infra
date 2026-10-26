"""流程节拍器：prefilter → select → executor 的自动串接。

除 cycle（上游观察）外，此前三个阶段全靠人工逐个调 CLI——288 个
prefilter_pending 存量无人推进。本模块把"批放行 → 逐任务筛选 → 执行器
消费到 idle"变成一轮可循环的节拍（`cli pipeline [--loop]`）。

一轮内 select（下载/筛选，I/O 密集）与 executor（真跑测试）重叠：
select 在后台线程持续产清单，executor 主线程就绪即消费——下载和测试
互相掩盖，不必"前一任务测完才下载下一个"。

容错语义：单阶段/单任务失败只记入本轮报告，不终止轮——下一轮重试或
人工处置；executor 单任务的 blocked/failed 也算消费完成，继续下一个。
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

# 筛选子进程登记（进程内存活；跨轮防重复派生；轮间由同任务筛选锁兜底）。
_SELECTION_IN_FLIGHT: dict = {}
_SELECTION_FAILURES: dict = {}


def make_runner(config: dict):
    """按 executor 配置构造 PytestRunner；test_python_env 未配置返回 None。"""
    exec_cfg = config.get("executor") or {}
    if not exec_cfg.get("test_python_env"):
        return None
    from .runner import PytestRunner

    return PytestRunner(
        exec_cfg["test_python_env"],
        exec_cfg.get("device_runner_args"),
        timeout_seconds=exec_cfg.get("function_timeout_seconds"),
        transfer_boot_dir=exec_cfg.get("transfer_boot_dir"),
    )


def boot_discovery_env(config: dict) -> dict:
    """Historical function name retained; no boot is injected."""
    from .device_policy import execution_env
    return execution_env()


def run_selection(
    config: dict,
    store,
    task_id: str,
    *,
    max_tests: int = 15,
    device: str = "auto",
    python_env: str | None = None,
    collect_timeout: int = 120,
    no_submit: bool = False,
) -> int:
    """`queue select` 的可复用主体（CLI 与 pipeline 共用）。返回退出码。

    步骤与语义同 cli._queue_cmd_select（2026-09-18 抽出）：状态门禁 →
    镜像 %B 取 PR 文本 → OBS 源码包解 test 树 → P0/P1 筛选 + 15 预算 →
    设备判定（auto=AST 语义 v2，可一例多设备）→ 基线覆盖注记 → 提交清单。
    """
    import json
    import os
    import shutil
    import sys
    import tarfile
    import tempfile

    from . import task_queue as q
    from .baseline_store import BaselineRouter
    from .config import store_paths
    from .obs import make_downloader
    from .selection import baseline_coverage
    from .selection import git as sel_git
    from .selection import plan_to_manifest, render_text, select_for_commit
    from .upstream import PyTorchMirror

    task = store.get_task(task_id)
    if task is None:
        print(f"任务不存在: {task_id}", file=sys.stderr)
        return 2
    if task.state != q.STATE_PENDING_ANALYSIS:
        print(f"任务 {task.task_id} 处于 {task.state}，只有 pending_analysis 可运行筛选", file=sys.stderr)
        return 2
    if not task.pytorch_commit:
        print(f"任务 {task.task_id} 缺少 pytorch_commit，无法筛选", file=sys.stderr)
        return 2

    paths = store_paths(config)
    # 同任务筛选互斥（防轮间重复派生；非阻塞，占用即跳过 rc=3）。
    # contextmanager 保证成功/异常/占用都关闭 fd（评审阻塞项 3）。
    import fcntl
    lock_dir = paths.data_root / "selection" / task_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_dir / ".selection.lock", os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("已有筛选进程在运行（锁占用）；本轮跳过", file=sys.stderr)
            return 3
        # 子进程模式：stderr 落任务文件（评审阻塞项 2——PIPE 只在退出后
        # 读取，>64KB 会撑爆管道令子进程永久阻塞）。
        log_path = paths.data_root / "selection" / task_id / "select-one.stderr.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _err = open(log_path, "ab")
        try:
            sys.stderr = _err
            return _run_selection_body(config, store, task, task_id, paths)
        finally:
            sys.stderr = sys.__stderr__
            _err.close()
    finally:
        os.close(lock_fd)


def _run_selection_body(
    config: dict,
    store,
    task,
    task_id: str,
    paths,
) -> int:
    """筛选主体（持锁后运行）：源码暂存 → P0/P1 筛选 → 设备判定 → 清单。"""
    sha = task.pytorch_commit
    mirror = PyTorchMirror(config=config)
    mirror.ensure()
    subject = sel_git(mirror.git_dir, "show", "-s", "--format=%s", sha).strip()
    # 完整 commit message（subject + 正文）作为 PR 文本信号：本地镜像
    # %B 可得，无需 gh/token；mentioned_in_pr 的 200 分加权依赖它。
    body = sel_git(mirror.git_dir, "show", "-s", "--format=%B", sha).strip()
    pr_text = body or subject

    # 测试树取自 OBS 源码包（任务级暂存——用户 2026-10-26：完成即删，
    # 不留共享 source_cache 实体副本）；executor 复用同一份归档避免重复
    # 下载。兼容：存量共享缓存存在则"收编"为任务本地副本后清点删除。
    sha10 = sha[:10]
    cache = paths.data_root / "selection" / task_id / "source-staging"
    tree = cache / "pytorch"
    changed_paths = []
    raw_changed = sel_git(mirror.git_dir, "diff", "--name-only", f"{sha}^", sha)
    changed_paths = [line.strip() for line in raw_changed.splitlines() if line.strip()]
    marker = cache / ".selection-sources-complete"
    downloader = make_downloader(config.get("obs"))
    if not (tree / "test").is_dir() or not marker.is_file():
        if downloader is None:
            print("OBS 下载器未配置（obs.downloader），无法获取源码包", file=sys.stderr)
            return 2
        pkg = cache / "pytorch-source.tar.gz"
        if not pkg.is_file():
            legacy = paths.data_root / "source_cache" / sha10 / "pytorch-source.tar.gz"
            if legacy.is_file():
                # 收编存量共享缓存副本（避免重复下载）；共享目录稍后由
                # 任务完成清理按"无在用引用"删除。
                cache.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy, pkg)
            else:
                downloader.download(task.obs_source_url, pkg)
        source_members = {f"pytorch/{path}" for path in changed_paths if not path.startswith("test/")}
        with tarfile.open(pkg, "r:gz") as tf:
            members = [m for m in tf.getmembers() if (
                m.name.startswith("pytorch/test/")
                or m.name in {"pytorch/aten/tools/run_tests.sh", "pytorch/.ci/pytorch/test.sh"}
                or m.name in source_members
            )]
            tf.extractall(cache, members=members, filter="data")
        marker.write_text(json.dumps({"commit": sha, "changed_source_files": sorted(source_members)}, ensure_ascii=False), encoding="utf-8")
    if not (tree / "test").is_dir():
        print(f"源码包解压后缺少 pytorch/test/: {cache}", file=sys.stderr)
        return 2

    import sys as _sys

    interpreter = (
        python_env
        or (config.get("executor") or {}).get("test_python_env")
        or _sys.executable
    )
    # P1-1：选端发现注入 boot（不设 ONLY_FOR，三通道变体都要）。
    discovery_env = boot_discovery_env(config)
    report = select_for_commit(
        mirror.git_dir, sha, subject, tree,
        pr_text=pr_text,
        max_tests=max_tests,
        collect_timeout=collect_timeout,
        interpreter=interpreter,
        discovery_env=discovery_env,
    )
    from .device_detect import make_device_resolver
    from .device_policy import ExecutionDeviceResolver, configured_devices
    resolver = ExecutionDeviceResolver(make_device_resolver(tree), None if device == "auto" else device,
                                       configured_devices(config))
    manifest = plan_to_manifest(report, device_of=resolver)
    report["device_diagnostics"] = list(resolver.diagnostics)

    if manifest:
        # 基线覆盖注记（只进计划文件；manifest schema 不动）。miss = 首次
        # 基线 = 无对比信号——选择质量的可观测性指标。
        router = BaselineRouter(config.get("baseline_dirs", []), paths.baselines_root)
        report["baseline_coverage"] = baseline_coverage(router, manifest)

    out_dir = paths.data_root / "selection" / task.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "test-plan.json"
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=out_dir, delete=False) as stream:
        temporary_plan = stream.name
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary_plan, plan_path)
    print(render_text(report))
    print(f"\n计划文件: {plan_path}")

    if not manifest:
        # 没有候选 = selector 不确定，不得当作“不需要测试”，也不提交空清单。
        print("无可用清单条目（候选未能展开或全部被跳过）；不提交，请人工查看计划文件", file=sys.stderr)
        return 2
    if no_submit:
        print("（--no-submit：清单未提交）")
        return 0
    done = store.submit_manifest(task.task_id, manifest, str(plan_path))
    print(f"{done.task_id} -> {done.state}（清单 {len(manifest)} 项）")
    return 0


def pipeline_once(config: dict, *, max_tasks_per_round: int = 50) -> dict[str, Any]:
    """一轮：① prefilter 批处理 → ② select 后台线程持续产清单 ∥ ③ executor 并行消费。

    ②③重叠（下载/测试互相掩盖）：清单一经提交，executor 的下一轮 run_once
    即可拾取，不必等全部筛选完成。executor idle 且 select 仍在产出时短暂
    等待后重试；select 线程结束且 executor idle 才收轮。
    """
    import time
    from .executor import Executor
    from .obs import make_downloader
    from .prefilter import prefilter_pending_tasks
    from .task_queue import STATE_PENDING_ANALYSIS, QueueStore

    store = QueueStore(config)
    report: dict[str, Any] = {
        "prefilter": {},
        "selected": [],
        "select_errors": [],
        "executed": [],
    }

    # ① 批放行存量（单任务失败在批内记账，见 prefilter_pending_tasks）。
    try:
        report["prefilter"] = prefilter_pending_tasks(config, queue=store)
    except Exception as e:  # 阶段失败不终止轮
        report["select_errors"].append(f"prefilter 阶段失败: {e}")

    # ② selector 后台线程：持续消费（轮内新到任务也筛选）；每个任务的
    # 筛选在独立进程组（评审阻塞项①③：隔离、无整任务超时、外部操作各
    # 有自身限制）。③ executor 主线程并行消费。
    import threading

    attempted: set[str] = set()  # 与主线程共享：收班判定=无未尝试候选
    stop_event = threading.Event()
    max_in_flight = max(int((config.get("selector") or {}).get("max_in_flight") or 2), 1)
    max_failures = max(int((config.get("selector") or {}).get("max_failures") or 3), 1)
    config_path = config.get("_config_path") or "settings.json"
    sel_command = config.get("_selection_command") or [
        sys.executable, "-m", "run_test.pipeline", "select-one",
        "--task", "{task_id}", "--config", config_path]

    def _reap() -> None:
        """回收已结束子进程：rc 记账；连续失败达上限转 blocked（证据留痕）。"""
        for tid, proc in list(_SELECTION_IN_FLIGHT.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del _SELECTION_IN_FLIGHT[tid]
            if rc == 3:
                report["select_errors"].append(f"{tid}: 已有筛选在运行，本轮跳过")
            elif rc == 0:
                report["selected"].append(tid)
            else:
                _SELECTION_FAILURES[tid] = _SELECTION_FAILURES.get(tid, 0) + 1
                message = f"{tid}: select 退出码 {rc}"
                if _SELECTION_FAILURES[tid] >= max_failures:
                    try:
                        store.mark_blocked(
                            tid, f"筛选连续失败 {_SELECTION_FAILURES[tid]} 次：{message}；"
                                 "转 blocked 待人工核实")
                    except Exception as mark_error:
                        report["select_errors"].append(f"{message}；转 blocked 失败: {mark_error}")
                    report["select_errors"].append(message)
                else:
                    report["select_errors"].append(message)

    def _select_worker() -> None:
        while True:
            _reap()
            candidates = [t for t in select_store.list_tasks(STATE_PENDING_ANALYSIS)
                          if t.task_id not in attempted
                          and t.task_id not in _SELECTION_IN_FLIGHT]
            if not candidates:
                if stop_event.is_set():
                    break
                time.sleep(0.5)
                continue
            for task in candidates:
                if stop_event.is_set() or len(_SELECTION_IN_FLIGHT) >= max_in_flight:
                    break
                attempted.add(task.task_id)
                cmd = [c.replace("{task_id}", task.task_id) if isinstance(c, str) else c
                       for c in sel_command]
                env = dict(os.environ)
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + (
                    ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                _SELECTION_IN_FLIGHT[task.task_id] = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, env=env,
                    start_new_session=True)
            time.sleep(0.5)

    selector = threading.Thread(
        target=_select_worker, name="pipeline-select", daemon=True,
    )
    selector.start()

    # ③ executor 主线程并行消费；blocked/failed 也算消费完成，继续下一个。
    runner = make_runner(config)
    executor = Executor(
        config, downloader=make_downloader(config.get("obs")), runner=runner,
    )
    cap = max(1, max_tasks_per_round)
    while len(report["executed"]) < cap:
        outcome = executor.run_once()
        if outcome.outcome == "idle":
            unattempted = [t for t in store.list_tasks(STATE_PENDING_ANALYSIS)
                           if t.task_id not in attempted
                           and t.task_id not in _SELECTION_IN_FLIGHT]
            if unattempted:
                time.sleep(0.5)  # 还有未筛选任务：等 selector 产出清单
                continue
            if selector.is_alive():
                stop_event.set()  # 候选已尽：通知 selector 收班
                time.sleep(0.5)
                continue
            break
        report["executed"].append({
            "task_id": outcome.task_id,
            "outcome": outcome.outcome,
            "detail": outcome.detail[:200],
        })
    stop_event.set()
    selector.join()
    return report


def main(argv=None) -> int:
    """`python -m run_test.pipeline select-one --task T --config C` 子进程入口。"""
    import argparse

    parser = argparse.ArgumentParser(prog="run_test.pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select-one", help="单任务筛选（子进程隔离入口）")
    sel.add_argument("--task", required=True)
    sel.add_argument("--config", default="settings.json")
    args = parser.parse_args(argv)
    if args.command == "select-one":
        from .config import load_config
        from .task_queue import QueueStore

        config = load_config(args.config)
        config["_config_path"] = args.config
        store = QueueStore(config)
        return run_selection(config, store, args.task)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
