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

from typing import Any


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


def _progress(config, task, stage):
    import json, time
    from .task_staging import task_root, atomic_json
    path = task_root(config, task) / 'progress.json'
    now = time.time()
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    if stage == 'starting':
        state = {}
    durations = state.get('stage_seconds', {})
    if state.get('stage'):
        durations[state['stage']] = durations.get(state['stage'], 0) + max(0, now - state['updated_at'])
    atomic_json(path, {'task_id': task.task_id, 'stage': stage, 'updated_at': now,
                      'stage_seconds': durations})
    print(f"[selection] {task.task_id} stage={stage} elapsed={durations}", flush=True)


def run_selection(config, store, task_id, **kwargs):
    from contextlib import ExitStack
    from .process_lock import executor_lease
    from .task_staging import task_root
    from .config import store_paths
    from .task_queue import QueueError
    try:
        task = store.get_task(task_id)
    except QueueError:
        return 2
    if task is None:
        return 2
    root = task_root(config, task)
    with ExitStack() as locks:
        if not locks.enter_context(executor_lease(root / '.selection.lock')):
            return 3
        # Slot locks survive parent restarts as long as the child is still alive.
        slots = max(1, int((config.get('selector') or {}).get('max_in_flight', 2)))
        for slot in range(slots):
            if locks.enter_context(executor_lease(store_paths(config).data_root / 'selection' / f'slot-{slot}.lock')):
                break
        else:
            return 3
        _progress(config, task, 'starting')
        try:
            rc = _run_selection_body(config, store, task_id, **kwargs)
            _progress(config, task, 'complete' if rc == 0 else 'failed')
            return rc
        except BaseException:
            _progress(config, task, 'failed')
            raise


def _run_selection_body(
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
    from .task_staging import stage_package, identity, checksum, atomic_json, task_root
    progress = lambda stage: _progress(config, task, stage)
    progress('mirror')
    sha = task.pytorch_commit
    mirror = PyTorchMirror(config=config)
    mirror.ensure()
    subject = sel_git(mirror.git_dir, "show", "-s", "--format=%s", sha).strip()
    # 完整 commit message（subject + 正文）作为 PR 文本信号：本地镜像
    # %B 可得，无需 gh/token；mentioned_in_pr 的 200 分加权依赖它。
    body = sel_git(mirror.git_dir, "show", "-s", "--format=%B", sha).strip()
    pr_text = body or subject

    progress('source_download')
    downloader = make_downloader(config.get('obs'))
    pkg = stage_package(config, task, 'pytorch-source.tar.gz', downloader)
    cache = pkg.parent
    tree = cache / 'pytorch'
    marker = cache / '.selection-sources-complete'
    expected = {'identity': identity(task, pkg.name), 'sha256': checksum(pkg)}
    try:
        complete = json.loads(marker.read_text(encoding='utf-8')) == expected
    except (OSError, ValueError):
        complete = False
    if not complete or not (tree / 'test').is_dir():
        progress('source_extract')
        if tree.exists():
            shutil.rmtree(tree)
        changed_paths = sel_git(mirror.git_dir, 'diff', '--name-only', f'{sha}^', sha).splitlines()
        source_members = {'pytorch/' + path for path in changed_paths if not path.startswith('test/')}
        with tarfile.open(pkg, 'r:gz') as tf:
            members = [m for m in tf.getmembers() if (
                m.name.removeprefix('./') == 'pytorch/test'
                or m.name.removeprefix('./').startswith('pytorch/test/')
                or m.name.removeprefix('./') in {'pytorch/aten/tools/run_tests.sh', 'pytorch/.ci/pytorch/test.sh'}
                or m.name.removeprefix('./') in source_members)]
            tf.extractall(cache, members=members, filter='data')
        atomic_json(marker, expected)
    if not (tree / "test").is_dir():
        print(f"源码包解压后缺少 pytorch/test/: {cache}", file=sys.stderr)
        return 2

    import sys as _sys

    interpreter = (
        python_env
        or (config.get("selector") or {}).get("python_env")
        or _sys.executable
    )
    # P1-1：选端发现注入 boot（不设 ONLY_FOR，三通道变体都要）。
    # Discovery may fall back to static identities; execution expands them using
    # the task's installed environment. Never install into or collect in the
    # active executor environment while it may be changing for another task.
    active_python = (config.get('executor') or {}).get('test_python_env')
    if active_python and os.path.abspath(interpreter) == os.path.abspath(active_python):
        raise ValueError('selector.python_env must be independent of executor.test_python_env')
    progress('test_selection')
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
    progress('wheel_prefetch')
    stage_package(config, task, 'wheels.tar.gz', downloader)
    if task.obs_wheels_npu_url:
        stage_package(config, task, 'wheels-npu.tar.gz', downloader)
    progress('manifest_submit')
    done = store.submit_manifest(task.task_id, manifest, str(plan_path))
    print(f"{done.task_id} -> {done.state}（清单 {len(manifest)} 项）")
    return 0


def pipeline_once(config: dict, *, max_tasks_per_round: int = 50) -> dict[str, Any]:
    from .executor import Executor
    from .obs import make_downloader
    from .selection_pool import get_pool
    report = {'prefilter': {}, 'selected': [], 'select_errors': [], 'executed': []}
    pool = get_pool(config)
    # Synchronize first scheduling tick only; never wait for an entire analysis.
    pool.started.wait(1)
    executor = Executor(config, downloader=make_downloader(config.get('obs')), runner=make_runner(config))
    for _ in range(max(1, max_tasks_per_round)):
        outcome = executor.run_once()
        if outcome.outcome == 'idle':
            break
        report['executed'].append({'task_id': outcome.task_id, 'outcome': outcome.outcome,
                                   'detail': outcome.detail[:200]})
    pool.collect(report)
    return report


def main(argv=None):
    import argparse
    from .config import load_config
    from .task_queue import QueueStore
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    select = sub.add_parser('select-one')
    select.add_argument('--task', required=True)
    select.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return run_selection(config, QueueStore(config), args.task)


if __name__ == '__main__':
    raise SystemExit(main())
