"""串行执行端：消费就绪任务，下载、安装、执行、写库、立即清理。

对应《OBS监测与串行测试队列说明》第 6–8 节：

- 整个执行端同一时间只允许一个任务处于 准备/安装/测试/写库/清理
  （由队列的 executor_lock 保证；后台监测仍可登记候选）；
- 排队时只保存小型任务信息，轮到任务才下载大体积包和源码；
- 执行顺序不可颠倒：锁定版本与对象 → 下载 → 核验（不齐不卸载不测试）→
  卸载旧 torch-npu/PyTorch（用包管理器，不手工删 site-packages）→
  安装本任务 wheel → 按清单执行 → 写库 → 立即清理 → 完成；
- “全部执行结束”不等于全部通过：含 fail/error/skip 的完整结果同样
  保存、刷新、清理；缺失结果不伪造四个零——单函数未取得结果降级为
  no_result（不计数、不刷基线，仅记无信号）；即使清单全部未取得结果，
  也持久化 PR 证据，不以 blocked 丢弃真实错误；
- 删除前必须核实绝对路径位于本任务登记的临时目录内；
- OBS 缺资源就记录异常，不换源、不转 GitHub。

依赖身份核实门禁：任务只有通过 verify_identity（即 OBS 判定规则已确认）
才能到达 pending_test，从而被本执行器领取。规则未确认时执行器永远空闲。
"""

from __future__ import annotations

import re
import json
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import task_queue as q
from .config import ConfigPendingError, store_paths
from .identifiers import Counts, baseline_log_filename
from .obs import ObsDownloadError, make_downloader
from .task_queue import QueueStore
from .record import FunctionResult, Submission, SubmitReport, submit_complete_result

_DEFAULT = object()


def _wait_free_npu_cards(need: int, timeout: float, interval: float):
    """等卡门禁默认探针，转发 run_test.npu_cards；等待过程打到 pipeline 日志。"""
    from .npu_cards import wait_free_cards

    return wait_free_cards(need, timeout, interval,
                           log=lambda msg: print(msg, flush=True))


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")


class ExecutorError(RuntimeError):
    pass


# One source function on one execution device may select at most 100
# concrete runtime instances. The full discovered set remains auditable.
MAX_INSTANCES_PER_FUNCTION = 100


def read_torch_npu_git_version(whl_path: Path) -> str | None:
    """从 torch_npu wheel 内 torch_npu/version.py 读取完整 git commit。

    wheel 即 zip；git_version 形如 '201ec18eb71afeab40cccaae8be8e5296cf97537'。
    读不到（文件缺失或字段为空）返回 None，调用方应视为无法绑定精确版本。
    """
    import re
    import zipfile

    try:
        with zipfile.ZipFile(whl_path) as z:
            names = [n for n in z.namelist() if n.endswith("torch_npu/version.py")]
            if not names:
                return None
            text = z.read(names[0]).decode(errors="replace")
    except (OSError, zipfile.BadZipFile):
        return None
    m = re.search(r"git_version\s*=\s*['\"]([0-9a-f]{40,64})['\"]", text)
    return m.group(1) if m else None


@dataclass(frozen=True)
class RunOutcome:
    task_id: str | None
    outcome: str  # idle / done / blocked / failed
    detail: str = ""


def _boot_fingerprint(boot_dir: str | None) -> dict | None:
    if not boot_dir:
        return None
    import hashlib, json
    digest = hashlib.sha256()
    root = Path(boot_dir)
    try:
        files = []
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode())
            digest.update(path.read_bytes())
            files.append(rel)
        manifest = None
        manifest_path = root / "BOOT_MANIFEST.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return {"sha256": digest.hexdigest(), "files": files, "manifest": manifest}


class Executor:
    def __init__(
        self,
        config: dict,
        queue_store: QueueStore | None = None,
        downloader=_DEFAULT,
        runner=None,
        persist: Callable[..., SubmitReport] | None = None,
        npu_probe: Callable[[int, float, float], tuple[list[int], str]] | None = None,
    ):
        self.config = config
        self.paths = store_paths(config)
        self.queue = queue_store or QueueStore(config)
        # 等卡门禁的探测函数（参数 need/timeout/interval → (空闲卡, 原因)），
        # 默认走 run_test.npu_cards；测试注入假探针避免 shell out。
        self.npu_probe = npu_probe or _wait_free_npu_cards
        self.downloader = (
            make_downloader(config.get("obs")) if downloader is _DEFAULT else downloader
        )
        self.runner = runner
        self._persist = persist or submit_complete_result
        self.card_pool = None  # Injectable resource pool for deterministic acceptance tests.

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run_once(self) -> RunOutcome:
        """执行一个任务（或完成一次续清理）。返回执行结果摘要。"""
        from .process_lock import lease_handle
        # Recovery must not erase another live executor's database lock.
        # Hold the OS lease across recovery, installation, execution and cleanup.
        with lease_handle(self.paths.queue_db.with_suffix(".executor.lock")) as fd:
            if fd is None:
                return RunOutcome(None, "idle", "另一个执行进程仍持有执行位")
            self._executor_fd = fd
            try:
                return self._run_once_owned()
            finally:
                self._executor_fd = None

    def _run_once_owned(self) -> RunOutcome:
        # 进程重启恢复：先核实残留状态（文档第 8 节）。
        self.queue.recover_interrupted(_lease_owned=True)

        # 清理阶段未完成的任务优先续清理；清理完成前不开始下一任务。
        for task in self.queue.list_tasks(q.STATE_CLEANING):
            self._resume_cleaning(task)
            if self.queue.get_task(task.task_id).state == q.STATE_CLEANING:
                return RunOutcome(None, 'idle', 'Cleanup pending; executor retained')

        task = self.queue.claim_next()
        if task is None:
            return RunOutcome(None, "idle", "无就绪任务或执行位被占用")
        try:
            return self._execute(task)
        except Exception as e:  # 兜底：未预期异常走中断路径
            self._abort(task, f"执行器未预期异常: {e}")
            return RunOutcome(task.task_id, "failed", str(e))

    def _execute(self, task: q.Task) -> RunOutcome:
        from .device_policy import configured_devices
        allowed = configured_devices(self.config)
        unsupported = [x.get("device") for x in (task.test_manifest or []) if x.get("device") not in allowed]
        if unsupported:
            reason = f"禁止执行设备 {unsupported!r}；只允许 cpu / accelerator，当前启用 {allowed}"
            self.queue.mark_blocked(task.task_id, reason)
            return RunOutcome(task.task_id, "blocked", reason)
        from .artifacts import wheel_identity
        import platform
        identity = wheel_identity(task.obs_wheels_url) or {}
        architecture = identity.get("architecture")
        if architecture and architecture != platform.machine():
            reason = f"wheel 架构 {architecture} 与本机 {platform.machine()} 不匹配；不下载、不安装"
            self.queue.mark_blocked(task.task_id, reason)
            return RunOutcome(task.task_id, "blocked", reason)
        # 阶段 1：准备（下载 + 完整性核验）。资源不齐 → blocked 等待，
        # 不卸载当前环境、不开始测试（文档第 6 节步骤 1–4）。
        ok, detail = self._timed(task, 'prepare', self._prepare, task)
        if not ok:
            return RunOutcome(task.task_id, "blocked", detail)

        # 阶段 2：卸载旧包、安装本任务 wheel（步骤 5–6）。
        # "blocked" = 待实施配置缺失（等待，不动环境）；"failed" = 安装/
        # 核验失败，属执行中断：保存诊断、清理临时资源、标 failed。
        status, detail = self._timed(task, 'install', self._install, task)
        if status == "blocked":
            return RunOutcome(task.task_id, "blocked", detail)
        if status == "failed":
            self._abort(task, detail)
            return RunOutcome(task.task_id, "failed", detail)

        # 阶段 3：按清单执行（步骤 7）。先定位测试树根：nodeid 是仓库
        # 相对路径，pytest 必须在 <源码树>/test 下运行（runner 会剥掉
        # 前导 test/；python -m pytest 把 cwd 前插 sys.path，从仓库根跑
        # 会让源码 torch/ 遮蔽已安装 wheel）。
        source_root = self._locate_test_root(task)
        if source_root is None:
            reason = (
                "任务源码目录中未找到测试树（source/pytorch/test 或 "
                "source/test），无法定位 pytest 运行目录"
            )
            self.queue.mark_blocked(task.task_id, reason)
            return RunOutcome(task.task_id, "blocked", reason)
        # 环境预检：环境级故障（CANN/driver/boot/测试依赖）在此分界为
        # blocked(env)，不进 testing——不与代码嫌疑的 failed 混淆。
        ok, reason = self._timed(task, 'preflight', self._preflight, task, source_root)
        if not ok:
            self.queue.mark_blocked(task.task_id, reason)
            return RunOutcome(task.task_id, "blocked", reason)
        # Parallel mode waits at the head job, not once for the whole PR.
        if not self._parallel_enabled():
            ok, reason, free_cards = self._timed(task, 'device_wait', self._wait_for_free_npu, task)
            if not ok:
                self.queue.mark_blocked(task.task_id, reason or "等待空闲 NPU 卡未成功")
                return RunOutcome(task.task_id, "blocked", reason)
        self.queue.transition(task.task_id, q.STATE_TESTING, "开始按清单执行")
        submission = self._timed(task, 'testing', self._run, task, source_root)
        if submission is None:
            # _run 自行处理了中断状态：blocked（全部 no_result，环境嫌疑）
            # 或 failed（诊断与 error_reason 见任务记录）。
            t = self.queue.get_task(task.task_id)
            if t is not None and t.state == q.STATE_BLOCKED:
                return RunOutcome(task.task_id, "blocked", t.error_reason or "")
            return RunOutcome(task.task_id, "failed", "测试执行中断，见诊断与任务 error_reason")

        # 阶段 4：写库（重试；耗尽后保留证据、不清理）。
        self.queue.transition(task.task_id, q.STATE_WRITING, "开始写库")
        try:
            self._timed(task, 'persist', self._persist_with_retry, task, submission)
        except ExecutorError as e:
            return RunOutcome(task.task_id, "failed", str(e))

        # 阶段 5：立即清理（成功写库后不等待、不设保留期）。
        from .task_staging import atomic_json, task_root
        atomic_json(task_root(self.config, task) / 'cleanup-intent.json',
                    {'outcome': 'done', 'reason': '写库完成，清理完成'})
        self.queue.transition(task.task_id, q.STATE_CLEANING, "写库完成，立即清理")
        try:
            self._timed(task, 'cleanup', self._cleanup_task_artifacts, task)
        except Exception as error:
            self._save_diagnostics(task, str(error), 'cleanup')
            return RunOutcome(task.task_id, 'cleaning', str(error))
        self.queue.transition(task.task_id, q.STATE_DONE, "本任务完成")
        return RunOutcome(task.task_id, "done", str(submission.pr_number))

    # ------------------------------------------------------------------
    # 阶段实现
    # ------------------------------------------------------------------

    def _timed(self, task, stage, operation, *args):
        import time
        from .task_staging import task_root, atomic_json
        path = task_root(self.config, task) / 'executor-progress.json'
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            state = {}
        started = time.monotonic()
        state.update(task_id=task.task_id, stage=stage, updated_at=time.time())
        atomic_json(path, state)
        print(f'[executor] {task.task_id} stage={stage}', flush=True)
        try:
            return operation(*args)
        finally:
            elapsed = time.monotonic() - started
            state.setdefault('stage_seconds', {})[stage] = elapsed
            state['updated_at'] = time.time()
            atomic_json(path, state)
            print(f'[executor] {task.task_id} stage={stage} seconds={elapsed:.3f}', flush=True)

    def _prepare(self, task: q.Task) -> tuple[bool, str]:
        """下载与核验。返回 (ok, detail)；不 ok 时任务已标记 blocked。"""
        if self.downloader is None:
            reason = "OBS 下载方式未配置（obs.downloader，如 obsutil）；按文档不启用任何替代下载路径"
            self.queue.mark_blocked(task.task_id, reason)
            return False, reason

        work_dir = Path(task.work_dir)
        packages = work_dir / "packages"
        source = work_dir / "source"
        temporary = work_dir / "temporary"
        for d in (packages, source, temporary):
            d.mkdir(parents=True, exist_ok=True)

        wheels_pkg = packages / "wheels.tar.gz"
        source_pkg = packages / "pytorch-source.tar.gz"
        # 拆分包格式（2026-10-22 起）：torch_npu 独立对象，与主体包同下。
        npu_pkg = packages / "wheels-npu.tar.gz"
        downloads = [(task.obs_wheels_url, wheels_pkg), (task.obs_source_url, source_pkg)]
        if task.obs_wheels_npu_url:
            downloads.append((task.obs_wheels_npu_url, npu_pkg))
        # 下载重试：次数/间隔来自 store.json（queue.download_retries /
        # download_retry_interval_seconds）；未配置（null）= 不重试，保持
        # "不猜默认值"约定。逐文件重试——源码包大，不因 wheels 失败重下它。
        queue_cfg = (self.config.get("queue") or {})
        max_attempts = 1 + max(int(queue_cfg.get("download_retries") or 0), 0)
        interval = max(float(queue_cfg.get("download_retry_interval_seconds") or 0), 0.0)

        def _download_with_retry(url: str, dest: Path) -> None:
            import time

            for attempt in range(1, max_attempts + 1):
                try:
                    self.downloader.download(url, dest)
                    return
                except ObsDownloadError:
                    if attempt >= max_attempts:
                        raise
                    dest.unlink(missing_ok=True)
                    if interval:
                        time.sleep(interval)

        try:
            from .task_staging import verified_package
            for url, pkg in downloads:
                staged = verified_package(self.config, task, pkg.name)
                if staged:
                    shutil.copy2(staged, pkg)
                    print(f'[reuse] {task.task_id} {pkg.name}', flush=True)
                else:
                    _download_with_retry(url, pkg)
        except ObsDownloadError as e:
            self.queue.bump_retry(task.task_id)
            reason = f"OBS 下载失败（共尝试 {max_attempts} 次）: {e}"
            self.queue.mark_blocked(task.task_id, reason)
            return False, reason

        # 对象与任务版本对应性核验：远端对象名中的短 commit 必须与任务
        # 锁定的完整 commit 一致（这只是完整性检查，不是身份放行依据）。
        # 拆分包的 torch_npu 对象名前缀是 npu 短 commit，pytorch 短 commit
        # 在第二个段——用 wheel_identity 取，取不到再退回文件名首个哈希。
        from .artifacts import wheel_identity
        for url, _pkg in downloads:
            short = (wheel_identity(url) or {}).get("pytorch_short")
            if short is None:
                m = re.search(r"([0-9a-f]{10,40})", Path(url.rsplit("/", 1)[-1]).stem)
                short = m.group(1) if m else None
            if short and not task.pytorch_commit.startswith(short):
                reason = (
                    f"对象 {url} 的版本标识 {short} 与任务锁定的 "
                    f"{task.pytorch_commit} 不对应，不安装、不测试"
                )
                self.queue.mark_blocked(task.task_id, reason)
                return False, reason

        # 解压并确认两个 wheel 与源码齐全（文档步骤 3）。
        try:
            wheels_dir = self._extract(wheels_pkg, packages / "wheels")
            self._extract(source_pkg, source)
            if task.obs_wheels_npu_url:
                self._extract(npu_pkg, packages / "wheels")
        except (tarfile.TarError, OSError) as e:
            reason = f"压缩包损坏或解压失败: {e}"
            self.queue.mark_blocked(task.task_id, reason)
            return False, reason

        torch_whl = self._find_whl(wheels_dir, "torch-")
        torch_npu_whl = self._find_whl(wheels_dir, "torch_npu-")
        if torch_whl is None or torch_npu_whl is None:
            reason = (
                f"wheel 包不齐（torch: {torch_whl}, torch_npu: {torch_npu_whl}）；"
                "资源未齐全，不卸载当前环境、不开始测试"
            )
            self.queue.mark_blocked(task.task_id, reason)
            return False, reason

        # 从包内 torch_npu wheel 读取精确 commit（torch_npu/version.py 的
        # git_version，完整 40 位；2026-09-17 实证），回写任务绑定信息。
        torch_npu_commit = read_torch_npu_git_version(torch_npu_whl)
        if torch_npu_commit is None:
            reason = (
                f"torch_npu wheel {torch_npu_whl.name} 内未找到 git_version，"
                "无法记录精确 torch-npu commit；不安装、不测试"
            )
            self.queue.mark_blocked(task.task_id, reason)
            return False, reason
        # torch_npu 短 commit 与包名一致性：拆分包从独立 npu 对象名取，
        # legacy/qualified 从主体包名取（包内自带 torch_npu）。
        named_npu = None
        if task.obs_wheels_npu_url:
            named_npu = (wheel_identity(task.obs_wheels_npu_url) or {}).get("torch_npu_short")
        if named_npu is None:
            named_npu = (wheel_identity(task.obs_wheels_url) or {}).get("torch_npu_short")
        if named_npu and not torch_npu_commit.startswith(named_npu):
            reason = "torch_npu wheel 内 commit 与 OBS 包名不一致；不安装、不测试"
            self.queue.mark_blocked(task.task_id, reason)
            return False, reason
        self.queue.set_task_torch_npu(
            task.task_id, torch_npu_commit,
            location=f"obs-wheel:{torch_npu_whl.name}",
        )
        return True, ""

    @staticmethod
    def _extract(pkg: Path, dest: Path) -> Path:
        if not tarfile.is_tarfile(pkg):
            raise tarfile.TarError(f"不是有效的 tar 包: {pkg}")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(pkg) as tf:
            tf.extractall(dest, filter="data")
        return dest

    @staticmethod
    def _find_whl(directory: Path, prefix: str) -> Path | None:
        matches = sorted(directory.rglob(f"{prefix}*.whl"))
        return matches[0] if matches else None

    def _install(self, task: q.Task) -> tuple[str, str]:
        """卸载旧 torch-npu/PyTorch 并安装本任务 wheel。

        返回 ("ok"|"blocked"|"failed", detail)。
        """
        python_env = (self.config.get("executor") or {}).get("test_python_env")
        if not python_env:
            reason = (
                "executor.test_python_env 未配置：测试 Python 环境路径需在实施配置中明确，"
                "不得默认使用系统 Python"
            )
            self.queue.mark_blocked(task.task_id, reason)
            return "blocked", reason

        work_dir = Path(task.work_dir)
        wheels_dir = work_dir / "packages" / "wheels"
        torch_whl = self._find_whl(wheels_dir, "torch-")
        torch_npu_whl = self._find_whl(wheels_dir, "torch_npu-")
        torchvision_whl = self._find_whl(wheels_dir, "torchvision-")
        if torch_whl is None or torch_npu_whl is None:
            return "failed", "准备阶段产物缺失，无法安装"

        import subprocess

        from .artifacts import wheel_identity
        identity = wheel_identity(task.obs_wheels_url) or {}
        if identity.get("python_tag"):
            probe = subprocess.run([python_env, "-c", "import sys; print(str(sys.version_info.major)+str(sys.version_info.minor))"],
                                   capture_output=True, text=True, timeout=30, check=False)
            if probe.returncode or probe.stdout.strip() != identity["python_tag"]:
                reason = f"wheel 要求 Python {identity['python_tag']}，测试解释器不匹配；未卸载现有包"
                self.queue.mark_blocked(task.task_id, reason)
                return "blocked", reason
        pip = [python_env, "-m", "pip"]
        logs: list[str] = []
        # 用包管理器卸载旧 torch-npu / PyTorch / torchvision（旧 torchvision
        # 与旧 torch 配套，须一并替换），而不是手工删除 site-packages 文件。
        logs.append(self._run_cmd(
            pip + ["uninstall", "-y", "torch_npu", "torch", "torchvision"],
            check=False,
        ))
        # 安装顺序与参数对齐构建方官方模式（_test-collect.yml:226-228）：
        #   torch 普通安装（带依赖解析，--find-links 优先用包内自带依赖
        #   wheel，避免联网拉取）；torch_npu / torchvision 用 --no-deps，
        #   杜绝 pip 依赖解析从软件源替换其他版本的 torch。
        try:
            logs.append(self._run_cmd(
                pip + ["install", "--find-links", str(wheels_dir), str(torch_whl)],
                check=True,
            ))
            logs.append(self._run_cmd(
                pip + ["install", "--no-deps", str(torch_npu_whl)],
                check=True,
            ))
            if torchvision_whl is not None:
                # torchvision 官方容错（wheel 不在包内时仅告警）。
                try:
                    logs.append(self._run_cmd(
                        pip + ["install", "--no-deps", str(torchvision_whl)],
                        check=True,
                    ))
                except ExecutorError as e:
                    logs.append(f"WARNING: torchvision 安装失败（按官方模式容错）: {e}")
        except ExecutorError as e:
            return "failed", f"pip 安装失败: {e}"

        # 测试框架依赖（executor.test_requirements，如 pytest、parameterized、
        # zstandard、pandas、pydot、z3-solver 等）：官方容器镜像预装，本机
        # 环境需显式安装；torch 已装好，pip 不会从软件源替换它。
        test_reqs = (self.config.get("executor") or {}).get("test_requirements") or []
        if test_reqs:
            try:
                logs.append(self._run_cmd(
                    pip + ["install", *test_reqs], check=True,
                ))
            except ExecutorError as e:
                return "failed", f"测试依赖安装失败: {e}"
        # PyTorch main can require C++20 while Triton-Ascend 3.6.0's runtime
        # launcher still requests C++17. Reapply the guarded compatibility step
        # after pip so a fresh installation cannot silently restore that flag.
        try:
            logs.append(self._run_cmd(
                [python_env, str(Path(__file__).with_name('triton_compat.py'))], check=True,
            ))
        except ExecutorError as e:
            return "failed", f"Triton 编译环境兼容检查失败: {e}"
        self._save_diagnostics(task, "\n".join(logs), "install")

        # 安装后核验：torch 的 git commit 必须与任务锁定版本一致，
        # 防止安装过程被替换成软件源上的其他版本。
        try:
            verify = subprocess.run(
                [python_env, "-c",
                 "import torch; print(torch.version.git_version)"],
                capture_output=True, text=True, timeout=300, check=False,
            )
        except subprocess.TimeoutExpired:
            return "failed", "安装核验（import torch）超时"
        installed = (verify.stdout or "").strip()
        if verify.returncode != 0:
            return "failed", f"安装核验失败：import torch 退出码 {verify.returncode}"
        if installed == "Unknown":
            # 构建方设置 BUILD_WITHOUT_SHA=1，torch wheel 不内嵌 commit
            #（2026-09-17 实证 torch-2.15.0a0+gitunknown）。commit 对应性
            # 已在准备阶段由对象名 sha10 前缀匹配保障，此处不重复核验。
            self._save_diagnostics(
                task, "torch git_version=Unknown（BUILD_WITHOUT_SHA 构建），"
                "跳过安装核验；对应性由对象名 sha 前缀匹配保障", "install-verify",
            )
        elif installed != task.pytorch_commit:
            return "failed", (
                f"安装核验失败：torch git_version={installed!r}，"
                f"任务锁定 {task.pytorch_commit!r}"
            )
        return "ok", ""

    @staticmethod
    def _run_cmd(cmd: list[str], *, check: bool) -> str:
        import subprocess

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        log = (
            f"$ {' '.join(cmd)}\nexit={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        if check and proc.returncode != 0:
            raise ExecutorError(f"命令失败 (exit={proc.returncode}):\n{log[-3000:]}")
        return log

    @staticmethod
    def _locate_test_root(task: q.Task) -> Path | None:
        """定位任务源码包解压后的测试树根（含 test/ 的仓库根）。

        OBS 源码包 tar 根为 pytorch/（成员名 pytorch/test/…）；兼容平铺
        布局。找不到返回 None（调用方 blocked，不安装环境的反面：环境
        在，但运行目录无法确定）。
        """
        source = Path(task.work_dir) / "source"
        for candidate in (source / "pytorch", source):
            if (candidate / "test").is_dir():
                return candidate
        return None

    # ------------------------------------------------------------------
    # 环境预检（安装后、测试前）：环境级故障 blocked(env) 的分界线
    # ------------------------------------------------------------------

    def _preflight_cmd(
        self, task: q.Task, tag: str, cmd: list[str], *, timeout: int,
        cwd: Path | None = None, env: dict | None = None,
        fail_marker: str | None = None,
    ) -> tuple[bool, str]:
        """跑一条预检命令：诊断落盘，返回 (ok, 摘要)。

        fail_marker：输出中命中即视为失败（boot 自检的 "  FAIL  " 行）。
        """
        import subprocess

        shown = " ".join(str(c) for c in cmd)
        try:
            proc = subprocess.run(
                list(cmd), capture_output=True, text=True,
                timeout=timeout, check=False, env=env,
                cwd=str(cwd) if cwd else None,
            )
        except subprocess.TimeoutExpired as e:
            self._save_diagnostics(task, f"$ {shown}\n[预检超时 {timeout}s]\n{e}", tag)
            return False, f"[预检超时 {timeout}s]"
        text = (
            f"$ {shown}\nexit={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        self._save_diagnostics(task, text, tag)
        output = proc.stdout or ""
        ok = proc.returncode == 0 and (fail_marker is None or fail_marker not in output)
        if ok:
            return True, ""
        summary = (proc.stderr or output).strip()[-500:]
        return False, f"exit={proc.returncode}: {summary}"

    def _preflight(self, task: q.Task, source_root: Path) -> tuple[bool, str]:
        """Run only the environment checks required by the manifest.

        CPU-only work must remain runnable when NPU/CANN is unavailable. NPU
        checks are required for accelerator and transfer-CUDA channels.
        """
        import os
        from .runner import _merge_pythonpath
        exec_cfg = self.config.get("executor") or {}
        python_env = exec_cfg.get("test_python_env")
        boot_dir = exec_cfg.get("transfer_boot_dir")
        work_dir = Path(task.work_dir)
        manifest = task.test_manifest or []
        devices = {item.get("device") for item in manifest}
        needs_npu = "accelerator" in devices

        if needs_npu:
            ok, out = self._preflight_cmd(
                task, "preflight-1-npu",
                [python_env, "-c", "import torch, torch_npu; assert torch_npu.npu.is_available(), 'npu unavailable'; print('npu devices:', torch_npu.npu.device_count())"],
                timeout=300, cwd=work_dir,
            )
            if not ok:
                return False, f"环境预检失败（torch_npu/NPU 不可用）: {out}"
            ok, out = self._preflight_cmd(
                task, "preflight-2-op",
                [python_env, "-c", "import torch, torch_npu; x=torch.randn(64,64,device='npu'); print('matmul sum:', float((x@x).sum().item()))"],
                timeout=300, cwd=work_dir,
            )
            if not ok:
                return False, f"环境预检失败（NPU 算子执行失败）: {out}"

        if manifest:
            from .device_policy import test_execution_env, pytest_command
            tf = manifest[0].get("runtime_nodeid") or manifest[0]["test_function"]
            nodeid = tf.split("::", 1)[0]
            nodeid = nodeid[len("test/"):] if nodeid.startswith("test/") else nodeid
            env = test_execution_env(manifest[0]["device"], source_root, tf)
            ok, out = self._preflight_cmd(task, "preflight-4-collect", pytest_command(python_env, manifest[0]["device"]) + [nodeid, "--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider"], timeout=600, cwd=source_root / "test", env=env)
            if not ok:
                # This is target-test evidence, not proof of an environment-wide failure.
                self._save_diagnostics(task, out, "preflight-4-collect-target")
        return True, ""

    def _wait_for_free_npu(self, task: q.Task) -> tuple[bool, str, list[int]]:
        """等卡门禁：accelerator 清单项执行前确认空闲 NPU 卡数量。

        executor.wait_free_npu_cards 未配置（或 0）= 不启用门禁（不猜默认值）。
        配置后：卡够 → 返回空闲卡列表；不够 → 每
        wait_free_npu_interval_seconds 轮询一次，超过
        wait_free_npu_timeout_seconds（0=一直等）→ 返回失败原因（blocked）。
        返回 (ok, 原因, 空闲卡列表)。
        """
        cfg = (self.config.get("executor") or {})
        need = int(cfg.get("wait_free_npu_cards") or 0)
        if need <= 0 or not any(
            x.get("device") == "accelerator" for x in (task.test_manifest or [])
        ):
            return True, "", []
        timeout = float(cfg.get("wait_free_npu_timeout_seconds") or 0)
        interval = max(float(cfg.get("wait_free_npu_interval_seconds") or 30), 1.0)
        cards, why = self.npu_probe(need, timeout, interval)
        if not cards:
            return False, why, []
        return True, "", cards

    def _parallel_enabled(self):
        from .cpu_capacity import worker_capacity
        configured = (self.config.get("executor") or {}).get("test_max_workers", 1)
        return configured == "auto" or worker_capacity(configured)["max_workers"] > 1

    def _execute_families(self, task, source_root, families, extra_env):
        from .runner import PytestRunner, FunctionRunOutcome
        from .test_scheduler import CardPool, CardLease, TestJob, run_ordered
        from .cpu_capacity import worker_capacity, test_thread_env, thread_budget
        cfg = self.config.get("executor") or {}
        capacity = worker_capacity(cfg.get("test_max_workers", 1))
        parallel = cfg.get("test_max_workers") == "auto" or capacity["max_workers"] > 1
        execution_env = test_thread_env(extra_env) if parallel else dict(extra_env or {})
        threads = thread_budget(execution_env)
        capacity_log = self.paths.diagnostics_root / task.task_id / "execution-capacity.json"
        capacity_log.parent.mkdir(parents=True, exist_ok=True)
        capacity_log.write_text(json.dumps({**capacity, "threads_per_test": threads,
                                            "thread_env": {k: v for k, v in execution_env.items()
                                                           if k.endswith("NUM_THREADS")}}, indent=2), encoding="utf-8")
        pool = None
        if parallel:
            has_npu = any(item["device"] == "accelerator" for item, _, _ in families)
            pool = self.card_pool or CardPool(self.paths.queue_dir / "npu-leases",
                                             ids=None if has_npu else [])
        jobs = []
        sizes = []
        for item, nodes, _ in families:
            tf, dev = item["test_function"], item["device"]
            sizes.append(len(nodes))
            for node in nodes:
                need = 0
                if parallel and dev == "accelerator":
                    need = item.get("required_npu_cards")
                    if need is None and hasattr(self.runner, "required_npu_cards"):
                        need = self.runner.required_npu_cards(node, dev, len(pool.ids))
                    elif need is not None and hasattr(self.runner, "required_npu_cards"):
                        need = max(need, self.runner.required_npu_cards(node, dev, 0))
                    if need is None:
                        need = len(pool.ids)
                    if need < 1 or need > len(pool.ids):
                        raise ExecutorError(f"{node}: needs {need} NPU cards; pool has {len(pool.ids)}")

                def execute(lease, node=node, dev=dev, tf=tf):
                    cards = lease.cards
                    env = dict(execution_env)
                    if parallel:
                        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, cards)) if dev == "accelerator" else ""
                    kwargs = dict(source_root=source_root, extra_env=env or None)
                    if isinstance(self.runner, PytestRunner):
                        kwargs["source_test_function"] = tf
                        kwargs["lease_fds"] = lease.fds
                    try:
                        return self.runner.run_function(node, dev, **kwargs)
                    except Exception as error:
                        return FunctionRunOutcome.incomplete(tf, dev, f"runner exception for {node}: {error}")

                jobs.append(TestJob(f"{dev}:{node}", need, execute, cpu_threads=threads))
        if parallel:
            log = self.paths.diagnostics_root / task.task_id / "scheduler.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            def event(**data):
                data["time"] = datetime.now(timezone.utc).isoformat()
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(data, ensure_ascii=False) + "\n")
                if data["state"] == "resource_wait":
                    print(f"[resource_wait] {data['key']} needs {data['required_cards']} cards; admission paused", flush=True)
            ceiling = (capacity["limits"]["online"] if cfg.get("test_max_workers") == "auto"
                       else capacity["max_workers"])
            flat = run_ordered(jobs, pool, max_workers=ceiling,
                               cpu_budget=lambda: worker_capacity(cfg.get("test_max_workers", 1))["available_cpus"],
                               interval=max(0.1, float(cfg.get("test_resource_poll_seconds", 1))), event=event)
        else:
            flat = [job.run(CardLease(())) for job in jobs]
        result, offset = [], 0
        for size in sizes:
            result.append(flat[offset:offset + size])
            offset += size
        return result

    def _run(self, task: q.Task, source_root: Path, extra_env: dict | None = None) -> Submission | None:
        """按清单执行测试，构造一份完整提交。中断时自行处理状态并返回 None。"""
        # 以 DB 最新绑定刷新快照：set_task_torch_npu 只写库不改调用方的
        # 旧内存对象，沿用旧快照会让 torch_npu_commit 停留 None，
        # 写库校验拒绝（2026-09-18 首跑实证）。
        fresh = self.queue.get_task(task.task_id)
        if fresh is not None:
            task = fresh
        if self.runner is None:
            reason = "测试执行接口未配置（runner / executor.device_runner_args）"
            self._abort(task, reason)
            return None
        if not task.test_manifest:
            self._abort(task, "任务缺少函数/device 清单，不得回退全量测试")
            return None

        results: list[FunctionResult] = []
        no_result: list[str] = []
        if hasattr(self.runner, "artifact_root"):
            self.runner.artifact_root = self.paths.diagnostics_root / task.task_id / "artifacts"
        if hasattr(self.runner, "begin_task"):
            self.runner.begin_task()
            fd = getattr(self, "_executor_fd", None)
            self.runner.process_lease_fds = () if fd is None else (fd,)
        families = []
        expansion_limits = []
        parallel = self._parallel_enabled()
        if parallel:
            from .cpu_capacity import test_thread_env
            extra_env = test_thread_env(extra_env)
        for item in task.test_manifest:
            tf, dev = item["test_function"], item["device"]
            explicit_nodes = list(item.get("runtime_nodeids") or [])
            if item.get("runtime_nodeid"):
                explicit_nodes.append(item["runtime_nodeid"])
            runtime_nodes = list(dict.fromkeys(explicit_nodes))
            family_complete = None
            if hasattr(self.runner, "discover_runtime_nodes"):
                try:
                    discovery_env = dict(extra_env or {})
                    if parallel and dev == "cpu":
                        discovery_env["ASCEND_RT_VISIBLE_DEVICES"] = ""
                    discovered = list(dict.fromkeys(self.runner.discover_runtime_nodes(
                        tf, dev, source_root=source_root, extra_env=discovery_env or None)))
                    if not discovered:
                        raise RuntimeError("runtime discovery returned no nodes")
                    if runtime_nodes and tf not in runtime_nodes:
                        stale = sorted(set(runtime_nodes) - set(discovered))
                        if stale:
                            raise RuntimeError(f"selected runtime nodes no longer collected: {stale}")
                    else:
                        runtime_nodes = discovered
                    family_complete = set(runtime_nodes) == set(discovered)
                except Exception as error:
                    log = f"runtime discovery failed for {tf} [{dev}]: {error}"
                    self._save_diagnostics(task, log, f"discover-{tf}-{dev}")
                    no_result.append(f"({tf}, {dev})")
                    results.append(FunctionResult(
                        test_function=tf, device=dev, counts=None, status="no_result",
                        reason=log, log_content=log, execution_status="collection_error",
                        coverage_status="unknown",
                    ))
                    # Never execute the static name after collection has rejected it.
                    continue
            elif not runtime_nodes:
                runtime_nodes = [tf]
            if len(runtime_nodes) > MAX_INSTANCES_PER_FUNCTION:
                expansion_limits.append({
                    "test_function": tf, "device": dev,
                    "selected_count": MAX_INSTANCES_PER_FUNCTION,
                    "requested_count": len(runtime_nodes),
                    "deferred_count": len(runtime_nodes) - MAX_INSTANCES_PER_FUNCTION,
                    "deferred_nodeids": runtime_nodes[MAX_INSTANCES_PER_FUNCTION:],
                })
                runtime_nodes = runtime_nodes[:MAX_INSTANCES_PER_FUNCTION]
                # A bounded sample is not complete source-function coverage.
                family_complete = False
            families.append((item, runtime_nodes, family_complete))

        # Keep canonical source identities for baseline compatibility. Actual
        # execution identities are explicit and separately named per channel.
        resolved_manifest = [
            {"test_function": item["test_function"], "device": item["device"],
             "execution_name": item["device"] + "::" + node,
             "runtime_nodeid": node, "family_complete": complete}
            for item, nodes, complete in families for node in nodes
        ]
        from .task_staging import atomic_json
        atomic_json(self.paths.diagnostics_root / task.task_id / 'execution-manifest.json',
                    {"task_id": task.task_id, "pytorch_commit": task.pytorch_commit,
                     "interpreter": getattr(self.runner, 'python_env', None),
                     "verified_against_execution": hasattr(self.runner, 'discover_runtime_nodes'),
                     "max_instances_per_function_device": MAX_INSTANCES_PER_FUNCTION,
                     "bounded_families": expansion_limits,
                     "items": resolved_manifest})
        family_outcomes = self._execute_families(task, source_root, families, extra_env)
        for (item, runtime_nodes, family_complete), outcomes in zip(families, family_outcomes):
            tf, dev = item["test_function"], item["device"]
            complete = all(outcome.completed for outcome in outcomes)
            logs = "\n\n".join(outcome.log_text for outcome in outcomes if outcome.log_text)
            observed = [outcome.observed_counts or outcome.counts for outcome in outcomes if (outcome.observed_counts or outcome.counts) is not None]
            all_instances = tuple(sorted({instance for outcome in outcomes for instance in outcome.instance_ids}))
            reports = [report for outcome in outcomes for report in outcome.event_reports]
            artifact_refs = [outcome.artifact_dir for outcome in outcomes if outcome.artifact_dir]
            fingerprints = {json.dumps(outcome.environment_fingerprint, sort_keys=True) for outcome in outcomes if outcome.environment_fingerprint is not None}
            env_fingerprint = outcomes[0].environment_fingerprint if len(fingerprints) == 1 else None
            provenances = {json.dumps(outcome.provenance, sort_keys=True) for outcome in outcomes if outcome.provenance is not None}
            provenance = outcomes[0].provenance if len(provenances) == 1 else None
            if not complete:
                # 单函数未取得完整结果：保存已观察到的计数和每个实例的原始事件，
                # 但 counts 仍为 None，避免把部分执行伪装成完整函数结果。
                self._save_diagnostics(task, logs, f"run-{tf}-{dev}")
                no_result.append(f"({tf}, {dev})")
                results.append(
                    FunctionResult(
                        test_function=tf, device=dev, counts=None,
                        status="no_result",
                        reason="未取得所有运行实例的完整结果（" + "; ".join(sorted({o.execution_status for o in outcomes})) + ")，不伪造四个零",
                        log_content=logs or "[无运行日志]",
                        execution_status=next((o.execution_status for o in outcomes if not o.completed), "runner_error"),
                        coverage_status="unknown",
                        instance_ids=all_instances,
                        environment_fingerprint=env_fingerprint,
                        observed_counts=Counts.sum(observed) if observed else None,
                        artifact_ref=artifact_refs[0] if artifact_refs else None,
                        artifact_refs=tuple(artifact_refs),
                        event_reports=reports,
                        provenance=provenance,
                    )
                )
                continue
            results.append(
                FunctionResult(
                    test_function=tf, device=dev,
                    counts=Counts.sum(outcome.counts for outcome in outcomes if outcome.counts is not None),
                    log_content=logs,
                    triage=sorted({sig for outcome in outcomes for sig in outcome.triage}),
                    execution_status="completed",
                    # Compare the selection with the complete collected family.
                    # Both bracket and underscore parameter samples are partial;
                    # a fully enumerated bracket family can still be complete.
                    coverage_status=("partial" if family_complete is False
                                     or (family_complete is None and any("[" in node for node in runtime_nodes))
                                     else "full" if all(outcome.coverage_status == "full" for outcome in outcomes)
                                     else "unknown"),
                    instance_ids=all_instances or tuple(runtime_nodes),
                    environment_fingerprint=env_fingerprint,
                    artifact_ref=artifact_refs[0] if artifact_refs else None,
                    artifact_refs=tuple(artifact_refs),
                    event_reports=reports,
                    provenance=provenance,
                )
            )

        if len(no_result) == len(task.test_manifest):
            # 全部 no_result 仍然持久化：PR 记录和原始 artifacts 是排障证据，
            # 提交不刷新任何基线，也不能用 blocked 丢弃真实错误。
            self._save_diagnostics(
                task,
                "清单全部函数未取得结果；结果将以 no_result 持久化\n" + "; ".join(no_result),
                "all-no-result",
            )

        # 本任务的 PR 日志目录（record 流程会整目录复制到 pr_logs/PR_<n>/）。
        work_dir = Path(task.work_dir)
        pr_log_dir = work_dir / "pr_logs"
        pr_log_dir.mkdir(parents=True, exist_ok=True)
        summary_lines = [
            f"task: {task.task_id}",
            f"PR: {task.pr_number}",
            f"pytorch_commit: {task.pytorch_commit}",
            f"torch_npu_commit: {task.torch_npu_commit}",
            "",
        ]
        for fr in results:
            if fr.is_no_result:
                summary_lines.append(
                    f"{fr.test_function} [{fr.device}] no_result: {fr.reason}"
                )
            else:
                summary_lines.append(
                    f"{fr.test_function} [{fr.device}] {fr.counts.as_dict()}"
                )
            (pr_log_dir / baseline_log_filename(fr.test_function, fr.device)).write_text(
                fr.log_content or "", encoding="utf-8"
            )
        # 合计行（2026-09-20）：no_result 独立可见——0 pass + 0 fail +
        # no_result=N 是"没测到"而非"全过"，摘要层面必须可区分。
        totals = Counts.sum(fr.counts for fr in results if not fr.is_no_result)
        no_result_total = sum(1 for fr in results if fr.is_no_result)
        summary_lines += [
            "",
            f"合计: {totals.as_dict()}；no_result={no_result_total}/{len(results)}"
            "（no_result 不产生计数，不等于四个零）",
        ]
        (pr_log_dir / "summary.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

        wheel_name = self._find_whl(work_dir / "packages" / "wheels", "torch_npu-")
        location = task.torch_npu_location or (
            wheel_name.name if wheel_name else "unknown"
        )
        # 合入时刻：任务已登记的优先；缺失时解析一次（Git 不可用则 None，
        # 防护回退祖先关系），并持久化回任务供重测沿用。
        merged_at = task.merged_at
        provenance = task.merged_at_provenance
        if not merged_at:
            try:
                from .upstream import PyTorchMirror
                merged_at = PyTorchMirror(config=self.config).commit_timestamp(
                    task.pytorch_commit)
                provenance = 'upstream-main-commit-time' if merged_at else None
            except Exception:
                merged_at = provenance = None
            if merged_at:
                self.queue.set_task_merged_at(task.task_id, merged_at, provenance)
        return Submission(
            pr_number=task.pr_number,
            pytorch_commit=task.pytorch_commit,
            torch_npu_environment={
                "location": location,
                "torch_npu_commit": task.torch_npu_commit,
            },
            function_results=results,
            pr_log_source_dir=pr_log_dir,
            merged_at=merged_at,
            merged_at_provenance=provenance,
        )

    def _persist_with_retry(self, task: q.Task, submission: Submission) -> SubmitReport:
        attempts = 3
        last: Exception | None = None
        for _ in range(attempts):
            try:
                return self._persist(self.config, submission)
            except Exception as e:  # 写库失败：保留证据重试（文档第 8 节）
                last = e
        reason = (
            f"写库失败（重试 {attempts} 次）: {last}；"
            f"结果证据保留于 {task.work_dir}，未清理"
        )
        self._save_diagnostics(task, str(last), "persist")
        # writing → failed：不进入清理，防止删除唯一结果副本。
        self.queue.transition(task.task_id, q.STATE_FAILED, reason)
        raise ExecutorError(reason)

    # ------------------------------------------------------------------
    # 清理与诊断
    # ------------------------------------------------------------------

    def _cleanup_task_artifacts(self, task: q.Task) -> None:
        from .task_staging import remove_staging
        self._safe_cleanup(task.work_dir)
        remove_staging(self.config, task)
        # Retire legacy source cache only when no other live task references it.
        sha = task.pytorch_commit or ''
        if re.fullmatch(r'[0-9a-f]{40,64}', sha):
            base = (self.paths.data_root / 'source_cache').resolve()
            cache = base / sha[:10]
            active = [t for t in self.queue.list_tasks() if t.task_id != task.task_id
                      and t.state not in q.TERMINAL_STATES
                      and (t.pytorch_commit or '').startswith(sha[:10])]
            if cache.exists() and not active:
                if cache.is_symlink() or cache.resolve().parent != base:
                    raise ExecutorError('Unsafe legacy source cache path')
                shutil.rmtree(cache)

    def _safe_cleanup(self, work_dir: str | Path) -> None:
        """删除本任务临时目录；绝对路径必须位于 task_work 根内。

        不得拼接未校验路径递归删除，不得扫描全盘清理（文档第 7 节）。
        """
        root = self.paths.task_work_root.resolve()
        target = Path(work_dir).resolve()
        if target == root or root not in target.parents:
            raise ExecutorError(
                f"拒绝删除任务临时目录之外的路径: {target}（task_work 根: {root}）"
            )
        if target.exists():
            shutil.rmtree(target)

    def _resume_cleaning(self, task: q.Task) -> None:
        try:
            self._cleanup_task_artifacts(task)
            # An aborted installation/test must remain failed after a cleanup
            # retry; deleting its files does not create a completed result.
            from .task_staging import task_root
            intent = task_root(self.config, task) / 'cleanup-intent.json'
            value = json.loads(intent.read_text()) if intent.exists() else {}
            aborted = value.get('outcome') == 'failed' or (task.error_reason or '').startswith('中断:')
            self.queue.transition(task.task_id, q.STATE_FAILED if aborted else q.STATE_DONE,
                                  value.get('reason', '清理完成（中断续清理）'))
        except Exception as e:
            attempts = self.queue.bump_cleanup_attempt(task.task_id)
            limit = int((self.config.get("queue") or {}).get("cleanup_retries") or 5)
            if attempts >= limit:
                self.queue.transition(
                    task.task_id, q.STATE_FAILED,
                    f"清理重试 {attempts} 次仍失败: {e}；请人工处理临时目录",
                )

    def _abort(self, task: q.Task, reason: str) -> None:
        """中断路径：保存诊断 → 清理本任务临时资源 → 标记 failed。"""
        self._save_diagnostics(task, reason, "abort")
        try:
            from .task_staging import atomic_json, task_root
            atomic_json(task_root(self.config, task) / 'cleanup-intent.json',
                        {'outcome': 'failed', 'reason': reason})
            self.queue.transition(task.task_id, q.STATE_CLEANING, f"中断: {reason}")
            if task.work_dir:
                self._cleanup_task_artifacts(task)
            self.queue.transition(task.task_id, q.STATE_FAILED, reason)
        except q.QueueError:
            # 状态已处于终态（可能此前已处理），诊断已保存即可。
            pass
        except Exception as error:
            # Keep cleaning pending, retain the abort outcome, and let the next
            # executor turn retry under its lease rather than escaping twice.
            self._save_diagnostics(task, str(error), 'cleanup')

    def _save_diagnostics(self, task: q.Task, text: str, tag: str) -> None:
        d = self.paths.diagnostics_root / task.task_id
        d.mkdir(parents=True, exist_ok=True)
        # tag 可能含函数路径中的 "/" 与 "::"，清理为安全文件名。
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
        (d / f"{safe_tag}-{_ts()}.log").write_text(str(text), encoding="utf-8")
