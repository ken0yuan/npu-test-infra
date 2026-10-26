"""上游 PyTorch main 镜像：clone/fetch、新 commit 发现与 PR 号解析。

方案（用户 2026-09-17 会话确认）：
- 定时（默认 1 小时）从 GitHub 拉取 pytorch main，只做 git 同步；
- 每轮找出新出现的 commit，从 subject 的 "(#PR号)" 解析 PR 号并记录
  （构建方 receive-trigger.yml 依赖同一约定：land bot push 到 main 的
  commit subject 一定含 "(#N)"）；无 PR 号的 commit 按用户确认完全跳过；
- 对已记录、未入队的 commit，在 OBS 桶中定位配套的
  wheels-<sha10>.tar.gz 与 pytorch-source-<sha10>.tar.gz，存在则入队；
- 镜像只作为身份与信息来源（blob:none 裸镜像，无工作区文件），
  不作为被测资源——被测 wheel 与源码仍只从 OBS 下载。

实证（2026-09-17，wheels-0051b809f7.tar.gz 与三个 workflow 全文）：
- 包体含 torch + torch_npu + torchvision + 纯 Python 依赖，同链配套构建；
- torch_npu wheel 内 torch_npu/version.py 的 git_version 为完整 40 位
  commit，由执行端下载后读取并回写任务（torch_npu_commit 入队时为空）；
- torch wheel 的 git_version 为 'Unknown'（BUILD_WITHOUT_SHA=1），
  不能用于安装核验，commit 对应性靠对象名 sha10 前缀匹配；
- 桶中存在两类包：Merged 标签路径（main 上 commit）与 ciflow trunk
  tag 路径（PR head，未必在 main）。本方案只处理 main fetch 到的
  commit，天然排除后者。
"""

from __future__ import annotations

import re
import os
import subprocess
from dataclasses import dataclass, field
import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import store_paths
from .task_queue import STATE_IDENTITY_PENDING, QueueStore

PYTORCH_UPSTREAM_URL = "https://github.com/pytorch/pytorch.git"

# land bot 格式：subject 尾部的 (#12345)。
_PR_IN_SUBJECT_RE = re.compile(r"\(#(\d+)\)")

_LOG_FMT = "%H%x00%ct%x00%s"


def detect_inversions(entries: list[tuple[str, int]]) -> list[dict]:
    """first-parent 新→旧序列里的时间倒序（子提交时间 < 父提交时间）。

    官方 main 的 committer 时钟在生产观测为零倒序（2026-10-26 Codex
    核实最近 1000 first-parent）；此处只做检测与持久化，不改排序。
    """
    inversions = []
    for i in range(1, len(entries)):
        sha, ts = entries[i]
        prev_sha, prev_ts = entries[i - 1]
        if ts > prev_ts:  # 越走越旧时间却变新 → 倒序
            inversions.append({"sha": sha, "commit_time": ts,
                               "child_time": prev_ts})
    return inversions


ORDER_SNAPSHOT_SCHEMA = "upstream-order/1"


def _order_digest(snapshot):
    payload = {key: snapshot[key] for key in
               ('schema', 'source', 'tip', 'checked_tip', 'checked', 'order', 'times')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_merge_order(mirror, *, limit: int = 1000) -> dict:
    """fetch 时一次校验并构建**合入序快照**（用户 2026-10-26 评审阻塞项
    1/4：成员绑定——持久化完整 first-parent SHA 序列，提交守卫仅在双方
    完整 commit 都在快照内时用位置比较排序，时间戳降级为元数据）。

    返回快照 dict（schema/source/tip/checked/order/times/inversions），
    由调用方经 persist_order_snapshot 原子落盘。
    """
    fmt = "%H%x00%ct"
    out = _git(mirror.git_dir, ["log", "--first-parent", f"--max-count={limit}",
                                f"--format={fmt}", "main"])
    shas, times = [], []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, ts = (line.split("\x00", 1) + [""])[:2]
        if ts.isdigit():
            shas.append(sha)
            times.append(int(ts))
    if not shas or any(not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', sha) for sha in shas):
        raise UpstreamError('Invalid or empty first-parent history')
    entries = list(zip(shas, times))
    snapshot = {"schema": ORDER_SNAPSHOT_SCHEMA,
            "source": mirror.repo_url,
            "tip": shas[0] if shas else None,
            "checked_tip": shas[0] if shas else None,
            "limit": limit,
            "checked": len(shas),
            "order": shas,          # 新→旧完整 first-parent 序列（成员绑定）
            "times": times,         # 元数据（committer 时刻，不参与排序）
            "inversions": detect_inversions(entries)}
    snapshot['sha256'] = _order_digest(snapshot)
    return snapshot


def persist_order_snapshot(config, snapshot: dict) -> Path:
    """原子落盘顺序快照（评审阻塞项 1：临时文件 + os.replace）。"""
    import tempfile
    paths = store_paths(config)
    target = paths.data_root / "upstream-order.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def load_order_snapshot(config, *, expected_source=PYTORCH_UPSTREAM_URL) -> dict | None:
    """读取并校验顺序快照；缺失/损坏/schema 不符 → None（调用方回退
    本地祖先，绝不只凭标签信任）。"""
    paths = store_paths(config)
    path = paths.data_root / "upstream-order.json"
    if not path.is_file():
        return None
    try:
        snap = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        order, times = snap['order'], snap['times']
        if (snap['schema'] != ORDER_SNAPSHOT_SCHEMA or snap['source'] != expected_source
                or not isinstance(order, list) or not order
                or any(not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', sha) for sha in order)
                or len(set(order)) != len(order)
                or type(snap['checked']) is not int or snap['checked'] != len(order)
                or snap['tip'] != order[0] or snap['checked_tip'] != order[0]
                or not isinstance(times, list) or len(times) != len(order)
                or any(type(ts) is not int or ts < 0 for ts in times)
                or snap.get('sha256') != _order_digest(snap)):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    # Clock inversions are diagnostic metadata. First-parent positions still
    # establish order; timestamps must never override that relationship.
    return snap


def order_position(snapshot: dict, sha: str) -> int | None:
    """完整 commit 在快照内的位置（新→旧）；未覆盖返回 None。"""
    try:
        return snapshot["order"].index(sha)
    except (KeyError, ValueError):
        return None


def _parse_log(out: str) -> list["UpstreamCommit"]:
    commits: list[UpstreamCommit] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, ts, subject = (line.split("\x00", 2) + ["", ""])[:3]
        commit_time = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
        m = _PR_IN_SUBJECT_RE.search(subject)
        pr = int(m.group(1)) if m else None
        commits.append(
            UpstreamCommit(sha=sha, pr_number=pr, commit_time=commit_time, subject=subject)
        )
    return commits


class UpstreamError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpstreamCommit:
    sha: str
    pr_number: int | None
    commit_time: str
    subject: str
    discovered_via: str = "hourly-fetch"


def _git(git_dir: Path, args: list[str], *, timeout: int = 1800) -> str:
    try:
        proc = subprocess.run(
            ["git", "--git-dir", str(git_dir), *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise UpstreamError(f"git 执行失败 ({args}): {e}") from e
    if proc.returncode != 0:
        raise UpstreamError(
            f"git {' '.join(args)} 失败 (exit={proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[-1500:]}"
        )
    return proc.stdout


@dataclass
class PyTorchMirror:
    repo_url: str = PYTORCH_UPSTREAM_URL
    mirror_dir: Path | None = None  # 默认 <data_root>/pytorch-main.git
    config: dict | None = None

    def __post_init__(self):
        if self.mirror_dir is None:
            paths = store_paths(self.config or {})
            self.mirror_dir = paths.data_root / "pytorch-main.git"

    @property
    def git_dir(self) -> Path:
        return Path(self.mirror_dir)

    def exists(self) -> bool:
        return (self.git_dir / "HEAD").is_file()

    def ensure(self) -> None:
        """镜像不存在时克隆（blob:none 裸镜像，只含 commit/tree 元数据）。"""
        if self.exists():
            origin = _git(self.git_dir, ["remote", "get-url", "origin"]).strip()
            # 本机 git 可能配置 url.<proxy>.insteadOf 把 GitHub URL 透明重写为
            # 代理形式（如 https://ghfast.top/https://github.com/...）。
            # 接受这种等价形式，但把 origin 显式改回上游 URL，保持镜像
            # 元数据与配置一致；fetch 是否经代理由本机 git 配置决定。
            if origin != self.repo_url and not origin.endswith(f"/{self.repo_url}"):
                raise UpstreamError(
                    f"镜像 origin 是 {origin!r}，与要求的 {self.repo_url!r} 不一致"
                )
            if origin != self.repo_url:
                _git(self.git_dir, ["remote", "set-url", "origin", self.repo_url])
            return
        self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            cloned = subprocess.run(
                [
                    "git", "clone", "--bare", "--filter=blob:none",
                    "--single-branch", "--branch", "main",
                    self.repo_url, str(self.git_dir),
                ],
                capture_output=True, text=True, timeout=7200, check=False,
            )
            if cloned.returncode:
                raise UpstreamError(f"镜像克隆失败 (exit={cloned.returncode}): {(cloned.stderr or cloned.stdout)[-1500:]}")
        except (OSError, subprocess.TimeoutExpired) as e:
            raise UpstreamError(f"镜像克隆失败: {e}") from e
        if not self.exists():
            raise UpstreamError("镜像克隆后仍不存在，请检查网络与磁盘")

    def fetch(self) -> str:
        """拉取最新 main（fetch 并前进本地 main 引用），返回新 tip。"""
        self.ensure()
        _git(self.git_dir, ["fetch", "--quiet", "origin", "main:refs/heads/main"])
        return self.tip()

    def tip(self) -> str:
        return _git(self.git_dir, ["rev-parse", "main"]).strip()

    def commit_before(self, when: datetime) -> str | None:
        """返回指定时刻之前 main 上最新的 commit（用于首轮回溯基线）。"""
        iso = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
        out = _git(
            self.git_dir, ["rev-list", "-1", f"--before={iso}", "main"]
        ).strip()
        return out or None

    def commits_between(self, old_tip: str, new_tip: str) -> list[UpstreamCommit]:
        """列出 (old_tip, new_tip] 之间的新 commit（旧→新排序）。"""
        out = _git(
            self.git_dir,
            ["log", "--reverse", f"--format={_LOG_FMT}", f"{old_tip}..{new_tip}"],
        )
        return _parse_log(out)

    def commits_since(self, when: datetime) -> list[UpstreamCommit]:
        """列出指定时刻之后的 main commit（首轮回溯用，时间窗口语义）。"""
        iso = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
        out = _git(
            self.git_dir,
            ["log", "--reverse", f"--format={_LOG_FMT}", f"--since={iso}", "main"],
        )
        return _parse_log(out)

    def is_on_main(self, sha: str) -> bool:
        """完整/短 sha 是否可解析且是 main 的祖先（用于桶中历史对象实证）。"""
        proc = subprocess.run(
            ["git", "--git-dir", str(self.git_dir), "rev-parse", "--verify",
             "--quiet", f"{sha}^{{commit}}"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            return False
        full = proc.stdout.strip()
        anc = subprocess.run(
            ["git", "--git-dir", str(self.git_dir), "merge-base", "--is-ancestor",
             full, "main"],
            capture_output=True, text=True, check=False,
        )
        return anc.returncode == 0

    def commit_timestamp(self, sha: str) -> str | None:
        """main 历史上该 commit 的 committer 时间（ISO UTC）。

        land bot 以 squash merge 落 main，该时刻即合入时刻；来源记录为
        upstream-main-commit-time。解析失败/对象不存在返回 None。
        """
        try:
            out = _git(self.git_dir, ["log", "-1", "--format=%ct", sha])
        except UpstreamError:
            return None  # git 不可用/对象缺失：时间未知，调用方保守处理
        line = out.strip().splitlines()[-1] if out.strip() else ""
        if not line.isdigit():
            return None
        return datetime.fromtimestamp(int(line), tz=timezone.utc).isoformat()

    def resolve_full_sha(self, short: str) -> str | None:
        """把短 sha 解析为完整 sha（歧义或不存在返回 None）。"""
        proc = subprocess.run(
            ["git", "--git-dir", str(self.git_dir), "rev-parse", "--verify",
             "--quiet", f"{short}^{{commit}}"],
            capture_output=True, text=True, check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    def merge_relation(self, a: str, b: str) -> str:
        """a 相对 b 的合入关系：'equal' | 'a_earlier' | 'a_later' | 'unknown'。

        main 线性历史上祖先关系即合入先后；git 调用失败或双向都不是
        祖先（如 commit 不在镜像历史）返回 'unknown'，调用方按保守处理。
        """
        if a == b:
            return 'equal'
        if self._is_ancestor_raw(a, b) is True:
            return 'a_earlier'
        if self._is_ancestor_raw(b, a) is True:
            return 'a_later'
        return 'unknown'

    def _is_ancestor_raw(self, older: str, newer: str):
        """True/False 为祖先判定；None 表示 git 失败（关系未知）。"""
        proc = subprocess.run(
            ["git", "--git-dir", str(self.git_dir), "merge-base", "--is-ancestor",
             older, newer],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0:
            return True
        return False if proc.returncode == 1 else None


# ---------------------------------------------------------------------------
# 每轮循环：fetch → 记录新 commit → 定位桶中对象 → 入队
# ---------------------------------------------------------------------------

@dataclass
class CycleReport:
    fetched_tip: str = ""
    new_commits: int = 0
    recorded: int = 0
    skipped_no_pr: int = 0
    pending_before: int = 0
    enqueued: list[dict] = field(default_factory=list)
    still_missing: list[str] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# 上游 commit 记录后等待桶中对象出现的最长时间；超过则不再查找
#（桶内对象保留期约 7 天，构建一般 1–2 小时内完成）。
OBJECT_WAIT_DAYS = 7


def run_cycle(
    config: dict,
    queue: QueueStore | None = None,
    mirror: PyTorchMirror | None = None,
    downloader=None,
    *,
    first_run_backfill_hours: float = 2.0,
) -> CycleReport:
    """一轮同步：fetch main → 记录新 commit（含 PR）→ 对未入队 commit
    定位桶中 wheels/source 对象，存在则以 main 观察到的身份入队。

    - 无 PR 号的 commit 只记 no_pr，不入队（用户确认：完全跳过）；
    - 对象不齐的 commit 保持 pending，下轮再查；发现超过 7 天标记 expired；
    - 入队后停在 prefilter_pending，等待分析端（下一个 agent 串接）。
    """
    from .obs import make_downloader

    paths = store_paths(config)
    queue = queue or QueueStore(config)
    mirror = mirror or PyTorchMirror(config=config)
    if downloader is None:
        downloader = make_downloader(config.get("obs"))
    report = CycleReport()

    # 1. fetch 并发现新 commit。
    report.fetched_tip = mirror.fetch()
    last_tip = queue.get_mirror_tip()
    if last_tip is None:
        # 首轮：回溯 backfill 窗口（时间语义：只记录最近 N 小时的 commit，
        # 干净起步，不处理全部历史）。
        cutoff = datetime.now(timezone.utc) - timedelta(hours=first_run_backfill_hours)
        new = mirror.commits_since(cutoff)
        via = "first-run-backfill"
    else:
        new = mirror.commits_between(last_tip, report.fetched_tip)
        via = "hourly-fetch"
    report.new_commits = len(new)
    for c in new:
        c = UpstreamCommit(
            sha=c.sha, pr_number=c.pr_number, commit_time=c.commit_time,
            subject=c.subject, discovered_via=via,
        )
    rec, no_pr = queue.record_upstream_commits([c.__dict__ for c in new])
    report.recorded = rec
    report.skipped_no_pr = no_pr
    queue.set_mirror_tip(mirror.repo_url, report.fetched_tip)

    # fetch 时一次合入序校验 + 原子快照（评审阻塞项 1：此前只定义未调用）。
    try:
        snapshot = validate_merge_order(mirror)
        persist_order_snapshot(config, snapshot)
        if snapshot["inversions"]:
            report.merge_order_inversions = snapshot["inversions"]
    except Exception as e:  # 校验失败不阻塞轮；提交守卫回退本地祖先
        report.errors.append(f"合入序快照失败: {e}")

    # 2. 对未入队 commit 定位桶中对象：先取桶快照（一次 ls），
    #    再在内存中匹配对象名，避免对每个 commit 逐个 stat。
    if downloader is None:
        report.errors.append(
            "OBS 下载方式未配置（obs.downloader），本轮无法定位桶中对象"
        )
        return report

    pending = queue.pending_upstream_commits()
    report.pending_before = len(pending)
    now = datetime.now(timezone.utc)
    bucket = f"obs://{config.get('obs', {}).get('bucket', 'pytorch-ci-cache')}/7day-cache"
    try:
        listing = downloader.list(f"{bucket}/")
    except Exception as e:
        report.errors.append(f"桶列表获取失败: {e}")
        return report

    for row in pending:
        sha10 = row["sha"][:10]
        from .artifacts import select_bundle, wheel_identity
        try:
            bundle = select_bundle(listing, bucket, sha10, config)
        except ValueError as error:
            report.errors.append(str(error))
            queue.update_upstream_commit(row["sha"], state="pending", note=str(error))
            continue
        wheels_url = bundle["wheels"]
        npu_url = bundle["wheels_npu"]
        source_url = f"{bucket}/pytorch-source-{sha10}.tar.gz"
        wheels_meta = listing.get(wheels_url) if wheels_url else None
        source_meta = listing.get(source_url)
        npu_meta = listing.get(npu_url) if npu_url else None
        # 拆分包格式下 torch_npu 独立成对象；主体包已选中但 npu 包未出现
        # 视为对象不齐，继续等待（同 source 缺失语义）。
        npu_required = bool(wheels_url) and (
            wheel_identity(wheels_url) or {}
        ).get("format") == "split-pytorch"
        if (wheels_meta is None or source_meta is None
                or (npu_required and npu_meta is None)):
            discovered = datetime.fromisoformat(row["discovered_at"])
            if discovered.tzinfo is None:
                discovered = discovered.replace(tzinfo=timezone.utc)
            if now - discovered > timedelta(days=OBJECT_WAIT_DAYS):
                queue.update_upstream_commit(
                    row["sha"], state="expired",
                    note="桶中长期无对应对象，不再查找（保留期约 7 天）",
                )
                report.expired.append(sha10)
            else:
                report.still_missing.append(sha10)
            continue

        # 对象齐全：登记候选并按 main 观察规则核实身份。
        task, created = queue.register_candidate(
            wheels_url, source_url,
            obs_wheels_version=wheels_meta.get("etag"),
            obs_source_version=source_meta.get("etag"),
            obs_wheels_npu_url=npu_url,
            obs_wheels_npu_version=npu_meta.get("etag") if npu_meta else None,
            obs_remote_time=wheels_meta.get("last_modified"),
        )
        if not created:
            # 该对象组合已登记过（例如上轮核实后中断重试）：
            # 已离开 identity_pending 的任务不重复核实，直接视为已入队。
            existing = queue.get_task(task.task_id)
            if existing is not None and existing.state not in (
                STATE_IDENTITY_PENDING,
            ):
                queue.update_upstream_commit(
                    row["sha"], state="queued", task_id=task.task_id,
                    note=f"任务已在流程中（{existing.state}）",
                )
                report.enqueued.append(
                    {
                        "task_id": task.task_id,
                        "pr_number": row["pr_number"],
                        "sha10": sha10,
                        "subject": row["subject"],
                    }
                )
                continue
        queue.register_identity_rule(
                "upstream-main-fetch",
                "每小时从 github.com/pytorch/pytorch main fetch；commit 直接来自 "
                "main 历史（已是 main 祖先），PR 号从 commit subject 的 (#N) 解析；"
                "仅处理 main fetch 到的 commit，天然排除 ciflow trunk 阶段的 "
                "PR head 包。torch_npu 随 wheels 包体自带，其精确 commit 由执行端"
                "从包内 torch_npu/version.py 的 git_version 读取后回写。",
                "用户 2026-09-17 会话确认（依据 Ascend/pytorch CI workflow 实证）",
            )
        task = queue.verify_identity(
            task.task_id, "upstream-main-fetch",
            {
                "pr_number": row["pr_number"],
                "pytorch_commit": row["sha"],
                "torch_npu_commit": None,  # 执行端从包内 wheel 解析后回写
                "binding": (
                    f"commit {row['sha'][:10]} 于 {row['discovered_at']} 由定时 "
                    f"fetch 从上游 main 观察到；subject: {row['subject']}"
                ),
            },
        )
        queue.update_upstream_commit(
            row["sha"], state="queued", task_id=task.task_id,
            note=f"对象齐全，已入队（wheels@{wheels_meta.get('last_modified')}）",
        )
        # 任务级合入时刻持久化（upstream-main-commit-time）：提交侧基线
        # 合入序比较优先用它，避免每次提交做实时 Git 祖先查询。
        if row.get("commit_time"):
            queue.set_task_merged_at(task.task_id, row["commit_time"],
                                     "upstream-main-commit-time")
        report.enqueued.append(
            {
                "task_id": task.task_id,
                "pr_number": row["pr_number"],
                "sha10": sha10,
                "subject": row["subject"],
            }
        )
    return report
