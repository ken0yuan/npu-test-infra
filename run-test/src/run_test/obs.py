"""OBS 下载层。

对应《OBS监测与串行测试队列说明》第 2 节硬性要求：
- 测试服务器只从 OBS 获取 PyTorch wheel、torch-npu wheel 和源码；
- 不访问 GitHub 拉取记录、不使用 GitHub token、不通过 ghfast 或
  git clone/pull 补源码、不在本地重建 wheel；
- 下载方式必须由部署配置显式给出（如 obsutil）。未配置时执行端在该阶段
  记 blocked，不偷偷启用任何联网认证路径，也不自动安装下载工具。
"""

from __future__ import annotations

import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

# 超时分级：stat/ls 是控制面查询；下载按最坏包体（779MB 源码包）× 链路
# 劣化（~500KB/s ≈ 26 分钟）校准。原先全局 3600s 在链路挂死时会让单个
# 任务烧 1 小时（2026-09-18 实证）；900s 又会误杀慢速链路下的大包下载
# （同日实证：779MB @ <900KB/s 两次尝试全超时）。
STAT_LS_TIMEOUT = 120
DOWNLOAD_TIMEOUT = 1800
DOWNLOAD_ATTEMPTS = 2
DOWNLOAD_RETRY_INTERVAL = 5


class ObsDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadedObject:
    url: str
    dest: Path
    size: int


class ObsutilDownloader:
    """通过 obsutil CLI 下载对象（obs://bucket/path → 本地路径）。"""

    def __init__(self, bin_path: str = "obsutil", *, regional_proxy_endpoint=None, bucket=None):
        self.bin_path = bin_path
        self.regional_proxy_endpoint = regional_proxy_endpoint
        self.bucket = bucket

    def _run(self, args: list[str], timeout: float) -> subprocess.CompletedProcess:
        try:
            if self.regional_proxy_endpoint:
                from .obs_transport import ObsRegionalProxy
                route = ObsRegionalProxy(self.regional_proxy_endpoint, self.bucket)
            else:
                route = nullcontext(None)
            with route as proxy:
                options = ['-proxy-url=' + proxy] if proxy else []
                return subprocess.run(
                    [self.bin_path, *args, *options], capture_output=True, text=True,
                    timeout=timeout, check=False,
                )
        except FileNotFoundError as e:
            raise ObsDownloadError(
                f"找不到下载工具 {self.bin_path!r}；OBS 下载方式未就绪，不自动安装"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise ObsDownloadError(f"obsutil 调用超时({timeout}s): {args}") from e

    def stat(self, url: str) -> dict | None:
        """查询对象元数据；不存在（或查询失败）返回 None。

        返回 {last_modified, etag, size}——ETag 作为远端对象版本标识
        （同名覆盖上传会变化，满足“对象更新必须重新核实”），
        LastModified 作为远端时间（不用本地发现时间冒充）。
        """
        proc = self._run(["stat", url], timeout=STAT_LS_TIMEOUT)
        if proc.returncode != 0:
            return None
        out = proc.stdout or ""
        import re

        def _field(name: str) -> str | None:
            m = re.search(rf"{name}\s*[:：]\s*(\S+)", out)
            return m.group(1) if m else None

        lm = _field("Last Modified")
        etag = _field("ETag")
        if etag:
            etag = etag.strip('"')
        size = _field(r"Size \(KB\)")
        return {
            "last_modified": lm,
            "etag": etag,
            "size_kb": size,
        }

    def list(self, prefix_url: str) -> dict[str, dict]:
        """列出一个前缀下的全部对象，返回 {url: {last_modified, etag, size}}。

        用于 cycle 的一次性桶快照（避免对每个 commit 逐个 stat）。
        解析 `obsutil ls` 的“对象名行 + 元数据行”两行式表格输出。
        """
        proc = self._run(["ls", prefix_url, "-limit=10000"], timeout=STAT_LS_TIMEOUT)
        if proc.returncode != 0:
            raise ObsDownloadError(
                f"obsutil ls 失败 (exit={proc.returncode}): {prefix_url}\n"
                f"{(proc.stderr or proc.stdout)[-1500:]}"
            )
        import re

        out = proc.stdout or ""
        entries = re.findall(
            r"^(obs://\S+)\s*\n\s+(\d{4}-\d{2}-\d{2}T\S+Z)\s+(\S+)\s+\S+\s+(\"[^\"]*\"|\S+)",
            out, re.MULTILINE,
        )
        result: dict[str, dict] = {}
        for url, mtime, size, etag in entries:
            result[url] = {
                "last_modified": mtime,
                "etag": etag.strip('"'),
                "size": size,
            }
        return result

    def download(self, url: str, dest: Path) -> DownloadedObject:
        if not isinstance(url, str) or not url.strip():
            raise ObsDownloadError("下载 URL 不能为空")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # 快速失败 + 一次重试：链路劣化时挂死等价于任务烧满超时（实证 1h/个）。
        last_error: ObsDownloadError | None = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                proc = self._run(["cp", "-f", url, str(dest)], timeout=DOWNLOAD_TIMEOUT)
            except ObsDownloadError as e:  # 超时（工具缺失不重试，直接抛）
                if "找不到下载工具" in str(e):
                    raise
                last_error = e
            else:
                if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
                    return DownloadedObject(url=url, dest=dest, size=dest.stat().st_size)
                last_error = ObsDownloadError(
                    f"obsutil 下载失败 (exit={proc.returncode}): {url}\n"
                    f"{(proc.stderr or proc.stdout)[-2000:]}"
                )
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(DOWNLOAD_RETRY_INTERVAL)
        assert last_error is not None
        raise last_error


def artifact_prefixes(obs_cfg: dict | None) -> tuple[str, str]:
    """Wheel and source namespaces; legacy single-bucket configurations still work."""
    obs_cfg = obs_cfg or {}
    legacy = obs_cfg.get('bucket', 'pytorch-ci-cache')
    def prefix(role):
        bucket = (obs_cfg.get(role) or {}).get('bucket', legacy)
        if not isinstance(bucket, str) or not bucket or any(c in bucket for c in '/?#@:'):
            raise ObsDownloadError(f'Invalid OBS {role} bucket: {bucket!r}')
        return f'obs://{bucket}/7day-cache'
    return prefix('wheels'), prefix('source')


def list_artifacts(obs_cfg, downloader):
    """Only publish a combined snapshot after every required namespace succeeds."""
    wheels, source = artifact_prefixes(obs_cfg)
    listing = {}
    for prefix in dict.fromkeys([wheels, source]):
        try:
            entries = downloader.list(prefix + '/')
            if any(not url.startswith(prefix + '/') for url in entries):
                raise ObsDownloadError('Listing returned objects outside its requested namespace')
            listing.update(entries)
        except Exception as error:
            raise ObsDownloadError(f'OBS listing failed for {prefix}: {error}') from error
    return listing, wheels, source


class RoutedObsDownloader:
    """Route exact bucket URLs, including old queued artifacts, without credential fallback."""
    def __init__(self, routes):
        self.routes = routes

    def _route(self, url):
        parsed = urlsplit(url)
        if (parsed.scheme != 'obs' or parsed.query or parsed.fragment
                or parsed.netloc not in self.routes or '..' in parsed.path.split('/')):
            raise ObsDownloadError(f'No configured OBS route for {url!r}')
        return self.routes[parsed.netloc]

    def list(self, url):
        return self._route(url).list(url)

    def stat(self, url):
        return self._route(url).stat(url)

    def download(self, url, dest):
        return self._route(url).download(url, dest)


def _make_downloader(cfg, bucket):
    if not cfg:
        return None
    kind = cfg.get("type")
    if kind == "obsutil":
        return ObsutilDownloader(cfg.get("bin", "obsutil"),
            regional_proxy_endpoint=cfg.get('regional_proxy_endpoint'),
            bucket=bucket)
    if kind == 'public_https':
        from .obs_public import PublicObsDownloader
        return PublicObsDownloader(cfg.get('endpoint', ''), bucket,
            regional_proxy_endpoint=cfg.get('regional_proxy_endpoint'))
    raise ObsDownloadError(
        f"未知 OBS 下载器类型 {kind!r}（当前支持 obsutil/public_https）；"
        "下载方式必须由部署配置显式给出"
    )


def make_downloader(obs_cfg: dict | None):
    """Explicit per-bucket routes; unconfigured legacy deployments still return None."""
    obs_cfg = obs_cfg or {}
    bucket = obs_cfg.get('bucket', 'pytorch-ci-cache')
    legacy = _make_downloader(obs_cfg.get('downloader'), bucket)
    if not obs_cfg.get('wheels') and not obs_cfg.get('source'):
        return legacy
    wheels, source = artifact_prefixes(obs_cfg)
    routes = {bucket: legacy} if legacy else {}
    for role, prefix in [('source', source), ('wheels', wheels)]:
        route = obs_cfg.get(role) or {}
        name = urlsplit(prefix).netloc
        if not route.get('downloader') and name in routes:
            continue
        selected = _make_downloader(route.get('downloader'), name)
        if selected is None:
            raise ObsDownloadError(f'OBS {role} bucket {name!r} requires an explicit downloader')
        if name in routes and not (role == 'source' and name == bucket):
            raise ObsDownloadError(f'Conflicting download routes for OBS bucket {name!r}')
        routes[name] = selected
    return RoutedObsDownloader(routes)
