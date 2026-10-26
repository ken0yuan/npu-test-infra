"""测试公共 fixture。全部在 tmp_path 中运行，不触碰真实环境。"""

from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

from run_test.baseline_store import BaselineRouter, ensure_baseline_db
from run_test.config import default_config, store_paths
from run_test.identifiers import Counts
from run_test.pr_store import ensure_pr_db
from run_test.task_queue import QueueStore
from run_test.record import FunctionResult, Submission

# 精确 commit 样例（40 位十六进制）。
COMMIT_A = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
COMMIT_B = "0b9a8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b"
COMMIT_NPU = "1234567890abcdef1234567890abcdef12345678"

# 分布在不同最小管理目录 / 不同文件 / 不同类的函数标识。
TF_A = "test/distributed/_tensor/test_tensor.py::test_basic"
TF_A_CLASS = "test/distributed/_tensor/test_tensor.py::TestMethods::test_basic"
TF_B = "test/distributed/fsdp/test_fsdp.py::test_wrap"
TF_B_SAME_NAME = "test/distributed/fsdp/test_other.py::test_wrap"

BASELINE_DIRS = ["test/distributed/_tensor", "test/distributed/fsdp"]


@pytest.fixture
def config(tmp_path):
    cfg = default_config()
    cfg["data_root"] = str(tmp_path / "data")
    cfg["baseline_dirs"] = list(BASELINE_DIRS)
    # 执行端配置：测试 Python 环境必须显式给出（文档要求不得默认系统 Python）。
    cfg["executor"]["test_python_env"] = sys.executable
    return cfg


@pytest.fixture
def paths(config):
    return store_paths(config)


@pytest.fixture
def router(config, paths):
    r = BaselineRouter(config["baseline_dirs"], paths.baselines_root)
    for d in r.dirs:
        ensure_baseline_db(r.managed_root(d))
    ensure_pr_db(paths.data_root)
    return r


@pytest.fixture
def queue(config):
    return QueueStore(config)


def make_submission(
    pr_number: int = 123456,
    pytorch_commit: str = COMMIT_A,
    items: list[tuple[str, str, Counts]] | None = None,
    pr_log_source_dir: Path | None = None,
) -> Submission:
    """构造一份完整提交；items 为 (test_function, device, counts) 列表。"""
    if items is None:
        items = [
            (TF_A, "cpu", Counts(3, 0, 0, 1)),
            (TF_A, "accelerator", Counts(2, 1, 0, 1)),
            (TF_B, "accelerator", Counts(0, 0, 1, 0)),
        ]
    if pr_log_source_dir is None:
        pr_log_source_dir = make_pr_log_dir()
    return Submission(
        pr_number=pr_number,
        pytorch_commit=pytorch_commit,
        torch_npu_environment={
            "location": "test-env",
            "torch_npu_commit": COMMIT_NPU,
        },
        function_results=[
            FunctionResult(
                test_function=tf, device=dev, counts=counts,
                log_content=f"log of {tf} on {dev}\ncounts={counts.as_dict()}",
                coverage_status="full", instance_ids=(tf,),
                environment_fingerprint={"channel": dev, "boot": None},
            )
            for tf, dev, counts in items
        ],
        pr_log_source_dir=pr_log_source_dir,
    )


def make_pr_log_dir(base: Path | None = None, tag: str = "run") -> Path:
    d = (base or Path(tempfile.mkdtemp(prefix="npu-store-test-"))) / f"pr-log-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.txt").write_text(f"PR log {tag}\n", encoding="utf-8")
    return d


def make_wheels_tar(dest: Path, sha10: str, torch_npu_commit: str = COMMIT_NPU) -> Path:
    """构造含 torch / torch_npu 两个 wheel 的 tar.gz。

    torch_npu wheel 是真实 zip 结构（内含 torch_npu/version.py 的
    git_version），与上游构建产物一致（2026-09-17 实证）；torch wheel
    仅占位（其 git_version 本来就是 'Unknown'）。
    """
    import zipfile

    staging = dest.parent / f"{dest.stem}-staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / f"torch-2.15.0a0+gitunknown-cp310-cp310-linux_aarch64.whl").write_bytes(b"fake")
    whl = staging / f"torch_npu-2.15.0a0-cp310-cp310-linux_aarch64.whl"
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr(
            "torch_npu/version.py",
            f"__version__ = '2.15.0a0'\n"
            f"git_version = '{torch_npu_commit}'\n",
        )
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(staging, arcname=".")
    return dest


def make_source_tar(dest: Path, sha10: str, *, test_tree: bool = True) -> Path:
    """构造源码包。tar 根为 pytorch/（与 OBS 包成员名 pytorch/test/… 一致）。

    test_tree=False 造出没有 test/ 的包（用于"找不到测试树 → blocked"用例）。
    """
    staging = dest.parent / f"{dest.stem}-staging"
    pkg = staging / "pytorch"
    (pkg / "torch").mkdir(parents=True, exist_ok=True)
    (pkg / "torch" / "version.py").write_text("__version__ = '2.7.0'\n")
    if test_tree:
        (pkg / "test").mkdir(exist_ok=True)
        (pkg / "test" / ".keep").write_text("")
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(staging, arcname=".")
    return dest


class FakeDownloader:
    """下载即生成本地假包；可注入失败 URL。"""

    def __init__(self, fail_urls: set[str] | None = None):
        self.fail_urls = fail_urls or set()
        self.calls: list[str] = []

    def download(self, url: str, dest: Path) -> None:
        self.calls.append(url)
        if url in self.fail_urls:
            from run_test.obs import ObsDownloadError

            raise ObsDownloadError(f"模拟下载失败: {url}")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        name = url.rsplit("/", 1)[-1]
        if name.startswith("wheels-"):
            sha10 = name[len("wheels-"):].split(".")[0]
            make_wheels_tar(dest, sha10)
        else:
            sha10 = name[len("pytorch-source-"):].split(".")[0]
            make_source_tar(dest, sha10)
