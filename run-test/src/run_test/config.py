"""存储与队列配置：加载、合并、校验与路径解析。

文档中标明“尚未确定/待实施配置”的项在默认配置中为 null，
代码遇到 null 相关路径时显式报“待配置”，不使用猜测的默认值。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_FILE_NAME = "settings.json"

PR_DB_NAME = "pr_results.db"
PR_LOGS_DIR_NAME = "pr_logs"
BASELINES_DIR_NAME = "baselines"
QUEUE_DIR_NAME = "queue"
QUEUE_DB_NAME = "queue.db"
TASK_WORK_DIR_NAME = "task_work"
ENV_FILE_NAME = "common_environment.json"
DIAGNOSTICS_DIR_NAME = "diagnostics"

DEFAULT_CONFIG: dict[str, Any] = {
    # 持久化数据根目录：PR 库、基线库、队列、日志、通用环境说明均在其下。
    "data_root": None,
    # 最小测试管理目录清单（段级前缀匹配函数文件路径，如 "test/distributed/_tensor"）。
    # 尚未确定：由用户通过 CLI / 配置逐步给定；实现不自行扫描仓库批量建库。
    "baseline_dirs": [],
    # PR 旧日志处理（文档 3.3 未定事项）：默认保守——不删除，
    # 旧内容移入 pr_logs/.superseded/PR_<n>/<时间戳>/，避免新旧混读。
    "pr_log_retention": "supersede_keep",
    "upstream": {
        # cycle 常驻模式的循环间隔（秒）。用户方案：暂定 1 小时。
        "cycle_interval_seconds": 3600.0,
        # 首轮回溯窗口（小时）：一次性 backfill，后续按 tip 增量。
        "first_run_backfill_hours": 2.0,
    },
    "queue": {
        # 执行循环空闲时的轮询间隔（秒）。
        "poll_interval_seconds": 30.0,
        # 下载/写库重试次数与间隔：待实施配置（文档第 8 节“重试次数、间隔尚未确定”）。
        "download_retries": None,
        "download_retry_interval_seconds": None,
        # 清理阶段自动重试次数上限；耗尽后标记 failed 并保留诊断。
        "cleanup_retries": 5,
    },
    "executor": {
        "allowed_devices": ["cpu", "accelerator"],
        # 指定测试 Python 环境（解释器绝对路径）。文档第 6 节：
        # 具体路径需实施配置明确，不得默认使用系统 Python。
        # 注意：OBS 包内 wheel 全部为 cp310（构建镜像 py3.10），必须 3.10。
        "test_python_env": None,
        # 测试框架依赖（wheel 安装后额外 pip install 的包清单）。
        # 官方测试容器镜像预装这些；本机串行执行环境需显式给出。
        # torch 已在本地装好，pip 不会从软件源替换它。
        "test_requirements": [
            "pytest",
            "hypothesis",
            "parameterized",
            "zstandard",
            "pandas",
            "pydot",
            "z3-solver",
        ],
        # triton-ascend 不在此列：与 torch 版本强配套，需按被测版本确定
        # 配套版本后单独安装（见 docs/archive/PIPELINE.md 待确认事项；venv310 已预装 3.2.0）。
        # device → 传给测试执行接口的附加参数（“已有测试执行接口”的对接点）。
        "device_runner_args": {},
        # 单函数执行超时（秒），None 表示不限制。
        "function_timeout_seconds": None,
        # transfer boot（sitecustomize）目录：device=cuda 的函数经它注入
        # CUDA→NPU 转换执行；cpu / accelerator 直接跑 test 源码，不注入
        # （用户 2026-09-17 决策）。None = cuda 函数不注入 boot。
        "transfer_boot_dir": None,
    },
    "skill": {
        # 过滤规则版本标识（钉定 commit），仅用于 skill_state 追溯。
        # 规则本体已收归 infra run-test/tools/pr_exclusion_filter.py（用户
        # 2026-09-18：逻辑不会改动，不再从 skill 仓库检出）。
        "ref": None,
    },
    "obs": {
        # OBS 桶名（对象位于 <bucket>/7day-cache/ 下）。
        "bucket": "pytorch-ci-cache",
        # OBS 下载方式，如 {"type": "obsutil", "bin": "/usr/local/bin/obsutil"}。
        # 未配置时执行端在该阶段记 blocked，不偷偷启用任何联网认证路径。
        "downloader": None,
        # 七天到期规则（以哪个远端时间为准、覆盖是否重置）待负责人确认；
        # 未确认时不计算 expires_at，不用“本地发现时间 + 七天”冒充。
        "expiry_rule": None,
    },
}


class ConfigError(ValueError):
    """配置缺失或非法。"""


class ConfigPendingError(RuntimeError):
    """某项实施配置尚未确定，相关功能不能执行。"""


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """按段合并 override 到 base（override 优先）。"""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"配置文件不存在: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"配置文件不是合法 JSON: {p}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是对象: {p}")
    return merge_config(default_config(), raw)


def validate_config(config: dict[str, Any]) -> None:
    root = config.get("data_root")
    if not isinstance(root, str) or not root.strip():
        raise ConfigError("data_root 必须是非空路径")
    if config.get("pr_log_retention") != "supersede_keep":
        raise ConfigError(
            "pr_log_retention 目前仅支持 supersede_keep（PR 旧日志保留策略为文档未定事项，"
            "默认保守不删除）"
        )
    dirs = config.get("baseline_dirs", [])
    if not isinstance(dirs, list):
        raise ConfigError("baseline_dirs 必须是列表")


def resolve_data_root(config: dict[str, Any]) -> Path:
    validate_config(config)
    return Path(config["data_root"]).expanduser().resolve()


@dataclass(frozen=True)
class StorePaths:
    data_root: Path
    pr_db: Path
    pr_logs_root: Path
    baselines_root: Path
    queue_dir: Path
    queue_db: Path
    task_work_root: Path
    env_file: Path
    diagnostics_root: Path

    @property
    def config_file(self) -> Path:
        return self.data_root.parent / CONFIG_FILE_NAME


def store_paths(config: dict[str, Any]) -> StorePaths:
    root = resolve_data_root(config)
    return StorePaths(
        data_root=root,
        pr_db=root / PR_DB_NAME,
        pr_logs_root=root / PR_LOGS_DIR_NAME,
        baselines_root=root / BASELINES_DIR_NAME,
        queue_dir=root / QUEUE_DIR_NAME,
        queue_db=root / QUEUE_DIR_NAME / QUEUE_DB_NAME,
        task_work_root=root / TASK_WORK_DIR_NAME,
        env_file=root / ENV_FILE_NAME,
        diagnostics_root=root / DIAGNOSTICS_DIR_NAME,
    )


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    validate_config(config)
    p.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return p
