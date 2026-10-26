"""通用环境说明文件（common_environment.json）。

契约见《本地测试结果与基线存储说明》第 4 节：
- 固定环境信息只写一份持久化说明文件，不建立环境数据库、不使用环境 ID；
- 内容包括实际使用的 CANN 版本、设备型号及固定配置、驱动和固件版本等；
- 具体值必须来自真实配置，禁止填写猜测值——采集不到的字段保持 null
  并记入 missing，绝不臆造。
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import ENV_FILE_NAME

CANN_VERSION_CANDIDATES = (
    Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
    Path("/usr/local/Ascend/cann/version.cfg"),
)
CANN_DIR_GLOB = "/usr/local/Ascend/cann-*"
DRIVER_VERSION_INFO = Path("/usr/local/Ascend/driver/version.info")
NPU_SMI = "npu-smi"

ENV_FIELDS = (
    "cann_version",
    "device_model",
    "device_count",
    "driver_version",
    "firmware_version",
)


class EnvironmentError(ValueError):
    pass


def env_file_path(data_root: Path) -> Path:
    return Path(data_root) / ENV_FILE_NAME


def _run(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _collect_cann() -> tuple[str | None, str | None]:
    """读取 CANN 版本：version.cfg 优先，其次安装目录名（如 cann-9.1.0-beta.1）。"""
    for cfg in CANN_VERSION_CANDIDATES:
        if cfg.is_file():
            text = cfg.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"^\s*Version\s*=\s*(\S+)\s*$", text, re.MULTILINE)
            if m:
                return m.group(1), str(cfg)
    import glob as _glob

    matches = sorted(_glob.glob(CANN_DIR_GLOB))
    for d in matches:
        m = re.search(r"cann-(.+)$", Path(d).name)
        if m:
            return m.group(1), f"目录名 {d}"
    return None, "未找到 version.cfg，/usr/local/Ascend/ 下也无 cann-* 目录"


def _first_npu_id() -> str | None:
    """从 `npu-smi info -m` 解析第一个 NPU ID（容器内 ID 可能不从 0 开始）。"""
    out = _run([NPU_SMI, "info", "-m"])
    if out is None:
        return None
    m = re.search(r"^\s*(\d+)\s+\d+\s+", out, re.MULTILINE)
    return m.group(1) if m else None


def _collect_devices() -> tuple[tuple[str | None, int | None], str | None]:
    """从 `npu-smi info -m` 解析设备型号与数量。

    -m 的映射表每行是 `NPU ID / Chip ID / Chip Logic ID / Chip Name`；
    只统计 `Ascend <型号>` 芯片行（排除 Mcu 等辅助芯片），按 NPU ID 去重。
    """
    out = _run([NPU_SMI, "info", "-m"])
    if out is None:
        return (None, None), "npu-smi info -m 执行失败或不存在"
    npu_ids: set[str] = set()
    models: set[str] = set()
    for m in re.finditer(
        r"^\s*(\d+)\s+\d+\s+\S+\s+Ascend\s+(\S+)\s*$", out, re.MULTILINE
    ):
        npu_ids.add(m.group(1))
        models.add(m.group(2))
    if not npu_ids:
        return (None, None), "npu-smi info -m 中未解析到 Ascend 芯片行"
    if len(models) > 1:
        return (
            (sorted(models)[0], len(npu_ids)),
            f"检测到多种型号 {sorted(models)}，仅记录第一种",
        )
    return (next(iter(models)), len(npu_ids)), None


def _collect_driver_firmware() -> tuple[tuple[str | None, str | None], str | None]:
    """驱动与固件版本。

    - 固件：`npu-smi info -t board -i <NPU ID>` 的 Firmware Version
      （NPU ID 先经 `npu-smi info -m` 解析——容器内 ID 可能不从 0 开始）；
    - 驱动：board 的 Software/Driver Version，兜底
      /usr/local/Ascend/driver/version.info 的 Version=。
    """
    npu_id = _first_npu_id()
    driver: str | None = None
    firmware: str | None = None
    notes: list[str] = []
    if npu_id is not None:
        out = _run([NPU_SMI, "info", "-t", "board", "-i", npu_id])
        if out is not None:
            fw = re.search(r"Firmware Version\s*[:：]\s*(\S+)", out)
            dv = re.search(r"(?:Driver|Software) Version\s*[:：]\s*(\S+)", out)
            firmware = fw.group(1) if fw else None
            driver = dv.group(1) if dv else None
        else:
            notes.append(f"npu-smi info -t board -i {npu_id} 执行失败")
    else:
        notes.append("无法从 npu-smi info -m 解析 NPU ID")
    if driver is None and DRIVER_VERSION_INFO.is_file():
        text = DRIVER_VERSION_INFO.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^\s*Version\s*=\s*(\S+)\s*$", text, re.MULTILINE)
        if m:
            driver = m.group(1)
    if driver is None or firmware is None:
        notes.append("驱动或固件版本未取得")
    return (driver, firmware), ("; ".join(notes) if notes else None)


def collect_environment() -> dict:
    """从本机真实配置采集环境信息；采集不到的字段为 null 并说明原因。"""
    fields: dict[str, object] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []

    cann, cann_info = _collect_cann()
    fields["cann_version"] = cann
    if cann is None:
        missing.append(f"cann_version: {cann_info}")
    else:
        sources["cann_version"] = cann_info

    (model, count), why = _collect_devices()
    fields["device_model"] = model
    fields["device_count"] = count
    sources["device_model"] = f"{NPU_SMI} info -m"
    sources["device_count"] = f"{NPU_SMI} info -m"
    if model is None:
        missing.append(f"device_model: {why}")
    if count is None:
        missing.append(f"device_count: {why}")

    (driver, firmware), why = _collect_driver_firmware()
    fields["driver_version"] = driver
    fields["firmware_version"] = firmware
    sources["driver_version"] = f"{NPU_SMI} info -t board / {DRIVER_VERSION_INFO}"
    sources["firmware_version"] = f"{NPU_SMI} info -t board"
    if driver is None or firmware is None:
        missing.append(f"driver_version/firmware_version: {why}")

    fields["other_fixed"] = {}  # 其他确实固定的环境信息，由用户按需补充
    return {
        "fields": fields,
        "sources": sources,
        "missing": missing,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "collector": "run_test.environment",
    }


def validate_environment(env: dict) -> list[str]:
    """返回缺失字段说明（字段值为 null 时视为缺失，不猜测填充）。"""
    problems = []
    fields = env.get("fields")
    if not isinstance(fields, dict):
        return ["环境说明缺少 fields 对象"]
    for name in ENV_FIELDS:
        if name not in fields:
            problems.append(f"缺少字段 {name}")
        elif fields[name] is None:
            problems.append(f"字段 {name} 未采集到（保持 null，禁止猜测值）")
    return problems


def write_environment_file(data_root: Path, env: dict | None = None) -> Path:
    path = env_file_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = env if env is not None else collect_environment()
    if validate_environment(content):
        # 允许写入带缺失字段的采集结果（值为 null + missing 说明），
        # 但不得因此臆造值；缺失情况在文件内自说明。
        pass
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_environment_file(data_root: Path) -> dict:
    path = env_file_path(data_root)
    if not path.is_file():
        raise EnvironmentError(f"通用环境说明文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
