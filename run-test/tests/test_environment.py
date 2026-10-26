"""通用环境说明：真实采集解析、缺失字段不猜测（存储文档第 4 节）。"""

from pathlib import Path

import run_test.environment as env_mod
from run_test.environment import (
    collect_environment,
    validate_environment,
    write_environment_file,
)

NPU_SMI_INFO = """+-------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc2                 Version: 24.1.rc2                                         |
+=====================================+======================+=======================+
| NPU     Name        | Health | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device      | Bus-Id | AICore(%)    Memory-Usage(MB)                        |
+======================+===============+======+=======================+
| 0     910B4         | OK     | 97.5         42                0    / 0      |
| 0     AI Core       | 0000:C1:00.0 | 0            875  / 15171                 |
| 1     910B4         | OK     | 96.8         40                0    / 0      |
| 1     AI Core       | 0000:C1:00.1 | 0            860  / 15171                 |
+======================+===============+======+=======================+
"""

BOARD_INFO = """
NPU ID                 : 4
Product Name           : IT21HMDC_B2X
Software Version       : 26.0.rc2
Firmware Version       : 7.3.0.1.220
"""

MAP_INFO = """
\tNPU ID                         Chip ID                        Chip Logic ID                  Chip Name
\t4                              0                              0                              Ascend 910B4
\t4                              1                              -                              Mcu
\t5                              0                              1                              Ascend 910B4
\t5                              1                              -                              Mcu
"""


def _mock_collect(monkeypatch, tmp_path, *, npu_smi=NPU_SMI_INFO, board=BOARD_INFO,
                  map_info=MAP_INFO, cann="Version=8.0.0.rc1\n", glob_results=None):
    import glob as glob_mod

    cfg = tmp_path / "version.cfg"
    cfg.write_text(cann)
    monkeypatch.setattr(env_mod, "CANN_VERSION_CANDIDATES", (cfg,))
    # 屏蔽真实机器上的兜底来源，保证测试隔离。
    monkeypatch.setattr(
        env_mod, "DRIVER_VERSION_INFO", tmp_path / "no-driver-version.info"
    )
    monkeypatch.setattr(glob_mod, "glob", lambda pattern: list(glob_results or []))

    def fake_run(cmd):
        if cmd[:4] == ["npu-smi", "info", "-t", "board"]:
            return board
        if cmd[:3] == ["npu-smi", "info", "-m"]:
            return map_info
        if cmd[:2] == ["npu-smi", "info"]:
            return npu_smi
        return None

    monkeypatch.setattr(env_mod, "_run", fake_run)


def test_collect_real_values(monkeypatch, tmp_path):
    _mock_collect(monkeypatch, tmp_path)
    env = collect_environment()
    f = env["fields"]
    assert f["cann_version"] == "8.0.0.rc1"
    assert f["device_model"] == "910B4"
    assert f["device_count"] == 2
    assert f["driver_version"] == "26.0.rc2"
    assert f["firmware_version"] == "7.3.0.1.220"
    assert validate_environment(env) == []
    # 每个字段都记录了来源（真实配置，非猜测）。
    assert env["sources"]["cann_version"].endswith("version.cfg")


def test_cann_version_from_directory_name(monkeypatch, tmp_path):
    """version.cfg 缺失时从 cann-* 目录名取版本（仍来自真实配置）。"""
    _mock_collect(
        monkeypatch, tmp_path, cann="no version here\n",
        glob_results=["/usr/local/Ascend/cann-9.1.0-beta.1"],
    )
    env = collect_environment()
    assert env["fields"]["cann_version"] == "9.1.0-beta.1"
    assert "cann-9.1.0-beta.1" in env["sources"]["cann_version"]


def test_missing_fields_stay_null_never_guessed(monkeypatch, tmp_path):
    _mock_collect(
        monkeypatch, tmp_path, npu_smi=None, board=None, map_info=None,
        cann="no version here\n",
    )
    env = collect_environment()
    f = env["fields"]
    assert f["cann_version"] is None
    assert f["device_model"] is None
    assert f["driver_version"] is None
    problems = validate_environment(env)
    assert len(problems) >= 4  # 缺失被显式列出，而不是被填成猜测值
    assert all("null" in p or "缺少" in p for p in problems)


def test_write_and_load_roundtrip(monkeypatch, tmp_path):
    _mock_collect(monkeypatch, tmp_path)
    path = write_environment_file(tmp_path)
    assert path.is_file()
    loaded = env_mod.load_environment_file(tmp_path)
    assert loaded["fields"]["device_model"] == "910B4"
    # 写入的是一份文件，不建立环境数据库、没有环境 ID。
    assert not (tmp_path / "environments.db").exists()
