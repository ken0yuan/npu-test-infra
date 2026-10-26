"""基线导入：junit 子用例聚合到 (test_function, device) 联合主键（存储文档第 5 节）。

快照 → 函数级基线库：只导入 status ∈ {passed, failed} 的文件；
占位/收集错误不伪造零；幂等重放按本次值覆盖。
"""

from __future__ import annotations

import json

import pytest

from run_test.baseline_import import (
    BaselineImportError,
    extract_source_functions,
    import_baseline,
    normalize_class,
    normalize_name,
    parse_junit_file,
)
from run_test.baseline_store import (
    baseline_db_path,
    baseline_logs_dir,
    get_baseline,
    list_baselines,
)
from run_test.config import store_paths
from run_test.identifiers import IdentifierError

from conftest import COMMIT_NPU

PYTORCH_COMMIT = "bf54051becd0c5b95a1f2f3a4b5c6d7e8f9a0b1c"

SRC_TEST_RPC = '''
import torch


class TestRPC:
    def test_remote(self):
        pass

    def test_allreduce(self):
        pass

    def test_timeout(self):
        pass


def test_top_level_cuda(self):
    pass
'''


@pytest.fixture
def import_env(tmp_path, config):
    """构造 baseline.json + junit/ + test-source 三件套；dirs 用父子共存清单。"""
    cfg = dict(config)
    cfg["baseline_dirs"] = ["test", "test/distributed"]
    src = tmp_path / "test-source"
    (src / "test" / "distributed").mkdir(parents=True)
    (src / "test" / "distributed" / "test_rpc.py").write_text(SRC_TEST_RPC)

    junit = tmp_path / "junit"
    junit.mkdir()

    def write_junit(rel_file: str, cases: str) -> None:
        (junit / (rel_file.replace("/", "_") + ".xml")).write_text(
            f'<testsuite name="s">{cases}</testsuite>'
        )

    def write_entry(rel_file: str, status: str) -> dict:
        return {"file": rel_file, "status": status, "counts": {}}

    return {
        "config": cfg,
        "paths": store_paths(cfg),
        "src": src,
        "junit": junit,
        "write_junit": write_junit,
        "write_entry": write_entry,
        "json": tmp_path / "baseline.json",
    }


def test_normalize_class_strips_device_suffix():
    # 真实 junit：设备实例化后缀无点嵌入类名（instantiate_device_type_tests）。
    assert normalize_class("test_rpc.TestRPCDeviceTypeCUDA") == "TestRPC"
    assert normalize_class("TestDropoutNNDeviceTypeCUDA") == "TestDropoutNN"
    assert normalize_class("TestPlain") == "TestPlain"
    assert normalize_class("pkg.mod.TestXDeviceTypeCPU") == "TestX"


def test_normalize_name_strips_param_and_device_suffix():
    src = {"test_remote", "test_allreduce"}
    assert normalize_name("test_remote_cuda[params=(2, 3)]", src) == "test_remote"
    assert normalize_name("test_remote_cuda", src) == "test_remote"
    assert normalize_name("test_remote", src) == "test_remote"
    # 原名本身命中源码集时保留原名（防误伤）。
    assert normalize_name("test_remote_cuda", {"test_remote_cuda"}) == "test_remote_cuda"
    # 都未命中：保守采用截断结果，交由 unmatched 统计。
    assert normalize_name("test_nope_cuda", src) == "test_nope"


def test_extract_source_functions(tmp_path):
    f = tmp_path / "t.py"
    f.write_text(SRC_TEST_RPC)
    names = extract_source_functions(f)
    assert {"test_remote", "test_allreduce", "test_timeout",
            "test_top_level_cuda"} <= names


def test_parse_junit_file_aggregates(import_env):
    import_env["write_junit"](
        "distributed/test_rpc.py",
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_remote_cuda"/>'
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_remote_cuda_float">'
        "<failure message=\"assert 1 == 2\">boom</failure></testcase>"
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_allreduce_cuda">'
        "<skipped message='no npu'/></testcase>"
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_timeout_cuda">'
        "<error message='timed out'/></testcase>"
        '<testcase name="distributed.test_rpc" time="0"/>',  # 占位：无类名
    )
    entry = {"file": "distributed/test_rpc.py", "status": "passed"}
    src_functions = extract_source_functions(
        import_env["src"] / "test" / "distributed" / "test_rpc.py"
    )
    aggregates, unmatched, no_src = parse_junit_file(
        entry, import_env["junit"] / "distributed_test_rpc.py.xml", src_functions
    )
    assert unmatched == 0 and no_src == 0
    # 占位被跳过，只剩 4 个函数。
    assert set(aggregates) == {
        "test/distributed/test_rpc.py::TestRPC::test_remote",
        "test/distributed/test_rpc.py::TestRPC::test_allreduce",
        "test/distributed/test_rpc.py::TestRPC::test_timeout",
    }
    remote = aggregates["test/distributed/test_rpc.py::TestRPC::test_remote"].counts
    assert (remote.pass_count, remote.fail_count) == (1, 1)  # 两个子用例聚合
    assert aggregates["test/distributed/test_rpc.py::TestRPC::test_allreduce"].counts.skip_count == 1
    assert aggregates["test/distributed/test_rpc.py::TestRPC::test_timeout"].counts.error_count == 1


def test_parse_junit_unmatched_counted(import_env):
    import_env["write_junit"](
        "distributed/test_rpc.py",
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_ghost_cuda"/>',
    )
    entry = {"file": "distributed/test_rpc.py", "status": "passed"}
    src_functions = extract_source_functions(
        import_env["src"] / "test" / "distributed" / "test_rpc.py"
    )
    _, unmatched, _ = parse_junit_file(
        entry, import_env["junit"] / "distributed_test_rpc.py.xml", src_functions
    )
    assert unmatched == 1


def test_import_baseline_end_to_end(import_env):
    import_env["write_junit"](
        "distributed/test_rpc.py",
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_remote_cuda"/>'
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_remote_cuda_f32">'
        "<failure message='x'>f</failure></testcase>",
    )
    data = {
        "meta": {
            "source_head": PYTORCH_COMMIT,
            "boot": {"version": 3},
            "env": {
                "torch": "2.15.0a0+gitbf54051",
                "torch_npu": "2.15.0+git42d8b33",
                "torch_npu_file": "/opt/py3.12/site-packages/torch_npu-2.15.0-cp310.whl",
            },
        },
        "files": [
            import_env["write_entry"]("distributed/test_rpc.py", "passed"),
        ],
    }
    import_env["json"].write_text(json.dumps(data))

    report = import_baseline(
        import_env["config"], import_env["json"], import_env["src"], COMMIT_NPU
    )
    assert report["files_imported"] == 1
    assert report["functions"] == 1
    assert report["unmatched"] == 0

    # 落在 test/distributed 子库（最长段级匹配），device 固定 cuda。
    paths = import_env["paths"]
    db = baseline_db_path(paths.baselines_root / "test" / "distributed")
    rec = get_baseline(
        _connect(db),
        "test/distributed/test_rpc.py::TestRPC::test_remote", "cuda",
    )
    assert rec is not None
    assert rec.counts.pass_count == 1 and rec.counts.fail_count == 1
    assert rec.pytorch_commit == PYTORCH_COMMIT
    assert rec.torch_npu_environment["torch_npu_commit"] == COMMIT_NPU
    assert rec.torch_npu_environment["location"] == "/opt/py3.12/site-packages"
    # 函数级日志存在且含子用例行。
    log = (paths.baselines_root / "test" / "distributed") / rec.log_path
    assert log.is_file()
    assert "test_remote_cuda" in log.read_text(encoding="utf-8")


def _connect(db_path):
    from run_test.db import connect
    return connect(db_path)


def test_file_status_marker_records(import_env):
    """用户 2026-09-17：collection_error / crashed / timeout 也要入库——
    以 __file__ 标记记录保存状态、根因与原始日志，不伪造函数结果。"""
    (import_env["json"].parent / "logs").mkdir()
    (import_env["json"].parent / "logs" / "test_a.log").write_text(
        "collected 1 item\nModuleNotFoundError: No module named 'torch_openreg'\n"
    )
    entries = [
        {"file": "distributed/test_a.py", "status": "collection_error", "rc": 2,
         "timed_out": False, "duration_sec": 1.5, "collected": 1, "passed": 0,
         "failed": 0, "errors": 1, "skipped": 0, "xfailed": 0,
         "pitfall": {"kind": "import_or_collection",
                     "exception": "ModuleNotFoundError: No module named 'torch_openreg'"},
         "first_failure": "collection failure :: ImportError",
         "log": "logs/test_a.log"},
        {"file": "distributed/test_b.py", "status": "crashed", "rc": -11,
         "timed_out": False, "duration_sec": 30.0, "collected": 60, "passed": 50,
         "failed": 0, "errors": 0, "skipped": 2, "xfailed": 0, "log": "logs/test_b.log"},
        {"file": "distributed/test_c.py", "status": "timeout", "rc": -9,
         "timed_out": True, "log": "logs/missing.log"},
    ]
    data = {
        "meta": {"source_head": PYTORCH_COMMIT, "env": {}},
        "files": entries,
    }
    import_env["json"].write_text(json.dumps(data))
    report = import_baseline(
        import_env["config"], import_env["json"], import_env["src"], COMMIT_NPU
    )
    assert report["files_imported"] == 0
    assert report["files_file_status"] == 3
    paths = import_env["paths"]
    conn = _connect(baseline_db_path(paths.baselines_root / "test" / "distributed"))
    a = get_baseline(conn, "test/distributed/test_a.py::__file__", "cuda")
    b = get_baseline(conn, "test/distributed/test_b.py::__file__", "cuda")
    c = get_baseline(conn, "test/distributed/test_c.py::__file__", "cuda")
    assert a is not None and b is not None and c is not None
    # 计数 = 采集器观察到的进程级真实计数。
    assert (a.counts.pass_count, a.counts.error_count) == (0, 1)
    assert (b.counts.pass_count, b.counts.skip_count) == (50, 2)
    assert c.counts.total == 0
    # 日志携带状态摘要、pitfall、first_failure 与原始日志内容。
    log_text = ((paths.baselines_root / "test" / "distributed") / a.log_path).read_text(
        encoding="utf-8"
    )
    assert "status: collection_error rc=2" in log_text
    assert "pitfall: import_or_collection" in log_text
    assert "first_failure: collection failure :: ImportError" in log_text
    assert "No module named 'torch_openreg'" in log_text


def test_passed_file_without_real_subcases_gets_marker(import_env):
    """junit 只有占位 testcase 的 passed 文件：落 __file__ 标记而非丢弃。"""
    import_env["write_junit"](
        "distributed/test_rpc.py",
        '<testcase name="distributed.test_rpc" time="0"/>',  # 占位：无类名
    )
    data = {
        "meta": {"source_head": PYTORCH_COMMIT, "env": {}},
        "files": [{"file": "distributed/test_rpc.py", "status": "passed",
                   "collected": 0, "passed": 0, "failed": 0, "errors": 0,
                   "skipped": 0, "xfailed": 0}],
    }
    import_env["json"].write_text(json.dumps(data))
    report = import_baseline(
        import_env["config"], import_env["json"], import_env["src"], COMMIT_NPU
    )
    assert report["files_imported"] == 0
    assert report["files_file_status"] == 1
    paths = import_env["paths"]
    conn = _connect(baseline_db_path(paths.baselines_root / "test" / "distributed"))
    rec = get_baseline(conn, "test/distributed/test_rpc.py::__file__", "cuda")
    assert rec is not None
    assert rec.counts.total == 0


def test_import_is_idempotent_overwrite(import_env):
    import_env["write_junit"](
        "distributed/test_rpc.py",
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_remote_cuda"/>',
    )
    data = {
        "meta": {"source_head": PYTORCH_COMMIT, "env": {}},
        "files": [import_env["write_entry"]("distributed/test_rpc.py", "passed")],
    }
    import_env["json"].write_text(json.dumps(data))
    import_baseline(import_env["config"], import_env["json"], import_env["src"], COMMIT_NPU)
    report2 = import_baseline(
        import_env["config"], import_env["json"], import_env["src"], COMMIT_NPU
    )
    assert report2["functions"] == 1
    paths = import_env["paths"]
    conn = _connect(baseline_db_path(paths.baselines_root / "test" / "distributed"))
    recs = list_baselines(conn)
    assert len(recs) == 1
    assert recs[0].counts.pass_count == 1  # 覆盖而非累加


def test_import_requires_full_npu_commit(import_env):
    """短 sha 不是精确 commit：写入记录路径必须拒绝（40/64 位校验）。"""
    import_env["write_junit"](
        "distributed/test_rpc.py",
        '<testcase classname="TestRPCDeviceTypeCUDA" name="test_remote_cuda"/>',
    )
    data = {
        "meta": {"source_head": PYTORCH_COMMIT, "env": {}},
        "files": [import_env["write_entry"]("distributed/test_rpc.py", "passed")],
    }
    import_env["json"].write_text(json.dumps(data))
    from run_test.baseline_store import BaselineStoreError
    with pytest.raises((BaselineStoreError, IdentifierError)):
        import_baseline(
            import_env["config"], import_env["json"], import_env["src"], "42d8b33"
        )


def test_import_requires_source_head(import_env, tmp_path):
    data = {"meta": {"env": {}}, "files": []}
    import_env["json"].write_text(json.dumps(data))
    with pytest.raises(BaselineImportError, match="source_head"):
        import_baseline(import_env["config"], import_env["json"], import_env["src"], COMMIT_NPU)
