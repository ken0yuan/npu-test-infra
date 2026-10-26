"""基线数据库：路由、联合主键与日志约束（存储文档第 5 节验收项）。"""

import pytest

from run_test.baseline_store import (
    BaselineRecord,
    BaselineRouter,
    BaselineStoreError,
    NoBaselineDirectoryError,
    baseline_db_path,
    baseline_log_filename,
    ensure_baseline_db,
    get_baseline,
    list_baselines,
    upsert_baseline,
)
from run_test.db import connect
from run_test.identifiers import Counts

from conftest import COMMIT_A, COMMIT_B, COMMIT_NPU, TF_A, TF_A_CLASS, TF_B, TF_B_SAME_NAME


def make_record(tf=TF_A, device="cpu", counts=Counts(1, 0, 0, 0), commit=COMMIT_A):
    return BaselineRecord(
        test_function=tf,
        device=device,
        pytorch_commit=commit,
        torch_npu_environment={"location": "env", "torch_npu_commit": COMMIT_NPU},
        counts=counts,
        log_path=f"logs/{baseline_log_filename(tf, device)}",
    )


class TestRouter:
    def test_routes_by_longest_prefix(self, paths):
        router = BaselineRouter(
            ["test/distributed/_tensor", "test/distributed/fsdp"], paths.baselines_root
        )
        assert router.route(TF_A).managed_dir == "test/distributed/_tensor"
        assert router.route(TF_B).managed_dir == "test/distributed/fsdp"

    def test_unknown_dir_raises(self, paths):
        router = BaselineRouter(["test/distributed/_tensor"], paths.baselines_root)
        with pytest.raises(NoBaselineDirectoryError):
            router.route("test/other/test_x.py::test_f")

    def test_parent_dir_fallback_with_child_dirs(self, paths):
        """父子目录共存（test/ 兜底 + 子库细分）：文件唯一归属最长匹配。"""
        router = BaselineRouter(
            ["test", "test/distributed/fsdp", "test/distributed"],
            paths.baselines_root,
        )
        # 子库文件命中更长前缀，不落入 test/ 兜底。
        assert router.route(TF_B).managed_dir == "test/distributed/fsdp"
        assert router.route("test/distributed/test_rpc.py::test_f").managed_dir == (
            "test/distributed"
        )
        # 兜底：不在任何子目录的顶层文件归 test/。
        assert router.route("test/testnn.py::test_f").managed_dir == "test"
        # 仍满足"不写入父子两库"：一个文件只有一个归属。
        assert router.route(TF_B).managed_dir != "test/distributed"

    def test_dirs_sorted_longest_first(self, paths):
        router = BaselineRouter(
            ["test/distributed", "test", "test/nn"], paths.baselines_root
        )
        assert router.dirs[0] == "test/distributed"
        assert router.route("test/nn/test_nn.py::test_f").managed_dir == "test/nn"


class TestBaselineCrud:
    @pytest.fixture
    def conn(self, router, paths):
        root = router.route(TF_A).root
        return connect(baseline_db_path(root))

    def test_primary_key_is_function_plus_device(self, conn):
        """联合主键 (test_function, device)；不同 device 互不覆盖。"""
        upsert_baseline(conn, make_record(TF_A, "cpu", Counts(1, 0, 0, 0)))
        upsert_baseline(conn, make_record(TF_A, "accelerator", Counts(0, 1, 0, 0)))
        recs = list_baselines(conn)
        assert len(recs) == 2
        assert get_baseline(conn, TF_A, "cpu").counts == Counts(1, 0, 0, 0)
        assert get_baseline(conn, TF_A, "accelerator").counts == Counts(0, 1, 0, 0)

    def test_upsert_replaces_not_accumulates(self, conn):
        upsert_baseline(conn, make_record(TF_A, "cpu", Counts(2, 0, 0, 1)))
        upsert_baseline(conn, make_record(TF_A, "cpu", Counts(0, 3, 0, 0), commit=COMMIT_B))
        rec = get_baseline(conn, TF_A, "cpu")
        assert rec.counts == Counts(0, 3, 0, 0)
        assert rec.pytorch_commit == COMMIT_B

    def test_class_variants_are_distinct_records(self, conn):
        """同文件不同类的同名函数是不同记录。"""
        upsert_baseline(conn, make_record(TF_A, "cpu"))
        upsert_baseline(conn, make_record(TF_A_CLASS, "cpu"))
        assert get_baseline(conn, TF_A, "cpu") is not None
        assert get_baseline(conn, TF_A_CLASS, "cpu") is not None
        assert (
            get_baseline(conn, TF_A, "cpu").log_path
            != get_baseline(conn, TF_A_CLASS, "cpu").log_path
        )

    def test_same_name_different_files_no_log_conflict(self, router):
        """不同文件的同名函数不共享/覆盖日志（验收 8-9）。"""
        c1 = connect(baseline_db_path(router.route(TF_B).root))
        upsert_baseline(c1, make_record(TF_B, "cpu"))
        upsert_baseline(c1, make_record(TF_B_SAME_NAME, "cpu"))
        r1 = get_baseline(c1, TF_B, "cpu")
        r2 = get_baseline(c1, TF_B_SAME_NAME, "cpu")
        assert r1.log_path != r2.log_path

    def test_log_filename_conflict_detected(self, conn):
        """文件名映射冲突时拒绝写入（严格无冲突保障）。"""
        rec = make_record(TF_A, "cpu")
        # 人为把 B 的记录写成与 A 相同的 log_path，模拟哈希碰撞。
        upsert_baseline(conn, rec)
        bad = make_record(TF_B, "cpu")
        bad = BaselineRecord(**{**bad.__dict__, "log_path": rec.log_path})
        with pytest.raises(BaselineStoreError, match="占用"):
            upsert_baseline(conn, bad)

    def test_invalid_device_rejected_by_db(self, conn):
        rec = make_record(TF_A, "cpu")
        bad = BaselineRecord(**{**rec.__dict__, "device": "npu"})
        with pytest.raises(BaselineStoreError):
            upsert_baseline(conn, bad)

    def test_separate_dbs_per_managed_dir(self, router):
        """按最小目录拆分：两个目录各有一个库。"""
        c1 = connect(baseline_db_path(router.route(TF_A).root))
        c2 = connect(baseline_db_path(router.route(TF_B).root))
        upsert_baseline(c1, make_record(TF_A, "cpu"))
        upsert_baseline(c2, make_record(TF_B, "cpu"))
        assert len(list_baselines(c1)) == 1
        assert len(list_baselines(c2)) == 1
        # 跨库互不可见。
        assert get_baseline(c1, TF_B, "cpu") is None
        assert get_baseline(c2, TF_A, "cpu") is None
