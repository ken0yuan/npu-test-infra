"""selection 移植层测试：本地 git 替身 + 源码树替身，不依赖 torch / 网络。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from run_test import selection

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}

FOO_V1 = "def alpha():\n    return 1\n"
FOO_V2 = "def alpha():\n    return 2\n\ndef beta():\n    return alpha()\n"
TEST_FOO = (
    "import torch\n"
    "from torch.foo import alpha\n\n"
    "class TestAlpha:\n"
    "    def test_alpha(self):\n"
    "        assert alpha() == 1\n"
    "    def test_other(self):\n"
    "        assert True\n"
)
TEST_PURE_V1 = (
    "class TestPure:\n"
    "    def test_a(self):\n"
    "        assert True\n"
    "    def test_b(self):\n"
    "        assert True\n"
)
TEST_PURE_V2 = TEST_PURE_V1.replace(
    "    def test_b(self):\n        assert True\n",
    "    def test_b(self):\n        assert False\n",
)


def _git(work: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=work, capture_output=True, text=True, env=GIT_ENV, check=True
    )
    return proc.stdout


@pytest.fixture
def env(tmp_path: Path):
    """commit(base) → commit((#123) …)：改 torch/foo.py + 收窄改 test/test_pure.py。"""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    (work / "torch").mkdir()
    (work / "test").mkdir()
    (work / "aten" / "tools").mkdir(parents=True)
    (work / "torch" / "foo.py").write_text(FOO_V1)
    (work / "test" / "test_foo.py").write_text(TEST_FOO)
    (work / "test" / "test_pure.py").write_text(TEST_PURE_V1)
    (work / "aten" / "tools" / "run_tests.sh").write_text("#!/bin/sh\nrun_if_exists test_aten\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "base")
    (work / "torch" / "foo.py").write_text(FOO_V2)
    (work / "test" / "test_pure.py").write_text(TEST_PURE_V2)
    _git(work, "commit", "-aqm", "(#123) change foo alpha and pure test")
    sha = _git(work, "rev-parse", "main").strip()
    bare = tmp_path / "mirror.git"
    subprocess.run(["git", "clone", "--bare", "-q", str(work), str(bare)], check=True)
    tree = tmp_path / "tree" / "pytorch"
    shutil.copytree(work, tree, ignore=shutil.ignore_patterns(".git"))
    return bare, tree, sha


FAKE_DISCOVER = ["TestAlpha.test_alpha", "TestAlpha.test_other", "TestPure.test_a", "TestPure.test_b"]


@pytest.fixture
def fake_discover(monkeypatch):
    calls: list[list[str]] = []

    def _fake(tree_root, command, timeout, cpp):
        calls.append(command)
        return list(FAKE_DISCOVER)

    monkeypatch.setattr(selection, "discover_cases", _fake)
    return calls


def test_plan_selects_direct_refs_and_changed_tests(env, fake_discover):
    bare, tree, sha = env
    report = selection.select_for_commit(
        bare, sha, "(#123) change foo alpha", tree, collect_timeout=1, interpreter="unused"
    )
    p0 = {item["path"] for item in report["p0"]}
    assert p0 == {"test/test_pure.py", "test/test_foo.py"}
    pure = next(i for i in report["p0"] if i["path"] == "test/test_pure.py")
    assert pure["test_nodes"] == ["TestPure.test_b"]
    assert pure["score"] == selection.P0_SCORE_NARROWED  # 窄化收窄：最高证据分
    foo = next(i for i in report["p0"] if i["path"] == "test/test_foo.py")
    assert "TestAlpha.test_alpha" in foo["test_nodes"]
    assert foo["score"] == selection.P0_SCORE_DIRECT_REF
    assert any(
        e["type"] == "direct_changed_symbol_reference" and e["symbol"] == "alpha"
        for e in foo["evidence"]
    )

    syms = {(i["source"], i["symbol"]) for i in report["missing_coverage"]}
    assert ("torch/foo.py", "beta") in syms
    assert ("torch/foo.py", "alpha") not in syms

    assert report["selection_granularity"] == "case"
    assert report["candidate_count"] == 2
    assert report["selected_count"] == 2
    assert report["deferred_tests"] == []
    assert report["unresolved_tests"] == []


def test_manifest_bridge_order_and_device(env, fake_discover):
    bare, tree, sha = env
    report = selection.select_for_commit(
        bare, sha, "(#123) change foo alpha", tree, collect_timeout=1, interpreter="unused"
    )
    manifest = selection.plan_to_manifest(report, device="cuda")
    # 证据分级排序：窄化被改测试文件(300) > 直接符号引用(250)。
    assert manifest == [
        {"test_function": "test/test_pure.py::TestPure::test_b", "device": "cuda"},
        {"test_function": "test/test_foo.py::TestAlpha::test_alpha", "device": "cuda"},
    ]
    assert report["manifest_skipped"] == []


def test_finalize_budget_round_robin(tmp_path, monkeypatch):
    def _fake(tree_root, command, timeout, cpp):
        n = Path(command[1]).stem.split("_")[1].upper()  # test/test_a.py → A
        return [f"Test{n}.test_x", f"Test{n}.test_y"]

    monkeypatch.setattr(selection, "discover_cases", _fake)
    report = {"p0": [], "p1": [], "missing_coverage": [], "ci_recommendations": [],
              "analysis_gaps": [], "notes": []}
    for n in ("a", "b", "c"):
        # 类名首字母大写与 discover 桩返回一致（TestA…），文件名保持小写。
        report["p1"].append({
            "path": f"test/test_{n}.py",
            "test_nodes": [f"Test{n.upper()}.test_x", f"Test{n.upper()}.test_y"],
            "evidence": [],
        })
    selection.finalize_plan(tmp_path, report, 4, 1, "", "unused")
    selected = report["p1"]
    assert len(selected) == 4
    paths = [item["path"] for item in selected]
    assert len(set(paths[:3])) == 3  # 轮转：前 3 条分属 3 个文件，大文件不吃满槽位
    deferred = report["deferred_tests"]
    assert len(deferred) == 2
    assert all(d["priority"] == "P1" and "global 4-case budget" in d["reason"] for d in deferred)
    assert report["selection_granularity"] == "case"
    assert report["candidate_count"] == 6
    assert report["selected_count"] == 4


def test_discovery_failure_static_fallback(env):
    """发现进程失败（环境问题）：静态回退 family 节点，候选不丢。

    family 名 ``Class.method`` 即合法 pytest nodeid；score 打 50 分折。
    执行端 A3 的 no_result 兜底真机收集不到的情况。
    """
    bare, tree, sha = env
    report = selection.select_for_commit(
        bare, sha, "(#123) x", tree, collect_timeout=5, interpreter="false"
    )
    assert report["candidate_count"] == 2  # 两个 P0 家族都静态回退成功
    assert report["unresolved_tests"] == []  # 静态回退不算 unresolved
    assert any("test discovery" in g["reason"] for g in report["analysis_gaps"])
    for item in report["p0"]:
        assert any(
            e["type"] == "static_fallback_after_discovery_failure"
            for e in item["evidence"]
        )
    manifest = selection.plan_to_manifest(report, device="cuda")
    assert {m["test_function"] for m in manifest} == {
        "test/test_pure.py::TestPure::test_b",
        "test/test_foo.py::TestAlpha::test_alpha",
    }


def test_discovery_success_family_missing_stays_unresolved(env, fake_discover):
    """发现成功但家族解析不到节点（方法改名/删除）：仍 unresolved，不硬造。"""
    bare, tree, sha = env
    report = selection.select_for_commit(
        bare, sha, "(#123) x", tree, collect_timeout=1, interpreter="unused",
        pr_text="",  # 桩 FAKE_DISCOVER 固定返回四节点，test_b 可解析
    )
    # 构造一个发现成功但方法名不存在的 P0 条目。
    report2 = {
        "p0": [{"path": "test/test_foo.py", "test_nodes": ["TestAlpha.test_gone"],
                "evidence": []}],
        "p1": [], "missing_coverage": [], "ci_recommendations": [],
        "analysis_gaps": [], "notes": [],
    }
    selection.finalize_plan(Path(tree), report2, 5, 1, "", "unused")
    assert report2["candidate_count"] == 0
    assert report2["unresolved_tests"] and (
        report2["unresolved_tests"][0]["test_node"] == "TestAlpha.test_gone"
    )


def test_fallback_nodes_throttled_to_smallest(tmp_path):
    """共享改动回退全节点：按方法行数限流取最小 3 个，其余记 gap。"""
    (tmp_path / "test").mkdir(parents=True)
    src = "class TestBig:\n"
    for i in range(6):
        src += f"    def test_m{i}(self):\n" + "        assert True\n" * (i + 1)
    (tmp_path / "test" / "test_big.py").write_text(src)
    # 改动落在类定义行（共享，无法收窄到方法）。
    item = selection.ChangedFile(
        path="test/test_big.py", status="M", changed_lines={1},
    )
    nodes, reason, total = selection.changed_test_nodes(tmp_path, item)
    assert total == 6
    assert len(nodes) == 3
    assert nodes == ["TestBig.test_m0", "TestBig.test_m1", "TestBig.test_m2"]
    assert "shared or deletion" in reason

    # build_report 层：fallback 标记 + 限流 gap 记账。
    report = selection.build_report(
        "sha", "base", [item], [], tmp_path, 1, 5000, 6,
    )
    entry = next(i for i in report["p0"] if i["path"] == "test/test_big.py")
    assert entry["fallback"] is True
    assert entry["score"] == selection.P0_SCORE_FALLBACK
    assert len(entry["test_nodes"]) == 3
    assert any("throttled" in g["reason"] for g in report["analysis_gaps"])


def test_p0_evidence_grading_drives_budget_order(tmp_path, monkeypatch):
    """finalize 全局预算按 P0 证据分排序：窄化 > 直接引用 > 回退 > P1 金丝雀。"""
    monkeypatch.setattr(
        selection, "discover_cases",
        lambda tree_root, command, timeout, cpp: ["TestX.test_n"],
    )
    report = {
        "p0": [
            {"path": "test/test_narrowed.py", "test_nodes": ["TestX.test_n"],
             "score": selection.P0_SCORE_NARROWED, "evidence": []},
            {"path": "test/test_direct.py", "test_nodes": ["TestX.test_n"],
             "score": selection.P0_SCORE_DIRECT_REF, "evidence": []},
            {"path": "test/test_fallback.py", "test_nodes": ["TestX.test_n"],
             "score": selection.P0_SCORE_FALLBACK, "fallback": True, "evidence": []},
        ],
        "p1": [
            {"path": "test/test_canary.py", "test_nodes": ["TestX.test_n"],
             "score": 20, "evidence": []},
        ],
        "missing_coverage": [], "ci_recommendations": [],
        "analysis_gaps": [], "notes": [],
    }
    selection.finalize_plan(tmp_path, report, 3, 1, "", "unused")
    assert [i["path"] for i in report["p0"]] == [
        "test/test_narrowed.py", "test/test_direct.py", "test/test_fallback.py",
    ]
    # 预算耗尽：P1 金丝雀进 deferred（回退 P0 优先于 P1）。
    assert [d["path"] for d in report["deferred_tests"]] == ["test/test_canary.py"]


def test_manifest_bridge_skips_unrepresentable():
    report = {
        "p0": [{"path": "test/test_nested.py", "test_nodes": ["Outer.Inner.test_n"], "evidence": []}],
        "p1": [
            {"path": "cpp/test_x", "kind": "gtest", "binary": "build/bin/test_x",
             "test_nodes": ["Suite.case"], "score": 1, "evidence": []},
            {"path": "test/test_param.py", "test_nodes": ["TestP.test_x[0]"], "evidence": []},
        ],
        "missing_coverage": [], "ci_recommendations": [], "analysis_gaps": [], "notes": [],
    }
    manifest = selection.plan_to_manifest(report, device="cuda")
    assert manifest == []
    skipped = report["manifest_skipped"]
    assert any(s["reason"].startswith("gtest") for s in skipped)
    assert any("nested-class" in s["reason"] for s in skipped)
    assert any("test_x[0]" in s["reason"] or "test_x" in s["node"] for s in skipped)


def test_pr_text_body_mentions_boost(env, fake_discover):
    """完整 PR 文本（%B 正文）提及具体用例 → mentioned_in_pr 加权证据。

    只给 subject 时该信号几乎不触发（B2 的动机）；pr_text 传入后正文
    里的用例名成为 200 分信号源。
    """
    bare, tree, sha = env
    report = selection.select_for_commit(
        bare, sha, "(#123) x", tree, collect_timeout=1, interpreter="unused",
        pr_text="(#123) x\n\nFixes TestPure.test_b by adjusting alpha.",
    )
    pure = next(i for i in report["p0"] if i["path"] == "test/test_pure.py")
    assert pure["score"] == selection.P0_SCORE_NARROWED  # 与 200 取 max 仍为 300
    assert any(e["type"] == "mentioned_in_pr" for e in pure["evidence"])


def test_baseline_coverage_annotation(tmp_path):
    """B4：清单逐条查基线库 hit/miss——miss 即首次基线、无对比信号。"""
    from run_test.baseline_store import (
        BaselineRecord,
        BaselineRouter,
        baseline_db_path,
        ensure_baseline_db,
        upsert_baseline,
    )
    from run_test.db import connect
    from run_test.identifiers import Counts

    hit_tf = "test/distributed/_tensor/test_tensor.py::test_basic"
    root = tmp_path / "baselines"
    router = BaselineRouter(["test/distributed/_tensor"], root)
    ensure_baseline_db(router.managed_root("test/distributed/_tensor"))
    conn = connect(baseline_db_path(router.managed_root("test/distributed/_tensor")))
    try:
        upsert_baseline(conn, BaselineRecord(
            test_function=hit_tf, device="cpu", pytorch_commit="a" * 40,
            torch_npu_environment={"location": "x", "torch_npu_commit": "b" * 40},
            counts=Counts(1, 0, 0, 0),
            log_path="logs/x.log",
        ))
    finally:
        conn.close()

    manifest = [
        {"test_function": hit_tf, "device": "cpu"},  # 命中
        {"test_function": "test/distributed/_tensor/test_tensor.py::test_other",
         "device": "cuda"},  # 未命中
    ]
    cov = selection.baseline_coverage(router, manifest)
    assert cov["hit"] == 1 and cov["miss"] == 1
    assert cov["miss_entries"] == [
        {"test_function": "test/distributed/_tensor/test_tensor.py::test_other",
         "device": "cuda"},
    ]

    # render_text 汇总行。
    text = selection.render_text({
        "pr_commit": "x", "p0": [], "p1": [], "missing_coverage": [],
        "ci_recommendations": [], "analysis_gaps": [], "baseline_coverage": cov,
    })
    assert "Baseline coverage: 1 hit / 1 miss" in text


def test_cpp_catalog_from_official_lists(env):
    bare, tree, sha = env
    catalog, gaps = selection.cpp_test_catalog(tree)
    # run_if_exists 需要本地 build/bin（测试服务器不构建）→ 只记 gap，不伪造套件
    assert catalog == {}
    assert any("run_if_exists" in g["reason"] for g in gaps)
    assert any("no literal baseline" in g["reason"] for g in gaps)


def test_shared_change_keeps_directly_changed_method():
    from run_test.selection import ChangedFile, changed_test_nodes
    tree = Path(__file__).parent / "_selection_shared_tmp"
    tree.mkdir(exist_ok=True)
    try:
        (tree / "test_shared.py").write_text(
            "class T:\n" + "".join(f"    def test_{x}(self):\n        pass\n" for x in ["a", "b", "changed", "d"]),
            encoding="utf-8",
        )
        nodes, _, _ = changed_test_nodes(
            tree,
            ChangedFile("test_shared.py", "M", changed_lines={5, 6}),
            max_fallback_nodes=1,
        )
        assert "T.test_changed" in nodes
    finally:
        import shutil
        shutil.rmtree(tree, ignore_errors=True)


def test_finalize_rejects_prefix_collisions(tmp_path, monkeypatch):
    """P1-2（2026-09-20）反例回归锁：裸 startswith 的两处误并入必须消除。

    - TestFoo 不得撞 TestFooBar / TestFooPrefixed（同前缀类）；
    - test_sub 不得撞 test_sub_scalar（同前缀方法，无设备 token）。
    """
    def _fake(tree_root, command, timeout, cpp, **kwargs):
        return [
            "TestFooBar.test_x_cpu",        # 同前缀类 → 不并
            "TestFooPrefixed.test_x_cpu",   # 同前缀类 → 不并
            "TestFoo.test_sub_scalar",      # 同前缀方法 → 不并
            "TestFoo.test_x_cpu",           # 真变体 → 并
            "TestFoo.test_x_cuda_float32",  # 真变体（设备+dtype）→ 并
        ]

    monkeypatch.setattr(selection, "discover_cases", _fake)
    report = {
        "p0": [{"path": "test/test_foo.py",
                "test_nodes": ["TestFoo.test_x", "TestFoo.test_sub"],
                "evidence": []}],
        "p1": [], "missing_coverage": [], "ci_recommendations": [],
        "analysis_gaps": [], "notes": [],
    }
    selection.finalize_plan(tmp_path, report, 10, 1, "", "unused")
    nodes = {item["test_nodes"][0] for item in report["p0"]}
    assert nodes == {"TestFoo.test_x_cpu", "TestFoo.test_x_cuda_float32"}
    # test_sub 家族：唯一收集到的 test_sub_scalar 非变体 → unresolved（不硬造）。
    assert any(u["test_node"] == "TestFoo.test_sub"
               for u in report["unresolved_tests"])
