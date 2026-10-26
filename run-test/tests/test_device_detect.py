"""设备判定 v2（AST 语义识别）与 plan_to_manifest 的 device_of 钩子。

v2 相对 v1 的关键差异（2026-09-18 外部评审意见）：
- 信号全部走 AST 节点：注释 / 文档字符串不再误判（test_comments_…）；
- instantiate_device_type_tests 逐类解析 only_for/except_for，不再按文件
  一刀切（test_instantiate_only_for_is_per_class）；
- 明确声明多作用范围时一例多设备输出（test_instantiate_multi_device…）；
- 分类结果带证据与缺省原因（basis / detail），缺省仍是 cuda（用户确认）。
"""

from __future__ import annotations

import textwrap

from run_test.device_detect import (
    DEVICE_ACCELERATOR,
    DEVICE_CPU,
    DEVICE_CUDA,
    classify_function,
    make_device_resolver,
)
from run_test.selection import plan_to_manifest


def _fn(body: str, decorators: str = "", cls: str | None = None) -> str:
    """构造一个最小测试文件源码（decorators 在有 cls 时装饰类）。"""
    inner = textwrap.dedent(body)
    if cls:
        dec = textwrap.dedent(decorators)
        return f"{dec}class {cls}:\n" + textwrap.indent(inner, "    ")
    return textwrap.dedent(decorators) + inner


def _instantiated(cls_body: str, cls: str, call_kwargs: str = "") -> str:
    """构造一个 instantiate_device_type_tests 参数化测试文件。"""
    return (
        "from torch.testing._internal.common_device_type import "
        "instantiate_device_type_tests\n"
        + _fn(cls_body, cls=cls)
        + f"\ninstantiate_device_type_tests({cls}, globals(){call_kwargs})\n"
    )


# ---------------------------------------------------------------------------
# 证据一：作用范围装饰器
# ---------------------------------------------------------------------------

def test_scope_decorator_only_cuda():
    src = _fn(
        """
        @onlyCUDA
        def test_foo(self):
            pass
        """,
        cls="TestFoo",
    )
    res = classify_function(src, "TestFoo", "test_foo")
    assert res.devices == [DEVICE_CUDA]
    assert res.basis == "decorator"


def test_scope_decorator_skip_cuda_if_means_cpu():
    # skipCUDAIf = cuda 变体被跳过 → 实际作用范围是非 cuda；
    # 装饰器声明优先于其参数里的 torch.cuda 信号。
    src = _fn(
        """
        @skipCUDAIf(torch.cuda.is_available())
        def test_foo(self, device):
            pass
        """,
        cls="TestFoo",
    )
    res = classify_function(src, "TestFoo", "test_foo")
    assert res.devices == [DEVICE_CUDA]
    assert res.basis == "decorator"


def test_class_scope_decorator_covers_methods():
    src = _fn(
        "def test_foo(self):\n    pass\n",
        decorators="@onlyCPU\n",
        cls="TestFoo",
    )
    res = classify_function(src, "TestFoo", "test_foo")
    assert res.devices == [DEVICE_CPU]
    assert res.basis == "decorator"


def test_method_and_class_decorators_union_multi_device():
    # 类装饰器 onlyCUDA + 方法装饰器 onlyCPU → 两个作用范围都保留。
    src = _fn(
        """
        @onlyCPU
        def test_foo(self, device):
            pass
        """,
        decorators="@onlyCUDA\n",
        cls="TestFoo",
    )
    res = classify_function(src, "TestFoo", "test_foo")
    assert res.devices == [DEVICE_CUDA, DEVICE_CPU]
    assert res.basis == "decorator"


# ---------------------------------------------------------------------------
# 证据三：源码信号（AST 语义，注释/文档字符串不参与）
# ---------------------------------------------------------------------------

def test_signal_torch_cuda():
    src = _fn("def test_f(self):\n    torch.cuda.synchronize()\n")
    res = classify_function(src, None, "test_f")
    assert (res.devices, res.basis) == ([DEVICE_CUDA], "signal")


def test_signal_cpu_keyword_and_method_call():
    kw = _fn('def test_a(self):\n    x = torch.zeros(3, device="cpu")\n')
    assert classify_function(kw, None, "test_a").devices == [DEVICE_CPU]
    call = _fn("def test_b(self):\n    y = out.cpu()\n")
    assert classify_function(call, None, "test_b").devices == [DEVICE_CPU]


def test_signal_cuda_wins_for_runnability():
    # .cuda() 是硬性运行要求：cpu 通道无 boot 无 CUDA 会直接失败。
    src = _fn("def test_m(self):\n    a = x.cuda()\n    b = a.cpu()\n")
    res = classify_function(src, None, "test_m")
    assert res.devices == [DEVICE_CUDA]
    assert res.basis == "signal"


def test_signal_accelerator_and_npu_strings():
    acc = _fn("def test_a(self):\n    torch.accelerator.current_accelerator()\n")
    assert classify_function(acc, None, "test_a").devices == [DEVICE_ACCELERATOR]
    npu = _fn('def test_n(self):\n    x = torch.zeros(2, device="npu")\n')
    assert classify_function(npu, None, "test_n").devices == [DEVICE_ACCELERATOR]
    pu1 = _fn('def test_p(self):\n    dev = "privateuse1"\n')
    assert classify_function(pu1, None, "test_p").devices == [DEVICE_ACCELERATOR]


def test_signal_cuda_index_and_ctor():
    src = _fn('def test_c(self):\n    d = torch.device("cuda:1")\n')
    assert classify_function(src, None, "test_c").devices == [DEVICE_CUDA]


def test_import_from_torch_cuda_counts():
    src = (
        "from torch.cuda import CUDAGraph\n\n"
        "def test_g(self):\n"
        "    with CUDAGraph():\n"
        "        y = x.cpu()\n"
    )
    # 若不识别 import，.cpu() 会让它误判 cpu，运行时 CUDAGraph 直接失败。
    assert classify_function(src, None, "test_g").devices == [DEVICE_CUDA]


def test_comments_and_docstrings_are_not_signals():
    src = (
        "def test_q(self):\n"
        "    # only cpu here; device='cpu'\n"
        '    """cpu"""\n'
        "    pass\n"
    )
    res = classify_function(src, None, "test_q")
    assert res.devices == [DEVICE_CUDA]
    assert res.basis == "default"


def test_default_cuda_with_reason_when_unmarked():
    src = _fn("def test_add(self):\n    assert torch.add(1, 2) == 3\n")
    res = classify_function(src, None, "test_add")
    assert res.devices == [DEVICE_CUDA]
    assert res.basis == "default"
    assert "缺省 cuda" in res.detail


# ---------------------------------------------------------------------------
# 证据二：instantiate_device_type_tests 逐类解析
# ---------------------------------------------------------------------------

def test_instantiate_only_for_is_per_class():
    # v1 按文件一刀切会把 TestB 也判成 cuda；v2 逐类解析。
    src = (
        _instantiated("def test_a(self, device):\n    pass\n",
                      "TestA", call_kwargs=", only_for=('cuda',)")
        + _instantiated("def test_b(self, device):\n    pass\n",
                        "TestB", call_kwargs=", only_for=('cpu',)")
    )
    a = classify_function(src, "TestA", "test_a")
    b = classify_function(src, "TestB", "test_b")
    assert (a.devices, a.basis) == ([DEVICE_CUDA], "instantiate")
    assert (b.devices, b.basis) == ([DEVICE_CPU], "instantiate")


def test_instantiate_multi_device_outputs_both():
    # 显式声明多作用范围 → 一例多设备。
    src = _instantiated("def test_m(self, device):\n    pass\n",
                        "TestM", call_kwargs=", only_for=('cuda', 'cpu')")
    res = classify_function(src, "TestM", "test_m")
    assert res.devices == [DEVICE_CUDA, DEVICE_CPU]
    assert res.basis == "instantiate"


def test_instantiate_except_for():
    src = _instantiated("def test_e(self, device):\n    pass\n",
                        "TestE", call_kwargs=", except_for=('cuda',)")
    res = classify_function(src, "TestE", "test_e")
    assert res.devices == [DEVICE_ACCELERATOR, DEVICE_CPU]
    assert res.basis == "instantiate"


def test_instantiate_unrestricted_defaults_cuda_and_body_literals_do_not_override():
    # 无限制实例化 → 走 _cuda_ 变体保持基线可比；方法体内的 device="cpu"
    # 属于函数实现，不决定实例化出哪些设备变体。
    src = _instantiated('def test_u(self, device):\n    x = torch.zeros(3, device="cpu")\n',
                        "TestU")
    res = classify_function(src, "TestU", "test_u")
    assert res.devices == [DEVICE_CUDA]
    assert res.basis == "default"


def test_method_scope_decorator_overrides_instantiate():
    src = _instantiated(
        """
        @onlyCPU
        def test_o(self, device):
            pass
        """,
        "TestO", call_kwargs=", only_for=('cuda',)",
    )
    res = classify_function(src, "TestO", "test_o")
    assert (res.devices, res.basis) == ([DEVICE_CPU], "decorator")


# ---------------------------------------------------------------------------
# 失败路径：缺省 cuda + 原因（用户确认保留缺省）
# ---------------------------------------------------------------------------

def test_failures_default_to_cuda_with_reason():
    broken = classify_function("def broken(:\n", None, "broken")
    assert broken.devices == [DEVICE_CUDA] and broken.basis == "default"
    src = _fn("def other(self):\n    pass\n")
    missing_fn = classify_function(src, None, "no_such_fn")
    assert missing_fn.devices == [DEVICE_CUDA] and "定位失败" in missing_fn.detail
    missing_cls = classify_function(src, "NoCls", "other")
    assert missing_cls.devices == [DEVICE_CUDA] and "定位失败" in missing_cls.detail


# ---------------------------------------------------------------------------
# 解析器与清单展开
# ---------------------------------------------------------------------------

def test_make_device_resolver_and_diagnostics(tmp_path):
    (tmp_path / "test_a.py").write_text(
        _fn(
            """
            def test_uses_cuda(self):
                torch.cuda.synchronize()

            def test_uses_accel(self):
                torch.accelerator.current_accelerator()
            """,
            cls="TestA",
        ),
        encoding="utf-8",
    )
    resolve = make_device_resolver(tmp_path)
    assert resolve("test_a.py", "TestA.test_uses_cuda") == [DEVICE_CUDA]
    assert resolve("test_a.py", "TestA.test_uses_accel") == [DEVICE_ACCELERATOR]
    assert resolve("missing.py", "test_x") == [DEVICE_CUDA]  # 文件缺失 → 缺省
    by_node = {d["node"]: d for d in resolve.diagnostics}
    assert by_node["TestA.test_uses_cuda"]["devices"] == [DEVICE_CUDA]
    assert by_node["TestA.test_uses_cuda"]["basis"] == "signal"
    assert by_node["test_x"]["basis"] == "default"
    assert "文件缺失" in by_node["test_x"]["detail"]


def _report_for(paths_nodes):
    return {
        "p0": [],
        "p1": [
            {"path": p, "test_nodes": [n], "evidence": []}
            for (p, n) in paths_nodes
        ],
        "missing_coverage": [], "ci_recommendations": [],
        "analysis_gaps": [], "notes": [],
    }


def test_plan_to_manifest_expands_multi_device():
    report = _report_for([("test/test_m.py", "TestM.test_both")])
    manifest = plan_to_manifest(report, device_of=lambda p, n: [DEVICE_CUDA, DEVICE_CPU])
    assert manifest == [
        {"test_function": "test/test_m.py::TestM::test_both", "device": "cuda"},
        {"test_function": "test/test_m.py::TestM::test_both", "device": "cpu"},
    ]


def test_plan_to_manifest_device_of_single_string_still_works():
    report = _report_for(
        [
            ("test/test_a.py", "TestA.test_cuda"),
            ("test/test_a.py", "TestA.test_accel"),
        ]
    )
    manifest = plan_to_manifest(
        report,
        device_of=lambda path, node: (
            DEVICE_CUDA if "cuda" in node else DEVICE_ACCELERATOR
        ),
    )
    assert manifest == [
        {"test_function": "test/test_a.py::TestA::test_cuda", "device": "cuda"},
        {"test_function": "test/test_a.py::TestA::test_accel",
         "device": "accelerator"},
    ]
    assert report["manifest_skipped"] == []


def test_plan_to_manifest_device_of_invalid_device_skips():
    report = _report_for([("test/test_a.py", "test_x")])
    manifest = plan_to_manifest(report, device_of=lambda path, node: ["cuda", "gpu"])
    assert manifest == []
    assert len(report["manifest_skipped"]) == 1
    assert report["manifest_skipped"][0]["reason"]


def test_plan_to_manifest_fixed_device_still_works():
    report = _report_for([("test/test_a.py", "test_x")])
    manifest = plan_to_manifest(report, device=DEVICE_CPU)
    assert manifest == [
        {"test_function": "test/test_a.py::test_x", "device": "cpu"}
    ]


def test_conditional_and_expected_failure_decorators_do_not_select_cpu():
    src = _fn("""
        @skipCUDAIf(False, "condition")
        @expectedFailCPU
        def test_foo(self):
            torch.cuda.synchronize()
    """, cls="TestFoo")
    result = classify_function(src, "TestFoo", "test_foo")
    assert result.devices == [DEVICE_CUDA]
    assert result.basis == "decorator"
