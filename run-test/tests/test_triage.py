"""triage 失败签名分类：类别命中、顺序敏感、函数级并集（C99 剔除）。

规则移植自 ~/tmp/analyze.py（transfer-CUDA 基线 25 类），顺序与类别 ID
必须保持一致——这里的用例同时是"移植忠实性"的回归锁。
"""

from run_test.triage import CATEGORY_OTHER, classify_text, classify_texts


def test_c01_private_cuda_api():
    assert classify_text(
        "AttributeError: module 'torch._C' has no attribute '_cuda_setStream'"
    ) == "C01-torchC-private-cuda-api"


def test_c03_and_c02_top_three():
    assert classify_text(
        "RuntimeError: No backend type associated with device type npu!"
    ) == "C03-pg-no-npu-backend"
    assert classify_text(
        "AttributeError: 'function' object has no attribute 'cache_clear'"
    ) == "C02-npu-inductor-cache-clear"


def test_c05_compound_requires_both_parts():
    """复合签名：两段都在（哪怕分处两行）才命中 C05。"""
    assert classify_text(
        "ProcessGroupGloo::allgather: unsupported device type npu"
    ) == "C05-gloo-unsupported-npu"
    # 只有前半 → 不命中 C05（落入后续规则或 C99）。
    assert classify_text("ProcessGroupGloo::something else") == CATEGORY_OTHER


def test_c09_first_and_shard_rules():
    assert classify_text("AssertionError: 'npu' != 'cuda'") == \
        "C09-device-assert-cuda-vs-npu"
    # C09 第二条（shard 元数据）排在 C24 之后——顺序敏感回归锁。
    assert classify_text(
        "local shard tensor device: npu:0, local shard metadata placement device: cuda"
    ) == "C09-device-assert-cuda-vs-npu"


def test_c20_before_c17_ordering():
    """C20–C24 在 C17–C19 之前：同时可命中时取 C20。"""
    text = "'>=' not supported between instances of 'NoneType' and 'int' in get_gpu_type; Didn't find engine for operation quantized::x"
    assert classify_text(text) == "C20-capability-none"


def test_c23_parametrize_regex():
    assert classify_text(
        "test_x() missing 2 required positional argument: 'device' and 'dtype'"
    ) == "C23-test-parametrize"


def test_c99_fallback():
    assert classify_text("AssertionError: torch.add wrong result") == CATEGORY_OTHER


def test_classify_texts_union_excludes_c99():
    texts = [
        "module 'torch._C' has no attribute '_cuda_getCompiledVersion'",  # C01
        "RuntimeError: No backend type associated with device type npu",   # C03
        "some unknown failure",                                            # C99
        "AttributeError: module 'torch._C' has no attribute '_cuda_setStream'",  # C01 重复
    ]
    assert classify_texts(texts) == [
        "C01-torchC-private-cuda-api",
        "C03-pg-no-npu-backend",
    ]
    # 全 C99 → 空并集（无已知签名 = 可能是真实代码问题）。
    assert classify_texts(["plain failure", "another one"]) == []


# ---------------------------------------------------------------------------
# 2026-09-20 增补（偏离 analyze.py 原版的差异，逐条回归锁）
# ---------------------------------------------------------------------------


def test_c25_autotune_gpu_type():
    """inductor autotune 进程池按字面 gpu_type 建池——43-PR 实跑 PR 179547
    的 8 个 tuning_pool 失败此前无一命中签名（已知环境墙必须可结构化）。"""
    assert classify_text(
        'raise ValueError(f"Unsupported gpu_type: {gpu_type}")\n'
        "ValueError: Unsupported gpu_type: npu"
    ) == "C25-autotune-gpu-type-unsupported"


def test_c26_guard_before_c10_and_c10_negative_guards():
    """cuda-designated 守卫 × transfer：mock call_count 断言失败。

    - call_count 断言的 "Scalars are not equal!" 必须落 C26（在 C10 之前），
      不得再误标为数值计算差异（实证：PR 195049 device_ok 误标 C10）；
    - C10 负向守卫：退出码断言（"Expected exit code"）不是数值差异，
      落 C99（实证：PR 196922）；
    - 真数值失配仍落 C10。
    """
    guard_text = (
        'File "test_native_utils.py", line 116, in test_device_ok_memoizes_per_device\n'
        "    self.assertEqual(query.call_count, 1)\n"
        "AssertionError: Scalars are not equal!\n\n"
        "Expected 1 but got 0.\nAbsolute difference: 1"
    )
    assert classify_text(guard_text) == "C26-cuda-designated-guard-transfer"

    exit_code_text = (
        "AssertionError: Scalars are not equal!\n\n"
        "Expected 0 but got 1.\nAbsolute difference: 1\n"
        "Expected exit code 0 but got 1 for pid: 978811"
    )
    assert classify_text(exit_code_text) == CATEGORY_OTHER

    numeric_text = (
        "AssertionError: Tensor-likes are not close!\n\n"
        "Max difference: 0.5 (tolerance 1e-5)"
    )
    assert classify_text(numeric_text) == "C10-numeric-mismatch"

    scalar_numeric = "AssertionError: Scalars are not equal!\n\nExpected 4 but got 3."
    assert classify_text(scalar_numeric) == "C10-numeric-mismatch"
