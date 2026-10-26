"""完整函数标识 / device / 计数 / commit / 日志名规则（存储文档 2.x 节）。"""

import pytest

from run_test.identifiers import (
    Counts,
    IdentifierError,
    baseline_log_filename,
    class_variant_match,
    method_variant_match,
    parse_test_function,
    validate_commit,
    validate_device,
    validate_test_function,
    validate_torch_npu_environment,
)


class TestTestFunction:
    def test_plain_function(self):
        assert validate_test_function("test/x/test_y.py::test_f") == "test/x/test_y.py::test_f"

    def test_with_class(self):
        assert validate_test_function("test/a.py::TC::test_m") == "test/a.py::TC::test_m"

    def test_class_must_be_preserved(self):
        # 同文件不同类的同名函数必须可区分。
        a = baseline_log_filename("test/a.py::T1::test_x", "cpu")
        b = baseline_log_filename("test/a.py::T2::test_x", "cpu")
        assert a != b

    def test_rejects_absolute_path(self):
        with pytest.raises(IdentifierError):
            validate_test_function("/abs/test.py::test_f")

    def test_rejects_dotdot(self):
        with pytest.raises(IdentifierError):
            validate_test_function("test/../x.py::test_f")

    def test_rejects_parametrized(self):
        with pytest.raises(IdentifierError):
            validate_test_function("test/x.py::test_f[param0]")

    def test_rejects_missing_function(self):
        with pytest.raises(IdentifierError):
            validate_test_function("test/x.py")

    def test_rejects_three_segments(self):
        with pytest.raises(IdentifierError):
            validate_test_function("test/x.py::A::B::test_f")

    def test_parse(self):
        p = parse_test_function("test/d/t.py::TC::test_m")
        assert (p.file_path, p.class_name, p.func_name) == ("test/d/t.py", "TC", "test_m")
        p2 = parse_test_function("test/d/t.py::test_m")
        assert (p2.file_path, p2.class_name, p2.func_name) == ("test/d/t.py", None, "test_m")


class TestDevice:
    @pytest.mark.parametrize("dev", ["cpu", "accelerator", "cuda"])
    def test_valid(self, dev):
        assert validate_device(dev) == dev

    @pytest.mark.parametrize("dev", ["npu", "NPU", "cuda:0", "910B4", "", None, 1])
    def test_invalid(self, dev):
        with pytest.raises(IdentifierError):
            validate_device(dev)


class TestCounts:
    def test_valid(self):
        c = Counts(1, 2, 3, 4)
        assert c.as_dict() == {"pass": 1, "fail": 2, "error": 3, "skip": 4}

    @pytest.mark.parametrize("args", [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)])
    def test_rejects_negative(self, args):
        with pytest.raises(IdentifierError):
            Counts(*args)

    def test_rejects_bool_and_str(self):
        with pytest.raises(IdentifierError):
            Counts(True, 0, 0, 0)
        with pytest.raises(IdentifierError):
            Counts("1", 0, 0, 0)

    def test_sum(self):
        assert Counts.sum([Counts(1, 1, 1, 1), Counts(2, 0, 0, 1)]) == Counts(3, 1, 1, 2)

    def test_zero_counts_is_not_pass(self):
        # 四列都为 0 不代表通过——Counts 只是数据结构，不携带状态语义。
        assert Counts.zero().total == 0


class TestCommit:
    def test_full_sha1(self):
        assert validate_commit("a" * 40) == "a" * 40

    def test_full_sha256(self):
        assert validate_commit("b" * 64) == "b" * 64

    @pytest.mark.parametrize("bad", ["", "main", "latest", "HEAD", "a" * 10, "a" * 39, "z" * 40])
    def test_rejects_imprecise(self, bad):
        with pytest.raises(IdentifierError):
            validate_commit(bad)


class TestTorchNpuEnvironment:
    def test_valid(self):
        env = validate_torch_npu_environment(
            {"location": "/envs/npu-main", "torch_npu_commit": "c" * 40}
        )
        assert env["torch_npu_commit"] == "c" * 40

    def test_missing_location(self):
        with pytest.raises(IdentifierError):
            validate_torch_npu_environment({"torch_npu_commit": "c" * 40})

    def test_missing_commit(self):
        with pytest.raises(IdentifierError):
            validate_torch_npu_environment({"location": "env"})

    def test_imprecise_commit(self):
        with pytest.raises(IdentifierError):
            validate_torch_npu_environment({"location": "env", "torch_npu_commit": "main"})


class TestLogFilename:
    def test_deterministic_and_unique(self):
        a = baseline_log_filename("test/a.py::test_x", "cpu")
        assert a == baseline_log_filename("test/a.py::test_x", "cpu")
        assert a != baseline_log_filename("test/a.py::test_x", "cuda")
        assert a != baseline_log_filename("test/b.py::test_x", "cpu")

    def test_same_name_different_file_no_share(self):
        # 不同文件的同名函数不得共享/覆盖日志。
        assert (
            baseline_log_filename("test/f1.py::test_x", "cpu")
            != baseline_log_filename("test/f2.py::test_x", "cpu")
        )

    def test_no_path_separators(self):
        name = baseline_log_filename("test/deep/dir/t.py::TC::test_m", "accelerator")
        assert "/" not in name and "::" not in name

    def test_long_identifier_truncated(self):
        name = baseline_log_filename("test/" + "x" * 300 + ".py::test_f", "cpu")
        assert len(name) < 200


# ---------------------------------------------------------------------------
# 设备变体匹配（2026-09-20 P1-2：selection 收集展开与 runner 发现共用）
# ---------------------------------------------------------------------------


class TestClassVariantMatch:
    def test_exact(self):
        assert class_variant_match("TestFoo", "TestFoo")

    def test_generated_device_class(self):
        # torch instantiate 真实生成规则：<原类名> + <device 大写>。
        assert class_variant_match("TestDistributions", "TestDistributionsCPU")
        assert class_variant_match("TestDistributions", "TestDistributionsCUDA")
        assert class_variant_match("TestDistributions", "TestDistributionsPRIVATEUSE1")

    def test_legacy_devicetype_form(self):
        assert class_variant_match("TestFoo", "TestFooDeviceTypeCUDA")

    def test_prefix_collision_rejected(self):
        assert not class_variant_match("TestFoo", "TestFooBar")
        assert not class_variant_match("TestFoo", "TestFooScalarHelper")

    def test_unrelated(self):
        assert not class_variant_match("TestFoo", "OtherTest")


class TestMethodVariantMatch:
    def test_exact(self):
        assert method_variant_match("test_x", "test_x")

    def test_device_variants(self):
        assert method_variant_match("test_x", "test_x_cpu")
        assert method_variant_match("test_x", "test_x_cuda")
        assert method_variant_match("test_x", "test_x_npu")
        assert method_variant_match("test_x", "test_x_privateuse1")

    def test_device_dtype_variants(self):
        assert method_variant_match("test_x", "test_x_cuda_float32")
        assert method_variant_match("test_x", "test_x_cpu_torch.float32")
        # parametrize 值在设备段之前也接受（name + param + "_" + device + dtype）。
        assert method_variant_match("test_x", "test_x_False_cuda")
        assert method_variant_match("test_x", "test_x_10_cuda_bfloat16")

    def test_prefix_collision_rejected(self):
        # P1-2 反例：共享前缀的独立方法不是变体（无设备 token）。
        assert not method_variant_match("test_sub", "test_sub_scalar")
        assert not method_variant_match("test_dirichlet", "test_dirichlet_10_torch.float")

    def test_requires_device_token(self):
        assert not method_variant_match("test_x", "test_x_scalar")
        assert not method_variant_match("test_x", "test_x_")
