"""测试函数标识、device、四项计数与日志文件命名的共用规则。

对应《本地测试结果与基线存储说明》第 2 节（共用标识与字段规则）：
- 完整函数标识使用仓库相对路径与 `/` 分隔符，有类名必须保留类名；
- device 只允许 cpu / accelerator / cuda 三个字面值；
- pass/fail/error/skip 均为非负整数计数列，不是状态字符串；
- 日志文件名必须由完整函数标识与 device 唯一确定。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

VALID_DEVICES = ("cpu", "accelerator", "cuda")

# 完整函数标识的两种形式（文档 2.1）：
#   test/path/test_file.py::test_function
#   test/path/test_file.py::TestClass::test_function
_TEST_FUNCTION_RE = re.compile(
    r"^(?P<path>[^:]+\.py)"
    r"(?:::(?P<class_name>[A-Za-z_][A-Za-z0-9_]*))?"
    r"(?:::(?P<func_name>[A-Za-z_][A-Za-z0-9_]*))$"
)

# 精确 commit：完整 SHA-1（40 位）或 SHA-256（64 位）十六进制。
# “main”“latest”、短 SHA 等不精确版本一律拒绝（文档 2.4）。
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IMPRECISE_COMMITS = {"", "main", "latest", "head", "master", "origin/main"}

# 参数化子用例不允许进入函数标识（文档 2.1：最小记录粒度是函数）。
_PARAMETRIZED_MARK = "["


class IdentifierError(ValueError):
    """标识、device、计数或版本信息不符合存储契约。"""


# ---------------------------------------------------------------------------
# 完整函数标识
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedTestFunction:
    file_path: str
    class_name: str | None
    func_name: str


def validate_test_function(value: object) -> str:
    """校验并原样返回完整函数标识。"""
    if not isinstance(value, str):
        raise IdentifierError(f"test_function 必须是字符串，收到 {value!r}")
    tf = value.strip()
    if _PARAMETRIZED_MARK in tf:
        raise IdentifierError(
            f"test_function 不得携带参数化参数（{tf!r}）；"
            "参数化子用例的结果应汇总到所属函数的四项计数中"
        )
    m = _TEST_FUNCTION_RE.match(tf)
    if not m:
        raise IdentifierError(
            f"test_function 格式非法: {value!r}，应为 "
            "test/path/test_file.py::test_function 或 "
            "test/path/test_file.py::TestClass::test_function"
        )
    _validate_file_path(m.group("path"))
    return tf


def _validate_file_path(path: str) -> None:
    if path.startswith("/") or "\\" in path:
        raise IdentifierError(f"文件路径必须是仓库相对路径: {path!r}")
    for seg in path.split("/"):
        if seg in ("", ".", ".."):
            raise IdentifierError(f"文件路径含空段或越段: {path!r}")


def parse_test_function(value: str) -> ParsedTestFunction:
    """拆出文件路径、类名（可为 None）与函数名。"""
    tf = validate_test_function(value)
    path, _, rest = tf.partition("::")
    if "::" in rest:
        class_name, _, func_name = rest.partition("::")
    else:
        class_name, func_name = None, rest
    return ParsedTestFunction(file_path=path, class_name=class_name, func_name=func_name)


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------

def validate_device(value: object) -> str:
    if not isinstance(value, str) or value not in VALID_DEVICES:
        raise IdentifierError(
            f"device 只允许 {list(VALID_DEVICES)} 三个字面值，收到 {value!r}；"
            "不得写 npu，也不得拼接卡号、型号或环境名"
        )
    return value


# ---------------------------------------------------------------------------
# 四项计数
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Counts:
    """pass / fail / error / skip 四个非负整数计数。

    计数以测试执行端提供的最终分类为准；error 不并入 fail，skip 不当作 pass。
    """

    pass_count: int
    fail_count: int
    error_count: int
    skip_count: int

    def __post_init__(self) -> None:
        for name in ("pass_count", "fail_count", "error_count", "skip_count"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise IdentifierError(f"{name} 必须为非负整数，收到 {v!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "Counts":
        try:
            return cls(
                pass_count=d["pass"],
                fail_count=d["fail"],
                error_count=d["error"],
                skip_count=d["skip"],
            )
        except KeyError as e:
            raise IdentifierError(f"计数缺少字段 {e}（应含 pass/fail/error/skip）") from e

    def as_dict(self) -> dict:
        return {"pass": self.pass_count, "fail": self.fail_count,
                "error": self.error_count, "skip": self.skip_count}

    @classmethod
    def zero(cls) -> "Counts":
        return cls(0, 0, 0, 0)

    @classmethod
    def sum(cls, items) -> "Counts":
        total = cls.zero()
        for it in items:
            total = cls(
                total.pass_count + it.pass_count,
                total.fail_count + it.fail_count,
                total.error_count + it.error_count,
                total.skip_count + it.skip_count,
            )
        return total

    @property
    def total(self) -> int:
        return self.pass_count + self.fail_count + self.error_count + self.skip_count

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.pass_count + other.pass_count,
            self.fail_count + other.fail_count,
            self.error_count + other.error_count,
            self.skip_count + other.skip_count,
        )


# ---------------------------------------------------------------------------
# 设备变体匹配（selection 收集展开与 runner 运行时发现共用，2026-09-20 P1-2）
# ---------------------------------------------------------------------------

# instantiate_device_type_tests 的真实生成规则（torch/testing/_internal/
# common_device_type.py 实读，非猜测）：
#   类名  = <原类名> + <device_type 大写>，如 TestDistributionsCPU /
#           TestDistributionsPRIVATEUSE1 / TestDistributionsCUDA；
#   方法名 = <原方法> + [parametrize 后缀] + "_" + <device_type> + [dtype 后缀]，
#           privateuse1 基类的方法后缀用改名后的后端名（本机 = npu）。
# 已知设备 token（torch get_device_type_test_bases 全集 + privateuse1 改名）。
_DEVICE_NAME_TOKENS = frozenset({
    "cpu", "cuda", "npu", "privateuse1", "xpu", "lazy", "mps", "hpu",
    "meta", "accelerator", "mtia",
})

# 变体尾段的值字符集：parametrize 值（False/10/abc）、dtype 名（torch.float32）。
_VARIANT_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._+-]+$")


def class_variant_match(source_class: str, concrete_class: str) -> bool:
    """concrete 是否属于 source 的同类节点。

    精确等值，或 instantiate 生成的 ``<source><DEVICE>`` 变体类（如
    TestFoo → TestFooCPU）；同前缀类（TestFooBar）不算。历史遗留的
    ``<source>DeviceType<DEV>`` 形态一并接受（device_detect 同款规则）。
    """
    if concrete_class == source_class:
        return True
    if not concrete_class.startswith(source_class):
        return False
    tail = concrete_class[len(source_class):]
    if tail[:10].lower() == "devicetype":
        tail = tail[10:]
    return tail.lower() in _DEVICE_NAME_TOKENS


def method_variant_match(source_method: str, concrete_method: str) -> bool:
    """concrete 是否属于 source 的同方法节点。

    精确等值，或设备/dtype/parametrize 变体（test_x_cuda、
    test_x_cpu_float32、test_x_<param>_cuda）；共享前缀的独立方法
    （test_sub vs test_sub_scalar）不算。尾段中必须至少含一个设备
    token 才视为设备变体——纯 parametrize 展开（test_dirichlet_10）
    不在本匹配器职责内。
    """
    if concrete_method == source_method:
        return True
    if not concrete_method.startswith(source_method + "_"):
        return False
    tail = concrete_method[len(source_method) + 1:]
    segments = [seg for seg in tail.split("_") if seg]
    if not segments:
        return False
    if not any(seg.lower() in _DEVICE_NAME_TOKENS for seg in segments):
        return False
    return all(_VARIANT_SEGMENT_RE.match(seg) is not None for seg in segments)


# ---------------------------------------------------------------------------
# 精确版本与环境信息
# ---------------------------------------------------------------------------

def validate_commit(value: object, field: str = "commit") -> str:
    """校验精确 git commit（完整 SHA-1/SHA-2 十六进制）。"""
    if not isinstance(value, str):
        raise IdentifierError(f"{field} 必须是字符串，收到 {value!r}")
    v = value.strip().lower()
    if v in _IMPRECISE_COMMITS:
        raise IdentifierError(f"{field} 必须是精确 commit，不得使用 {value!r}")
    if not _COMMIT_RE.match(v):
        raise IdentifierError(
            f"{field} 必须是完整 40/64 位十六进制 commit，收到 {value!r}"
        )
    return v


def validate_torch_npu_environment(value: object) -> dict:
    """校验 torch_npu_environment：必须含环境定位信息与精确 torch-npu commit。

    直接保存环境信息本身，不是指向另一张环境表的 ID（文档 2.4）。
    """
    if not isinstance(value, dict):
        raise IdentifierError(
            f"torch_npu_environment 必须是对象，含 location 与 torch_npu_commit，收到 {value!r}"
        )
    location = value.get("location")
    if not isinstance(location, str) or not location.strip():
        raise IdentifierError(
            "torch_npu_environment.location 必须是非空的环境路径或名称"
        )
    commit = validate_commit(value.get("torch_npu_commit"), "torch_npu_commit")
    normalized = dict(value)
    normalized["location"] = location.strip()
    normalized["torch_npu_commit"] = commit
    return normalized


# ---------------------------------------------------------------------------
# 日志文件命名
# ---------------------------------------------------------------------------

_SLUG_MAX = 120


def baseline_log_filename(test_function: str, device: str) -> str:
    """由完整函数标识 + device 生成无冲突的基线日志文件名。

    形如 ``<可读 slug>.<device>.<sha256 前 16 位>.log``：
    slug 仅提供可读性，唯一性由完整标识 + device 的哈希保证；
    写入侧还会在库内检查文件名未被其他 (test_function, device) 占用，
    双重保障“不同文件或类的同名函数不共享日志”。
    """
    validate_test_function(test_function)
    validate_device(device)
    slug = _slugify(test_function)
    digest = hashlib.sha256(
        f"{test_function}\x00{device}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{slug}.{device}.{digest}.log"


def _slugify(test_function: str) -> str:
    out = []
    for ch in test_function:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:_SLUG_MAX]


def pr_log_dir_name(pr_number: int) -> str:
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
        raise IdentifierError(f"PR 号必须是正整数，收到 {pr_number!r}")
    return f"PR_{pr_number}"
