"""测试函数的执行设备判定 v2 —— AST 语义识别（纯脚本，无 AI/人工介入）。

对应流程第三步"如何判定使用哪个硬件跑哪个测试"（用户 2026-09-18：所谓
分析就是设备判定，必须脚本化）。v1 为正则优先级扫描；2026-09-18 按外部
评审意见重构为 AST 语义识别，分层与规则如下。

一、功能分类（classify_*，只看证据，回答"测试函数声明/要求了哪些设备"）。
   证据全部来自 AST 节点：注释不是 AST 节点、文档字符串被显式跳过，
   ``# only cpu``、docstring 里的 "cuda" 之类文本不会误判为信号。证据分三类：

   1. 作用范围装饰器（最高优先，精确到方法；类装饰器对类内方法生效）：
      onlyCUDA / skipCPUIf / expectedFailCUDA → cuda；
      onlyCPU / skipCUDAIf / expectedFailCPU → cpu。
      （skipCUDAIf = cuda 变体被跳过，实际作用范围是非 cuda。）
   2. instantiate_device_type_tests 参数化（逐类解析 only_for/except_for，
      不按文件一刀切）：类方法的设备集合 = only_for / except_for 映射到
      cuda/accelerator/cpu 后的集合；显式声明多个通道则一例多设备输出
      （保留多作用范围）。无限制实例化（或限制映射后不含本地通道，如
      only_for=("xla",)）→ cuda：基线的 DeviceTypeCUDA 实例按 cuda 键
      入库，走 _cuda_ 变体保持 (tf, cuda) 键可比。方法级范围装饰器可
      覆盖类级结果；方法体内的 device="cpu" 之类软信号不覆盖——那属于
      函数实现，不决定实例化出哪些设备变体。
   3. 源码信号（API 调用 / 设备参数 / 设备导入）：
      torch.cuda、import torch.cuda、x.cuda()、"cuda"/cuda:N 串 → cuda；
      torch.accelerator、torch.npu、x.npu()、"accelerator"/"npu"/
      "privateuse1" 串 → accelerator；x.cpu()、device="cpu" → cpu。
      多种信号并存时按 cuda > accelerator > cpu 取唯一执行通道：
      显式 .cuda() 在 cpu 通道会直接失败（无 boot、无 CUDA 硬件），
      cuda 信号是硬性运行要求；accelerator 原生环境含 CPU，次之。
      一例多设备只来自明确的多作用范围声明（装饰器组合 / only_for 多
      通道），不来自信号混用。

二、执行通道策略（最后一步才决定，独立于证据）：无证据 / 文件缺失 /
   定位失败 / 解析失败 → 一律缺省 cuda（用户 2026-09-18 确认保留：boot
   环境下未标记函数的行为与基线采集一致，(tf, cuda) 键与 transfer-CUDA
   基线直接可比），同时把原因写入诊断，不静默、不阻塞、不输出"未知"设备。

三、诊断（make_device_resolver().diagnostics）：逐用例记录
   {path, node, devices, basis, detail}，basis ∈ decorator / instantiate /
   signal / default；由 `cli queue select` 附进 data/selection/<task>/
   test-plan.json 的 device_diagnostics 字段。

已知边界：不追踪跨函数调用与模块级变量（模块级 import 的绑定名例外：
函数内裸用即视为设备信号，见 _module_import_aliases）；不分析 PR diff
内容——选哪些函数测由 selection.py 的 P0/P1 按 diff 驱动完成，测试树取自
PR commit 的源码包、函数本身已含 PR 对测试的全部改动。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEVICE_CUDA = "cuda"
DEVICE_ACCELERATOR = "accelerator"
DEVICE_CPU = "cpu"

# 通道优先级：多信号并存时取先命中者（理由见模块 docstring 第一.3 条）。
_SIGNAL_PRIORITY = (DEVICE_CUDA, DEVICE_ACCELERATOR, DEVICE_CPU)

# 作用范围装饰器（末端名 → 设备）。skipCUDAIf = cuda 下跳过 → 非 cuda 作用域。
_SCOPE_DECORATORS = {
    "onlyCUDA": DEVICE_CUDA,
    "onlyCPU": DEVICE_CPU,
}
# torch.<attr> 模块引用 / `from torch import <name>` → 设备。
_MODULE_ATTRS = {
    "cuda": DEVICE_CUDA,
    "accelerator": DEVICE_ACCELERATOR,
    "npu": DEVICE_ACCELERATOR,
}
# x.<attr>() 方法调用 → 设备（torch.<attr> 除外，走 _MODULE_ATTRS）。
_METHOD_CALLS = {
    "cuda": DEVICE_CUDA,
    "cpu": DEVICE_CPU,
    "npu": DEVICE_ACCELERATOR,
}
# 设备参数关键字（device="cpu" / device_type='cuda' / devices=[...]）。
_DEVICE_KEYWORDS = {"device", "device_type", "devices"}
# 设备构造调用（torch.device("cuda") / device_type("cuda")）。
_DEVICE_CTORS = {"device", "device_type"}
# 设备串字面量 → 设备（cuda:N 由正则归一为 cuda）。
_STRING_DEVICES = {
    "cpu": DEVICE_CPU,
    "accelerator": DEVICE_ACCELERATOR,
    "npu": DEVICE_ACCELERATOR,
    "privateuse1": DEVICE_ACCELERATOR,
}
_CUDA_INDEX_RE = re.compile(r"^cuda(?::\d+)?$")

_INSTANTIATE_NAME = "instantiate_device_type_tests"
# instantiate 的"全部设备类型"在本地三通道中的展开（except_for 从中扣除）。
_ALL_CHANNELS = (DEVICE_CUDA, DEVICE_ACCELERATOR, DEVICE_CPU)


@dataclass
class Classification:
    """功能分类结果：设备集合（可多个）+ 证据类别 + 人读原因。"""

    devices: list[str]
    basis: str  # decorator / instantiate / signal / default
    detail: str

    def as_dict(self) -> dict:
        return {"devices": list(self.devices), "basis": self.basis, "detail": self.detail}


def _default(detail: str) -> Classification:
    """缺省通道：cuda（用户确认保留），原因入诊断。"""
    return Classification([DEVICE_CUDA], "default", f"{detail}；按基线可比性缺省 cuda")


def _string_device(text: str) -> str | None:
    if _CUDA_INDEX_RE.match(text):
        return DEVICE_CUDA
    return _STRING_DEVICES.get(text)


def _const_strings(value: ast.expr) -> list[str]:
    """取表达式里的字符串字面量：单值 device="cuda" 或容器 devices=["cpu", "cuda"]。"""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return [value.value]
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return [
            elt.value
            for elt in value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
    return []


def _walk_code(node: ast.AST) -> Iterator[ast.AST]:
    """深度优先遍历，跳过文档字符串（Expr(Constant str)）。

    注释不是 AST 节点天然不参与——这是 v2 相对 v1 正则扫描的关键修正。
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) \
                and isinstance(child.value.value, str):
            continue
        yield child
        yield from _walk_code(child)


def _decorator_name(dec: ast.expr) -> str | None:
    """装饰器末端名：@a.b.onlyCUDA / @skipCUDAIf(cond) → onlyCUDA / skipCUDAIf。"""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _scope_decorators(decorator_list: list[ast.expr]) -> tuple[set[str], list[str]]:
    """从装饰器列表提取 (作用范围设备集合, 命中的装饰器名)。"""
    devices: set[str] = set()
    names: list[str] = []
    for dec in decorator_list or []:
        name = _decorator_name(dec)
        if name in _SCOPE_DECORATORS:
            devices.add(_SCOPE_DECORATORS[name])
            names.append(name)
    return devices, names


def _module_import_aliases(tree: ast.Module) -> dict[str, str]:
    """模块级 import 的 {本地绑定名 → 设备}（torch.cuda / torch.npu 等族）。

    模块级 import 在函数节点之外，_collect_signals 看不到；函数内裸用这些
    绑定名（如 from torch.cuda import CUDAGraph 后的 CUDAGraph()）按此表
    记为设备信号。函数内 import 已由 _collect_signals 的 Import/ImportFrom
    分支直接覆盖，不依赖本表。import torch.cuda（无 asname）绑定名是
    torch，其 torch.cuda 属性访问走既有 Attribute 分支，不需入表。
    """
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch":
                for alias in node.names:
                    if alias.name in _MODULE_ATTRS:
                        aliases[alias.asname or alias.name] = _MODULE_ATTRS[alias.name]
            else:
                device = (
                    DEVICE_CUDA if module.startswith("torch.cuda")
                    else DEVICE_ACCELERATOR
                    if module.startswith(("torch.accelerator", "torch.npu"))
                    else None
                )
                if device:
                    for alias in node.names:
                        aliases[alias.asname or alias.name] = device
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) > 1 and parts[0] == "torch" and parts[1] in _MODULE_ATTRS:
                    if alias.asname:
                        aliases[alias.asname] = _MODULE_ATTRS[parts[1]]
    return aliases


def _collect_signals(
    roots: list[ast.AST], aliases: dict[str, str] | None = None,
) -> set[str]:
    """AST 遍历收集源码信号设备（API 调用 / 设备参数 / 设备导入）。

    aliases 来自模块级 import（_module_import_aliases）：函数内裸用这些
    绑定名的 Name 视为对应设备信号。
    """
    signals: set[str] = set()
    alias_map = aliases or {}
    for root in roots:
        for node in _walk_code(root):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "torch" \
                        and node.attr in _MODULE_ATTRS:
                    signals.add(_MODULE_ATTRS[node.attr])
            elif isinstance(node, ast.Name):
                # 模块级 import 绑定的设备名（如 CUDAGraph / cuda 别名）。
                if node.id in alias_map:
                    signals.add(alias_map[node.id])
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if func.attr in _METHOD_CALLS and not (
                        isinstance(func.value, ast.Name) and func.value.id == "torch"
                    ):
                        signals.add(_METHOD_CALLS[func.attr])
                    if func.attr in _DEVICE_CTORS:
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                dev = _string_device(arg.value)
                                if dev:
                                    signals.add(dev)
                elif isinstance(func, ast.Name) and func.id in _DEVICE_CTORS:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            dev = _string_device(arg.value)
                            if dev:
                                signals.add(dev)
            elif isinstance(node, ast.keyword):
                if node.arg in _DEVICE_KEYWORDS:
                    for text in _const_strings(node.value):
                        dev = _string_device(text)
                        if dev:
                            signals.add(dev)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("torch.cuda"):
                    signals.add(DEVICE_CUDA)
                elif module.startswith(("torch.accelerator", "torch.npu")):
                    signals.add(DEVICE_ACCELERATOR)
                elif module == "torch":
                    for alias in node.names:
                        if alias.name in _MODULE_ATTRS:
                            signals.add(_MODULE_ATTRS[alias.name])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) > 1 and parts[0] == "torch" and parts[1] in _MODULE_ATTRS:
                        signals.add(_MODULE_ATTRS[parts[1]])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # 裸设备串（parametrize 列表、device 比较值等）；文档字符串已被剔除。
                dev = _string_device(node.value)
                if dev:
                    signals.add(dev)
    return signals


def _kw_strings(call: ast.Call, name: str) -> list[str] | None:
    """关键字参数的字符串字面量（only_for="cuda" / ("cuda", "cpu")）。

    关键字存在但值不是字面量（引用变量）→ []；关键字不存在 → None。
    """
    for kw in call.keywords:
        if kw.arg == name:
            return _const_strings(kw.value)
    return None


def _instantiate_table(tree: ast.Module) -> dict[str, tuple[set[str] | None, str]]:
    """扫描 instantiate_device_type_tests 调用，逐类得出设备集合。

    返回 {类名: (设备集合或 None, 摘要)}；None 表示该类的实例化没有映射到
    本地通道的限制（无 only_for/except_for、引用变量、或只含 xla 等外部
    设备）——统一走 cuda 策略（_cuda_ 变体，基线可比）。
    """
    table: dict[str, tuple[set[str] | None, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_instantiate = (
            isinstance(func, ast.Name) and func.id == _INSTANTIATE_NAME
        ) or (
            isinstance(func, ast.Attribute) and func.attr == _INSTANTIATE_NAME
        )
        if not is_instantiate:
            continue
        class_args = [node.args[0]] if node.args else []
        if isinstance(class_args[0], (ast.Tuple, ast.List)):
            class_args = list(class_args[0].elts)
        names = [a.id for a in class_args if isinstance(a, ast.Name)]
        if not names:
            continue
        only = _kw_strings(node, "only_for")
        excl = _kw_strings(node, "except_for")
        if only:
            devices = set()
            for text in only:
                dev = _string_device(text)
                if dev:
                    devices.add(dev)
            how = f"only_for={only!r}"
        elif excl:
            excluded = set()
            for text in excl:
                dev = _string_device(text)
                if dev:
                    excluded.add(dev)
            devices = set(_ALL_CHANNELS) - excluded
            how = f"except_for={excl!r}"
        elif only is not None or excl is not None:
            devices = set()
            how = "only_for/except_for 非字符串字面量，未解析"
        else:
            devices = set()
            how = "无 only_for/except_for（全设备类型实例化）"
        for name in names:
            table.setdefault(name, (devices or None, how))
    return table


def classify_function(
    file_source: str, class_name: str | None, func_name: str,
) -> Classification:
    """按 (类名, 函数名) 从测试文件 AST 判定设备（规则见模块 docstring）。

    文件不可解析 / 类或函数定位不到 → 缺省 cuda + 原因（不排除、不报错）。
    """
    try:
        tree = ast.parse(file_source)
    except SyntaxError as error:
        return _default(f"文件解析失败（SyntaxError: {error.msg}）")

    cls_node: ast.ClassDef | None = None
    func_node = None
    if class_name:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                cls_node = node
                break
        if cls_node is None:
            return _default(f"类 {class_name} 定位失败")
        for node in cls_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == func_name:
                func_node = node
                break
    else:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == func_name:
                func_node = node
                break
    if func_node is None:
        where = f"{class_name}.{func_name}" if class_name else func_name
        return _default(f"函数 {where} 定位失败")

    def _sorted(devices: set[str]) -> list[str]:
        return sorted(devices, key=_SIGNAL_PRIORITY.index)

    # 1) 作用范围装饰器（方法级并入类级；可多设备 → 多通道输出）。
    scopes, names = _scope_decorators(func_node.decorator_list)
    conditional_names = [n for n in (_decorator_name(d) for d in func_node.decorator_list)
                         if n in {"skipCUDAIf", "skipCPUIf", "expectedFailCUDA", "expectedFailCPU"}]
    if cls_node is not None:
        cls_scopes, cls_names = _scope_decorators(cls_node.decorator_list)
        scopes |= cls_scopes
        names = cls_names + names
        conditional_names.extend(n for n in (_decorator_name(d) for d in cls_node.decorator_list)
                                 if n in {"skipCUDAIf", "skipCPUIf", "expectedFailCUDA", "expectedFailCPU"})
    if scopes:
        return Classification(_sorted(scopes), "decorator", f"作用范围装饰器 {names}")

    # 2) instantiate_device_type_tests 参数化（逐类，不按文件一刀切）。
    if cls_node is not None:
        table = _instantiate_table(tree)
        if cls_node.name in table:
            devices, how = table[cls_node.name]
            if devices:
                return Classification(
                    _sorted(devices), "instantiate",
                    f"instantiate_device_type_tests({cls_node.name}, {how})",
                )
            return Classification(
                [DEVICE_CUDA], "default",
                f"instantiate_device_type_tests({cls_node.name}, {how})；"
                "基线按 _cuda_ 变体采集，走 cuda 通道保持可比",
            )

    # 3) 源码信号（唯一通道，按可运行性优先级 cuda > accelerator > cpu）。
    #    模块级 import 的绑定名先建表：函数内裸用即信号（模块级 import 在
    #    函数节点之外，不扫全文件——未用到该名的函数不受影响）。
    roots: list[ast.AST] = [func_node]
    if cls_node is not None and cls_node.decorator_list:
        roots.extend(cls_node.decorator_list)
    signals = _collect_signals(roots, _module_import_aliases(tree))
    if signals:
        dev = next(d for d in _SIGNAL_PRIORITY if d in signals)
        others = sorted(signals - {dev}, key=_SIGNAL_PRIORITY.index)
        detail = f"源码信号 → {dev}"
        if others:
            detail += f"（并存 {others}，按可运行性优先级取 {dev}）"
        return Classification([dev], "decorator" if conditional_names else "signal",
                              (f"条件/预期结果装饰器 {conditional_names}；" if conditional_names else "") + detail)

    # 4) 无证据 → 执行通道策略缺省 cuda（用户确认保留）。
    if conditional_names:
        return Classification([DEVICE_CUDA], "decorator", f"条件/预期结果装饰器 {conditional_names}；未改变设备集合")
    return _default("无设备信号")


def _source_has_identity(source: str, class_name: str | None, func_name: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    body = tree.body
    if class_name is not None:
        cls = next((n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
        if cls is None:
            return False
        body = cls.body
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func_name for n in body)


def _variant_identity(source: str, class_name: str, method: str, declared=None) -> tuple[str, str, str | None] | None:
    """Map concrete names to real declarations, including pure parametrization."""
    from .runner import declared_methods_from_source
    try:
        if declared is None:
            declared = declared_methods_from_source(source)
    except (SyntaxError, ValueError):
        return None
    base_class, token = class_name, None
    if class_name not in declared:
        from .identifiers import class_variant_match
        owners = [name for name in declared if name and class_variant_match(name, class_name)]
        if not owners:
            return None
        # TestNNDeviceTypeCPU derives from TestNNDeviceType, not TestNN.
        # Resolve the longest real declaration before interpreting its suffix.
        base_class = max(owners, key=len)
        suffix = class_name[len(base_class):]
        if suffix.lower().startswith('devicetype'):
            suffix = suffix[10:]
        token = suffix.upper()
    methods = declared.get(base_class, set())
    if class_name == base_class and method in methods:
        return None
    candidates = [name for name in methods if method == name or method.startswith(name + "_")]
    if not candidates:
        return None
    return base_class, max(candidates, key=len), token


class DeviceResolver:
    """(path, test_node) → 设备列表；逐用例记录诊断（含缺省原因）。"""

    def __init__(self, tree_root: Path):
        self.tree_root = Path(tree_root)
        self.diagnostics: list[dict] = []
        self._sources: dict[str, str | None] = {}
        self._identities: dict[tuple[str, str], str] = {}

    def _source(self, path: str) -> str | None:
        if path not in self._sources:
            try:
                self._sources[path] = (self.tree_root / path).read_text(encoding="utf-8")
            except OSError:
                self._sources[path] = None
        return self._sources[path]

    def __call__(self, path: str, node: str) -> list[str]:
        source = self._source(path)
        variant_device = None
        cls = None
        method = node.split(".")[-1]
        if source is None:
            result = _default(f"文件缺失/不可读: {path}")
        else:
            parts = node.split(".")
            method = parts[-1]
            cls = parts[0] if len(parts) == 2 else None
            # instantiate_device_type_tests generates suffixes, but a real
            # source class/method may legitimately have the same suffix. Strip
            # only when the unsuffixed identity exists in the AST.
            original_cls, original_method = cls, method
            base_cls, base_method = cls, method
            variant_token = None
            if cls:
                from .runner import declared_class_methods
                variant = _variant_identity(source, cls, method, declared_class_methods(self.tree_root / path))
                if variant is not None:
                    base_cls, base_method, variant_token = variant
            if variant_token is not None:
                variant_device = {"CUDA": DEVICE_CUDA, "CPU": DEVICE_CPU,
                                  "PRIVATEUSE1": DEVICE_ACCELERATOR,
                                  "NPU": DEVICE_ACCELERATOR,
                                  "XPU": "xpu"}.get(variant_token, variant_token.lower())
            cls, method = base_cls, base_method
            result = classify_function(source, cls or None, method)
            if variant_device is not None:
                # A concrete collected variant is one execution channel; never
                # copy its runtime node to all class-level only_for devices.
                result = Classification([variant_device], "instantiate",
                                        f"收集到设备变体 {variant_device}；源码身份 {cls or method}")
        source_identity = f"{path}::{cls+'::' if cls else ''}{method}"
        self._identities[(path, node)] = source_identity
        self.diagnostics.append({"path": path, "node": node, **result.as_dict(),
                                 "source_identity": source_identity,
                                 "declared_device": result.devices,
                                 "execution_channel": result.devices[0] if len(result.devices) == 1 else "multiple"})
        return result.devices

    def source_test_function(self, path: str, node: str) -> str | None:
        return self._identities.get((path, node))


def make_device_resolver(tree_root: Path) -> DeviceResolver:
    """生成 (path, test_node) -> list[device] 解析器（供 plan_to_manifest）。

    test_node 形如 "Class.method" 或 "method"；path 相对 tree_root。
    失败一律缺省 cuda（用户 2026-09-18 确认保留），原因记入 .diagnostics。
    """
    return DeviceResolver(tree_root)
