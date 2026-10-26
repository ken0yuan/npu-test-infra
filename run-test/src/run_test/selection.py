"""从已合入 main 的 PR commit 产出 ≤15 个具体测试用例的选择器。

移植自用户指定的筛选器 ~/pytorch-commit-pre-check/scripts/（2026-09-17）：
``detect_test_impact.py`` 的 P0/P1 证据分级 + ``cpp_test_selection.py`` 的
C++ 候选与 ``finalize_plan`` 全局预算（15 上限）。选择语义与原实现一致：

- P0：被改测试文件的收窄节点 + 直接引用被改 Python 符号的测试方法。
  P0 条目按证据打分（窄化 300 > 直接符号引用 250 > 新文件 240 > 回退
  100）参与 finalize 的全局预算排序；共享 setup / 删改回退全节点时按
  方法行数限流（默认每文件 3 个最小方法），防止大文件雪崩挤掉其余
  候选——全局 15 预算同时裁 P0/P1，排序即优先级。
- P1：import 了被改模块但无直接符号引用的金丝雀（p1_score 打分，
  每源最多 1 个、方法行数上限），以及 C++ 官方套件/PR 提及候选。
- missing_coverage / ci_recommendations / analysis_gaps 全量记账。
- ``finalize_plan`` 把 Python 家族展开为具体用例后做全局 ≤max_tests
  预算：按 (优先级, -score, 路径) 排序 + 每文件轮转，前 limit 条入
  p0/p1，其余记 deferred_tests。

与原 skill 的差异（本仓环境适配）：

- 原 skill 面向开发者工作树（merge-base + staged + untracked）；本仓的
  PR 是已合入 main 的 commit，文件集与逐行改动取自本地 blob:none 裸
  镜像 ``git diff <sha>^ <sha>``（首父差异，merge commit 时即 PR 落地差异）。
- 变更后源码内容 ``git show <sha>:<path>``（按需懒取 blob）；测试树
  来自 OBS 源码包解压的 ``pytorch/test/``，不从 GitHub 拉任何测试资源。
- PR 文本 = commit subject + message 正文（本地镜像 ``git show -s
  --format=%B`` 可得全文，无需 gh/token），由调用方拼好经 ``pr_text``
  传入；未传时退回 subject。官方目录 + 路径关键词继续，缺失记入
  analysis_gaps。
- 测试服务器不构建 C++（无 build/bin），gtest 候选无法本地枚举，按原
  降级路径转入 ci_recommendations。
- Python 家族展开的解释器默认取 ``executor.test_python_env``；发现失败
  （环境问题）静态回退 family 节点（score 打 50 分折），记 analysis_gaps，
  不阻塞其余选择。
- ``plan_to_manifest`` 把结论桥接为队列清单（``test/x.py::Class::func``
  + device）；清单层只允许一级类名与固定 device 字面值，放不下的节点
  记入 manifest_skipped。
"""

from __future__ import annotations

import ast
import re
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .identifiers import (
    IdentifierError,
    class_variant_match,
    method_variant_match,
    validate_device,
    validate_test_function,
)


EXCLUDED_PREFIXES = ("third_party/", "build/", ".git/", "agent_space/")
TEST_PREFIXES = ("test/",)
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".cu", ".h", ".hpp", ".cuh"}


# ---------------------------------------------------------------------------
# git（裸镜像）
# ---------------------------------------------------------------------------


def git(git_dir: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=git_dir, text=True, capture_output=True, timeout=1800
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def _parse_name_status_details(output: str) -> list[tuple[str, str, str | None]]:
    result: list[tuple[str, str, str | None]] = []
    for raw in output.splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        status = parts[0]
        if status.startswith(("R", "C")) and len(parts) >= 3:
            result.append((status[0], parts[2], parts[1]))
        elif len(parts) >= 2:
            result.append((status[0], parts[1], None))
    return result


def _parse_name_status(output: str) -> list[tuple[str, str]]:
    return [(status, path) for status, path, _ in _parse_name_status_details(output)]


@dataclass
class ChangedFile:
    path: str
    status: str
    module: str = ""
    changed_lines: set[int] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)
    old_path: str | None = None
    old_module: str = ""
    old_symbols: set[str] = field(default_factory=set)


def collect_changed_files(git_dir: Path, sha: str, tree_root: Path | None = None) -> list[ChangedFile]:
    """首父差异 ``sha^ sha`` 的变更文件集；.py 文件附逐行改动与（可选）符号。"""
    parent = git(git_dir, "rev-parse", "--verify", "--quiet", f"{sha}^", check=False).strip()
    if not parent:
        raise RuntimeError(f"commit {sha} 无首父提交，无法计算首父差异")
    entries: dict[str, tuple[str, str | None]] = {}
    for status, path, old_path in _parse_name_status_details(git(git_dir, "diff", "--name-status", parent, sha)):
        entries[path] = (status, old_path)

    changed: list[ChangedFile] = []
    for path, (status, old_path) in sorted(entries.items()):
        if path.startswith(EXCLUDED_PREFIXES) and not (old_path and not old_path.startswith(EXCLUDED_PREFIXES)):
            continue
        item = ChangedFile(path=path, status=status, old_path=old_path)
        if path.endswith(".py"):
            item.module = path_to_module(path)
            item.changed_lines = changed_new_lines(git_dir, parent, sha, path, status)
        if old_path and old_path.endswith(".py"):
            item.old_module = path_to_module(old_path)
            old_text = git(git_dir, "show", f"{parent}:{old_path}", check=False)
            if old_text:
                item.old_symbols = changed_python_symbols(old_text, old_path, set(range(1, len(old_text.splitlines()) + 1)))
            if status == "D":
                item.module = item.old_module
                item.symbols = set(item.old_symbols)
        elif status == "D" and path.endswith(".py"):
            old_text = git(git_dir, "show", f"{parent}:{path}", check=False)
            if old_text:
                item.module = path_to_module(path)
                item.symbols = changed_python_symbols(old_text, path, set(range(1, len(old_text.splitlines()) + 1)))
        if status in {"M", "R"} and path.endswith(".py"):
            new_text = git(git_dir, "show", f"{sha}:{path}", check=False)
            old_name = old_path or path
            old_text = git(git_dir, "show", f"{parent}:{old_name}", check=False)
            if new_text:
                new_all = changed_python_symbols(new_text, path, set(range(1, len(new_text.splitlines()) + 1)))
                item.symbols |= changed_python_symbols(new_text, path, item.changed_lines)
            else:
                new_all = set()
            if old_text:
                old_all = changed_python_symbols(old_text, old_name, set(range(1, len(old_text.splitlines()) + 1)))
                if status == "M":
                    deleted = old_all - new_all
                    item.old_symbols = deleted
                    item.symbols |= deleted
                elif not item.old_symbols:
                    item.old_symbols = old_all
        changed.append(item)
    if tree_root is not None:
        attach_symbols(tree_root, changed)
    return changed


def path_to_module(path: str) -> str:
    if not path.endswith(".py"):
        return ""
    module = path[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return module


def changed_new_lines(git_dir: Path, parent: str, sha: str, path: str, status: str) -> set[int]:
    del status  # 新增文件在首父差异中天然以全量 hunk 呈现，无需特判
    diff = git(git_dir, "diff", "--unified=0", "--no-ext-diff", parent, sha, "--", path)
    lines: set[int] = set()
    for line in diff.splitlines():
        match = HUNK_RE.match(line)
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        # 纯删除锚定到两个幸存邻居。
        lines.update(range(start, start + count) if count else (max(start, 1), start + 1))
    return lines


# ---------------------------------------------------------------------------
# Python 符号提取（内容读自 OBS 源码树；git show 仅用于 diff 本身）
# ---------------------------------------------------------------------------


def read_tree_file(tree_root: Path, rel_path: str) -> str | None:
    try:
        return (tree_root / rel_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def attach_symbols(tree_root: Path, changed: list[ChangedFile]) -> None:
    """为非测试 .py 变更文件补 changed_symbols（内容 = 该 commit 的源码树）。"""
    for item in changed:
        if not item.path.endswith(".py") or item.path.startswith(TEST_PREFIXES):
            continue
        text = read_tree_file(tree_root, item.path)
        if text is not None:
            item.symbols |= changed_python_symbols(text, item.path, item.changed_lines)


def _overlaps(node: ast.AST, changed_lines: set[int]) -> bool:
    start = getattr(node, "lineno", 0)
    start = min([start] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = getattr(node, "end_lineno", start)
    return any(start <= line <= end for line in changed_lines)


def _parse_python(text: str, filename: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(text, filename=filename)


def changed_python_symbols(text: str, filename: str, changed_lines: set[int]) -> set[str]:
    try:
        tree = _parse_python(text, filename)
    except SyntaxError:
        return set()

    symbols: set[str] = set()

    def visit_body(body: list[ast.stmt], *, module_level: bool = False) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _overlaps(node, changed_lines):
                    symbols.add(node.name)
                continue
            if isinstance(node, ast.ClassDef):
                if node.lineno in changed_lines:
                    symbols.add(node.name)
                visit_body(node.body)
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.lineno in changed_lines:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        symbols.add(target.id)
                continue
            if module_level and isinstance(node, ast.ImportFrom) and node.lineno in changed_lines:
                # 大写内部 import 通常是兼容性 re-export。
                if (node.module or "").startswith("torch."):
                    symbols.update(
                        alias.asname or alias.name
                        for alias in node.names
                        if (alias.asname or alias.name)[:1].isupper()
                    )

    visit_body(tree.body, module_level=True)
    return symbols


# ---------------------------------------------------------------------------
# 测试树扫描与直接引用匹配
# ---------------------------------------------------------------------------


@dataclass
class TestInfo:
    path: str
    imported_modules: set[str] = field(default_factory=set)
    imported_names: dict[str, tuple[str, str]] = field(default_factory=dict)
    module_aliases: dict[str, str] = field(default_factory=dict)
    direct_refs: list[dict[str, str]] = field(default_factory=list)
    line_count: int = 0
    test_nodes: dict[str, tuple[int, int]] = field(default_factory=dict)
    module_refs: dict[str, set[str]] = field(default_factory=dict)


class TestVisitor(ast.NodeVisitor):
    def __init__(self, changed_modules: dict[str, set[str]]) -> None:
        self.changed_modules = changed_modules
        self.info = TestInfo(path="")
        self.test_stack: list[str] = []
        self.class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            module = alias.name
            local = alias.asname or module.split(".")[0]
            if module in self.changed_modules:
                self.info.imported_modules.add(module)
                self.info.module_aliases[local] = module
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        if module in self.changed_modules:
            self.info.imported_modules.add(module)
            for alias in node.names:
                local = alias.asname or alias.name
                self.info.imported_names[local] = (module, alias.name)
        for alias in node.names:
            possible_module = f"{module}.{alias.name}" if module else alias.name
            if possible_module in self.changed_modules:
                local = alias.asname or alias.name
                self.info.imported_modules.add(possible_module)
                self.info.module_aliases[local] = possible_module
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        is_test = node.name.startswith("test")
        if is_test:
            name = ".".join([*self.class_stack, node.name])
            self.test_stack.append(name)
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            self.info.test_nodes[name] = (start, node.end_lineno or node.lineno)
        self.generic_visit(node)
        if is_test:
            self.test_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Name(self, node: ast.Name) -> Any:
        imported = self.info.imported_names.get(node.id)
        if imported:
            module, symbol = imported
            self._add_module_ref(module)
            if symbol in self.changed_modules.get(module, set()):
                self._add_ref(module, symbol, node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        dotted = ast.unparse(node)
        for module in self.changed_modules:
            if dotted.startswith(module + "."):
                self._add_module_ref(module)
                symbol = dotted[len(module) + 1 :].split(".")[0]
                if symbol in self.changed_modules[module]:
                    self._add_ref(module, symbol, node.lineno)
        if isinstance(node.value, ast.Name):
            module = self.info.module_aliases.get(node.value.id)
            if module:
                self._add_module_ref(module)
            if module and node.attr in self.changed_modules.get(module, set()):
                self._add_ref(module, node.attr, node.lineno)
        self.generic_visit(node)

    def _add_module_ref(self, module: str) -> None:
        if self.test_stack:
            self.info.module_refs.setdefault(module, set()).add(self.test_stack[-1])

    def _add_ref(self, module: str, symbol: str, line: int) -> None:
        if not self.test_stack:
            return
        node = self.test_stack[-1]
        ref = {"module": module, "symbol": symbol, "test_node": node, "line": str(line)}
        if ref not in self.info.direct_refs:
            self.info.direct_refs.append(ref)


def is_runnable_test_path(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py")


def analyze_tests(tree_root: Path, changed_sources: list[ChangedFile]) -> list[TestInfo]:
    changed_modules: dict[str, set[str]] = {}
    for item in changed_sources:
        if item.module:
            changed_modules.setdefault(item.module, set()).update(item.symbols)
        if item.old_module:
            changed_modules.setdefault(item.old_module, set()).update(item.old_symbols)
    tests: list[TestInfo] = []
    test_root = tree_root / "test"
    if not test_root.exists():
        return tests

    for path in sorted(test_root.rglob("*.py")):
        rel = path.relative_to(tree_root).as_posix()
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        if not is_runnable_test_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = _parse_python(text, rel)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        visitor = TestVisitor(changed_modules)
        visitor.info.path = rel
        visitor.info.line_count = len(text.splitlines())
        visitor.visit(tree)
        tests.append(visitor.info)
    return tests


def changed_test_nodes(
    tree_root: Path, item: ChangedFile, *, max_fallback_nodes: int = 3,
) -> tuple[list[str], str, int]:
    """收窄被改测试文件的可运行节点。

    返回 (nodes, reason, total)：total > len(nodes) 表示回退全节点后被
    限流截断（按方法行数最小优先保留 max_fallback_nodes 个）——共享
    setup / 类装饰器 / 删改无法安全收窄时防止大文件雪崩挤掉全局预算。
    """
    text = read_tree_file(tree_root, item.path)
    if text is None:
        return [], "test file could not be parsed", 0
    try:
        tree = _parse_python(text, item.path)
    except SyntaxError:
        return [], "test file could not be parsed", 0
    visitor = TestVisitor({})
    visitor.visit(tree)
    selected = []
    covered: set[int] = set()
    for name, (start, end) in visitor.info.test_nodes.items():
        hits = item.changed_lines.intersection(range(start, end + 1))
        if hits:
            selected.append(name)
            covered.update(hits)
    if item.status == "A":
        return sorted(visitor.info.test_nodes), "new test file", 0
    if item.changed_lines - covered:
        all_nodes = sorted(visitor.info.test_nodes)
        reason = "shared or deletion changes; all statically identified test methods"
        # Directly changed methods are mandatory evidence. The size limit only
        # selects additional canaries for shared/deletion changes.
        extras = [n for n in all_nodes if n not in selected]
        extras.sort(key=lambda n: (visitor.info.test_nodes[n][1] - visitor.info.test_nodes[n][0] + 1, n))
        keep = selected + extras[:max(0, max_fallback_nodes - len(selected))]
        return sorted(set(keep)), reason, len(all_nodes) if len(keep) < len(all_nodes) else 0
    return sorted(selected), "changed test methods", 0


# ---------------------------------------------------------------------------
# P1 打分与报告组装
# ---------------------------------------------------------------------------


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", value.replace("_", " "))
        if len(token) > 2 and token.lower() not in {"torch", "test", "python"}
    }


def p1_score(source: ChangedFile, test: TestInfo) -> tuple[int, int, str]:
    source_stem = Path(source.path).stem.lower()
    test_stem = Path(test.path).stem.lower()
    source_tokens = _tokens(source.module)
    test_tokens = _tokens(test_stem)
    score = 0
    if source_stem in test_stem:
        score += 10
    score += 3 * sum(token in test_stem for token in source_tokens)
    score += 2 * len(source_tokens & test_tokens)
    if source.module.startswith("torch._inductor") and test.path.startswith("test/inductor/"):
        score += 4
    if source.module.startswith("torch._dynamo") and test.path.startswith("test/dynamo/"):
        score += 4
    imported_specific = sum(1 for module, _ in test.imported_names.values() if module == source.module)
    score += min(imported_specific, 3)
    return (-score, test.line_count, test.path)


# P0 证据分级（finalize 全局预算按 score 排序；值即优先级）：
# 窄化收窄 > 直接符号引用 > 新测试文件 > 回退全节点。回退条目另加
# fallback 标记并在 changed_test_nodes 内限流，防止大文件雪崩。
P0_SCORE_NARROWED = 300
P0_SCORE_DIRECT_REF = 250
P0_SCORE_NEW_FILE = 240
P0_SCORE_FALLBACK = 100
_P0_REASON_SCORES = {
    "changed test methods": P0_SCORE_NARROWED,
    "new test file": P0_SCORE_NEW_FILE,
    "shared or deletion changes; all statically identified test methods": P0_SCORE_FALLBACK,
}


def build_report(
    sha: str,
    merge_base: str,
    changed: list[ChangedFile],
    tests: list[TestInfo],
    tree_root: Path,
    max_p1_per_source: int,
    max_p1_lines: int,
    min_p1_score: int,
    max_fallback_nodes: int = 3,
) -> dict[str, Any]:
    changed_tests = {
        item.path
        for item in changed
        if item.path.startswith(TEST_PREFIXES)
        and item.path.endswith(".py")
        and is_runnable_test_path(Path(item.path))
    }
    changed_sources = [item for item in changed if item.module and not item.path.startswith(TEST_PREFIXES)]

    p0_by_path: dict[str, dict[str, Any]] = {}
    analysis_gaps: list[dict[str, str]] = []
    for path in sorted(changed_tests):
        item = next(item for item in changed if item.path == path)
        nodes, reason, total = changed_test_nodes(
            tree_root, item, max_fallback_nodes=max_fallback_nodes
        )
        if not nodes:
            analysis_gaps.append({"path": path, "reason": reason + "; no runnable test method resolved"})
            continue
        entry = {
            "path": path,
            "test_nodes": nodes,
            "evidence": [{"type": "changed_test_file", "reason": reason}],
            "score": _P0_REASON_SCORES.get(reason, P0_SCORE_FALLBACK),
        }
        if total:
            entry["fallback"] = True
            analysis_gaps.append(
                {
                    "path": path,
                    "reason": (
                        f"fallback nodes throttled: kept the {len(nodes)} smallest "
                        f"methods of {total}; the rest are not in the plan"
                    ),
                }
            )
        p0_by_path[path] = entry

    direct_coverage: dict[tuple[str, str], list[dict[str, str]]] = {}
    for test in tests:
        for ref in test.direct_refs:
            key = (ref["module"], ref["symbol"])
            direct_coverage.setdefault(key, []).append({"path": test.path, **ref})
            entry = p0_by_path.setdefault(
                test.path,
                {"path": test.path, "test_nodes": [], "evidence": [], "score": 0},
            )
            if ref["test_node"] not in entry["test_nodes"]:
                entry["test_nodes"].append(ref["test_node"])
            # 直接引用证据取 max：与被改测试文件证据叠加时保留更高分。
            entry["score"] = max(entry.get("score", 0), P0_SCORE_DIRECT_REF)
            evidence = {
                "type": "direct_changed_symbol_reference",
                "source_module": ref["module"],
                "symbol": ref["symbol"],
                "line": int(ref["line"]),
            }
            if evidence not in entry["evidence"]:
                entry["evidence"].append(evidence)

    p1_by_path: dict[str, dict[str, Any]] = {}
    ci_recommendations: list[dict[str, Any]] = []
    for source in changed_sources:
        imported_candidates = [
            test
            for test in tests
            if source.module in test.imported_modules
            and test.module_refs.get(source.module, set()) - set(p0_by_path.get(test.path, {}).get("test_nodes", []))
        ]
        all_candidates = [
            test for test in imported_candidates if -p1_score(source, test)[0] >= min_p1_score
        ]

        def method_size(test: TestInfo) -> int:
            nodes = test.module_refs[source.module] - set(p0_by_path.get(test.path, {}).get("test_nodes", []))
            return min(test.test_nodes[n][1] - test.test_nodes[n][0] + 1 for n in nodes)

        oversized = [test for test in all_candidates if method_size(test) > max_p1_lines]
        candidates = [test for test in all_candidates if method_size(test) <= max_p1_lines]
        candidates.sort(key=lambda test: p1_score(source, test))
        for test in candidates[:max_p1_per_source]:
            entry = p1_by_path.setdefault(
                test.path,
                {"path": test.path, "test_nodes": [], "evidence": []},
            )
            nodes = test.module_refs[source.module] - set(p0_by_path.get(test.path, {}).get("test_nodes", []))
            node = min(nodes, key=lambda n: (test.test_nodes[n][1] - test.test_nodes[n][0], n))
            if node not in entry["test_nodes"]:
                entry["test_nodes"].append(node)
            evidence = {
                "type": "direct_import_canary",
                "source_module": source.module,
                "score": -p1_score(source, test)[0],
                "reason": "selected as a minimal import/initialization compatibility canary",
            }
            if evidence not in entry["evidence"]:
                entry["evidence"].append(evidence)
        if not candidates and oversized:
            oversized.sort(key=lambda test: p1_score(source, test))
            canary = oversized[0]
            ci_recommendations.append(
                {
                    "path": canary.path,
                    "source_module": source.module,
                    "reason": f"smallest referenced test method exceeds local P1 limit ({method_size(canary)} lines)",
                }
            )

    missing: list[dict[str, Any]] = []
    for source in changed_sources:
        if not source.symbols:
            missing.append(
                {
                    "source": source.path,
                    "symbol": "<module-level-change>",
                    "reason": "no changed Python symbol could be resolved to a direct test reference",
                }
            )
            continue
        for symbol in sorted(source.symbols):
            if not direct_coverage.get((source.module, symbol)):
                missing.append(
                    {
                        "source": source.path,
                        "symbol": symbol,
                        "reason": "no existing test directly references this changed symbol",
                    }
                )

    for item in changed:
        if Path(item.path).suffix in CPP_SUFFIXES:
            missing.append({"source": item.path, "symbol": "<native-change>", "reason": "official suite/path/PR selection does not prove coverage of changed native lines"})
        elif item.path.startswith(TEST_PREFIXES):
            if item.status == "D":
                analysis_gaps.append({"path": item.path, "reason": "deleted test file has no surviving runtime node; inspect parent commit callers"})
            elif item.old_path:
                analysis_gaps.append({"path": item.old_path, "reason": f"test path renamed to {item.path}; parent path is not executed directly"})
            continue
        elif not item.path.endswith(".py"):
            analysis_gaps.append(
                {
                    "path": item.path,
                    "reason": "automatic source-to-test mapping is not implemented for this file type",
                }
            )
        elif item.module and not item.symbols:
            analysis_gaps.append(
                {
                    "path": item.path,
                    "reason": "the diff could not be resolved to changed Python symbols",
                }
            )

    for item in changed_sources:
        if item.status == "D":
            analysis_gaps.append({"path": item.path, "reason": "deleted source analyzed from parent commit; direct behavior coverage is not proven"})
        elif item.old_path:
            analysis_gaps.append({"path": item.old_path, "reason": f"renamed source tracked as {item.path}; parent symbols retained for reference analysis"})

    return {
        "pr_commit": sha,
        "base_ref": "first-parent",
        "merge_base": merge_base,
        "changed_files": [
            {
                "path": item.path,
                "status": item.status,
                "module": item.module,
                "changed_symbols": sorted(item.symbols),
                "old_path": item.old_path,
                "old_module": item.old_module,
                "old_symbols": sorted(item.old_symbols),
            }
            for item in changed
        ],
        "p0": sorted(p0_by_path.values(), key=lambda item: item["path"]),
        "p1": sorted(p1_by_path.values(), key=lambda item: item["path"]),
        "missing_coverage": missing,
        "ci_recommendations": ci_recommendations,
        "analysis_gaps": analysis_gaps,
        "notes": [
            "P2 is intentionally omitted from the local plan.",
            "P1 is capped per changed source and is a canary set, not every importer.",
            "Static references do not prove behavioral assertions; review missing_coverage manually.",
            "Diff and symbols come from the local main mirror; the test tree comes from the OBS source package of the same commit.",
        ],
    }


# ---------------------------------------------------------------------------
# C++ 官方套件（目录来源不变；本机无 build/bin → 全部降级为 CI 建议）
# ---------------------------------------------------------------------------


def cpp_test_catalog(tree_root: Path) -> tuple[dict[str, dict], list[dict]]:
    catalog: dict[str, dict] = {}
    gaps: list[dict] = []
    for relative, baseline in (("aten/tools/run_tests.sh", True), (".ci/pytorch/test.sh", False)):
        path = tree_root / relative
        try:
            lines = path.read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            gaps.append({"path": relative, "reason": f"cannot read official C++ test list: {error}"})
            continue
        for line in lines:
            tokens = _shlex_tokens(line)
            names = []
            if "--cpp" in tokens and "test/run_test.py" in tokens:
                names = [token for token in tokens if re.fullmatch(r"cpp/[A-Za-z0-9_/-]+", token)]
            elif baseline and tokens and tokens[0] == "run_if_exists" and len(tokens) == 2:
                # run_if_exists 依赖本地 build/bin；测试服务器不构建，记 gap。
                gaps.append({"path": "cpp/" + tokens[1], "reason": "run_if_exists requires a local build/bin; unavailable on the test server"})
                continue
            for name in names:
                if ".." in Path(name).parts:
                    continue
                entry = catalog.setdefault(name, {"baseline": baseline, "source": relative})
                entry["baseline"] = entry["baseline"] or baseline
    if not any(item["baseline"] for item in catalog.values()):
        gaps.append({"path": "aten/tools/run_tests.sh", "reason": "no literal baseline C++ test targets were resolved"})
    return catalog, gaps


def _shlex_tokens(line: str) -> list[str]:
    import shlex

    try:
        return shlex.split(line, comments=True)
    except ValueError:
        return []


def path_keywords(value: str) -> set[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    ignored = {"aten", "torch", "test", "tests", "cpp", "src", "native", "build", "bin", "internal", "cu", "hpp"}
    return {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", value)
            if len(word) > 2 and word.lower() not in ignored}


def mentioned(text: str, name: str) -> bool:
    return bool(re.search(r"(?<![\w/])" + re.escape(name) + r"(?![\w/])", text))


def discover_cases(tree_root: Path, command: list[str], timeout: int, cpp: bool,
                  *, env: dict[str, str] | None = None, cwd: Path | None = None) -> list[str]:
    try:
        proc = subprocess.run(command, cwd=cwd or tree_root, env=env, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"test discovery failed: {error}") from error
    if proc.returncode:
        raise RuntimeError(f"test discovery exited {proc.returncode}: {(proc.stderr or proc.stdout)[-1500:]}")
    cases = []
    suite = ""
    for raw in proc.stdout.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if cpp:
            if re.fullmatch(r"[\w/]+\.", line):
                suite = line
            elif suite and re.fullmatch(r"\s+[\w/]+", line):
                case = suite + line.strip()
                if not any(part.startswith("DISABLED_") for part in re.split(r"[./]", case)):
                    cases.append(case)
            else:
                suite = ""
        elif re.fullmatch(r"\w+\.test\w*", line):
            cases.append(line)
    if not cases:
        raise RuntimeError("test discovery returned no runnable concrete cases")
    return sorted(set(cases))


def add_cpp_candidates(tree_root: Path, report: dict, changed_paths: list[str], tests: list[TestInfo],
                       pr_text: str, timeout: int) -> None:
    catalog, gaps = cpp_test_catalog(tree_root)
    report["analysis_gaps"].extend(gaps)
    source_keywords = set().union(*(path_keywords(path) for path in changed_paths))
    for name, metadata in catalog.items():
        path_score = len(source_keywords & path_keywords(name))
        pr_match = mentioned(pr_text, name) or mentioned(pr_text, "build/bin/" + name[4:])
        if not (metadata["baseline"] or path_score or pr_match):
            continue
        binary = "build/bin/" + name[4:]
        try:
            nodes = discover_cases(tree_root, [str(tree_root / binary), "--gtest_list_tests"], timeout, cpp=True)
        except RuntimeError as error:
            report["analysis_gaps"].append({"path": name, "reason": str(error)})
            report["ci_recommendations"].append({"path": name, "reason": "official C++ candidate could not be enumerated locally"})
            continue
        for node in nodes:
            case_mention = mentioned(pr_text, node)
            score = (200 if case_mention else 100 if pr_match else 0) + 10 * path_score + int(metadata["baseline"])
            evidence = [{"type": "official_cpp_test_list", "source": metadata["source"], "baseline": metadata["baseline"]}]
            if path_score:
                evidence.append({"type": "similar_path", "keywords": sorted(source_keywords & path_keywords(name))})
            if case_mention or pr_match:
                evidence.append({"type": "mentioned_in_pr", "match": node if case_mention else name})
            report["p1"].append({"path": name, "binary": binary, "kind": "gtest", "test_nodes": [node], "score": score, "evidence": evidence})

    for test in tests:
        path_score = len(source_keywords & path_keywords(test.path))
        file_mention = mentioned(pr_text, test.path) or mentioned(pr_text, Path(test.path).name)
        for node in test.test_nodes:
            node_mention = mentioned(pr_text, node) or mentioned(pr_text, node.replace(".", "::"))
            if not (path_score or file_mention or node_mention):
                continue
            evidence = []
            if path_score:
                evidence.append({"type": "similar_path", "keywords": sorted(source_keywords & path_keywords(test.path))})
            if file_mention or node_mention:
                evidence.append({"type": "mentioned_in_pr", "match": node if node_mention else test.path})
            report["p1"].append({"path": test.path, "test_nodes": [node], "score": (200 if node_mention else 100 if file_mention else 0) + 10 * path_score, "evidence": evidence})


# ---------------------------------------------------------------------------
# finalize：家族展开 + 全局预算（15 上限，超出记 deferred_tests）
# ---------------------------------------------------------------------------


def finalize_plan(tree_root: Path, report: dict, limit: int, timeout: int, pr_text: str = "",
                  interpreter: str | None = None, discovery_env: dict[str, str] | None = None,
                  discovery_cwd: Path | None = None) -> None:
    """展开 Python 家族为具体用例，再应用一个全局预算。

    发现进程失败时保留静态源码身份并降低优先级；它不是可直接执行的 nodeid。
    执行端必须在任务的源码/解释器/设备环境中重新收集并解析完整实例集。
    发现成功后对整组实例统一按声明归属，避免只命中带设备 token 的一部分。
    """
    import sys as _sys

    python = interpreter or _sys.executable
    from .runner import declared_class_methods, declared_node_match
    cache: dict[str, list[str] | str] = {}
    declarations = {}
    ranked: dict[tuple[str, str], tuple[str, dict]] = {}
    unresolved = []
    for priority in ("p0", "p1"):
        for entry in report[priority]:
            path = entry["path"]
            nodes_with_origin: list[tuple[str, bool]] = []  # (node, 静态回退?)
            if entry.get("kind") == "gtest":
                nodes_with_origin = [(n, False) for n in entry["test_nodes"]]
            else:
                if path not in cache:
                    try:
                        if discovery_env is None and discovery_cwd is None:
                            cache[path] = discover_cases(tree_root, [python, path, "--discover-tests"], timeout, cpp=False)
                        else:
                            cache[path] = discover_cases(tree_root, [python, path, "--discover-tests"], timeout, cpp=False, env=discovery_env, cwd=discovery_cwd)
                    except RuntimeError as error:
                        cache[path] = str(error)
                        report["analysis_gaps"].append({"path": path, "reason": str(error)})
                discovered = cache[path]
                # setdefault(path, parse(...)) evaluates parse for EVERY case.
                # Native path matching can emit thousands of entries per file.
                # Failed discovery uses static identities and needs no parsing.
                if not isinstance(discovered, str) and path not in declarations:
                    declarations[path] = declared_class_methods(tree_root / path)
                declared = declarations.get(path, {})
                for family in entry["test_nodes"]:
                    owner, _, method = family.rpartition(".")
                    # P1-2（2026-09-20）：精确变体匹配取代裸 startswith——
                    # TestFoo 不再撞 TestFooBar，test_sub 不再撞 test_sub_scalar。
                    matches = [] if isinstance(discovered, str) else [node for node in discovered
                        if declared_node_match(declared, owner, method,
                                               node.rpartition(".")[0], node.rpartition(".")[2])]
                    if not matches:
                        if isinstance(discovered, str):
                            # 发现失败：静态回退（family 即 nodeid），不计
                            # unresolved——候选没有丢，只是展开降级。
                            nodes_with_origin.append((family, True))
                            continue
                        unresolved.append({"path": path, "test_node": family, "priority": priority.upper(), "reason": "could not resolve method family to concrete cases"})
                    nodes_with_origin.extend((m, False) for m in matches)
            for node, is_static in nodes_with_origin:
                key = (path, node)
                item = {**entry, "kind": entry.get("kind", "python"), "test_nodes": [node], "evidence": list(entry["evidence"])}
                if is_static:
                    item["score"] = max(item.get("score", 0) - 50, 0)
                    item["evidence"].append({"type": "static_fallback_after_discovery_failure"})
                if mentioned(pr_text, node) or mentioned(pr_text, node.replace(".", "::")):
                    item["score"] = max(item.get("score", 0), 200)
                    item["evidence"].append({"type": "mentioned_in_pr", "match": node})
                if key not in ranked:
                    ranked[key] = (priority, item)
                else:
                    existing = ranked[key][1]
                    existing["score"] = max(existing.get("score", 0), item.get("score", 0))
                    for evidence in item["evidence"]:
                        if evidence not in existing["evidence"]:
                            existing["evidence"].append(evidence)
    # 同分文件轮转，防止一个大文件吃光全部槽位。
    ordered = sorted(ranked.values(), key=lambda pair: (pair[0], -pair[1].get("score", 0), pair[1]["path"], pair[1]["test_nodes"][0]))
    rounds: dict[tuple[str, int, str], int] = {}
    with_rounds = []
    for priority, item in ordered:
        key = (priority, -item.get("score", 0), item["path"])
        round_number = rounds.get(key, 0)
        rounds[key] = round_number + 1
        with_rounds.append((priority, -item.get("score", 0), round_number, item["path"], item["test_nodes"][0], item))
    with_rounds.sort(key=lambda row: row[:5])
    report["p0"], report["p1"] = [], []
    report["deferred_tests"] = []
    for index, (priority, _, _, _, _, item) in enumerate(with_rounds):
        if index < limit:
            report[priority].append(item)
        else:
            report["deferred_tests"].append({**item, "priority": priority.upper(), "reason": f"exceeds the global {limit}-case budget"})
    report["unresolved_tests"] = unresolved
    report["selection_granularity"] = "case"
    report["max_tests"] = limit
    report["candidate_count"] = len(ranked)
    report["selected_count"] = min(len(ranked), limit)
    report["notes"].append("The budget counts concrete cases; deferred and unresolved tests are not covered by this local run.")
    report["discovery_environment"] = {
        "interpreter": python,
        "cwd": str(discovery_cwd or tree_root),
        "boot_injected": bool(discovery_env and discovery_env.get("PYTHONPATH")),
        "device_filter": (discovery_env or {}).get("PYTORCH_TESTING_DEVICE_ONLY_FOR", "none"),
        "verified_against_execution": False,
        "verification_note": "selection-time discovery is advisory; Executor re-discovers in the installed runtime",
    }


# ---------------------------------------------------------------------------
# 组装入口与清单桥接
# ---------------------------------------------------------------------------


def select_for_commit(
    git_dir: Path,
    sha: str,
    subject: str,
    tree_root: Path,
    *,
    pr_text: str = "",
    max_tests: int = 15,
    max_p1_per_source: int = 1,
    max_p1_lines: int = 5000,
    min_p1_score: int = 6,
    max_fallback_nodes: int = 3,
    collect_timeout: int = 120,
    interpreter: str | None = None,
    discovery_env: dict[str, str] | None = None,
    discovery_cwd: Path | None = None,
) -> dict[str, Any]:
    """镜像 diff + OBS 源码树 → finalize 后的完整计划（与原 plan JSON 兼容）。

    pr_text：完整 PR 文本信号（subject + commit message 正文，镜像 ``git
    show -s --format=%B`` 可得，无需 gh/token）；缺省退回 subject。
    mentioned_in_pr 的 200 分加权依赖它——只给 subject 时几乎不触发。
    """
    merge_base = git(git_dir, "rev-parse", f"{sha}^").strip()
    changed = collect_changed_files(git_dir, sha, tree_root)
    sources = [item for item in changed if not item.path.startswith(TEST_PREFIXES)]
    tests = analyze_tests(tree_root, sources)
    report = build_report(
        sha,
        merge_base,
        changed,
        tests,
        tree_root,
        max(max_p1_per_source, 0),
        max(max_p1_lines, 0),
        max(min_p1_score, 0),
        max_fallback_nodes=max(max_fallback_nodes, 1),
    )
    text = pr_text or subject
    native = [item.path for item in changed if Path(item.path).suffix in CPP_SUFFIXES]
    if native:
        # PR 正文来自镜像 %B（调用方拼好）；缺失降级与原实现一致。
        add_cpp_candidates(tree_root, report, native, tests, text, max(collect_timeout, 1))
    finalize_plan(tree_root, report, max(max_tests, 1), max(collect_timeout, 1), text, interpreter, discovery_env, discovery_cwd)
    return report


def plan_to_manifest(
    report: dict,
    device: str = "cpu",
    device_of: Callable[[str, str], "str | list[str]"] | None = None,
) -> list[dict]:
    """finalize 后的 p0/p1 Python 用例 → 队列清单；放不下的节点记 manifest_skipped。

    device_of(path, test_node) 提供时逐用例判定 device（设备判定脚本），
    返回单设备或设备列表——列表即一例多通道，清单按设备逐条展开；
    否则整个清单用固定 device。
    """
    entries: list[dict] = []
    skipped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for priority in ("p0", "p1"):
        for item in report[priority]:
            record = {"path": item["path"], "node": item["test_nodes"][0], "priority": priority.upper()}
            if item.get("kind", "python") == "gtest":
                # 测试服务器不构建 C++，gtest 用例只留在 ci_recommendations。
                skipped.append({**record, "reason": "gtest cases are not runnable on the test server; see ci_recommendations"})
                continue
            parts = item["test_nodes"][0].split(".")
            method, classes = parts[-1], parts[:-1]
            if len(classes) > 1:
                skipped.append({**record, "reason": "nested-class node exceeds the one-class manifest schema"})
                continue
            runtime_tf = "::".join([item["path"], *classes, method])
            tf = runtime_tf
            try:
                if device_of is not None:
                    got = device_of(item["path"], item["test_nodes"][0])
                    devices = [got] if isinstance(got, str) else list(got)
                    resolver_identity = getattr(device_of, "source_test_function", None)
                    if resolver_identity is not None:
                        mapped = resolver_identity(item["path"], item["test_nodes"][0])
                        if mapped:
                            tf = mapped
                else:
                    devices = [device]
                tf = validate_test_function(tf)
                devices = [validate_device(d) for d in devices]
            except IdentifierError as error:
                skipped.append({**record, "reason": str(error)})
                continue
            for dev in devices:
                key = (tf, dev)
                existing = next((entry for entry in entries if (entry["test_function"], entry["device"]) == key), None)
                if existing is None:
                    seen.add(key)
                    entry = {"test_function": tf, "device": dev}
                    if runtime_tf != tf:
                        entry.update({"source_test_function": tf, "runtime_nodeid": runtime_tf,
                                      "device_basis": "ast_or_collected_variant"})
                    entries.append(entry)
                else:
                    # Preserve every concrete collected instance instead of
                    # de-duplicating it by source function/device.
                    node = runtime_tf
                    nodes = existing.setdefault("runtime_nodeids", [])
                    first = existing.get("runtime_nodeid") or existing["test_function"]
                    if first not in nodes:
                        nodes.append(first)
                    if node not in nodes:
                        nodes.append(node)
    report["manifest"] = entries
    report["manifest_skipped"] = skipped
    return entries


def baseline_coverage(router, manifest: list[dict]) -> dict:
    """清单逐条查基线库 hit/miss（只读注记，不改清单 schema）。

    miss = 选中即首次基线 = 无对比信号（整条流水线的产出价值打折）——
    43584 条既有基线是现成资产，这里把它变成选择质量的可观测性指标。
    """
    from .baseline_store import (
        NoBaselineDirectoryError,
        baseline_db_path,
        get_baseline,
    )
    from .db import connect

    hit = 0
    miss_entries: list[dict] = []
    for item in manifest:
        tf, dev = item["test_function"], item["device"]
        try:
            route = router.route(tf)
        except NoBaselineDirectoryError:
            miss_entries.append(
                {"test_function": tf, "device": dev, "reason": "no baseline directory"}
            )
            continue
        conn = connect(baseline_db_path(route.root))
        try:
            rec = get_baseline(conn, tf, dev)
        finally:
            conn.close()
        if rec is None:
            miss_entries.append({"test_function": tf, "device": dev})
        else:
            hit += 1
    return {"hit": hit, "miss": len(miss_entries), "miss_entries": miss_entries}


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"PR commit: {report.get('pr_commit', '?')}",
        f"Concrete cases: {report.get('selected_count', '?')} selected / {report.get('candidate_count', '?')} candidates",
        f"Deferred: {len(report.get('deferred_tests', []))}; unresolved: {len(report.get('unresolved_tests', []))}",
        f"Manifest: {len(report.get('manifest', []))} entries; skipped: {len(report.get('manifest_skipped', []))}",
    ]
    cov = report.get("baseline_coverage")
    if cov is not None:
        lines.append(
            f"Baseline coverage: {cov['hit']} hit / {cov['miss']} miss"
            "（miss = 首次基线，无对比信号）"
        )
    lines.extend(["", "P0 (must run):"])
    if not report["p0"]:
        lines.append("  (none)")
    for item in report["p0"]:
        nodes = ", ".join(item["test_nodes"])
        lines.append(f"  - {item['path']} [{nodes}]")
        for evidence in item["evidence"]:
            lines.append(f"      {evidence['type']}: {evidence.get('symbol', '')}".rstrip())

    lines.extend(["", "P1 (minimal canaries):"])
    if not report["p1"]:
        lines.append("  (none)")
    for item in report["p1"]:
        lines.append(f"  - {item['path']} [{', '.join(item['test_nodes'])}]")
        modules = sorted({e.get("source_module", e["type"]) for e in item["evidence"]})
        lines.append(f"      imports: {', '.join(modules)}")

    lines.extend(["", "Missing direct coverage:"])
    if not report["missing_coverage"]:
        lines.append("  (none)")
    for item in report["missing_coverage"]:
        lines.append(f"  - {item['source']}::{item['symbol']}")
    lines.extend(["", "CI recommendations:"])
    if not report["ci_recommendations"]:
        lines.append("  (none)")
    for item in report["ci_recommendations"]:
        lines.append(f"  - {item['path']}: {item['reason']}")
    lines.extend(["", "Automatic analysis gaps:"])
    if not report["analysis_gaps"]:
        lines.append("  (none)")
    for item in report["analysis_gaps"]:
        lines.append(f"  - {item['path']}: {item['reason']}")
    return "\n".join(lines)
