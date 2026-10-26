#!/usr/bin/env python3
"""根据单个拉取请求的 diff 文件判断该请求是否可以从代码影响分析范围中排除。

调用方只需传入一个 UTF-8 编码的 diff 文件，不需要准备拉取请求详情文件，也不会
发起任何网络请求。脚本从文件头提取全部改动路径，然后执行路径判断。

支持以下文件头格式：

1. 标准 Git 统一 diff：``diff --git a/旧路径 b/新路径``；
2. 简化文件头：``--- 文件路径 +新增行数 -删除行数``；
3. 不含 ``diff --git`` 时，使用成对的 ``--- a/旧路径`` 和 ``+++ b/新路径``。

当非空文件列表完全属于以下任意一类时，返回可以排除：

1. 全部文件都位于 ``docs/`` 目录；
2. 全部文件都位于 ``test/``、``.ci/``、``.github/`` 或 ``docs/`` 目录；
3. 全部文件路径都包含 ``cuda/``、``mps/``、``xpu/``、``hip/``、``tpu/``
   或 ``rocm/`` 设备后端目录标记。

判断采取保守策略：无法提取文件路径、文件头格式不完整，或者 diff 中包含
``[TRUNCATED]``、``[NO_PATCH]`` 等不完整标记时，均返回 ``false``。

使用方式：

    python pr_exclusion_filter.py path/to/PR_123456.diff

标准输出只有一个 JSON 布尔值：

* ``true``：可以排除；
* ``false``：应保留在分析范围内。

也可以直接导入并调用 :func:`should_exclude_diff_file`。本脚本只依赖 Python 标准库。
"""

import argparse
import json
import re
import shlex
from pathlib import Path


# 排除规则的唯一权威是 references/pipeline-policies.md「前置过滤规则边界（封闭列表）」；
# prefilter_and_batch.py 是同一封闭列表在"pr_details 文件列表"粒度的实现（流水线内使用），
# 两边前缀集合必须保持一致，改动任一处须同步另一处并对照 pipeline-policies.md 封闭列表。
DOCUMENTATION_PREFIXES = ("docs/",)
NON_PRODUCTION_PREFIXES = ("test/", ".ci/", ".github/", "docs/")
DEVICE_PATH_MARKERS = ("cuda/", "mps/", "xpu/", "hip/", "tpu/", "rocm/")

INCOMPLETE_MARKERS = ("[TRUNCATED]", "[NO_PATCH]")
SUMMARY_HEADER_PATTERN = re.compile(r"^--- (.+) \+\d+ -\d+$")


class ChineseArgumentParser(argparse.ArgumentParser):
    """将命令行帮助中的固定标题显示为中文。"""

    def format_help(self) -> str:
        """返回中文标题的命令行帮助文本。"""
        help_text = super().format_help()
        return (
            help_text.replace("usage: ", "用法：", 1)
            .replace("positional arguments:\n", "位置参数：\n", 1)
            .replace("options:\n", "可选参数：\n", 1)
        )


def _all_paths_start_with(paths: list[str], prefixes: tuple[str, ...]) -> bool:
    """判断非空文件列表中的每条路径是否都以指定目录之一开头。"""
    return bool(paths) and all(
        any(path.startswith(prefix) for prefix in prefixes) for path in paths
    )


def _all_paths_are_device_specific(paths: list[str]) -> bool:
    """判断非空文件列表中的每条路径是否都包含设备后端目录标记。"""
    return bool(paths) and all(
        any(marker in path.lower() for marker in DEVICE_PATH_MARKERS)
        for path in paths
    )


def should_exclude_paths(paths: list[str]) -> bool:
    """根据完整的修改文件路径列表返回是否可以排除。

    参数：
        paths: 拉取请求修改的全部文件路径。

    返回：
        所有路径完全属于文档、测试与持续集成配置或设备后端专属路径时返回
        ``True``，否则返回 ``False``。空列表返回 ``False``。

    异常：
        TypeError: ``paths`` 不是纯字符串列表。
    """
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise TypeError("文件路径必须是纯字符串列表")

    return (
        _all_paths_start_with(paths, DOCUMENTATION_PREFIXES)
        or _all_paths_start_with(paths, NON_PRODUCTION_PREFIXES)
        or _all_paths_are_device_specific(paths)
    )


def _strip_git_prefix(path: str) -> str:
    """去除统一 diff 文件路径中的 ``a/`` 或 ``b/`` 前缀。"""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _decode_single_path(raw_path: str) -> str:
    """解析可能带引号的单个 diff 路径。"""
    raw_path = raw_path.split("\t", 1)[0].strip()
    if not raw_path:
        return ""
    try:
        parts = shlex.split(raw_path)
    except ValueError:
        return ""
    if len(parts) != 1:
        return raw_path
    return parts[0]


def _append_unique(paths: list[str], seen: set[str], path: str) -> None:
    """按首次出现顺序添加非空且未重复的文件路径。"""
    if path and path != "/dev/null" and path not in seen:
        seen.add(path)
        paths.append(path)


def extract_changed_paths(diff_text: str) -> tuple[list[str], bool]:
    """从 diff 文本中提取改动文件路径，并返回路径信息是否完整。

    优先使用标准 ``diff --git`` 文件头；不存在时依次尝试简化文件头和成对的
    ``---``、``+++`` 文件头。只要发现截断标记、缺失补丁标记或格式损坏的标准
    文件头，完整性就返回 ``False``。

    返回：
        二元组 ``(去重后的文件路径列表, 路径信息是否完整)``。

    异常：
        TypeError: ``diff_text`` 不是字符串。
    """
    if not isinstance(diff_text, str):
        raise TypeError("diff 内容必须是字符串")

    has_incomplete_marker = any(marker in diff_text for marker in INCOMPLETE_MARKERS)
    lines = diff_text.splitlines()

    paths: list[str] = []
    seen: set[str] = set()
    git_headers_found = 0
    malformed_git_header = False

    for line in lines:
        if not line.startswith("diff --git "):
            continue
        git_headers_found += 1
        try:
            parts = shlex.split(line)
        except ValueError:
            malformed_git_header = True
            continue
        if len(parts) < 4 or parts[0:2] != ["diff", "--git"]:
            malformed_git_header = True
            continue
        _append_unique(paths, seen, _strip_git_prefix(parts[3]))

    if git_headers_found:
        is_complete = bool(paths) and not malformed_git_header and not has_incomplete_marker
        return paths, is_complete

    for line in lines:
        match = SUMMARY_HEADER_PATTERN.match(line)
        if match:
            _append_unique(paths, seen, match.group(1).strip())

    if paths:
        return paths, not has_incomplete_marker

    old_path = ""
    for line in lines:
        if line.startswith("--- "):
            old_path = _strip_git_prefix(_decode_single_path(line[4:]))
            continue
        if not line.startswith("+++ "):
            continue

        new_path = _strip_git_prefix(_decode_single_path(line[4:]))
        selected_path = old_path if new_path == "/dev/null" else new_path
        _append_unique(paths, seen, selected_path)
        old_path = ""

    is_complete = bool(paths) and not has_incomplete_marker
    return paths, is_complete


def should_exclude_diff_text(diff_text: str) -> bool:
    """根据完整 diff 文本返回是否可以排除。

    文件路径信息不完整时直接返回 ``False``，不会使用部分数据作出排除结论。
    """
    paths, is_complete = extract_changed_paths(diff_text)
    return is_complete and should_exclude_paths(paths)


def should_exclude_diff_file(diff_file: str | Path) -> bool:
    """读取一个 diff 文件并返回是否可以排除。

    文件按 UTF-8 解码；无法解码的个别字节会被替换，但不影响标准 ASCII 文件头
    的路径提取。
    """
    path = Path(diff_file)
    diff_text = path.read_text(encoding="utf-8", errors="replace")
    return should_exclude_diff_text(diff_text)


def main() -> int:
    """读取命令行中的 diff 文件并输出布尔判断结果。"""
    parser = ChineseArgumentParser(
        add_help=False,
        description="根据拉取请求 diff 文件中的修改路径判断是否可以排除",
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument(
        "diff_file",
        type=Path,
        help="单个拉取请求的 diff 文件",
    )
    args = parser.parse_args()

    try:
        result = should_exclude_diff_file(args.diff_file)
    except OSError as error:
        parser.exit(2, f"错误：无法读取 diff 文件：{error}\n")

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
