"""把 transfer-CUDA 基线快照导入按最小目录拆分的基线数据库。

对应《本地测试结果与基线存储说明》第 5 节的基线库契约，输入是
`~/npu-baseline/run-full/` 的文件级快照（baseline.json + junit/ + logs/）：

- status ∈ {passed, failed} 且 junit 有真实子用例的文件：子用例聚合到
  函数级记录（文档 2.1）；剥 [param] 参数化后缀与 _cuda 设备后缀，用
  test-source 的 AST 函数集校验归属；类名剥 DeviceType<DEV> 后缀；
- 其余文件（collection_error / no_cuda_tests / crashed / timeout，以及
  junit 无真实子用例的 passed/failed）：以**文件级标记记录**入库——
  test_function 固定为 `test/<file>::__file__`，计数是采集器观察到的
  进程级真实计数，日志携带 status/rc/pitfall/first_failure/原始日志
  （用户 2026-09-17：这些诊断信息同样有价值，不得丢弃）。`__file__`
  标记保证它们永不与函数级结果混淆，也不冒充任何函数的四个零；
- device 统一记 `cuda`：该基线是 CUDA 测试经 boot transfer 到 NPU 执行
  的产物，与 PR 轮 device=cuda → transfer 模式同口径（用户 2026-09-17
  确认：cuda 用 transfer，cpu/accelerator 直接跑源码不 transfer）；
- 直接写基线库（ensure_baseline_db + upsert_baseline），不写 PR 库
  （基线不是某个 PR 的结果）；
- 日志先落位（*.tmp.<ts>）再 os.replace，幂等重放安全；重复导入按
  (test_function, device) 覆盖，计数不累加。
"""

from __future__ import annotations

import ast
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .baseline_store import (
    BaselineRecord,
    BaselineRouter,
    baseline_db_path,
    baseline_logs_dir,
    ensure_baseline_db,
    upsert_baseline,
)
from .config import store_paths
from .db import connect
from .identifiers import Counts, baseline_log_filename

# 只有这两个状态的文件能聚合出函数级结果；其余状态（collection_error /
# no_cuda_tests / crashed / timeout）以及"junit 无真实子用例"的文件，
# 以文件级标记记录入库（用户 2026-09-17：这些信息同样有价值，必须入库），
# 不再丢弃。标记记录与函数记录同库同 device，但函数名固定为 __file__，
# 计数是采集器观察到的进程级真实计数，日志携带 pitfall 与原始日志。
IMPORTABLE_STATUSES = ("passed", "failed")
BASELINE_DEVICE = "cuda"
FILE_MARKER = "__file__"
_LOG_COPY_CAP = 262144  # 原始日志超过该字节数时截头尾拷贝
_DEVICE_SUFFIX_RE = re.compile(r"_cuda(?=_|$)")
_CLASS_SUFFIX_RE = re.compile(r"DeviceType[A-Za-z0-9]+$")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class BaselineImportError(ValueError):
    pass


@dataclass
class FunctionAggregate:
    """一个 (test_function, device) 的聚合计数与子用例明细（用于日志）。"""

    test_function: str
    counts: Counts = field(default_factory=Counts.zero)
    lines: list[str] = field(default_factory=list)


def extract_source_functions(src_file: Path) -> set[str]:
    """从测试源码 AST 提取全部函数/方法名，作为子用例归属校验集。"""
    try:
        tree = ast.parse(src_file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def normalize_class(classname: str) -> str:
    """junit classname 末段类名：剥 instantiate_device_type_tests 的
    DeviceType<DEV> 实例化后缀（TestDropoutNNDeviceTypeCUDA → TestDropoutNN）。"""
    cls = classname.split(".")[-1] if classname else ""
    return _CLASS_SUFFIX_RE.sub("", cls)


def normalize_name(raw: str, src_functions: set[str]) -> str:
    """junit testcase 名 → 源码函数名：剥 [param] 与 _cuda 设备后缀。

    截断结果须命中源码函数集才采用；原名本身命中则保留原名（防误伤
    函数名本身含 _cuda 的罕见情况）；都未命中时先试"源码函数名最长
    前缀"归属（真实数据存在不带 [..] 的参数化 id，如
    test_xxx_backend_nccl-lazy），仍不中则保守返回截断结果，
    由调用方统计 unmatched / 跳过非法标识符。
    """
    stripped = raw.split("[", 1)[0]
    m = _DEVICE_SUFFIX_RE.search(stripped)
    base = stripped[: m.start()] if m else stripped
    if base in src_functions:
        return base
    if stripped in src_functions:
        return stripped
    candidates = [f for f in src_functions if base.startswith(f)]
    if candidates:
        return max(candidates, key=len)
    return base


def parse_junit_file(
    entry: dict, junit_path: Path, src_functions: set[str]
) -> tuple[dict[str, FunctionAggregate], int, int]:
    """把一个文件的 junit 解析聚合为 {test_function: FunctionAggregate}。

    返回 (聚合结果, 归属未命中源码集的子用例数, 无源码校验集的子用例数)。
    占位 testcase（收集错误包装：name 为 module 路径、无类名）跳过。
    """
    root = ET.parse(junit_path).getroot()
    file_path = "test/" + entry["file"]
    aggregates: dict[str, FunctionAggregate] = {}
    unmatched = no_src = 0
    for tc in root.iter("testcase"):
        name = tc.get("name") or ""
        classname = tc.get("classname") or ""
        # 占位：收集错误时 pytest 用 module 路径当 name 且无类名。
        if not name.startswith("test_") or "." in name or not classname:
            continue
        cls = normalize_class(classname)
        stripped = name.split("[", 1)[0]
        base = normalize_name(name, src_functions)
        if not src_functions:
            no_src += 1
        elif base != stripped and base not in src_functions:
            unmatched += 1
        if not _IDENT_RE.match(base):
            # 无法构成合法 test_function（如含连字符的参数化 id 且无前缀
            # 可归属）：计入 unmatched 并跳过，不伪造记录。
            unmatched += 1
            continue
        tf = f"{file_path}::{cls}::{base}" if cls else f"{file_path}::{base}"
        agg = aggregates.setdefault(tf, FunctionAggregate(test_function=tf))
        tags = {child.tag for child in tc}
        if "error" in tags:
            agg.counts = Counts(agg.counts.pass_count, agg.counts.fail_count,
                                agg.counts.error_count + 1, agg.counts.skip_count)
            status = "ERROR"
        elif "failure" in tags:
            agg.counts = Counts(agg.counts.pass_count, agg.counts.fail_count + 1,
                                agg.counts.error_count, agg.counts.skip_count)
            status = "FAILED"
        elif "skipped" in tags:
            agg.counts = Counts(agg.counts.pass_count, agg.counts.fail_count,
                                agg.counts.error_count, agg.counts.skip_count + 1)
            status = "SKIPPED"
        else:
            agg.counts = Counts(agg.counts.pass_count + 1, agg.counts.fail_count,
                                agg.counts.error_count, agg.counts.skip_count)
            status = "PASSED"
        msg = ""
        for child in tc:
            if child.tag in ("failure", "error", "skipped"):
                msg = (child.get("message") or "").strip().splitlines()
                msg = msg[0][:300] if msg else ""
                break
        agg.lines.append(f"{status} {name}" + (f": {msg}" if msg else ""))
    return aggregates, unmatched, no_src


def _write_log_staged(logs_dir: Path, filename: str, content: str) -> str:
    """日志先写 *.tmp.<ts> 再原子替换（对齐 record.py 的先落位约束）。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    final = logs_dir / filename
    tmp = logs_dir / f"{filename}.tmp.{os.getpid()}"
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, final)
    return final


def _entry_counts(entry: dict) -> Counts:
    """采集器观察到的进程级真实计数（crashed/timeout 也可能有部分计数）。"""
    return Counts(
        pass_count=int(entry.get("passed") or 0),
        fail_count=int(entry.get("failed") or 0),
        error_count=int(entry.get("errors") or 0),
        skip_count=int(entry.get("skipped") or 0),
    )


def file_status_log_content(entry: dict, snapshot_root: str | Path) -> str:
    """文件级标记记录的日志：状态摘要 + pitfall + 原始采集日志（超长截头尾）。"""
    rel = entry["file"]
    lines = [
        f"# baseline file-status: test/{rel} [{BASELINE_DEVICE}]",
        f"# status: {entry.get('status') or 'unknown'} rc={entry.get('rc')} "
        f"timed_out={entry.get('timed_out')} duration_sec={entry.get('duration_sec')}",
        f"# collected={entry.get('collected', 0)} passed={entry.get('passed', 0)} "
        f"failed={entry.get('failed', 0)} errors={entry.get('errors', 0)} "
        f"skipped={entry.get('skipped', 0)} xfailed={entry.get('xfailed', 0)}",
    ]
    pitfall = entry.get("pitfall") or {}
    if pitfall.get("kind"):
        lines.append(f"# pitfall: {pitfall['kind']}")
        exc = (pitfall.get("exception") or "").strip()
        if exc:
            lines.append("# pitfall.exception（前 2000 字符）:")
            lines.extend("  " + ln for ln in exc[:2000].splitlines())
    first_failure = (entry.get("first_failure") or "").strip()
    if first_failure:
        lines.append(f"# first_failure: {first_failure[:500]}")
    log_rel = entry.get("log") or ""
    lines.append(f"# source log: {log_rel or '(无)'}")
    src = Path(snapshot_root) / log_rel if log_rel else None
    if src and src.is_file():
        try:
            raw = src.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raw = f"(原始日志读取失败: {e})"
        if len(raw) > _LOG_COPY_CAP:
            half = _LOG_COPY_CAP // 2
            lines.append(f"--- 原始日志（超 {_LOG_COPY_CAP} 字节，截头尾） ---")
            lines.append(raw[:half] + "\n...[截断]...\n" + raw[-half:])
        else:
            lines.append("--- 原始日志 ---")
            lines.append(raw)
    return "\n".join(lines) + "\n"


def _persist_record(
    route, tf: str, counts: Counts, log_content: str,
    pytorch_commit: str, torch_npu_environment: dict,
    coverage_status: str = "unknown",
) -> str:
    """写单条基线记录：日志先落位，再事务内 upsert（幂等覆盖）。"""
    ensure_baseline_db(route.root)
    filename = baseline_log_filename(tf, BASELINE_DEVICE)
    _write_log_staged(baseline_logs_dir(route.root), filename, log_content)
    conn = connect(baseline_db_path(route.root))
    try:
        conn.execute("BEGIN IMMEDIATE")
        upsert_baseline(conn, BaselineRecord(
            test_function=tf,
            device=BASELINE_DEVICE,
            pytorch_commit=pytorch_commit,
            torch_npu_environment=torch_npu_environment,
            counts=counts,
            log_path=f"logs/{filename}",
            coverage_status=coverage_status,
        ))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return filename


def import_baseline(
    config: dict,
    baseline_json: str | Path,
    src_root: str | Path,
    torch_npu_commit: str,
    *,
    router: BaselineRouter | None = None,
) -> dict:
    """执行基线导入，返回统计报告 dict。

    torch_npu_commit 必须是完整 40/64 位精确 commit（快照 meta 只含版本
    串短 sha，精确 commit 由调用方从安装环境核实后显式提供）。
    幂等：同一 (test_function, device) 重复导入按本次值覆盖。
    """
    paths = store_paths(config)
    router = router or BaselineRouter(config.get("baseline_dirs") or [], paths.baselines_root)
    data = json.loads(Path(baseline_json).read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    src_root = Path(src_root)
    junit_dir = Path(baseline_json).parent / "junit"

    pytorch_commit = meta.get("source_head") or meta.get("env", {}).get("torch_git")
    if not pytorch_commit:
        raise BaselineImportError("baseline.json meta 缺少 source_head，无法记录精确 pytorch_commit")
    env = meta.get("env") or {}
    npu_file = env.get("torch_npu_file") or ""
    torch_npu_environment = {
        "location": str(os.path.dirname(npu_file) or npu_file or "site-packages"),
        "torch_npu_commit": torch_npu_commit,
    }

    report: dict = {
        "files_total": 0, "files_imported": 0, "files_file_status": 0,
        "files_skipped": defaultdict(int),
        "functions": 0, "unmatched": 0, "no_src": 0,
        "skipped": [], "by_dir": defaultdict(int),
    }
    seen_functions: set[str] = set()

    for entry in data.get("files", []):
        report["files_total"] += 1
        rel = entry["file"]
        file_path = "test/" + rel
        status = entry.get("status")

        aggregates: dict[str, FunctionAggregate] = {}
        if status in IMPORTABLE_STATUSES:
            junit_path = junit_dir / (rel.replace("/", "_") + ".xml")
            if junit_path.is_file():
                src_functions = extract_source_functions(src_root / "test" / rel)
                try:
                    aggregates, unmatched, no_src = parse_junit_file(
                        entry, junit_path, src_functions
                    )
                    report["unmatched"] += unmatched
                    report["no_src"] += no_src
                except ET.ParseError:
                    aggregates = {}
                    report["files_skipped"]["broken_junit"] += 1
            else:
                report["files_skipped"]["no_junit"] += 1

        if not aggregates:
            # 文件级标记记录：没有可聚合的函数结果，但状态/根因/原始日志
            # 照样入库（__file__ 标记，不与函数结果混淆、不伪造函数结果）。
            tf = f"{file_path}::{FILE_MARKER}"
            if tf in seen_functions:
                continue
            seen_functions.add(tf)
            route = router.route(tf)
            _persist_record(
                route, tf, _entry_counts(entry),
                file_status_log_content(entry, Path(baseline_json).parent),
                pytorch_commit, torch_npu_environment, "unknown",
            )
            report["files_file_status"] += 1
            report["by_dir"][route.managed_dir] += 1
            continue

        report["files_imported"] += 1
        for tf, agg in aggregates.items():
            if tf in seen_functions:
                continue
            seen_functions.add(tf)
            route = router.route(tf)
            log_content = (
                f"# baseline import: {tf} [{BASELINE_DEVICE}]\n"
                f"# source: run-full transfer-CUDA 基线 (pytorch {pytorch_commit}, "
                f"torch_npu {env.get('torch_npu')}, boot v{meta.get('boot', {}).get('version')})\n"
                + "\n".join(agg.lines) + "\n"
            )
            _persist_record(
                route, tf, agg.counts, log_content,
                pytorch_commit, torch_npu_environment, "unknown",
            )
            report["functions"] += 1
            report["by_dir"][route.managed_dir] += 1
    report["files_skipped"] = dict(report["files_skipped"])
    report["by_dir"] = dict(report["by_dir"])
    return report
