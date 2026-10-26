"""run_test：PR 测试结果与基线存储 + OBS 监测串行测试队列。

依据两份设计说明文档实现：
- 《本地测试结果与基线存储说明.md》：PR 结果库（pr_number 主键）、
  按最小目录拆分的基线库（(test_function, device) 联合主键）、
  通用环境说明文件、日志布局与完整提交的六步刷新流程（record）。
- 《OBS监测与串行测试队列说明.md》：SQLite 持久化队列、状态机、
  身份核实强制门禁（规则未确认不放行）、串行执行端。

实施边界（OBS 判定规则尚未确认，2026-09-17 用户答复）：
- 已实现：存储端全部、队列结构与状态机、执行端骨架、skill 拉取校验；
- 未实现（依赖待确认规则）：候选身份判定逻辑本身、监测放行、
  筛选输入对接（prefilter_and_batch 的数据入口）。
"""

from .identifiers import (
    Counts,
    IdentifierError,
    baseline_log_filename,
    parse_test_function,
    validate_commit,
    validate_device,
    validate_test_function,
    validate_torch_npu_environment,
)
from .pr_store import (
    PrResultRecord,
    ensure_pr_db,
    get_pr_result,
    list_pr_results,
    upsert_pr_result,
)
from .baseline_store import (
    BaselineRecord,
    BaselineRouter,
    ensure_baseline_db,
    get_baseline,
    list_baselines,
)
from .record import (
    BaselineDiff,
    FunctionResult,
    Submission,
    SubmitReport,
    SubmissionValidationError,
    submit_complete_result,
)
from .task_queue import (
    ACTIVE_STATES,
    ALL_STATES,
    IdentityGateError,
    QueueStore,
    Task,
    TERMINAL_STATES,
)
from .environment import collect_environment, write_environment_file
from .executor import Executor, RunOutcome
from .obs import make_downloader

__all__ = [
    "Counts", "IdentifierError", "baseline_log_filename",
    "parse_test_function", "validate_commit", "validate_device",
    "validate_test_function", "validate_torch_npu_environment",
    "PrResultRecord", "ensure_pr_db", "get_pr_result", "list_pr_results",
    "upsert_pr_result",
    "BaselineRecord", "BaselineRouter", "ensure_baseline_db",
    "get_baseline", "list_baselines",
    "BaselineDiff", "FunctionResult", "Submission", "SubmitReport",
    "SubmissionValidationError", "submit_complete_result",
    "ACTIVE_STATES", "ALL_STATES", "IdentityGateError", "QueueStore",
    "Task", "TERMINAL_STATES",
    "collect_environment", "write_environment_file",
    "Executor", "RunOutcome", "make_downloader",
]
