# 流水线异步筛选与任务产物清理：阶段验收

2026-10-26：原服务器 Claude 已明确将 pipeline/executor/cli/test_pipeline 四文件交接给 Codex，交接提交 `08eff91`。本文记录其后的修复；尚不是整体验收完成证明。

## 修复后的行为

- 前置过滤与筛选协调器在后台持续刷新候选，执行器独立消费已就绪任务。单次 `pipeline_once` 不等待整项筛选结束；CLI 常驻消费间隔最多一秒，上游观察周期保持独立。
- 筛选通过真实 `python -m run_test.pipeline select-one --task ... --config ...` 子进程运行，有效配置保存到运行目录，避免错误读取当前目录的其他 settings。配置副本权限 0600，不打印内容、不提交 Git。
- 没有整任务总超时。OBS、Git、测试发现沿用各自外部操作超时。一个等待的筛选不会阻塞另一个槽位及已就绪测试；如果全部槽位都在等待，不会无限增加进程。没有以测试专用分支绕开生产调度。
- 默认同时两个筛选，待执行任务与在飞筛选合计受预取上限约束。任务锁防重复筛选，槽位锁限制跨父进程重启后的并发；stdout/stderr 写任务文件，不用等待进程退出后才读取的 PIPE。
- 每个任务先下载源码、筛选，再预取 wheel。包身份包含任务 ID、完整 commit、对象 URL/version、大小与 SHA256；成功的临时下载原子替换后才登记可复用，部分下载不当成成功缓存。执行器重新核验记录后复用源码及 wheel，不重复下载。
- 筛选默认使用控制解释器，或显式独立的 `selector.python_env`。拒绝与执行器环境使用相同解释器路径，预取不运行 pip；发现无法导入依赖时保留原有静态身份回退，由执行端在任务实际环境重新展开并执行。此降级会写入报告的分析缺口，不能宣称已完整发现实例。
- 写库完成后清理任务 work_dir 和 source-staging，保留测试计划、阶段耗时、子进程日志和锁文件。遗留 source_cache 仅在同 commit 前缀无其他在用任务时清理。清理失败保持 cleaning，不假报 done；唯一结果副本写库失败时继续保留。

## 配置与排查（后续 AI 必须先读）

`settings.json` 的 `selector` 可设置：

```json
{"max_in_flight": 2, "prefetch_tasks": 2, "max_failures": 3, "retry_seconds": 30}
```

`prefetch_tasks` 不小于 `max_in_flight`。`python_env` 可选，应指向独立于执行器的解释器，不要为预取修改活动测试环境。不要恢复已否决的 `task_timeout_seconds=900`。

主要文件位于 `data/runtime/selection/<task_id>/`：

- `test-plan.json`：筛选依据与清单证据，测后保留。
- `progress.json`、`executor-progress.json`：当前阶段与实测阶段秒数。
- `select-one.log`：子进程标准输出和错误，测后保留。
- `selection-state.json`：协调器记录的失败次数及下次重试时间。连续失败达上限会转 blocked；人工修复后重试应检查该文件和队列状态。
- `source-staging/`：源码归档/部分测试树/wheel 及身份校验记录，任务结果持久化后删除，不能用作永久缓存。

`selection_pool.py` 只调度、预取并记账；真正安装、等卡、测试、写库仍由持有原 executor lease 的单执行器完成。不要从预取线程调用安装方法，也不要在停止父调度器时杀硬件测试。飞书与基线的约束见同目录报告验收记录及修复日志。

## 验收证据与边界

服务器独立目录：`data/runtime/maintenance/codex-pipeline-isolated-6ca244ca/acceptance.log`，29 项通过，2.97 秒。交接后的生产仓库针对性回归：

```bash
PYTHONPATH=run-test/src venv/control/bin/python -m pytest -q \
 run-test/tests/test_pipeline.py run-test/tests/test_executor.py \
 run-test/tests/test_queue.py run-test/tests/test_selection.py \
 run-test/tests/test_npu_gate.py --tb=short
```

75 项通过，5.07 秒。日志：`data/runtime/maintenance/codex-pipeline-acceptance-20261026.log`。

真实子进程夹具验证：一个等待进程期间另一任务及新到任务推进；1.1 MB stderr 不堵塞；预取上限含已就绪任务；失败留痕及 blocked；任务锁忙/异常后释放；有效配置传播。真实下载替身、真实队列/写库链路验证：校验失败不复用、部分包不复用、源码筛选到执行只下载一次、wheel 复用、清理保留计划、清理失败后恢复。设备门禁相关回归仍通过。

本机 Windows 全组合有一项现有架构命名测试失败；服务器 Linux 控制环境对应测试通过。未将 Windows 结果包装为全通过。

部署前逐文件校验交接哈希，备份 `data/runtime/maintenance/code-backups/codex-pipeline-20261026T150110Z-19ff7787/`。Codex 本轮只部署源码，未重启生产服务、未中断真实测试、未修改运行基线库。真实队列安全边界重载、生产实测耗时/空间回收数据、合入顺序持久化的综合验收仍待完成；未推送 GitHub。

## Additional restart-boundary review

A constructed coordinator-restart fixture reproduced an off-by-one prefetch limit: one surviving locked worker plus one ready task allowed another task to start. The scheduler now counts externally locked workers against both process slots and ready-plus-running capacity. The fixture failed before the change and passed afterward; all 30 isolated pipeline/executor tests passed in 3.43s. Evidence: data/runtime/maintenance/codex-pipeline-isolated-c9fce354/acceptance.log. Code backup: data/runtime/maintenance/code-backups/codex-pipeline-20261026T151145Z-fc2bf79a/. This was an isolated fixture, not a reported production incident. No service restart or hardware test was performed.
