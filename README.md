# npu-test-infra

用于 PyTorch PR 的包发现、测试筛选、串行执行、结果记录和基线维护。执行设备仅支持 `cpu` 和 `accelerator`（昇腾 NPU）；禁止 CUDA 执行。等待配套包的规则仍为 7 天。

## 目录

- `run-test/`：流水线源码、服务脚本、工具和回归测试。
- `plugin/api-test-relations/`：API ↔ 测试关系采集插件（Git submodule）。
- `plugin/trace_api_chain/`：调用链采集、离线重放与查询工具（Git submodule）。
- `data/`：队列、结果库、基线、源码缓存、日志和验收输出，不提交 Git。
- `venv/control/`、`venv/test/`：控制面与被测环境，不提交 Git。
- `docs/`：使用说明与规则；`docs/archive/` 是本地历史归档，不提交 Git。

## 获取源码和子模块

```bash
git clone --recurse-submodules git@github.com:ken0yuan/npu-test-infra.git
cd npu-test-infra
```

已有主仓库时，在更新主仓库后执行：

```bash
git submodule update --init --recursive
git submodule status
```

主仓库锁定每个插件的提交，不自动跟随插件的最新 main。当前 API 关系插件锁定 0.1.3（`6ae759d`）；调用链插件初始锁定 `e9ac4dd`。插件改动应在各自仓库提交并推送，再由主仓库提交更新后的子模块引用。

## Linux 安装

在 Linux 上使用 Python 3.11。当前 OBS wheel 为 cp311；CPU 架构必须与 wheel 相符。

```bash
python3.11 -m venv venv/control
venv/control/bin/python -m pip install -r docs/requirements-validation.txt
python3.11 -m venv venv/test
venv/test/bin/python -m pip install -r docs/requirements-test.txt
# 在专用测试环境安装与目标源码匹配的 torch、torch_npu 及测试依赖。
venv/control/bin/python run-test/prepare.py --device cpu   --test-python "$PWD/venv/test/bin/python" --obsutil /absolute/path/to/obsutil
bash verify.sh
```

`prepare.py` 会在项目根生成本机 `settings.json` 并初始化数据库；已有配置保持不变。真实任务会替换测试环境的包，不应使用系统 Python 作为测试环境。OBS 凭据由本机 obsutil 配置提供，不写进仓库。

根入口默认使用 `venv/control/bin/python`；可用 `RUN_TEST_PYTHON` 显式覆盖。默认设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，accelerator 执行时显式加载 `torch_npu`，避免在无 CANN 的 CPU 环境自动加载失败。

## 启动和查询

```bash
bash start.sh start
bash start.sh status
bash run.sh results
bash run.sh baseline list
bash run-test/tools/baseline-refresh.sh PR_NUMBER
bash start.sh stop
```

`start.sh` 管理 watchdog、cycle、pipeline、bug_monitor、brief 五常驻。服务日志在 `data/service/`，简报在 `data/logs/简报.log`。`run.sh` 是业务命令入口；`verify.sh` 运行工具回归和四场景隔离验收，输出在 `data/validation/`。

基线合入序保护（2026-10-26）：结果提交自带 `merged_at`（来源
`upstream-main-commit-time`，任务登记时持久化），提交前与基线的合入时刻
比较——更早提交保留现有基线（行与日志均不回退）并留痕；时刻缺失/无效/
同刻歧义时回退 Git 祖先关系，仍不可得则保守保留。存量行回填：
`bash run.sh merged-at-backfill`（先备份、只填 NULL、幂等、不触发报告刷新）。
结果库为 `data/runtime/pr_results.db`，基线位于 `data/runtime/baselines/`，队列位于 `data/runtime/queue/`。同 PR 重跑替换当前结果，保留历史日志；`no_result` 不推动基线；`done` 表示任务处理结束，不等于测试通过。

## 基线刷新与回滚

`bash run.sh baseline rollforward --all --wait-seconds 600` 会读取当前已保存的 PR 结果，按合入时间从早到晚刷新；没有基线库时自动创建。整批在临时目录完成并校验，成功后原子切换数据库和日志。运行中的测试持有执行位时等待，不中断测试。

刷新前后自动保存持久快照。用 `bash run.sh baseline history` 查看 id，使用 `bash run.sh baseline rollback SNAPSHOT_ID --wait-seconds 600` 恢复指定基线；回滚前也会保存当前状态，返回 `undo_snapshot` 便于撤销。快照在 `data/runtime/baseline-history/`，不会随临时对比报告一起清理。

给 AI 的完整步骤、成功判据、数据准入规则和恢复范围见 [基线维护说明](docs/run_test/baseline-history.md)。每次维护都必须在 `docs/run_test/修复日志.md` 留下快照和验证记录。

## 真实问题台账

`bug_monitor` 持续扫描任务、测试计划和真实 JUnit/收集/执行产物。计划生成前的选例异常也会立即留证。

```bash
bash run.sh ledger scan
bash run.sh ledger status
bash run.sh ledger resolve U-new-000001 --note '修复与回验说明' --evidence /absolute/data_root/verified.log
```

`data/runtime/issue-ledger/issues.db` 保存发生记录；原始证据按 SHA256 留快照，任务清理或 PR 重跑后仍可核对。
问题台账 `问题总览.md`（一个问题一节，出现次数与涉及 PR 聚合）由数据库生成，默认在 `data/runtime/issue-ledger/views/`；可用本机配置
`issue_ledger.docs_root` 指向 `docs/run_test/台账/`。新部署使用 `U-new-` / `S-new-`，不冒用遗失的历史编号。
重复扫描不重复立账；已知 C 签名和显式配置的文件级 X 规则保留观察记录，不另立 U/S。
跳过、超时、未取得结果均不算通过；已知签名也不自动证明环境故障或排除 PR 因果。
正常预算导致的候选延期不会自动判为选例错误。自动监测不会安装包或改变测试结果与排除策略。

测试依赖见 `docs/requirements-test.txt`，只安装到 `venv/test`，并同步到本机配置的
`executor.test_requirements`，确保逐任务换包后仍安装它们。框架包只用 OBS 配套包；
硬件专用包和不明缺失模块需单独核对，不按模块名猜包安装。

## 飞书报告生成

已按历史飞书测试报告还原 PR 链接、提交环境、六列统计表、基线前后对比、
折叠测试详情与日志代码块。生成过程只读结果库与本次 PR 日志，不访问认证配置。

```bash
bash run.sh report render --pr 191432
bash run.sh report scan
```

每份报告位于 `data/runtime/reports/PR_<编号>/<内容哈希>/`：
`report.json` 为可核对数据，`report.html` 为可离线打开的预览，
`feishu-plan.json` 为飞书原生内容块的有序发布计划。`current.json` 指向当前版本；
内容不变不重写，重跑保留旧版本。发布计划中的表格单元格必须在创建表格返回 ID 后填充；
`render/scan` 只生成本地文件；远端发布使用下列独立命令。

本机配置加入 `"reports": {"enabled": true}` 后，常驻 `bug_monitor` 每轮自动生成报告；
单个 PR 的报告失败会记在服务日志中，不改变测试结果或阻断其他 PR。

统计表按“函数 × 设备”的测试项归类，与下方执行实例计数分开：
Error > Failed > No Result > Skipped > Passed。部分实例跳过归入 Skipped，全部实例
通过才归入 Passed；中断前观察计数单独展示，不冒充完整结果。
只使用本次 `baseline-diff.json` 的历史快照；缺失、不可比或身份冲突均明确处理，
不拿已更新的当前基线反推过去。日志缺失显示证据提示，长日志分块保留；HTML 转义，
终端颜色码仅从展示文本中去除，SHA256 仍对应原文件。

样例来源、与旧实现的边界及统计口径见 [报告还原说明](docs/report-generation.md)。

### 发布到飞书

将飞书专用配置保存在仓库根目录 `.env`（例如 `/home/workspace-ycy/npu-test-infra/.env`），
文件权限设为 0600。可先复制 `.env.example` 再填写真实值；`.gitignore` 已忽略 `.env` 与 `.env.*`，仅允许无凭据的示例模板入库。其中只需 `FEISHU_APP_ID`、`FEISHU_APP_SECRET`、
`FEISHU_DOMAIN`、`FEISHU_TEST_FOLDER_TOKEN`。不要把认证内容提交 Git 或写入运行日志。
在本机 `settings.json` 配置 `reports.feishu.env_file` 指向仓库 `.env` 的绝对路径。`settings.json` 同样由 Git 忽略；监控进程每轮重新读取配置。

```bash
bash run.sh report publish --pr 191432
bash run.sh report publish-scan
```

验证手工发布成功后，将 `reports.enabled` 和 `reports.feishu.enabled` 设为 `true`，
重载 `bug_monitor` 即启用自动发布。回执及断点保存在 `data/runtime/feishu-publish/`。
上传目标只用 `FEISHU_TEST_FOLDER_TOKEN`，不写深度分析文件夹。
先在“报告上传暂存”中构建并完整回读验证，再移入目标目录，将同名旧报告移入
“历史测试报告”，保留旧文档及链接。内容不变不重复发布；超时后的未知写入结果必须
经远端回读确认，无法确认时留下错误和暂存内容，不盲目再写。

## 插件

两个插件独立存放，暂未自动接入常驻 executor。API 关系插件需要采集时，在专用测试环境安装并显式启用：

```bash
venv/test/bin/python -m pip install -e plugin/api-test-relations
venv/test/bin/python -m api_test_relations doctor
```

真实测试执行命令与 `settle` / `query` / `audit` 的用法见 [插件迁移说明](docs/plugin-migration.txt) 和 [插件 README](plugin/api-test-relations/README.md)。每轮使用 `data/plugin/` 下的新输出目录，保留完整参数化测试名。插件标签不会把 CPU 测试转换为 NPU 测试。

调用链插件的安装、采集、重放与查询用法见 [调用链 README](plugin/trace_api_chain/README.md)。本次仅纳入子模块，没有将其加入测试环境或启用采集。

## 验证与边界

当前已验证 Linux ARM64 CPU 的真实测试与插件采集；NPU 需要具备 CANN 和实际昇腾设备的目标机验收。四场景验收中的下载、装包和硬件检查使用替身，不能充当真实硬件执行证据。

目录迁移说明见 [布局说明](docs/LAYOUT.md)，运行规则见 [流水线策略](docs/pipeline-policies.md)。历史材料与原包校验清单仅在本地 `docs/archive/` 中保留，旧路径不代表当前入口。

## 提交 GitHub

主仓库只提交入口脚本、`settings.example.json`、`run-test/`、`.gitmodules`、`plugin/` 的子模块引用和当前文档；插件源码在各自仓库中维护。`.gitignore` 排除本机配置、凭据、虚拟环境、运行数据和历史归档。提交前用 `git status --short` 核对文件；不要使用 `git add -f` 绕过这些规则。
