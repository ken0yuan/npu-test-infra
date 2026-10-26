# 2026-10-26 综合修复与验收

> **AI 协作水印：AI-Assisted-By: OpenAI Codex**
>
> 前期基础实现由服务器 Claude 与 Codex 分阶段完成；用户要求交接后，本文列出的最终修复、独立验收、生产复核和本地交付由 OpenAI Codex 执行。Git 加密签名使用仓库所有者既有 GPG 身份，不冒充 OpenAI 官方签名或认证。

用户要求服务器 Claude 停止修改，由 Codex 接手修复、独立验收、拉到本机签名并正常推送 GitHub。原会话在 `2026-10-26T15:19:41.731Z` 确认停止；交接提交为 `bce3557ce66da2dff2faec696bbec3bd54077fee`。本文记录最终行为及证据，历史日志中的阶段判断不代表最终验收。

## 改动位置索引

以下是本次累计交付的功能路径；最终本机签名提交是在服务器已有提交历史上追加修正，不重写前期作者或签名。

| 文件/模块 | 改动与解决的问题 | 验证 |
|---|---|---|
| `run-test/src/run_test/upstream.py` | fetch 后保存完整 first-parent 顺序；原子写入，严格校验来源、SHA、数量、tip 与摘要 | 真实临时 Git fetch、坏快照、人工同刻/倒序 |
| `record.py`、`pr_store.py`、`baseline_store.py` | commit 与 merged_at/来源绑定；新旧比较先于 DB/日志覆盖；旧 PR 仍可出报告；回填备份、迁移、锁和指纹保护 | `test_record_flow.py`、`test_order_acceptance.py`、独立故障探针 |
| `baseline_history.py`、`baseline_rollforward.py` | 显式前滚、永久快照、带 undo 的回滚、原子恢复数据库与日志，共用执行 lease | `test_baseline_history.py`、`test_rollforward.py`，不在生产演示回滚 |
| `pipeline.py`、`selection_pool.py`、`cli.py` | 筛选子进程和执行端解耦；持续收新任务、并发/预取预算、重试留痕、跨父进程重启限额 | 真实子进程阻塞/大 stderr/新到任务/锁测试，生产推进 |
| `selection.py` | 修正 setdefault 参数的重复 AST 解析，按文件惰性缓存；发现失败时保留静态候选但不做无用解析 | 真实 8845 条条目整理 12.119 秒；候选集合/延期记录不变 |
| `task_staging.py` | task/commit/URL/version/哈希绑定的暂存复用、部分下载不准入、任务路径与 symlink 边界 | 校验损坏、半包、不同 commit、外部路径拒绝 |
| `executor.py` | 复用包、各阶段实测耗时、落库后清理；清理失败续做，中断任务不能因清理成功误报 done | 实际队列/写库夹具、清理失败恢复、生产复用和清理观察 |
| `result_docs.py`、`feishu_publish.py` | 长文档分篇与导航、完整版本暂存/验证/晋升/归档、升级续传、40 子块批次、历史回执零写入 | Unicode 全文、预算、篇数变化、丢响应与晋升中断；生产历史零重传 |
| `run-test/tests/acceptance.py` | 用独立发现 venv 验证新隔离契约，继续运行真实 Git/pytest/SQLite 四场景 | 完整 `verify.sh` |
| `README.md`、本文、`修复日志.md`、`待修复.md` | 纠正旧阶段结论、链接实际入口、写明证据/限制/操作步骤与 AI 水印 | 对照源码、日志、进程及提交记录 |

上述相对源码模块均位于 `run-test/src/run_test/`；测试位于 `run-test/tests/`。`.env`、settings、运行数据、私钥及无关插件脏改动不属于交付内容。

## 功能与使用

### 合入顺序、回填和回滚

- `cycle` fetch 官方 PyTorch main 后原子保存 `data/runtime/upstream-order.json`：完整 first-parent SHA 序列、官方来源、tip、时间元数据、格式及摘要。当前覆盖最近 1000 个提交，不是 2000；没有宣称可通过 settings 改此窗口。
- 普通提交中，双方完整 SHA 都属于有效快照时按位置比较；时间相同或人工构造的时钟倒序也不会改变先后。缺失、损坏、来源冲突或未覆盖回退本地祖先；无法判断则保留现有基线。提交侧不 fetch、不联网。官方镜像本次 1000 条检查为零时间倒置；构造倒序测试不是生产事故。
- 同 commit 重试可以刷新并保留合入时间；换 commit 不继承原基线或 PR 的时间。旧结果仍保存 PR 报告，但不能覆盖新基线数据库或日志。
- `bash run.sh merged-at-backfill` 默认保护已经生成的报告。维护锁覆盖原库备份、schema 迁移、读取及写入，备份唯一，异常释放锁。`--refresh-reports` 是显式维护选项，不能作为日常扫描默认值。
- 显式基线前滚/快照/回滚入口、范围和恢复点见 [基线维护说明](baseline-history.md)。该维护命令有独立的显式排序与回滚语义；不要将人工回滚和普通任务防回退混为一谈。本轮没有在生产演示回滚或批量回填。

### 筛选、预取与测后清理

- `pipeline --loop` 每秒检查可消费任务；后台预筛持续接受新候选，默认最多两个筛选子进程。执行器不等待全部筛选结束，仍仅一个执行器持 OS lease，保留设备等待门禁。
- `selector.max_in_flight`、`prefetch_tasks` 限制并发及 ready+running 总量；父进程重启后，仍持任务锁的存活子进程也计入预算。没有整任务筛选总超时；既有 Git/下载/单次收集等外部操作超时仍保留。
- 筛选使用控制解释器或独立 `selector.python_env`，不能指向活动 `executor.test_python_env`；缺少依赖时按选择器静态候选规则处理，执行端在本任务环境重新收集。筛选不安装包。
- 同任务下载按 task/commit/URL/version/size/SHA256 校验，筛选到执行复用；`.partial` 不视为成功。日志写文件，不使用无人消费的 PIPE。失败次数和退避保存在 `selection-state.json`；上限后 blocked，不把无法选例伪装为通过或“不需测试”。
- 结果可靠落库后删除本任务 wheel、解压文件、源码归档、源码树及工作副本；保留计划、阶段秒数、原始结果与诊断。清理失败保持 cleaning 并重试。安装/测试中断后的续清理保持 failed，不能误变 done；任务暂存路径拒绝指向外部的符号链接。

### 飞书完整报告

- 长文档生成主篇/续篇及目录、前后篇和返回主篇链接；20k 日志保持一个可见块，没有固定 8k 拆碎。仅按容量要求无损分配块/元素，跨篇联合文本不截断。
- 完整版本集先暂存、逐篇回读校验，再晋升并归档旧主篇和旧续篇。覆盖篇数增加/减少、多篇变单篇、丢响应续传和晋升中断；多篇移动不是远端原子事务，短暂共存时旧版仍在，恢复后对账完成才记 published。
- 已发布旧回执升级不重写；未完成旧计划以完整 plan hash 隔离，防止同指纹但不同布局重复追加。上传批次最多 40 个子块。
- `reports.feishu.doc_max_blocks/doc_max_chars` 是项目预算，默认 50/500000，并计入表格、目录及导航。单块 100000 UTF-16 单元依据官方帮助搜索索引；未做租户最大边界写测，不称 500k 为飞书硬限制。详见 [报告验收](report-acceptance-20261026.md)。
- `.env` 留在服务器仓库根、0600 并忽略，settings 和运行库不上传。监控仍串行发布；独立发布服务和监控超时诊断属于现存改进项，不宣称已完成。

## 独立验收证据

全量 Linux 隔离目录：`data/runtime/maintenance/codex-full-isolated-e123308b/`（北京时间已跨入 2026-10-27，文件名沿用本次修复开始日）。

| 验证 | 结果与证据 |
|---|---|
| 完整 pytest | **423 passed，0 failed/error/skipped，31.024 秒**；`data/validation/tests.xml`、`acceptance.log` |
| 四场景端到端 | passed/failed/timeout/install_failure 均按预期结束；真实临时 Git、筛选、pytest、SQLite、清理；下载/装包/硬件为替身，不冒充 NPU 实测 |
| 实际 fetch→快照→提交 | 临时 Git 人工同时间/倒序、较新后较旧、坏/缺/未覆盖快照、原子替换失败；DB 与日志均不回退，旧 PR 能生成报告 |
| 故障安全 | 备份异常保留 traceback 仍释放 lease；旧 schema 先备份再迁移；默认回填不改已生成报告指纹；回滚与异常恢复测试纳入全量 |
| 异步筛选 | 真实子进程：一个等待时后续/新到任务推进、1.1MB stderr、重启存活 worker 预算、有效配置与锁释放 |
| 报告完整性 | 700k 混合 Unicode 联合全文、实际容量/导航/目录、旧暂存丢响应升级、布局变化续传、晋升中断及历史回执零写入 |
| 复用与清理 | 真队列/真写库+下载替身证明不重复下载、无半包复用、保存结果后清理、失败续清理状态正确、外部 symlink 不写入 |

完整验收曾暴露旧 `acceptance.py` 将筛选和执行设为同一解释器，触发新保护；已改为独立无 pip 发现 venv 后通过，不撤销保护以迁就测试。Windows 本机不用于替代 Linux 全量验收；本机负责源码、Git 与签名。

早期独立证据仍保留：`codex-acceptance-eb79409.json`、`codex-feishu-review-v3-20261026.json`、`codex-pipeline-isolated-c9fce354/acceptance.log`。不把旧阶段测试数量叠加为测试总数。

## 生产安全边界与观察

生产变更在持 executor lease 且 preparing/testing/writing/cleaning 均为零时完成。逐文件与交接提交哈希一致才替换，备份为 `data/runtime/maintenance/code-backups/codex-final-a57d460050/`。等待报告进程睡眠且各目标进程无活跃子进程后，只重载 pipeline/cycle/bug_monitor；brief 保持原进程，未杀正在执行的测试。

- pipeline `387379 → 412669`，cycle `387378 → 412668`，bug_monitor `387380 → 412670`。
- 部署时官方顺序 tip 为 `2eda30aa8bb9262ff24131d6cc93ef41fe26ea71`，1000 条零时间倒置；在执行位释放前已写入有效快照。
- 重载后的首次生产报告扫描：14 checked、14 unchanged、0 generated、0 published、errors=[]，没有历史批量重传。
- 新候选在已有筛选运行时被预筛；两个原队头零候选任务退出后，后续 `55b40a` 与 `d4295d` 开始筛选。生产源码下载约 16–18 秒、提取约 20–28 秒，阶段数据可查 `selection/<task_id>/progress.json`。这些是本次真实计时，不能与旧 6–10 分钟下载记录当作同条件性能对照。
- 部署及持续观察证据：`data/runtime/maintenance/codex-final-deployment.json`、`codex-final-observation.json`。

### 生产观察发现并修复的第二个筛选瓶颈

首次重载后两个有效 PR 持续耗 CPU。隔离取栈定位到 `finalize_plan` 中 `declarations.setdefault(path, declared_class_methods(...))`：`setdefault` 的参数每次都会求值，原生文件匹配产生逐用例条目时重复解析同一文件。

真实 PR 200299 捕获 8845 条条目、149 个文件，单文件最多出现 935 次。旧隔离运行反复停在同一文件 AST 解析；最终修复按文件惰性缓存，发现失败走静态候选时根本不需要声明解析。修复后对同一捕获计划整理耗时 **12.119 秒**，输出 8844 个去重候选、15 个入选和 8829 个延期记录；没有删候选或缩减规则。成功发现和失败发现两种回归分别验证每文件解析一次/零次及候选集合不变。证据为 `codex-selection-stages.log`、`codex-selection-fixed-benchmark.json`、`codex-selection-fixed-plan.json`。

在 executor 与 scheduler 双锁、无活动执行任务、两个目标仍为 pending_analysis 且无收集子进程的边界，停止仅这两个旧筛选进程并重载协调器；已下载暂存保留。pipeline 更新到 `419676`，备份 `code-backups/codex-selector-cache-0a2225c3/`。部署助手曾在 supervisor 5 秒重启间隔读取不存在的 worker 键，复核新 PID 和源文件哈希后补齐部署证据；没有重复部署。修复后的两个任务源码阶段分别 **1.266/1.431 秒**，跳过重新下载和提取；不是将下载缓存永久保留。

### 真实任务与飞书全链路

PR **200299**，task `t-20261026-072104-55b40a`，完整 commit `310e1478ac3c7aacfb24467711264b32624e55eb`，在 `2026-10-26T16:18:44.185929Z` 完成落库和清理。任务来源为 `upstream-main-commit-time`，合入时间 `2026-10-22T21:00:13Z`；顺序由已校验 first-parent 快照判断，不能仅靠此时间覆盖。

- 实际结果 **14 passed、0 failed、0 error、8 skipped、2 no_result**。这不是全部测试通过；22 个有效结果建立基线，2 个 no_result 未刷新基线，诊断保留。
- 筛选 60.258 秒；准备 81.874 秒、安装 23.977 秒、预检 29.452 秒、设备等待 9.638 秒、测试 543.948 秒、持久化 0.053 秒、清理 7.074 秒。三份包均复用；后续 PR 201867、200512 的筛选同时推进。
- 工作目录和 source-staging 均已删除，保留 26 个日志目录条目和测试计划。删除前记录的文件逻辑字节总量为 **4,826,346,839（约 4.83 GB）**；这是逻辑内容大小下界，不是磁盘物理块回收测量。
- 本次新结果通过既有 publisher 发布为 `PR#200299` 和 `PR#200299-1/-2/-3`，四篇回读 `content_verified=true`，物理块数分别 **49、49、50、37**，符合每篇 50 块预算。完整计划、内容与导航校验通过后晋升，旧主篇按正常版本替换归档；没有批量重发其他历史 PR。
- [真实主报告](https://mcnh5rzpwcgn.feishu.cn/docx/WRfmd2eFqowU5fxtXJzcKpbgnGc)。证据：`data/runtime/maintenance/codex-live-task-before.json`、`codex-live-task-after.json`、`codex-live-report-200299.json`，以及该指纹的 `feishu-publish` published 回执。

## 本机签名交付

本机仓库保留服务器完整提交链，再追加最终修复与本文；真实远端为 `https://github.com/ken0yuan/npu-test-infra.git`，目标 main。既有 GPG 公钥指纹为 `1FBCCC5A834F9A49C49E401211425BE649D35941`（GitHub 已登记）。最终提交带 `AI-Assisted-By: OpenAI Codex`，使用用户身份签名；交付时校验签名、重新 fetch 并正常快进 push，不改写历史。最终 commit SHA 与 GitHub 签名验证结果记录在交付回执，避免文档自引用提交哈希。

源码、文档和测试纳入 Git；运行证据保留服务器，凭据、settings、运行库与无关插件改动不上传。基线回滚只做隔离测试；租户单块最大边界没有额外写测，不能由本次四篇正常发布推断硬上限。

## 给后续 AI 的操作约束

1. 先读本文件及任务 progress/队列/新鲜服务日志，不能用陈旧 checked 数断言当前状态；源码通过不等于生产进程已加载。
2. 维护从 `run.sh` 入口持锁；部署只在安全任务边界，验证 PID/start_ticks。不得杀测试、删除未落库唯一证据或批量重发历史报告。
3. 每次修复记录问题、复现、改动、回归、部署与遗留项到 [修复日志](修复日志.md)，同步 agent 持久记忆。原服务器 Claude 已按用户要求停止实施，不应再次并行修改。
4. GitHub 目标 `ken0yuan/npu-test-infra` 的 main；fetch 后确认快进关系，使用本机既有签名身份签名并 verify-commit，正常 push。不要 force、不要混入 `plugin/trace_api_chain` 的无关脏改动或本机配置/密钥。
