# 飞书多文档修复与阶段验收（2026-10-26）

状态：报告代码和隔离流程验收通过；生产发布边界尚待核验。整个流水线改进尚未完成，不能把本文当作最终交付或 GitHub 已上传证明。

## 原问题与修复

基于交接提交 `4a7ff9b`，Codex 接手 `result_docs.py`、`feishu_publish.py` 及报告测试；原服务器 Claude 继续实现流水线提速、清理与合入顺序校验，避免交叉编辑。

- 旧规划器把一个 700,000 字符测试整体放进约 700,000 字符续篇，只标记超预算。现在按字符容量拆分元素，完整保留内容与样式，并计入表格单元格、空文本块、根节点、目录、续篇上下文及导航。
- 普通 20,000 字符日志保持一个可见代码块。只有文档容量或单块容量要求时才分段，不存在固定 8,000 字符日志分块。
- `generate` 正确读取 `reports.feishu`；兼容旧顶层 `feishu` 配置，正式嵌套配置优先。
- 已发布旧回执在升级后直接复用，零写入，不因渲染器升级而批量重传历史报告。
- 未完成的旧上传保留原回执及远端暂存文档。新计划使用完整计划 SHA256 隔离暂存目录，避免旧请求已提交但响应丢失时重复追加；同计划重试继续对账恢复。
- 全部文档回读通过后逐篇移入目标目录，全部晋升后才归档旧版本。飞书移动不是跨文档事务：中断时新旧文件可能暂时并存，旧版本仍保留、状态仍为 staging，重试对账完成后才标记 published。
- 续篇含回主篇、前篇/后篇链接；主篇目录链接全部续篇。旧续篇按确切 PR 名称与数字后缀归档，不影响其他 PR。

## 配置和运行方式（供后续 AI 使用）

入口不变：`bash run.sh report render --pr PR_NUMBER` 生成本地报告，发布使用 `bash run.sh report publish --pr PR_NUMBER`。不要为验收执行全量 `publish-scan`，也不要删除历史 published 回执强制重传。

配置位于本地 `settings.json` 的 `reports.feishu`，凭据只从被 Git 忽略的仓库 `.env` 读取：

```json
{"doc_max_blocks": 50, "doc_max_chars": 500000, "block_max_chars": 100000}
```

前两个值是项目可调整的分篇预算，不宣称是新版飞书文档的硬限制。最低支持 42 块、2,048 字符；过小配置会在上传前明确报错。字符计数采用 UTF-16 单元，避免 emoji 等补充字符低估容量。块内文本元素属于传输表达，不产生额外可见代码块。

单块 100,000 字符依据：飞书官方 [Docs limits FAQs](https://www.feishu.cn/hc/en-US/articles/360049067974-docs-limits-faqs) 的搜索索引正文明确给出新版文档每块 100,000 字符及 40,000 块/文档。2026-10-26 直接打开该页提取为空，因此这是官方页面索引证据，不声称本次对租户执行了极限写入测试。历史租户 90,019 字符块只是成功实例，不是上限。每次追加最多 40 个子块是本实现的传输批次。

断点：`data/runtime/feishu-publish/<folder>/PR_<n>/<fingerprint>.json`。升级前暂存回执保留为 `.superseded-<hash>.json`，远端旧暂存也保留供排查。计划 `feishu-report-plan/3`，暂存目录名包含计划哈希。最终 published 回执指向主篇及全部续篇。

## 验收证据

2026-10-26 本地 53 项通过（3.72 秒），服务器控制环境同 53 项通过（3.13 秒）：

```bash
PYTHONPATH=run-test/src venv/control/bin/python -m pytest -q \
  run-test/tests/test_result_docs.py run-test/tests/test_feishu_publish.py --tb=short
```

服务器日志：`data/runtime/maintenance/codex-feishu-acceptance-20261026.log`。覆盖真实规划器→Publisher→有状态内存 API，含混合 Unicode 大日志的逐篇实际块数/字符量、联合全文、40 块请求边界、旧发布回执、丢响应断点、预算变化、篇数变化及晋升中断。未向真实飞书写入。

独立探针另复验六项，全部不再复现：20k 日志分块、700k 单篇超预算、汇总超预算、追加请求过大、配置未接线、v1 暂存升级重复追加。700k ASCII 示例正文分为 456 / 400074 / 300058 字符（还需加目录/导航，回归测试另核算实际完整文档）；配置 20k 预算的生成结果与直接规划结果一致，均为 9 篇。证据：`data/runtime/maintenance/codex-feishu-review-v3-20261026.json`。

代码部署前校验三文件的交接哈希，原件保存在 `data/runtime/maintenance/code-backups/codex-report-20261026T140318Z-e3b4c08a/`。本轮 Codex 未重启服务、未杀测试、未修改基线 DB、未上传真实飞书报告。

## 尚未关闭

生产进程在安全任务边界加载新代码、真实租户文档发布观察仍待完成；流水线提速/有界预取/测后清理、持久化合入顺序校验及最终综合验收仍由原 agent 与 Codex 继续推进。全部约定功能验收完成后才推送 GitHub。修复日志和 agent 持久记忆需引用本记录，禁止把先前只检查“篇数 >= 2”的测试当作容量通过证据。
