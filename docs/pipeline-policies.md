# 流水线全局策略

## 全局红线（违反任一条 → 输出无效）

> ⚠️ **核心门禁（Step 5m）**：`merge_analysis.py --validate-only` 强制要求——①图模式且 impact_tag ∈ {Shim, Test, Check} 且触碰 NPU 核心依赖文件（Layer 1 Gate，与 Step 4 路由共用 `_touches_npu_core`，集合一致）的 PR 必须已回填 `graph_analysis`；②`_needs_cann_analysis` 判定为真（CANN API Gate，与 Step 4 路由共用同一函数、集合一致；**CANN Queue 准入唯一判据是 impact_tag ∈ {Shim, Test, Check}**，本判据的唯一权威即本节——`torch_npu_dependency` / `adaptation_layer` 是 Step 3 初判，不参与准入，否则该 PR 再无任何环节复核、初判误判直接变漏标；是否依赖 CANN 的判定口径唯一权威见 [cann-analysis-rules.md](cann-analysis-rules.md)「子代理 Prompt」）的 PR 必须已回填 `cann_analysis`（单栏结论契约见该文件「回填字段」）。任一缺失 → exit(1)，无法进入 Step 6。**故只要 `<RUN_DIR>/prepared/queues/graph_queue.json` / `cann_queue.json` 非空，Step 5 深度分析就不可跳过。** 各步骤如何衔接、各自产出见 `SKILL.md`「流程导航」表。

1. **禁止脚本替代 AI 分析**：`pr_summary`、`impact_analysis`、`impact_tag`（若需推理）必须由 AI 生成。不得使用脚本/模板/正则/f-string 批量生成分析文本。前置过滤（docs-only、test/CI-only、设备专属路径）是唯一允许的规则匹配，此列表封闭不可扩展。**违规后果**：脚本/规则生成的分析文本视为无效输出，必须废弃并由 AI 重做；确需保留的必须在输出中标记 `⚠️ 自动分类，需人工复核`，并告知用户。
2. **输出字段精确值**：`impact_tag` ∈ {Shim, Test, Check, Monitor, Docs}，禁止后缀。`function_module` ∈ 10 大分类，禁止自创。`pr_summary` 必须是中文摘要（30-80 字），禁止透传英文标题。分析文本**禁止模糊措辞，也禁止模板化拼接措辞**（把 function_module 名称与"验证影响"拼接、无具体测试文件名/耦合链路/NPU 文件路径的空洞文本）——禁止清单与示例的唯一权威是 [analysis-rules.md](analysis-rules.md) 规则 6，此处不复列。
3. **数据完整透传**：`diff_summary`、`pr_url`、`files`、`lines_changed`、`labels` 必须从 Step 2 完整透传到最终输出，**不得省略或修改**，merge_analysis.py 以 pr_details.json 为权威源回填缺失字段。
4. **子代理失败自动回退**：Step 3 子代理全部或部分失败时，禁止暂停等待人工确认，必须由主会话直接分批逐条 AI 分析。禁止因 PR 数量大而自行改用脚本/模板替代。
5. **全程禁止暂停询问用户**：任何情况下（脚本 exit(1)、门禁不通过、子代理失败、需要执行下一步骤、Agent 框架弹出确认提示等），**禁止暂停等待人工确认**。必须自动诊断原因、执行缺失的 Step、重试脚本，一口气跑完 Step 0 → 6 直到 Excel 输出。典型场景：Step 5m `--validate-only` 因图模式 PR 缺 graph_analysis 而 exit(1) → 自动回到 Step 5 补做缺失 PR 的深度分析 → 重跑 Step 5m → Step 6。Agent 框架询问"需要继续执行 Step X 吗？"时，一律视为已授权，直接继续。
6. **Check/Test/Shim 必须含耦合链路**：`impact_analysis` 必须包含从直接引用 → 调用上下文 → 调用链追溯 → 初始化/注册入口 → 影响范围的完整链路。**Monitor 必须含 AI 代码判断**：不只是 grep 结论，需要说明变更属于哪个子系统、NPU 为什么不受影响。仅写一句 grep 匹配结果或空洞的"不影响"视为无效输出。
7. **adaptation_suggestion 必须独立写作**：图模式 PR 的 `graph_analysis.adaptation_suggestion` 不得从 `impact_analysis` 截断、正则提取关键词拼接或缩写得来。必须是 AI 阅读完整分析后重新写的 1-3 句精炼适配建议，精准到文件路径和方法名。
8. **每条 PR 的分析文本必须唯一**：不同 PR 的 `impact_analysis` 不得共享相同的文本骨架（去掉文件路径引用后）。`merge_analysis.py` 自动扫描重复骨架，命中 → exit(1)。
9. **Step 5.7 只派生不改 tag**：`need_adaptation`/`need_verification`/`priority` 必须由 AI 子代理综合具体证据推理产出，不得用纯规则/阈值脚本批量判定，也**禁止仅按 impact_tag/adaptation_type 等标签机械套用**。本步骤不重新评估、不修改 `impact_tag` 及其他任何既有字段，不引入新门禁失败、无需重跑校验。三字段值域、优先级来源与确定性取值例外、子代理失败回退的全部细则以 [review-rules.md](review-rules.md) 为唯一权威，此处不复列。
10. **Step 1 合入时间取 Commits API**：PR 列表的合入时间必须取 Commits API 的 `commit.committer.date`，禁止改用 PR API 的 `merged_at`（squash merge 导致其为 null）。已固化在 `fetch_prs.py`，禁止跳过或修改。
11. **Step 2 设备专属 PR 过滤禁改**：`parse_prs.py` 已固化 MPS/MTIA/XPU 专属 PR 的过滤逻辑，禁止跳过或修改排除规则。

> 详细约束和违反后果见各阶段文档（`stages/`）与对应规则文件（`references/`）。门禁由 `merge_analysis.py` 和 `generate_excel.py` 执行。

---

## 前置过滤规则边界（封闭列表）

以下 4 类允许规则匹配跳过 AI 分析，**此列表封闭，禁止扩展**：

| 条件 | function_module | impact_tag | 
|------|----------------|------------|
| 所有文件在 `docs/` 下 | `社区生态` | `Docs` |
| 所有文件在 `test/`/`.ci/`/`.github/` 下（无核心代码） | `社区生态` | `Monitor` |
| 所有文件是设备专属路径（`cuda/`/`mps/`/`xpu/`/`hip/`/`tpu/`） | 按路径判定 | `Monitor` |
| 纯 Revert PR（title/body 含 revert/rollback 关键词，且非高风险路径） | 按原 PR 判定 | `Monitor` |

**任何不满足上述 4 类的 PR（即含任何核心代码路径），必须进入 AI 分析，禁止用规则匹配批量生成。** 本表为前置过滤的**唯一权威**，由 `scripts/prefilter_and_batch.py`（Step 2.6，pr_details 文件列表粒度）与 `scripts/pr_exclusion_filter.py`（单 PR diff 文件粒度，供外部 skill 调用）实现，两脚本前缀集合须与本表保持一致；跳过 PR 的最小记录 JSON 格式见 [`analysis-rules.md`](analysis-rules.md) 规则 5「跳过记录格式」。

---

## 门禁失败修复流程（patch_analysis.py）

`merge_analysis.py` / `generate_excel.py` 的门禁 exit(1) 时，修复动作分两层：

1. **内容（AI 工作）**：针对门禁列出的失败项，AI 重新生成对症的分析文本，直接用 Write 工具写成补丁文件 `<RUN_DIR>/quality/patches/patch.json`（格式 `{"PR编号": {"字段": "新值"}}`，字段级整体替换）。补丁内容必须是重新生成的分析——禁止机械改写旧文本凑长度、禁止从 impact_analysis 截断拼接 adaptation_suggestion（红线 1/7）、禁止按标签批量默认三字段（红线 9）。
2. **机制（脚本）**：`python scripts/patch_analysis.py --run-dir $RUN_DIR --revalidate` 应用补丁并自动复跑门禁（`--run-dir` 模式下 analysis/patch 路径自动推导，`--analysis`/`--patch` 保留为显式覆盖项）；仍未通过 → 继续修补，禁止暂停。

**禁止为此手写一次性 Python 脚本**（历史上的 fix_*.py 家族即此反模式——内容嵌进脚本代码，绕开审查且每次重写）。

---



