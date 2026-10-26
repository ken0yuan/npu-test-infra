# 按合入顺序刷新与回滚基线

基线位于 `settings.json` 的 `data_root/baselines/`，按 `baseline_dirs` 路由到各库。
首次刷新时自动建库；已有库保留未涉及的键。无需先清空数据库。

## 给人和 AI 的执行步骤

在仓库根目录运行：

```bash
# 从当前 PR 结果库读取全部 PR，等正在运行的任务完成后按合入时间升序刷新
bash run.sh baseline rollforward --all --wait-seconds 600

# 或指定已有测试结果的 PR；输入顺序可以任意
printf '%s\n' '{"prs":[201997,191432]}' > /tmp/baseline-prs.json
bash run.sh baseline rollforward --input /tmp/baseline-prs.json --wait-seconds 600

# 查看永久保存的快照，选择明确的 id
bash run.sh baseline history

# 手动建一个恢复点
bash run.sh baseline snapshot --reason before-manual-maintenance --wait-seconds 600

# 以 history 中的真实 id 替换 SNAPSHOT_ID
bash run.sh baseline rollback SNAPSHOT_ID --wait-seconds 600
```

1. 以 `applied: true` 且 `errors: []` 判断整批刷新成功，不以某个 PR 的输出存在判断成功。
2. 保存返回的 `before_snapshot` 和 `after_snapshot`。回到刷新前用前者；回到刷新后用后者。
3. 回滚返回 `restored`、`undo_snapshot` 和恢复后的各库行数。撤销回滚时回滚到 `undo_snapshot`。
4. 无执行位时等待到 `--wait-seconds` 上限，然后退出报错，不杀测试、不假报成功。默认不等待。
5. 每次维护将目的、PR 顺序、快照 id、验证结果追加到 `docs/run_test/修复日志.md`。

## 顺序与数据规则

- 按 `merged_at` 升序；同刻按 PR 号升序。合入时间来源沿用现有约定：输入显式时间 > PR 库 > `upstream_commits` 对应提交时间。缺时间报错，不拿测试完成时间冒充合入时间。
- `--all` 在取得执行锁后读取当前结果，避免等待期间漏掉刚完成的 PR。同一 PR 的结果库只保留当前一份结果，因此这不是恢复所有历次重跑的历史。
- 仅完整覆盖且实例集合、环境指纹符合可比较规则的结果可更新基线。`no_result`、部分覆盖、环境不一致均不能强制写入。
- 原始对比档案的 `before` 保持其历史意义，`replay_before` 是本次回放实际读到的基线；`baseline_action` 反映本次动作。
- 缺少档案时从 PR 库重建候选结果，只更新基线，不重新提交或覆盖 PR 结果及原始日志。
- 整批先在临时目录回放和校验，再原子切换。任一 PR 或日志处理错误都不发布，正式基线保持刷新前状态，返回 `applied: false`；发布后收据或快照写入异常时恢复刷新前快照。合入时间元数据的补齐仍保留。
- 这是显式维护命令的排序和快照功能。日常执行器提交结果的触发方式不变；不要把某次快照误认为它之后的实时状态。

## 快照、范围和安全性

快照保存在 `data_root/baseline-history/<id>/`，包含 `manifest.json`、各基线 SQLite 库和日志。
SQLite 使用 backup API，包含已提交 WAL 内容；恢复前检查文件 SHA256、数据库完整性、行数、日志和目录配置。
刷新、手动快照、回滚与正常执行器共用 OS 执行锁。不要绕过入口另开脚本直接写基线文件。

回滚恢复整个基线树，因此会覆盖目标快照之后的基线变化；它先为当前基线再建一份安全快照。
回滚不改 PR 结果库、任务队列、安装环境、源代码或 PR 原始日志，不重跑测试。
Linux 上用 `renameat2(RENAME_EXCHANGE)` 原子交换完整目录，使数据库和日志成套切换；不支持该操作时拒绝恢复。
快照目录不含服务器登录凭据。刷新收据 `refresh.json`、回滚收据 `rollback.json` 长期保留于快照目录。

`data_root/rollforward/<run_id>/` 是临时对比报告，可以按现有规则分析后清理；
**不要连带删除 `baseline-history/`**。没有事前快照的旧状态不能凭空恢复。
