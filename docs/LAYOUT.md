# 目录布局与迁移

2026-10-26 按 plugin / data / venv / docs / run-test 划分，Python 包名仍为 run_test。

| 原位置 | 当前位置 |
|---|---|
| src/run_test | run-test/src/run_test |
| tests/run_test | run-test/tests |
| scripts/recovery/acceptance.py | run-test/tests/acceptance.py |
| scripts/ops | run-test/services |
| scripts/pr_exclusion_filter.py | run-test/tools/pr_exclusion_filter.py |
| scripts/ops/baseline-refresh.sh | run-test/tools/baseline-refresh.sh |
| prepare.py / pyproject.toml | run-test/ 下同名文件 |
| start-pipeline.sh | 功能已合入 start.sh 和 run-test/env.sh；原件归档 |
| data/ctl-venv311 / data/testenv-py311 | 重建为 venv/control / venv/test |
| data/简报.log | data/logs/简报.log |
| 旧迁移、审查、名称修复验收材料 | docs/archive/20261026/runtime-history.tar.gz |
| 原 README、交接、验收、恢复说明、参考 ZIP 与 SHA256SUMS | docs/archive/20261026/ |

`data/runtime/` 原位保留，尤其是结果库、队列和基线；数据库内的历史日志和源码路径不做批量替换。当前配置中的测试 Python 路径更新到新环境。

`docs/archive/` 为本地历史材料，不进入 Git；完整迁移前备份保存在项目之外。原始 SHA256SUMS 只用于原交付包，不能用于整理后的目录。

目录调整没有启用额外采集器，也没有更改 CPU / accelerator 策略、7 天等待规则、结果覆盖或基线更新语义。插件的自动流水线接入是独立工作。
