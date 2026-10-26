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

结果库为 `data/runtime/pr_results.db`，基线位于 `data/runtime/baselines/`，队列位于 `data/runtime/queue/`。同 PR 重跑替换当前结果，保留历史日志；`no_result` 不推动基线；`done` 表示任务处理结束，不等于测试通过。

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
