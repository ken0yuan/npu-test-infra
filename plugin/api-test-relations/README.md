# API ↔ Test Relations

独立 pytest 插件：在已有测试执行期间记录 **台账 API ↔ 测试实例** 的关系。
目标部署环境是 **ARM64 Linux + 昇腾 NPU**；采集核心与设备无关，x86 Linux 也可以使用。
插件不设置 CUDA/NPU 设备、不启动后台调度、不安装或替换 PyTorch/torch_npu、不采集测试通过失败结果。
不需要 `-g1`、GDB、CUDA 工具链。它观察的是 Python/C callable 身份，不是底层 NPU 算子调用栈。

## 文件树

```text
api-test-relations-plugin/
├── pyproject.toml                   # 安装信息、pytest 插件、CLI 入口
├── README.md                        # 安装、接入、查询
├── ORIGINS.json                     # 从旧采集器提取的文件及原始校验值
├── VALIDATION.json                  # 本次实际验证范围，不代替 ARM/NPU 验收
├── docs/
│   ├── data-contract.md             # 输出规范和关联语义
│   └── limitations.md               # 能力边界、已知缺陷
├── src/api_test_relations/
│   ├── pytest_plugin.py             # 显式启用、测试源码清单、结束时合并
│   ├── cli.py                       # doctor / query / status / audit / settle
│   ├── settle.py                    # 进程退出后，从原始证据生成新汇总
│   ├── storage.py                   # SQLite 分片合并、子进程回执、API 查询
│   ├── recovery.py                  # 已退出进程的热日志恢复与原证据保留
│   ├── audit.py                     # 只读结构检查
│   ├── catalogs/pytorch-api-ledger.json # 冻结的 10,362 条台账
│   └── runtime/
│       ├── relation_plugin.py       # pytest setup/call/teardown 归属
│       ├── relation_collector.py    # callable 身份、别名、线程上下文
│       ├── relation_identity.c      # CPython 身份适配器，在目标 CPU 上编译
│       ├── relation_guard.h         # 原生 profiler 转发及 opcode 守护
│       ├── relation_context.py      # 当前测试归属上下文
│       ├── relation_process.py      # 子进程、进程池任务、回执
│       ├── relation_launch.py       # Python 启动命令解析
│       ├── relation_child_entry.py  # 子进程采集入口
│       └── bootstrap/sitecustomize.py
├── examples/test_ascend_smoke.py     # 在真实 NPU 服务器上运行的小样本
├── scripts/check_smoke_relations.py  # 检查小样本的 API 归属
└── tests/                           # 独立回归，无需真实 NPU
```

## 环境要求

- Linux，ARM64 或 x86_64；**本版仅支持 CPython 3.11**。
- pytest 7–9；C 编译器（`cc` 或 `CC` 指定）及当前 Python 的开发头文件。
- 服务器已经能用兼容的 PyTorch、torch_npu、CANN 运行目标测试。
- 在服务器上原生构建小型 C 适配器；不携带本机的 x86 `.so`。适配器没有 CPU 汇编或固定源码偏移，但依赖 CPython 3.11 的内部结构。
- Python 3.10/3.12 等版本需要额外适配，当前会明确拒绝运行，不会退化成不可靠的名字匹配。

在服务器**原有测试环境**中安装，以下安装不会依赖或替换 torch：

```bash
python -m pip install ./api_test_relations-0.1.2-py3-none-any.whl
api-test-relations doctor
```

也可以进入源码目录执行 `python -m pip install .`。`doctor` 只编译并加载身份适配器，不导入 PyTorch，不证明 NPU 测试已经验证。

## 接入现有测试命令

保留原来可以运行 NPU 测试的环境变量、参数和筛选条件，追加下面几个插件参数即可：

```bash
cd /path/to/pytorch
python -m pytest test/YOUR_EXISTING_NPU_TEST.py \
  --api-relations-out /data/api-relations/run-001 \
  --api-relations-repo "$PWD" \
  --api-relations-import torch_npu \
  --api-relations-label ascend-npu
```

`YOUR_EXISTING_NPU_TEST.py` 替换为服务器上已能运行的测试。`--api-relations-import torch_npu` 只是先导入扩展；`--api-relations-label` 只是元数据。**插件不会把原本的 CPU/CUDA 测试自动改成 NPU 测试**。分布式和不适用测试仍由原命令筛掉；插件不会自行增删测试。

安装后 pytest 自动发现插件，但只有提供 `--api-relations-out` 才启用采集。如果环境设置了 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，另加 `-p api_test_relations.pytest_plugin`。

每次 pytest 使用新的、空的输出目录。此精简版不提供自动重跑、旧基线导入、调度或 pytest-xdist；多个独立 pytest 进程分别使用不同输出目录。内部启动的 Python 子进程和进程池任务由采集桥接跟踪。

默认使用随包台账；可用 `--api-relations-catalog /path/to/catalog.json` 替换。默认排除 `torch.compile`。只解析台账中的 `torch.*` 条目；额外 `torch_npu.*` 台账需要新的命名空间适配，当前会记录 `outside_namespace`。

## 查询

pytest 的 `sessionfinish` 发生在解释器退出之前，长期存活的子进程可能随后才写入分片或回执。等待 pytest 和相关子进程退出后，推荐先生成最终汇总：

```bash
api-test-relations settle --out /data/api-relations/run-001 \
  --destination /data/api-relations/run-001-final
```

`settle` 不执行测试、不覆盖原目录；复制原始证据并重建汇总，记录来源和校验值。原始 gap 保留，过时的派生状态重新计算；空分片明确记为缺口。仍有生产者 PID 存在时拒绝执行，不会擅自终止进程。后续查询可使用 `run-001-final`。

```bash
api-test-relations status --out /data/api-relations/run-001
api-test-relations query --out /data/api-relations/run-001 --api torch.Tensor.fill_ --json fill-relations.json
api-test-relations audit --out /data/api-relations/run-001
```

返回完整参数化 nodeid、`pytorch/` 开头的测试文件、阶段、API ID、同一对象的别名、首个调用点和采集缺口。未观测到关系不等于没有测试调用该 API；`capture_finished` 表示采集会话收尾，不是零遗漏声明。

结束时等待已记录子进程最多 120 秒，可用 `--api-relations-child-wait` 调整。超时不杀服务器进程，保留缺口和原始分片。异常中断保留已写分片；不要将它当成完整采集，或对原目录直接重新运行。

## ARM + 昇腾实机验收

在插件源码目录、已配置好的 NPU 环境执行：

```bash
api-test-relations doctor
python -m pytest examples/test_ascend_smoke.py \
  --api-relations-repo "$PWD" \
  --api-relations-out /data/api-relations/npu-smoke-001 \
  --api-relations-import torch_npu \
  --api-relations-label ascend-npu
python scripts/check_smoke_relations.py --out /data/api-relations/npu-smoke-001
```

这里的根目录是插件源码目录，因此示例路径规范化为 `pytorch/examples/...`，真实测试接入时应指定真正的 PyTorch 根目录。

本次在隔离的 x86 Linux / CPython 3.11 环境验证采集逻辑，真实 ARM/NPU 验收尚未执行。NPU 专属扩展、替换后的绑定和目标测试的采集缺口，需要以上实机检查确认。

0.1.1 修复注册表并发刷新、cProfile 抢占、SQLite 热日志合并三类问题；0.1.2 增加进程退出后的证据重汇总。39 项隔离回归及另外 2 项真实 PyTorch/CUDA 集成检查通过；还进行了真实上游测试对照。修复范围与剩余边界见 `RELEASE_NOTES.md` 和 `docs/limitations.md`。已有基线不会自动恢复或重采。
