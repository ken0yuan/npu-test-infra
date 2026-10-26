# 输出规范

每次 pytest 独立建立一个目录：

```text
run-001/
├── manifest.json           # 仓库、平台、解释器、台账/采集器哈希、排除项
├── environment.json        # torch / 显式导入扩展的版本及加载路径
├── test-sources.json        # 测试源文件路径、SHA256、完整参数化实例名
├── api-catalog.json         # 此次使用的冻结台账
├── relations.sqlite        # 汇总关系，不保存测试断言/结果
├── progress.json           # 采集状态、计数
├── capture-error.json      # 仅发生采集初始化/合并异常时存在
├── runtime/                # 此次冻结代码和当前 CPU 编译的身份适配器
└── shards/
    ├── <pid>[-token].sqlite # 进程独立原始证据
    ├── launches/           # 启动命令和父测试归属
    ├── receipts/           # 子进程启动/结束回执
    ├── tasks/              # 进程池任务提交归属
    ├── task-receipts/      # worker 执行回执
    └── sqlite-recovery/   # 0.1.1：恢复前原件、副本与恢复核对记录
```

主关系键为 `(job, nodeid, phase, api_id)`，此独立插件一次运行的 job 为 `pytest`。`phase` 是 setup/call/teardown；分别保存，查询可判断关系发生在正文还是 fixture。
`aliases` 表示当时解析为同一 callable 的台账名称，并不证明用户采用了其中每一个拼写。`site` 是首个证据调用位置，不是每次调用的完整轨迹。

`tests` 保存完整 nodeid 和文件；`phases` 保存 entered/closed；`registry` 保存精确解析、未解析、排除、原生符号等状态；`gaps` 保存采集问题。
`processes`、`process_details`、`process_tasks` 保留因果归属。启动记录本身不代表采集成功。
`retained` 表沿用原存储格式但本版始终为空；本版不导入旧基线。

查询结果中的 `unobserved_means_no_test` 永远为 false。只读 audit 检查孤立关系、台账和路径，不证明全量零漏采，也不替代原始分片逐项核对。

代码、台账和源码位置均按实际运行生成/解析；没有按某个 API 的固定源文件行号建立关系。不同 PyTorch/torch_npu 版本会产生不同解析状态，必须各自保存运行证据，不混用结果。

0.1.2 的 `settle` 输出是独立派生目录，增加 `settlement.json`（原目录、原 manifest 校验值、原进度、复制文件校验值、汇总实现版本）。manifest 的 `package_version` 仍表示原采集器版本；`finalizer_version` 单独表示重汇总代码版本。原始分片不被修改，原摘要也保留。`capture_pid` 用于拒绝对仍在执行的采集进程收尾。
