# OBS 套包准入配置（2026-11-12）

按用户决定，当前 py311/aarch64 部署移除
`obs.wheel_name_contains` 中的 `-torch-npu-builder-`，有效配置为：

```json
{
  "obs": {
    "wheel_name_contains": ["-py311-", "aarch64"]
  }
}
```

因此 `torch-npu-builder` 和 `torch-npu-test` 命名的套包均参与选择。
`settings.example.json` 同步当前部署的 Python/架构条件；其他部署应按实际环境设置。

沿用现有 `select_bundle`，不改变通用解析和配对算法：

- PyTorch 对象必须匹配该 main 提交的短 SHA，源码对象也必须存在。
- 拆分的 PyTorch、torch-npu 成员必须具有相同 Python、架构、构建环境和格式版本。
- 多套完整候选按可信 OBS 上传时间选取最新完整套包。
- 新出现但尚未齐套的构建不会挤掉已有完整套包。
- 执行端继续核验 wheel 标签、包内 torch-npu 提交与对象名绑定。

这是构建名称准入条件的调整，不代表已经验证新增包中的全部测试。
线上先备份原配置，在 cycle/pipeline 空闲边界更新并重载，使用实际待包提交核对入队。
当前 OBS 路由、历史结果、已归档问题与其他筛选逻辑保持原有行为。

Assisted-by: OpenAI Codex
