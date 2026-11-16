# Wheel 与源码分桶发布

维护：OpenAI Codex，2026-11-16。Assisted-by: OpenAI Codex

构建方已迁移 wheel 发布位置。旧桶中没有新 wheel，不能据此判定上游
没有产包。入口为 Ascend/pytorch 的 `pytorch_ci_trigger_pr.yml`，调用
`ci-test` 分支的 `receive-trigger.yml`。

| 资源 | 前缀 | 下载方式 |
| --- | --- | --- |
| PyTorch/torch-npu wheels | `obs://pytorch-torchnpu/7day-cache/` | 匿名 HTTPS，`obs.cn-southwest-2.myhuaweicloud.com` |
| 配套 PyTorch 源码 | `obs://pytorch-ci-cache/7day-cache/` | 原 obsutil 凭据和原区域路由 |

`obs.wheels` 配置新 wheel 桶及 `public_https` 下载器；`obs.source`
指定源码桶并复用旧 `obs.downloader`。原 `obs.bucket` 和下载方式保留，
已有队列中的旧 wheel URL 仍按其实际桶下载。未配置分桶时保持旧行为。
详见根目录 `settings.example.json`。无需把旧凭据发送到新桶，也无需新密钥。

cycle 和 obs-scan 分别完整获取两个命名空间的列表，再合并快照。
任何一侧列表失败时不发布部分成功的快照，不把查询失败记成“对象不存在”。
匿名列表支持 XML 命名空间及分页；HEAD 的 404 与认证/网络错误分别处理。
下载通过临时文件完成，校验非空和响应长度后再替换目标文件。

配对仍严格使用同一个 PyTorch commit 前缀；torch-npu 必须同时匹配
PyTorch 前缀、Python、架构、构建环境和 vN。源码只查旧桶对应前缀，
不同提交或新桶中的同名源码不能补位。完整套包排序继续使用远端上传
时间，不能用本地发现时间替代。上游 main 身份核实、包内 torch-npu
完整 commit 回写、选例证据、有效名额补位和执行锁均保持原流程。

真实验证使用 PyTorch `cebf82915f`、torch-npu
`147abdd56fc0f0a64d6dd9cec34110db63d03607`。参考校验记录位于服务器
`/home/workspace-ycy/handoffs/nightly-source-migration-20261116/verified-data/verified-build.json`。
验证在独立目录进行，核对双桶发现、HEAD、真实下载、归档和 wheel
内容身份、配套源码 SHA-256 及隔离队列入队；不切换正在使用的测试环境，
不重跑历史测试，也不改写其结果。
