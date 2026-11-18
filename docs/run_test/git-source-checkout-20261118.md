# 直接检出同 commit 的 PyTorch 源码

维护：OpenAI Codex，2026-11-18。Assisted-by: OpenAI Codex

上游 `Ascend/pytorch@ci-test` 的 `_build.yml` 和 `_test-collect.yml`
已改为 Git 获取 PyTorch 源码，不再发布对应 OBS 源码 tarball。
继续把源码包作为 wheel 入队的先决条件，会永久等待已经停产的资源。

部署设置 `obs.source.provider: git`。OBS wheel 仍走新桶、匿名 HTTPS；
历史任务中已经绑定的 OBS 源码 URL 仍按原路由下载。新任务直接绑定
`git-source://pytorch/<完整commit>` 和 `git:<commit>:<tree>:checkout-v1`，
不等待源码 tarball，也不打包本地源码 tarball。

主循环先确认 PyTorch/torch-npu wheel 配套完整，随后验证同一完整提交
在本地完整 main 历史中，绑定源码 tree。任务选例和执行准备各自通过
本地共享对象克隆及 detached checkout 获得任务独立源码目录。
复用已有完整仓库中的对象，不按 PR 向 GitHub 重新克隆；镜像增量 fetch
仍由既有上游同步负责。不能用 main 当前文件替代任务 commit，也不能
把 GitCode 的 `Ascend/pytorch`（torch-npu 插件）当成上游 PyTorch 仓库。

任务目录写入 `.run-test-git-source.json`，记录完整 commit、tree、源码
范围和子模块 pins。与上游测试阶段一致，不自动初始化子模块，不用
某分支最新内容替代 pin。`source_scope: tracked-tree`、
`submodule_policy: not_initialized` 明确表示仅检出主仓跟踪内容，
不声称包含所有第三方子模块。执行诊断另存源码身份记录。

源码目录已有其他身份或跟踪文件被修改时明确失败；不能把脏源码当成
原提交复用。新检出先在临时目录完成，确认 HEAD 后再发布到任务路径。
wheel 架构、Python、构建环境、包格式及包内 torch-npu commit 校验保留。

cycle 日志和 fetch-health 增加缺资源分类（wheel、配套 npu、Git 源码），
避免再以笼统的“等待”掩盖供应缺失。历史结果、失败归档和任务身份不改写。
