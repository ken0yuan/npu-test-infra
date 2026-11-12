# 完整上游历史镜像

维护：OpenAI Codex，2026-11-12。Assisted-by: OpenAI Codex

PyTorch main 镜像首次使用完整 bare clone：不使用 `--filter`、`--depth`
或 `--shallow-since`。只跟踪 main 分支，但保留其完整祖先和文件对象。
以后复用同一个镜像，定时增量 fetch，不为每个 PR 重新克隆。

Git 镜像提供合入身份、父提交、完整 diff 和历史文件内容。OBS 源码包
仍提供与被测构建对应的测试源码；不能把没有 `.git` 的源码快照当成历史。

原来的 `blob:none` 镜像有提交和目录树，却缺少文件内容。`git show`
或涉及文件内容的 diff 会触发按需 fetch；GitHub 不通时，选例因此失败。
新镜像须能在上游不可访问时读取历史文件和正常增量 fetch 得到的新文件。

首次克隆在同级临时目录完成，成功后才发布到正式路径。失败不得留下
可被误认为已就绪的正式镜像。发现已有 partial/shallow 镜像时明确报错，
不再无提示地依赖按需下载。promisor 标记也要检查；仅删除配置不能补齐对象。

## 旧镜像迁移

1. 检查可用空间，另建完整裸镜像。不要使用旧 partial 镜像作为 reference，
   不要手动删除其 promisor 配置或标记。Git 2.34 不支持 `fetch --refetch`，
   普通 fetch 不保证补齐已经协商为“存在”的历史文件对象。
2. 新仓库不得有 shallow 边界、partial/promisor 配置、promisor pack 标记
   或 objects/info/alternates。执行 `git fsck --full --no-reflogs --no-dangling`。
   新 main 必须包含旧 main，并确认队列使用的提交及文件均可本地读取。
3. 等 cycle 和 pipeline 无子任务时暂停两个 worker，持有执行锁后切换；
   在切换前再次检查旧 tip 未变化。旧目录改名保留，完整镜像移入原路径。
   保留仓库自己的 HTTP/1.1 和正式源路由设置，不能复制旧 partial 配置。
4. 同步部署代码并恢复两个 worker，确认新进程使用完整镜像。
   历史队列、日志和测试结果不改写，也不因此重跑已解决的失败。

完整镜像解决选例读取文件时的按需网络请求。定时同步新提交仍需要
GitHub 可达；网络故障应继续明确报告，不能把旧 tip 当作最新提交。
