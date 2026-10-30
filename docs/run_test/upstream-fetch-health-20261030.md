# 上游抓取重试与健康状态

2026-10-30 核查发现，服务器 Git 全局 insteadOf 将 GitHub 请求转给
ghfast.top；该代理间歇连接超时。原实现单次 fetch 失败就退出本轮，
常驻循环随后等待一小时；失败发生在 OBS 配套包检查之前，因此已发现、
等待配套包的提交也无法在这一轮继续检查。

`queue_warning` 仅统计最后入队时间。队列数量没有变化不能证明上游没有
新提交，抓取失败期间应明确标为“最新提交是否积压未知”。

本次修改：

- 每轮默认最多尝试 3 次 fetch，失败后的等待为 5、10 秒；单次 fetch
  超时默认 180 秒。只重试连接、DNS、超时和部分 HTTP/传输错误。
  认证、仓库身份、引用错误直接报告。配置位于 upstream 的
  fetch_attempts、fetch_retry_seconds、fetch_timeout_seconds。
- 将独立抓取回执原子写入 data_root/upstream-fetch-health.json，记录
  本轮尝试、最近成功时间和 tip、最近失败、连续失败轮数。成功抓取即使
  没有新提交也会刷新健康时间；失败不会前移队列中的镜像观察位置。
- bug_monitor 与 brief 增加 upstream_health/upstream_warning，抓取失败
  无需等待两小时无入队即可体现；成功记录超过轮询间隔的 1.5 倍也报警。
  未生成新回执时，使用 mirror_state.last_fetch_at 判断过期。
- 保留原 Git 路由与上游身份，不修改全局代理，不引入未经验证的第三方源。
  网络中断仍可能耗尽重试；本次不声称已经消除外部代理故障。

本次不改变历史测试结果、已解决归档、测试基线或 NPU 状态，也不重跑历史
PR。维护回归使用隔离数据库和本地 Git 仓库，不依赖外网抓取。

Assisted-by: OpenAI Codex
