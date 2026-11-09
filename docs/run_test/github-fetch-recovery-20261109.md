# GitHub 上游抓取恢复（2026-11-09）

服务器镜像保持直连官方 PyTorch GitHub。近期既有连接超时，也有
`curl 16 Error in the HTTP2 framing layer` 与 `HTTP/2 stream ... not closed cleanly`。
后两种错误没有命中既有瞬时错误规则，导致本应重试的轮次提前退出。

本次将 HTTP/2 中断及 curl 16、18、56、92 传输错误纳入原有有限重试。
仍保留最大尝试次数、退避、失败告警和上次成功提交；认证等永久错误不重试。
部署时仅在 PyTorch 裸镜像设置 `http.version=HTTP/1.1`，原配置先备份。
该设置减少 HTTP/2 故障，不能保证不可达的网络自动恢复。

恢复期间从本机可访问的官方 GitHub 获取 main，并按 SHA256 校验增量包、
验证祖先关系，补入 `50d50e7229` 之后的两个首父提交，镜像前进到 `cbee6335ae`。
正常 cycle 登记两个提交，其中 PR #202698 的配套 OBS 包齐全并成功入队。
这次中转记录独立保留；只有服务器真实抓取成功才能清除网络失败状态。
被测包和源码继续从 OBS 获取，历史测试不重跑。

证据目录：`data/runtime/maintenance/codex-github-recovery-20261109/`。

Assisted-by: OpenAI Codex
