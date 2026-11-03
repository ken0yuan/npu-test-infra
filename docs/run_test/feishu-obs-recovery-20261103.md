# 飞书发布与 OBS 查询恢复（2026-11-03）

## 飞书

当前未发布版本仅 PR #202503，6 篇计划中的第 5 篇创建请求结果不确定，
`part4-doc_pending` 使后续扫描保持停止，避免重复创建。其余 8 条 staging
记录属于非当前版本，不能合计为 9 个当前发布故障。

本次在发布锁内备份原 journal，对专属版本目录进行两次完整远端回读，
确认仅有已登记的前 4 篇后，记录人工恢复证据并解除这一个标记。
原 6 篇全部续传、内容回读与目标目录对账通过，保留最初北京时间上传日期。
发布前后 PR 原始数据库记录 SHA256 相同，没有重跑历史测试。
创建接口的不确定结果仍保守核对，不对所有 pending 创建做盲目自动重试。

证据：`data/runtime/maintenance/codex-feishu-obs-20261103/` 下的
`feishu-journal-before-recovery.json`、`feishu-remote-reconcile.json`、
`feishu-recovery-result.json`。

## OBS

默认桶域名的地址连接超时，同区域官方服务端点能连接，并可在保留原桶
主机名及证书校验的情况下完成请求。直接查询此前持续 120 秒超时，
并非已经证实对象改名或不存在。

新增可选的 `obs.downloader.regional_proxy_endpoint`：每次 obsutil 命令
临时启动环回 CONNECT 通道，仅接受配置桶的 HTTPS 主机名，将字节流
转交官方区域端点。TLS 与原桶主机名校验仍由 obsutil 完成，通道不解密
流量、不记录请求头或凭据，命令结束关闭连接与监听。
不改变系统 hosts、DNS、全局代理，也不启用其他来源或降低 TLS 校验。

部署配置示例（保持原 bin、bucket 等配置）：

```json
{
  "obs": {
    "bucket": "pytorch-ci-cache",
    "downloader": {
      "type": "obsutil",
      "regional_proxy_endpoint": "https://obs.ap-southeast-1.myhuaweicloud.com"
    }
  }
}
```

实测列表有 1706 个对象，581 个 split-pytorch 与 577 个 split-torchnpu，
全部 wheel 对象能被既有规则解析。已发现的 pending 中有 12 套完整配套包。
其余情况包括真实缺配套成员，以及超出部署构建环境过滤范围；未放宽
Python、架构、构建环境、格式版本或同提交配对要求。

同时记录 OBS 查询健康状态：列表查询失败时显示配套包是否存在未知，
不再将未查询成功的 pending 输出为“等待 0”。Git 成功不掩盖 OBS 失败。
该通道解决当前桶域名地址不通的情况；区域服务端点自身不可用时仍需告警。

Assisted-by: OpenAI Codex
