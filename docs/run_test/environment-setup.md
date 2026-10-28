# 测试环境安装要求

当前服务器为 Ubuntu 22.04 / aarch64 / Python 3.11，测试解释器位于
`/home/workspace-ycy/npu-test-infra/venv/test/bin/python`。控制环境保持独立。
源码、torch、torch_npu、torchvision 继续使用对应 commit 的 OBS 包。

## Triton-Ascend 3.6.0

用户指定使用 `triton-ascend==3.6.0`。共享测试依赖以
[`../requirements-test.txt`](../requirements-test.txt) 为准。该 wheel 要求
`triton==3.6.0`、`numpy==1.26.4`、`scipy==1.13.1`（Python 3.11）、
`attrs==24.2.0`、`decorator==5.1.1`、`psutil==6.0.0`、
`pytest>=8.3.2,<9` 和 `pytest-xdist==3.6.1`；本部署锁定 pytest 8.4.2。

在仓库根目录执行，安装期间须持有流水线的 executor lease，避免改动运行中
任务共用的 Python 环境：

```bash
venv/test/bin/python -m pip config --site set global.extra-index-url \
  https://mirrors.huaweicloud.com/ascend/repos/pypi
venv/test/bin/python -m pip install triton==3.6.0
venv/test/bin/python -m pip install -r docs/requirements-test.txt
venv/test/bin/python run-test/src/run_test/triton_compat.py
```

先安装基础 triton，再安装 Triton-Ascend。两者共享 `triton` 导入目录，
Triton-Ascend 提供 Ascend 后端；安装 Ascend 包后不要单独强制重装基础
triton，否则可能覆盖 Ascend 实现。日常每个 PR 的 `executor.test_requirements`
须保留 `triton-ascend==3.6.0` 以及与其一致的 pytest 版本要求，避免恢复旧的
`pytest==9.0.3` 配置。

安装后须核验两个 distribution 的版本均为 3.6.0、Ascend 后端可导入，
并在一张空闲 NPU 上实际编译运行向量加法，比较 Triton 与 torch 的结果。
仅能 `import triton` 不代表编译器、CANN 或设备路径已验证。

### 当前 PyTorch main 的 C++20 兼容步骤

本机 OBS PyTorch 2.16.0a0 的 `ATen/ATen.h` 明确要求 C++20；官方
Triton-Ascend 3.6.0 的 `triton/backends/ascend/utils.py` 则固定使用
`-std=c++17` 编译运行时辅助扩展，会在真正启动 kernel 前报错。
上述 `triton_compat.py` 仅在该精确版本且 ATen 要求 C++20 时，将这一处
参数改为 `-std=c++20`，保留带原文件 SHA256 的备份；未知编译逻辑拒绝修改。
执行器每次安装完成后自动检查，因此重装 wheel 后也会重新应用；不会修改
OBS 的 torch / torch_npu 包，也不修改被测用例。

2026-10-28 在 Ascend 910B4 上实测：4096 元素 float32 Triton 向量加法通过，
目标后端为 `npu`，与 torch 参考值的最大误差约 `2.38e-7`。基础算子验收不等于
所有 PyTorch main 的编译器私有接口均已兼容。

官方资料：[安装说明](https://triton-ascend.readthedocs.io/en/v3.6.0/installation_guide.html)、
[版本兼容表](https://triton-ascend.readthedocs.io/en/v3.6.0/release_note.html)。
当前 OBS 测试的是 PyTorch main 构建；基础算子验收通过不代表所有 main 分支
Inductor / Triton 私有接口都与 3.6.0 兼容。

## OpenReg 的环境边界

`torch_openreg` 是用于验证 PrivateUse1 后端注册的测试扩展。当前 NPU 环境中，
PrivateUse1 已注册为 `npu`，不能在同一进程再改为 `openreg`。不要为了运行
OpenReg 测试覆盖现有 NPU 注册。此类测试应使用隔离的、不加载 torch_npu 的
环境及配套编译的 OpenReg 扩展；其失败不能直接作为 NPU 算子回归结论。
