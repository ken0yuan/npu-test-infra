"""失败签名分类：把"环境 bug"从日志海洋变成结构化字段。

规则移植自 ``/home/y00981897/tmp/analyze.py`` 的 ``rules()``（2026-09-16
版）——transfer-CUDA→NPU 全量基线 1297 文件的失败分类产物，25 类、
**顺序敏感、首个命中生效**；前三类（C01/C02/C03）占全部失败约 74%。
类别 ID、匹配语义与判定顺序逐条保持一致，保证与基线期分析可比
（包括 C09 的第二条 shard 规则排在 C24 之后、C20–C24 先于 C17–C19
这些非常规位置）。

**2026-09-20 增补（偏离 analyze.py 原版，逐条注明）**：

- C25-autotune-gpu-type-unsupported：插在 C02 后。43-PR 实跑中 PR 179547
  的 8 个 inductor autotune 失败（`Unsupported gpu_type: npu`）无一命中
  原有签名——已知环境墙必须可结构化。
- C26-cuda-designated-guard-transfer：插在 C10 前。cuda-designated 守卫
  在 transfer 下恒拒绝 npu，mock/capability 断言失败（文本特征仅为
  call_count 断言，**命中需人工复核**，描述里已注明）。
- C10 谓词加负向守卫（call_count / Expected exit code）："Scalars are
  not equal!" 同样出自 assertEqual(call_count)/退出码断言，不能都算
  数值计算差异（实证：PR 195049 device_ok 误标、PR 196922 退出码断言）。

用途（调用方）：

- ``runner``：junit 失败文本逐条分类 → ``FunctionRunOutcome.triage``
  （函数级已知签名并集）；
- ``record``：本次 triage 与旧基线日志的签名对比——同签名 = 基线已知
  环境失败（无新信号），计数变化或新签名 = 回归候选。

语义边界：``C99-other`` 表示"无已知环境签名"——在函数级并集里被剔除
（classify_texts 不含 C99），因为"无签名"恰恰可能是真实代码问题。
"""

from __future__ import annotations

import re
from typing import Callable

CATEGORY_OTHER = "C99-other"


def _has(*needles: str) -> Callable[[str], bool]:
    return lambda t: any(n in t for n in needles)


def _re(pattern: str) -> Callable[[str], bool]:
    compiled = re.compile(pattern)
    return lambda t: bool(compiled.search(t))


# (类别 ID, 判定谓词, 人读说明)。顺序敏感：首个命中生效。
SIGNATURES: list[tuple[str, Callable[[str], bool], str]] = [
    ("C01-torchC-private-cuda-api",
     _has("module 'torch._C' has no attribute '_cuda_"),
     "torch._C 私有 CUDA API 缺失（无 NPU 桩）"),
    ("C02-npu-inductor-cache-clear",
     _has("'function' object has no attribute 'cache_clear'"),
     "torch_npu inductor 后端加载失败（get_gpu_type.cache_clear）"),
    ("C25-autotune-gpu-type-unsupported",
     _has("Unsupported gpu_type:"),
     "inductor autotune 进程池不识别 npu（TuningProcessPool 按字面 gpu_type 建池；2026-09-20 增补，非 analyze.py 原类）"),
    ("C03-pg-no-npu-backend",
     _has("No backend type associated with device type npu"),
     "c10d ProcessGroup 查不到 npu 后端"),
    ("C04-pg-unknown-backend-variant",
     _re(r"Unknown backend type hccl"),
     "PG 后端名新变体未映射（hccl2/hccl-lazy，boot v1 只修了 hccl-legacy）"),
    ("C05-gloo-unsupported-npu",
     lambda t: "ProcessGroupGloo::" in t and "unsupported device type npu" in t,
     "Gloo 进程组不支持 npu 张量"),
    ("C06-hccl-comm-init",
     _has("HCCL function error"),
     "HCCL 通信域初始化失败"),
    ("C07-aclnn-op-error",
     _has("NPU function error: call aclnn"),
     "aclnn 算子执行失败（NPU kernel error）"),
    ("C08-c10d-op-unsupported",
     lambda t: "not currently supported on the NPU backend" in t
     or bool(re.search(r"Backend hccl does not support", t)),
     "c10d 算子在 NPU/HCCL 后端未实现"),
    ("C09-device-assert-cuda-vs-npu",
     lambda t: bool(re.search(r"!= device\(type='cuda'", t))
     or "'npu' != 'cuda'" in t or "'cuda' != 'npu'" in t
     or bool(re.search(r"Expects (the parameter to already be moved to )?device cuda:?\d* but got npu", t))
     or "but found one of them on device: npu" in t
     or "but found at least one of them on device: npu" in t
     or bool(re.search(r"on device cuda:?\d* \(.{0,40}\) but found", t))
     or bool(re.search(r"Expected one of.*cuda.*but got npu", t))
     or "Expects cuda:0 but got npu:0" in t,
     "设备断言失败：测试期望 cuda，transfer 后实际 npu"),
    ("C26-cuda-designated-guard-transfer",
     _has("call_count"),
     "cuda-designated 能力守卫 × transfer：device_ok 型守卫见 npu 恒拒绝、"
     "mock/capability 断言失败（文本证据仅 call_count 断言特征，命中需人工复核；"
     "2026-09-20 增补，非 analyze.py 原类）"),
    ("C10-numeric-mismatch",
     # call_count / 子进程退出码断言也产 "Scalars are not equal!"，但语义不是
     # 数值计算差异——负向守卫防止误标（2026-09-20，实证 PR 195049/196922）。
     lambda t: ("Tensor-likes are not close!" in t or "Max difference:" in t
                or ("Scalars are not equal!" in t
                    and "call_count" not in t and "Expected exit code" not in t)),
     "数值结果与期望不一致（NPU 计算差异）"),
    ("C11-no-cuda-linkage",
     _has("PyTorch is not linked with support for cuda devices",
          "PyTorch was compiled without CUDA support"),
     "原生路径/RPC 子进程报 PyTorch 未链接 CUDA"),
    ("C12-rpc-tensorpipe",
     lambda t: "device map" in t or "tensorpipe" in t.lower()
     or "_faulty_tensorpipe_construct" in t,
     "RPC/tensorpipe 不兼容（设备映射/后端参数/超时）"),
    ("C13-pg-reinit",
     _has("trying to initialize the default process group twice"),
     "default 进程组重复初始化（单进程串跑无 spawn 的编排差异）"),
    ("C14-stream-api",
     _has("Uninitialized default NPU stream", "Failed to create StreamBlock",
          "'StreamContext' object has no attribute"),
     "NPU stream API 不兼容/未初始化"),
    ("C15-profiler-args",
     lambda t: "_prepare_profiler()" in t
     or "'profile' object has no attribute 'events'" in t
     or "kineto" in t.lower(),
     "torch.profiler/kineto 在 NPU 下不兼容（参数/事件接口）"),
    ("C16-jit-script-disabled",
     lambda t: "Freezing expects a ScriptModule" in t
     or ("torch.jit.script" in t and "disabled" in t)
     or ("'function' object has no attribute 'graph'" in t
         and "_register_jit_decomposition_for_jvp" in t)
     or ("decompositions_for_jvp" in t and "has no attribute 'graph'" in t),
     "transfer_to_npu 禁用 jit.script（jvp 分解/Freezing 连带失败）"),
    # 原实现顺序：C20–C24 在 C17–C19 之前（顺序敏感，勿"整理"）。
    ("C20-capability-none",
     lambda t: ("'>=' not supported between instances of 'NoneType' and 'int'" in t
                and "get_gpu_type" in t)
     or ("get_device_capability" in t and "NoneType" in t),
     "NPU get_device_capability 返回 None（inductor comm_analysis 崩）"),
    ("C21-aclnn-sleep-missing",
     _has("aclnnSleep"),
     "aclnnSleep 不在 libopapi.so（torch.cuda._sleep 无 NPU 实现）"),
    ("C22-backend-string-unmapped",
     lambda t: bool(re.search(r"Backend [\w:,]+ not supported!", t))
     or "'custom' != 'nccl'" in t,
     "backend 字符串 cuda:nccl 未改写 / 后端名断言失败"),
    ("C23-test-parametrize",
     lambda t: bool(re.search(r"missing \d+ required positional argument: 'device'", t))
     or bool(re.search(r"missing \d+ required positional arguments?: 'device' and 'dtype'", t)),
     "测试参数化未注入 device/dtype 参数（ao/sparsity）"),
    ("C24-double-grad-dtype",
     lambda t: "grad_dtype 'Double'" in t or ("grad_dtype" in t and "Double" in t),
     "double 精度 grad_dtype 不匹配（NPU double 支持差异）"),
    # C09 第二条独立规则（shard 元数据）——原实现排在 C24 之后。
    ("C09-device-assert-cuda-vs-npu",
     _re(r"local shard tensor device: npu:\d+, local shard metadata placement device: cuda"),
     "设备断言失败：期望 cuda 实际 npu（shard 元数据）"),
    ("C17-quantized-x86-engine",
     _has("Didn't find engine for operation quantized::"),
     "量化算子缺引擎（仅 X86 实现）"),
    ("C18-openreg-backend-missing",
     _has("openreg"),
     "torch_openreg 私有后端未安装（accelerator API 被 NPU 抢占）"),
    ("C19-cuda-host-register",
     _has("cudaHostRegister"),
     "cudaHostRegister 不可用（pinned-memory 注册为 None）"),
]


def classify_text(text: str) -> str:
    """单段文本 → 首个命中的类别 ID；无命中 C99-other（与 analyze.py 一致）。"""
    for category, predicate, _ in SIGNATURES:
        if predicate(text):
            return category
    return CATEGORY_OTHER


def classify_texts(texts: list[str]) -> list[str]:
    """逐条首命中后取函数级并集；C99 剔除（无已知签名 ≠ 环境噪声）。"""
    seen: list[str] = []
    for text in texts:
        category = classify_text(text)
        if category != CATEGORY_OTHER and category not in seen:
            seen.append(category)
    return seen
