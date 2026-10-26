"""本地等待空闲 NPU 卡（用户 2026-10-26 要求）：执行前检查卡占用，
卡数量够就运行，不够就轮询等待。

空闲判据（2026-10-26 在 8×910B4 实证两种输出）：
- `npu-smi info -t proc-mem -i <id>` 空闲时含 "No process in device"，
  忙时出现 "Process id:<pid>  Process name:…" 行；
- 查询失败按占用处理，不冒险开跑。

作为模块供 executor 在 accelerator 任务执行前调用；也可独立当脚本：
    python -m run_test.npu_cards --need 4 --timeout 3600
成功把空闲卡 ID 列表打到 stdout（逗号分隔）后 exit 0，超时 exit 1。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

NPU_SMI = "npu-smi"

# `npu-smi info -m` 映射表的 Ascend 芯片行（同 environment.py 口径），
# NPU ID 按 Chip 行去重——容器内 ID 可能不从 0 开始。
_ASCEND_CHIP_ROW = re.compile(r"^\s*(\d+)\s+\d+\s+\S+\s+Ascend\s+\S+\s*$", re.MULTILINE)


def _run(cmd: list[str], *, timeout: float = 60.0) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def npu_ids(smi: str = NPU_SMI) -> list[int]:
    out = _run([smi, "info", "-m"])
    if out is None:
        return []
    return sorted({int(m.group(1)) for m in _ASCEND_CHIP_ROW.finditer(out)})


def card_free(npu_id: int, smi: str = NPU_SMI) -> bool:
    """空闲 = 无进程占用 且 健康 OK。任一查询失败按占用处理，不冒险开跑。"""
    out = _run([smi, "info", "-t", "proc-mem", "-i", str(npu_id)])
    if out is None or "No process in device" not in out or "Process id:" in out:
        return False
    health = _run([smi, "info", "-t", "health", "-i", str(npu_id)])
    if health is None:
        return False
    m = re.search(r"Health\s*:\s*(\S+)", health)
    return m is not None and m.group(1) == "OK"


def free_npu_cards(smi: str = NPU_SMI) -> list[int]:
    return [i for i in npu_ids(smi) if card_free(i, smi)]


def wait_free_cards(need: int = 1, timeout: float = 0.0, interval: float = 30.0,
                    smi: str = NPU_SMI, log=print) -> tuple[list[int], str]:
    """阻塞到空闲卡数 ≥ need。返回 (空闲卡列表, "")；超时返回 ([], 原因)。

    timeout<=0 表示一直等。interval 为轮询间隔（秒，最小 1）。
    可见卡总数 < need 时立即返回诊断（等再久也不会够）。
    """
    if need <= 0:
        raise ValueError("need 必须是正整数")
    total = len(npu_ids(smi))
    if total < need:
        return [], f"需要 {need} 张空闲 NPU 卡，本机仅可见 {total} 张（含被占用），等待不会使其增加"
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    started = time.monotonic()
    while True:
        cards = free_npu_cards(smi)
        if len(cards) >= need:
            return cards, ""
        waited = f"，已等 {int(time.monotonic() - started)}s" if deadline is not None else ""
        log(f"[等待卡] 需 {need} 张空闲 NPU，当前空闲 {len(cards)}/{total}{waited}")
        if deadline is not None and time.monotonic() >= deadline:
            return [], f"等待 {need} 张空闲 NPU 卡超时（{int(timeout)}s）"
        time.sleep(max(interval, 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="等待空闲 NPU 卡：数量足够立即输出卡 ID 并退出，否则轮询等待")
    ap.add_argument("--need", type=int, default=1, help="需要的空闲卡数（默认 1）")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="最长等待秒数；0=一直等（默认）")
    ap.add_argument("--interval", type=float, default=30.0, help="轮询间隔秒数")
    args = ap.parse_args(argv)
    cards, why = wait_free_cards(args.need, args.timeout, args.interval)
    if cards:
        print(",".join(map(str, cards)))
        return 0
    print(why, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
