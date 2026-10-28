"""Device readiness in an isolated child, after the caller owns card leases."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

RESULT_PREFIX = "NPU_READINESS_RESULT:"
DEVICE_PREFIX = "NPU_READINESS_DEVICE:"
PROCESS_PREFIX = "NPU_READINESS_PROCESS:"
PROBE_CODE = r'''
import os, sys, json, traceback
print("NPU_READINESS_PROCESS:" + str(os.getpid()), flush=True)
checked = []
device = None
phase = "import"
try:
    import torch, torch_npu
    phase = "availability"
    assert torch_npu.npu.is_available(), "NPU unavailable"
    expected = int(sys.argv[2])
    assert torch_npu.npu.device_count() == expected, "NPU visible mask/count mismatch"
    for device in json.loads(sys.argv[1]):
        print("NPU_READINESS_DEVICE:" + str(device), flush=True)
        phase = "set_device"
        torch_npu.npu.set_device(device)
        phase = "current_stream"
        torch_npu.npu.current_stream()
        phase = "allocation"
        x = torch.ones((2, 2), device="npu:" + str(device))
        phase = "compute"
        y = x @ x
        phase = "synchronize"
        torch_npu.npu.synchronize(device)
        assert float(y.sum().item()) == 8.0, "NPU readiness result mismatch"
        checked.append(device)
except Exception as error:
    traceback.print_exc()
    print("NPU_READINESS_RESULT:" + json.dumps(dict(ok=False, checked=checked,
          failed_logical_device=device, phase=phase, error=str(error))), flush=True)
    sys.exit(73)
print("NPU_READINESS_RESULT:" + json.dumps(dict(ok=True, checked=checked,
      failed_logical_device=None, phase="complete", error="")), flush=True)
'''

_TRANSIENT = re.compile(r"507033|TsdOpen|TSD.*(?:fail|closed)|HDC.*(?:fail|closed)|CreateSession.*fail", re.I)
_ENVIRONMENT = re.compile(r"507033|TsdOpen|HDC|TSD|NPU unavailable|visible mask/count mismatch|No module named ['\"](?:torch|torch_npu)", re.I)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    reason: str = ""
    log_text: str = ""
    failed_cards: tuple = ()
    execution_status: str = "env_init_error"


class NpuReadiness:
    def __init__(self, python, root, *, executor_fds=(), timeout=300, cache_seconds=60,
                 retry_delay=1, run=None):
        self.python, self.root = python, Path(root)
        self.executor_fds = tuple(executor_fds)
        self.timeout, self.cache_seconds = timeout, cache_seconds
        self.retry_delay, self.run = retry_delay, run or subprocess.run
        self._lock = threading.Lock()
        self._ready = {}

    def _cann_logs(self, pid, destination, env):
        """Copy only this child PID's logs; never sweep other running tests."""
        if not pid:
            return "", []
        roots = [Path.home() / "ascend/log"]
        if env.get("ASCEND_PROCESS_LOG_PATH"):
            roots.append(Path(env["ASCEND_PROCESS_LOG_PATH"]))
        content, saved = [], []
        for root in dict.fromkeys(roots):
            for part in ("debug/plog", "run/plog", "plog"):
                for source in sorted((root / part).glob(f"plog-{pid}_*.log")):
                    try:
                        target = destination / (part.replace("/", "-") + "-" + source.name)
                        shutil.copy2(source, target)
                        content.append(source.read_text(errors="replace"))
                        saved.append(str(target))
                    except OSError:
                        continue
        return "\n".join(content), saved

    def ensure(self, lease, env):
        cards = tuple(lease.cards)
        if not cards:
            raise ValueError("NPU readiness requires a nonempty card lease")
        if env.get("ASCEND_RT_VISIBLE_DEVICES") != ",".join(map(str, cards)):
            raise ValueError("NPU readiness mask does not match the card lease")
        now = time.monotonic()
        with self._lock:
            logical = [i for i, card in enumerate(cards)
                       if now - self._ready.get(card, float('-inf')) >= self.cache_seconds]
        if not logical:
            return ProbeResult(True)
        logs = []
        for attempt in range(2):
            destination = self.root / uuid.uuid4().hex
            destination.mkdir(parents=True, exist_ok=True)
            cmd = [self.python, "-c", PROBE_CODE, json.dumps(logical), str(len(cards))]
            options = ({"pass_fds": tuple(dict.fromkeys((*self.executor_fds, *lease.fds)))}
                       if os.name != "nt" else {})
            try:
                proc = self.run(cmd, env=env, capture_output=True, text=True,
                                timeout=self.timeout, check=False, **options)
                rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
            except subprocess.TimeoutExpired as error:
                rc, out, err = None, error.stdout or "", error.stderr or ""
                err = (err.decode(errors="replace") if isinstance(err, bytes) else err) + "\nNPU readiness timeout"
            except OSError as error:
                rc, out, err = None, "", str(error)
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            record, pid, entered = {}, None, None
            for line in out.splitlines():
                try:
                    if line.startswith(RESULT_PREFIX):
                        value = json.loads(line[len(RESULT_PREFIX):])
                        if isinstance(value, dict):
                            record = value
                    elif line.startswith(PROCESS_PREFIX):
                        pid = int(line[len(PROCESS_PREFIX):])
                    elif line.startswith(DEVICE_PREFIX):
                        entered = int(line[len(DEVICE_PREFIX):])
                except (ValueError, TypeError):
                    continue
            checked = record.get("checked", [])
            checked = checked if isinstance(checked, list) and all(type(i) is int and i in logical for i in checked) else []
            success = rc == 0 and record.get("ok") is True and set(checked) == set(logical)
            if not success and not record:
                err += "\nNPU readiness did not return a valid completion receipt"
            cann, paths = self._cann_logs(pid, destination, env) if not success else ("", [])
            detail = cann + "\n" + err + "\n" + out
            proof = dict(phase="probe", physical_cards=cards, logical_devices=logical,
                         visible_devices=env["ASCEND_RT_VISIBLE_DEVICES"], attempt=attempt + 1,
                         pid=pid, returncode=rc, result=record, stdout=out, stderr=err,
                         cann_logs=paths, completed=success)
            (destination / "result.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
            logs.append(f"NPU readiness attempt {attempt + 1}, cards={cards}, evidence={destination}\n{detail}")
            with self._lock:
                for i in checked:
                    self._ready[cards[i]] = time.monotonic()
            if success:
                return ProbeResult(True, log_text="\n\n".join(logs))
            failed = record.get("failed_logical_device", entered)
            failed_cards = (cards[failed],) if type(failed) is int and failed in logical else ()
            if attempt == 0 and _TRANSIENT.search(detail):
                time.sleep(self.retry_delay)
                continue
            first = next((line.strip() for pattern in (r"HDC|TsdOpen|CreateSession", r"507033|NPU unavailable")
                          for line in detail.splitlines() if re.search(pattern, line, re.I)), None)
            reason = first or str(record.get("error") or err).strip()[-700:]
            category = "env_init_error" if _ENVIRONMENT.search(detail) else "npu_probe_error"
            return ProbeResult(False, reason or "NPU readiness failed", "\n\n".join(logs), failed_cards, category)
