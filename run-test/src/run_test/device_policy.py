"""Execution policy; historical database CUDA keys remain readable."""
import os
from pathlib import Path
from .config import ConfigError

EXECUTION_DEVICES = ("cpu", "accelerator")

def configured_devices(config):
    devices = (config.get("executor") or {}).get("allowed_devices", list(EXECUTION_DEVICES))
    if not isinstance(devices, list) or not devices:
        raise ConfigError("executor.allowed_devices 必须是非空设备列表")
    for device in devices:
        require_execution_device(device)
    return devices

def require_execution_device(device):
    if device not in EXECUTION_DEVICES:
        raise ConfigError(f"禁止执行设备 {device!r}；只允许 cpu / accelerator")
    return device

def execution_env(device=None):
    if device is not None:
        require_execution_device(device)
    env = dict(os.environ)
    for key in ("PYTEST_PLUGINS", "PYTEST_ADDOPTS", "PYTORCH_TESTING_DEVICE_FOR_CUSTOM",
                "PYTORCH_TESTING_DEVICE_EXCEPT_FOR", "PYTORCH_TESTING_DEVICE_ONLY_FOR"):
        env.pop(key, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Backend registration belongs to the explicitly selected execution channel.
    # accelerator registers torch_npu in pytest_command, inside its card lease.
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = {"cpu":"cpu", "accelerator":"npu"}.get(device, "cpu,npu")
    env["PYTHONPATH"] = os.pathsep.join(x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x)
    return env

def test_execution_env(device, source_root, test_function):
    """Use the same sibling-import path for discovery, preflight and execution.

    Run from test/, never the source checkout root (which would shadow the
    installed torch wheel). Some packaged tests import their siblings absolutely.
    """
    env = execution_env(device)
    if source_root is not None:
        relative = test_function.split("::", 1)[0]
        if relative.startswith("test/"):
            relative = relative[5:]
        test_root = (Path(source_root) / "test").resolve()
        parent = (test_root / relative).resolve().parent
        if parent != test_root and test_root not in parent.parents:
            raise ValueError("test path escapes source test directory")
        entries = [str(parent), *env.get("PYTHONPATH", "").split(os.pathsep)]
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(x for x in entries if x))
    return env


def pytest_command(python, device):
    require_execution_device(device)
    if device == "cpu":
        return [python, "-m", "pytest"]
    # Explicit native NPU registration, with no CUDA transfer or pytest extension.
    return [python, "-c", "import torch, torch_npu, pytest, sys; "
            "assert torch_npu.npu.is_available(), 'NPU unavailable'; "
            "sys.exit(pytest.main(sys.argv[1:]))"]

class ExecutionDeviceResolver:
    def __init__(self, resolver, fixed=None, allowed=None):
        self.resolver = resolver
        self.fixed = require_execution_device(fixed) if fixed else None
        self.allowed = list(EXECUTION_DEVICES) if allowed is None else allowed
        self.diagnostics = []

    def __call__(self, path, node):
        found = self.resolver(path, node)
        history = getattr(self.resolver, "diagnostics", [])
        evidence = dict(history[-1]) if history else {}
        # Legacy generic tests defaulted to CUDA solely for baseline comparability.
        # Generic device parametrization now generates CPU and native NPU variants.
        if evidence.get("basis") == "default" or evidence.get("level") == "default":
            found = ["cpu", "accelerator"]
        selected = [d for d in found if d in self.allowed]
        if self.fixed:
            selected = [self.fixed] if self.fixed in selected else []
        self.diagnostics.append({**evidence, "path":path, "node":node,
            "execution_devices":selected, "excluded_devices":[d for d in found if d not in EXECUTION_DEVICES],
            "policy":"cpu / accelerator only; CUDA is never relabeled"})
        return selected

    def source_test_function(self, path, node):
        return self.resolver.source_test_function(path, node)
