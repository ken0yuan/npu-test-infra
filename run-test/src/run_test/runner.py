"""Native CPU/accelerator pytest runner with built-in JUnit results."""

from __future__ import annotations
from .device_policy import require_execution_device, test_execution_env, pytest_command

import ast
import os
import json
import hashlib
import subprocess
import tempfile
import signal
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigPendingError
from .identifiers import (
    Counts,
    class_variant_match,
    method_variant_match,
    validate_device,
    validate_test_function,
)


@dataclass(frozen=True)
class FunctionRunOutcome:
    test_function: str
    device: str
    completed: bool
    counts: Counts | None
    log_text: str
    returncode: int | None
    # 已知环境失败签名（triage.py 25 类）的函数级并集；空 = 无已知签名
    # （可能是真实代码问题）。仅 completed 时有意义。
    triage: tuple[str, ...] = ()
    execution_status: str = "completed"
    artifact_dir: str | None = None
    coverage_status: str = "unknown"
    instance_ids: tuple[str, ...] = ()
    environment_fingerprint: dict | None = None
    observed_counts: Counts | None = None
    event_reports: tuple[dict, ...] = ()
    provenance: dict | None = None

    @classmethod
    def incomplete(cls, test_function: str, device: str, log_text: str,
                   returncode: int | None = None, execution_status: str = "runner_error",
                   artifact_dir: str | None = None, observed_counts: Counts | None = None,
                   event_reports: tuple[dict, ...] = (),
                   environment_fingerprint: dict | None = None,
                   provenance: dict | None = None) -> "FunctionRunOutcome":
        return cls(
            test_function=test_function, device=device, completed=False,
            counts=None, log_text=log_text, returncode=returncode,
            execution_status=execution_status, artifact_dir=artifact_dir,
            observed_counts=observed_counts, event_reports=event_reports,
            environment_fingerprint=environment_fingerprint,
            provenance=provenance,
        )


def parse_junit_counts(xml_path: Path) -> Counts:
    """解析 junitxml 并聚合为四项计数。

    error / failure / skipped 子元素分别对应 error / fail / skip，
    其余 testcase 记为 pass。parametrize 的每个子用例是一条 testcase。
    """
    root = ET.parse(xml_path).getroot()
    # pytest emits separate testcase elements for call failure and teardown
    # error. They are one runtime instance, so aggregate by classname/name and
    # apply error > failure > skip > pass precedence while retaining both
    # tracebacks in parse_junit_failure_texts/artifacts.
    grouped: dict[tuple[str, str], set[str]] = {}
    for tc in root.iter("testcase"):
        key = (tc.get("classname", ""), tc.get("name", ""))
        grouped.setdefault(key, set()).update(child.tag for child in tc)
    pass_c = fail_c = err_c = skip_c = 0
    for tags in grouped.values():
        if "error" in tags:
            err_c += 1
        elif "failure" in tags:
            fail_c += 1
        elif "skipped" in tags:
            skip_c += 1
        else:
            pass_c += 1
    return Counts(pass_c, fail_c, err_c, skip_c)


def parse_junit_failure_texts(xml_path: Path) -> list[str]:
    """提取 junitxml 里 error/failure 子用例的文本（message + 正文）。

    供 triage 签名分类——比 stdout/stderr 更精确（逐子用例定位）。
    """
    root = ET.parse(xml_path).getroot()
    texts: list[str] = []
    for tc in root.iter("testcase"):
        for child in tc:
            if child.tag in ("error", "failure"):
                message = child.get("message") or ""
                body = (child.text or "").strip()
                texts.append(f"{message}\n{body}" if message else body)
    return texts


def _merge_pythonpath(boot_dir: str) -> str:
    """boot 目录前置到现有 PYTHONPATH；剥空项防止尾部冒号炸 import
    （空路径项 = 当前目录，会让测试树内 import torch 解析错位）。"""
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return os.pathsep.join([boot_dir, *existing])



_COMMON_DEVICE_ENV = ("PYTORCH_TESTING_DEVICE_ONLY_FOR", "PYTORCH_TESTING_DEVICE_EXCEPT_FOR", "PYTORCH_TESTING_DEVICE_FOR_CUSTOM")

# 家族级发现/执行必须与通道同设备集（P2-3）。accelerator 通道的 instantiate
# 设备名 = PrivateUse1 改名后的后端名，本机实证：
#   venv310/bin/python -c "import torch; print(torch._C._get_privateuse1_backend_name())"
# 输出 "npu"（torch_npu 注册）。若换机/换后端，重跑上面命令确认。
_CHANNEL_ONLY_FOR = {"cpu": "cpu", "accelerator": "npu"}

def _boot_fingerprint(boot_dir: str | None) -> dict | None:
    """Return a stable boot fingerprint without leaking absolute paths."""
    if not boot_dir:
        return None
    root = Path(boot_dir)
    if not root.is_dir():
        return {"missing": True}
    digest = hashlib.sha256()
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel_path = path.relative_to(root)
        if "__pycache__" in rel_path.parts or path.suffix == ".pyc":
            continue
        if path.suffix not in {".py", ".json", ".toml", ".yaml", ".yml", ".txt"} and path.name != "BOOT_MANIFEST":
            continue
        rel = rel_path.as_posix()
        digest.update(rel.encode())
        digest.update(path.read_bytes())
        files.append(rel)
    return {"sha256": digest.hexdigest(), "files": files}

def _environment_fingerprint(device: str, boot_dir: str | None) -> dict:
    return {
        "channel": device,
        "boot": _boot_fingerprint(boot_dir) if device == "cuda" else None,
        "python_version": "unknown",
        "actual_backend": "unknown",
    }


def declared_class_methods(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, set[str]]:
    """Read declarations and local imported bases without importing test code."""
    path = path.resolve()
    if path in _seen:
        return {}
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return {}
    external = {}
    needed = {base.id for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
              for base in node.bases if isinstance(base, ast.Name)}
    roots = [path.parent]
    for parent in path.parents:
        if parent not in roots:
            roots.append(parent)
        if parent.name == "test":
            roots.append(parent.parent)
            break
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        aliases = [alias for alias in node.names if (alias.asname or alias.name) in needed]
        if not aliases:
            continue
        search = roots
        if node.level:
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            search = [base]
        module = Path(*node.module.split("."))
        candidates = [file for base in search for file in
                      [base / module.with_suffix(".py"), base / module / "__init__.py"]]
        imported = next((file for file in candidates if file.is_file()), None)
        if imported is not None:
            methods = declared_class_methods(imported, _seen | {path})
            for alias in aliases:
                external[alias.asname or alias.name] = methods.get(alias.name, set())
    return declared_methods_from_source(source, external)



def declared_methods_from_source(source: str, external=None) -> dict[str, set[str]]:
    tree = ast.parse(source)
    def scope_nodes(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child
            else:
                yield from scope_nodes(child)
    declarations = list(scope_nodes(tree))
    classes = {n.name: n for n in declarations if isinstance(n, ast.ClassDef)}
    def methods(name, visiting):
        if name in visiting:
            return set()
        if name not in classes:
            return set((external or {}).get(name, ()))
        node = classes[name]
        names = {n.name for n in scope_nodes(node)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for base in node.bases:
            if isinstance(base, ast.Name):
                names.update(methods(base.id, visiting | {name}))
        return names
    return {"": {n.name for n in declarations if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))},
            **{name: methods(name, set()) for name in classes}}


def declared_node_match(declared, source_class, source_method, cls, method):
    """Resolve every variant against declarations, not only an empty strict match."""
    if source_class != cls:
        if not source_class or not class_variant_match(source_class, cls):
            return False
        # A real class named TestFooCPU is not automatically a variant of TestFoo.
        if cls in declared:
            return False
    if not declared:
        return method_variant_match(source_method, method)
    pool = declared.get(cls, declared.get(source_class, set()))
    if source_method not in pool:
        return False
    owners = [name for name in pool if method == name or method.startswith(name + "_")]
    return bool(owners) and max(owners, key=len) == source_method


_COLLECTION_METADATA_PREFIX = "RUN_TEST_ORIGINS="
# Standard pytest collection, followed by Python reflection in the same process.
# No pytest plugin/hook and no test execution. functools.wraps/__wrapped__ retain
# the original identity of PyTorch's parametrize/device-generated methods. This
# also handles inherited methods from imported bases, which local AST cannot.
_COLLECT_WITH_ORIGINS = r"""
import inspect, json, pathlib, sys
import pytest
if sys.argv.pop(1) == "accelerator":
    import torch, torch_npu
    assert torch_npu.npu.is_available(), "NPU unavailable"
target = pathlib.Path(sys.argv[1]).resolve()
rc = pytest.main(sys.argv[1:])
origins = {}
source_info = {}
def remember(key, value, owner=None):
    try:
        if isinstance(value, (classmethod, staticmethod)):
            value = value.__func__
        original = inspect.unwrap(value)
        required = getattr(value, "num_required_devices", None)
        if required is None:
            required = getattr(original, "num_required_devices", None)
        # Thousands of generated nodes often unwrap to the same function.
        # Reflect its source once; missing source never erases its identity.
        if required is None:
            try:
                if original not in source_info:
                    parameters = inspect.signature(original).parameters
                    source = inspect.getsource(original)
                    sensitive = any(token in source for token in (
                        "device_count", "world_size", "init_process_group", "spawn(",
                        "get_all_devices", "set_device", "npu:1", "cuda:1"))
                    source_info[original] = "device" in parameters and not sensitive and "distributed" not in str(target)
                if source_info[original]:
                    required = 1
            except (AttributeError, TypeError, ValueError, OSError):
                pass
        if type(required) is not int or required < 1:
            required = None
        origins[key] = {"name": getattr(original, "__name__", ""),
                        "qualname": getattr(original, "__qualname__", ""),
                        "source_file": getattr(getattr(original, "__code__", None), "co_filename", None),
                        "declared_device": getattr(owner, "device_type", getattr(owner, "device", None))
                            if isinstance(getattr(owner, "device_type", getattr(owner, "device", None)), str) else None,
                        "required_npu_cards": required}
    except (AttributeError, TypeError, ValueError, OSError):
        origins[key] = {"name": "", "qualname": ""}
for module in tuple(sys.modules.values()):
    filename = vars(module).get("__file__") if module is not None else None
    if not isinstance(filename, str) or pathlib.Path(filename).resolve() != target:
        continue
    for name, value in tuple(vars(module).items()):
        if inspect.isclass(value):
            for method in dir(value):
                if method.startswith("test"):
                    remember(name + "::" + method, inspect.getattr_static(value, method), value)
        elif name.startswith("test") and callable(value):
            remember(name, value)
print("\nRUN_TEST_ORIGINS=" + json.dumps(origins, sort_keys=True))
sys.exit(rc)
"""


class PytestRunner:
    """逐函数运行 pytest，device 参数由配置映射给出。"""

    def __init__(
        self,
        python_env: str | None,
        device_args: dict[str, list[str]] | None = None,
        extra_args: list[str] | None = None,
        timeout_seconds: int | None = None,
        transfer_boot_dir: str | Path | None = None,
        artifact_root: str | Path | None = None,
    ):
        if not python_env or not str(python_env).strip():
            raise ConfigPendingError(
                "executor.test_python_env 未配置：测试 Python 环境的具体路径"
                "需在实施配置中明确，不得默认使用系统 Python"
            )
        self.python_env = str(python_env)
        self.device_args = {
            k: list(v or []) for k, v in (device_args or {}).items()
        }
        self.extra_args = list(extra_args or [])
        self.timeout_seconds = timeout_seconds
        # transfer boot（sitecustomize）目录：仅 device=cuda 使用；
        # cpu / accelerator 直接跑 test 源码，不注入（用户 2026-09-17 决策）。
        self.transfer_boot_dir = None  # historical argument accepted but never injected
        self.artifact_root = Path(artifact_root) if artifact_root else None
        self._collection_lock = threading.RLock()
        self._collection_cache = {}
        self._runtime_requirements = {}
        self._file_receipts = {}
        self._task_collection_enabled = False
        self.process_lease_fds = ()

    def _lease_options(self, fds=()):
        return {"pass_fds": tuple(set(self.process_lease_fds + tuple(fds)))} if os.name != "nt" else {}

    def begin_task(self):
        """Receipts never survive a PR/wheel installation boundary."""
        with self._collection_lock:
            self._collection_cache.clear()
            self._runtime_requirements.clear()
            self._file_receipts.clear()
            self._task_collection_enabled = True

    def required_npu_cards(self, nodeid, device, total_cards):
        if device == "cpu":
            return 0
        return self._runtime_requirements.get((device, nodeid)) or total_cards

    def _collect_file(self, tf, dev, root, rel_file, extra_env):
        env = test_execution_env(dev, root, tf)
        env.update(extra_env or {})
        cmd = [self.python_env, "-c", _COLLECT_WITH_ORIGINS, dev, rel_file, "--collect-only",
               "-q", "-o", "addopts=", "-p", "no:cacheprovider",
               *self.device_args.get(dev, []), *self.extra_args]
        source = root / "test" / rel_file
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else "missing"
        key = (str(root.resolve()), rel_file, digest, tuple(cmd), tuple(sorted(env.items())))
        # Single-flight: threads targeting the same file/mask share one actual
        # collection. Bound to this runner's task, interpreter, args and full env.
        with self._collection_lock:
            if self._task_collection_enabled and key in self._collection_cache:
                return self._collection_cache[key]
            try:
                proc = subprocess.run(cmd, cwd=str(root / "test"), env=env,
                                      capture_output=True, text=True,
                                      timeout=self.timeout_seconds, check=False,
                                      **self._lease_options())
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"runtime discovery timeout ({self.timeout_seconds}s): {rel_file}") from error
            if proc.returncode in (0, 5) and self._task_collection_enabled:
                if self.artifact_root:
                    receipt = self.artifact_root / "collections" / (hashlib.sha256(repr(key).encode()).hexdigest() + ".json")
                    receipt.parent.mkdir(parents=True, exist_ok=True)
                    receipt.write_text(json.dumps({"argv":cmd,"cwd":str(root/'test'),
                        "source_sha256":digest,"device":dev,"visible_devices":env.get("ASCEND_RT_VISIBLE_DEVICES"),
                        "returncode":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr},ensure_ascii=False,indent=2),encoding="utf-8")
                    proc.receipt_path = str(receipt)
                self._collection_cache[key] = proc
            return proc

    def _source_function_key(self, runtime_nodeid: str) -> str:
        """Validate the source function while allowing pytest ``[param]`` nodes."""
        value = str(runtime_nodeid).strip()
        # Parameter ids may contain punctuation; the source key is the final
        # function token before the bracket. Store/return the source key only.
        base = re.sub(r"(::[A-Za-z_]\w*)\[.*\]$", r"\1", value)
        return validate_test_function(base)

    def discover_runtime_nodes(
        self, test_function: str, device: str, *, source_root: str | Path,
        extra_env: dict | None = None,
    ) -> list[str]:
        """Collect native CPU/NPU variants using standard pytest stdout."""
        tf = self._source_function_key(test_function)
        dev = require_execution_device(validate_device(device))
        root = Path(source_root)
        orig_file = tf.split("::", 1)[0]
        rel_file = orig_file
        # discovery runs from the ``test`` directory; strip repository prefix.
        if rel_file.startswith("test/"):
            rel_file = rel_file[5:]
        source_class = tf.split("::")[1] if len(tf.split("::")) == 3 else ""
        source_method = tf.rsplit("::", 1)[-1]

        proc = self._collect_file(tf, dev, root, rel_file, extra_env)
        self._file_receipts[(tf, dev, tuple(sorted((extra_env or {}).items())))] = getattr(proc, "receipt_path", None)
        collected = []
        origins = None
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if line.startswith(_COLLECTION_METADATA_PREFIX):
                origins = json.loads(line[len(_COLLECTION_METADATA_PREFIX):])
                continue
            if "::" in line:
                file_seg = line.split("::", 1)[0]
                if file_seg == rel_file or file_seg.endswith("/" + rel_file):
                    collected.append(line)
        if proc.returncode not in (0, 5):
            raise RuntimeError(
                f"runtime discovery exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '')[-1000:]}"
            )
        nodes = []
        declared = declared_class_methods(root / "test" / rel_file)
        for node in collected:
            parts = node.split("::")
            cls = parts[-2] if len(parts) == 3 else ""
            method = parts[-1].split("[", 1)[0]
            if len(parts) not in (2, 3):
                continue
            key = "::".join([cls, method]) if cls else method
            origin = (origins or {}).get(key, {})
            # Copied template methods can have entirely different runtime class
            # names. Trust exact unwrapped ownership in this source file, not a
            # shared method name or an unrelated class prefix.
            source_qualname = f"{source_class}.{source_method}" if source_class else source_method
            origin_file = origin.get("source_file")
            template_origin = (bool(source_class) and source_method in declared.get(source_class, set())
                               and origin.get("qualname") == source_qualname
                               and bool(origin_file)
                               and Path(origin_file).resolve() == (root / 'test' / rel_file).resolve())
            backend = origin.get('declared_device')
            if template_origin and backend in {'cpu', 'cuda', 'mps', 'npu', 'privateuse1', 'xpu'}:
                if backend not in ({'cpu'} if dev == 'cpu' else {'npu', 'privateuse1'}):
                    continue
            if source_class != cls and not template_origin:
                if not source_class or not class_variant_match(source_class, cls) or cls in declared:
                    continue
            if origins is None:
                # Supports collection adapters without reflection metadata.
                matches = declared_node_match(declared, source_class, source_method, cls, method)
            else:
                # Exact runtime identities are valid too (explicit selected samples).
                matches = method == source_method or origin.get("name") == source_method
            if matches:
                full_node = "::".join([orig_file, *parts[1:]])
                nodes.append(full_node)
                origin_key = "::".join([cls, method]) if cls else method
                self._runtime_requirements[(dev, full_node)] = (origins or {}).get(origin_key, {}).get("required_npu_cards")
        if not nodes:
            raise RuntimeError(
                f"runtime discovery found no node for {tf} [{dev}]: "
                f"rc={proc.returncode}, collected={len(collected)}; "
                f"tail={(proc.stderr or proc.stdout or '')[-400:]!r}"
            )
        return sorted(set(nodes))

    def run_function(self, test_function: str, device: str, *,
                     source_root: str | Path | None = None,
                     artifact_dir: str | Path | None = None,
                     extra_env: dict | None = None,
                     source_test_function: str | None = None,
                     lease_fds=()) -> FunctionRunOutcome:
        """Run ordinary pytest with built-in collection and JUnit; no capture extension.

        extra_env：调用方注入的额外环境变量（如等卡门禁钉定的
        ASCEND_RT_VISIBLE_DEVICES），覆盖同名默认变量。
        """
        runtime_nodeid = str(test_function).strip()
        tf = self._source_function_key(source_test_function or runtime_nodeid)
        dev = require_execution_device(validate_device(device))
        fingerprint = _environment_fingerprint(dev, self.transfer_boot_dir if dev == "cuda" else None)
        cwd = str(Path(source_root) / "test") if source_root else None
        nodeid = runtime_nodeid[5:] if source_root and runtime_nodeid.startswith("test/") else runtime_nodeid
        env = test_execution_env(dev, source_root, tf)
        if extra_env:
            env = {**env, **extra_env}
        artifact = Path(artifact_dir) if artifact_dir else None
        if artifact is None and self.artifact_root:
            suffix = hashlib.sha256(runtime_nodeid.encode()).hexdigest()[:16]
            artifact = self.artifact_root / f"{dev}-{tf.replace('/', '_').replace(':', '_')[:150]}-{suffix}"
        if artifact:
            artifact.mkdir(parents=True, exist_ok=True)
        real_subprocess = getattr(subprocess.run, "__module__", "subprocess") == "subprocess"
        expected = []
        out = err = ""
        rc = None
        status = "runner_error"
        observed = None
        with tempfile.TemporaryDirectory(prefix="run-test-") as temp:
            xml = Path(temp) / "junit.xml"
            cmd = pytest_command(self.python_env, dev) + [nodeid, f"--junitxml={xml}",
                   "-q", "--no-header", "-o", "addopts=", "-p", "no:cacheprovider",
                   *self.device_args.get(dev, []), *self.extra_args]
            try:
                if real_subprocess:
                    collect = pytest_command(self.python_env, dev) + [nodeid, "--collect-only", "-q",
                               "-o", "addopts=", "-p", "no:cacheprovider", *self.device_args.get(dev, [])]
                    if source_test_function and source_root and self._task_collection_enabled:
                        try:
                            valid = self.discover_runtime_nodes(tf, dev, source_root=source_root, extra_env=extra_env)
                            if runtime_nodeid not in valid:
                                raise RuntimeError(f"node absent from execution-environment collection: {runtime_nodeid}")
                            result = subprocess.CompletedProcess(collect, 0, nodeid + "\n", "")
                        except Exception as error:
                            result = subprocess.CompletedProcess(collect, 5, "", str(error))
                    else:
                        result = subprocess.run(collect, cwd=cwd, env=env, capture_output=True,
                                                text=True, timeout=self.timeout_seconds, **self._lease_options(lease_fds))
                    file_part = nodeid.split("::", 1)[0]
                    expected = [ln.strip() for ln in result.stdout.splitlines() if "::" in ln and
                                (ln.strip().split("::", 1)[0] == file_part or
                                 ln.strip().split("::", 1)[0].endswith("/" + file_part))]
                    if artifact:
                        (artifact / "collection.json").write_text(json.dumps({"argv":collect,"nodeids":expected,
                            "returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr,
                            "source_test_function":source_test_function,
                            "receipt":"task_file_collection" if source_test_function and self._task_collection_enabled else "node_collection",
                            "file_receipt":self._file_receipts.get((tf,dev,tuple(sorted((extra_env or {}).items())))),
                            "visible_devices":env.get("ASCEND_RT_VISIBLE_DEVICES")},ensure_ascii=False,indent=2),encoding="utf-8")
                    if result.returncode or not expected:
                        rc = result.returncode;out = result.stdout;err = result.stderr
                        status = "collection_error"
                    else:
                        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                 text=True, env=env, cwd=cwd, start_new_session=True,
                                                 **self._lease_options(lease_fds))
                        try:
                            out, err = child.communicate(timeout=self.timeout_seconds)
                        except subprocess.TimeoutExpired:
                            if os.name == "nt":
                                subprocess.run(["taskkill","/PID",str(child.pid),"/T","/F"],capture_output=True,check=False)
                                if child.poll() is None: child.kill()
                            else:
                                try: os.killpg(child.pid, signal.SIGTERM)
                                except ProcessLookupError: pass
                            try: out, err = child.communicate(timeout=10)
                            except subprocess.TimeoutExpired:
                                if os.name == "nt": child.kill()
                                else:
                                    try: os.killpg(child.pid, signal.SIGKILL)
                                    except ProcessLookupError: pass
                                out, err = child.communicate(timeout=10)
                            raise subprocess.TimeoutExpired(cmd,self.timeout_seconds,output=out,stderr=err)
                        rc = child.returncode
                else:
                    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                                            timeout=self.timeout_seconds, check=False, start_new_session=True)
                    rc, out, err = result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired as exc:
                status = "timeout"
                out = exc.stdout or "";err = exc.stderr or ""
                if isinstance(out,bytes):out=out.decode(errors="replace")
                if isinstance(err,bytes):err=err.decode(errors="replace")
            except OSError as exc:
                status = "runner_error";err = str(exc)
            boot_note = ""
            log = boot_note + (f"[cwd: {cwd}]\n" if cwd else "") + f"$ {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}"
            instances = set()
            triage = ()
            if xml.is_file():
                try:
                    observed = parse_junit_counts(xml)
                    root = ET.parse(xml).getroot()
                    for case in root.iter("testcase"):
                        for n in expected:
                            parts = n.split("::")
                            class_ok = len(parts)==2 or case.get("classname", "").endswith(parts[-2])
                            if class_ok and case.get("name")==parts[-1]:instances.add(n)
                    from .triage import classify_texts
                    triage = tuple(classify_texts(parse_junit_failure_texts(xml)))
                except ET.ParseError as exc:
                    log += f"\n[junitxml 解析失败: {exc}]"
            if status != "timeout":
                if rc in (0,1) and observed and observed.total:
                    # Without every collected case there is no complete result (e.g. -x).
                    complete = not real_subprocess or (instances==set(expected) and len(instances)==observed.total)
                    status = "completed" if complete else "interrupted"
                elif status != "collection_error":
                    status = "interrupted" if rc == 2 else "collection_error" if rc == 5 else "runner_error"
            if artifact:
                (artifact / "command.json").write_text(json.dumps({"argv":cmd,"cwd":cwd,"device":dev},ensure_ascii=False,indent=2),encoding="utf-8")
                (artifact / "stdout.log").write_text(out or "",encoding="utf-8")
                (artifact / "stderr.log").write_text(err or "",encoding="utf-8")
                if xml.is_file():(artifact / "junit.xml").write_bytes(xml.read_bytes())
                (artifact / "result.json").write_text(json.dumps({"execution_status":status,"returncode":rc,
                    "counts":observed.as_dict() if observed and status=="completed" else None,
                    "observed_counts":observed.as_dict() if observed else None,"instance_ids":sorted(instances)},indent=2),encoding="utf-8")
            if status != "completed":
                return FunctionRunOutcome.incomplete(tf,dev,log,rc,execution_status=status,
                    artifact_dir=str(artifact) if artifact else None,observed_counts=observed,environment_fingerprint=fingerprint)
            return FunctionRunOutcome(tf,dev,True,observed,log,rc,triage=triage,
                artifact_dir=str(artifact) if artifact else None,coverage_status="full" if expected else "unknown",
                instance_ids=tuple(sorted(instances)),environment_fingerprint=fingerprint)
