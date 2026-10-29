"""Explicit Python suite layouts shared by indexing, collection and execution."""
from pathlib import Path

SUITES = {"pytorch": "test/", "tools": "tools/test/"}


def suite_for(path):
    return next((name for name, prefix in SUITES.items() if path.startswith(prefix)), None)


def layout(root, path):
    root = Path(root).resolve()
    suite = suite_for(path)
    if suite is None:
        raise ValueError(f"Unregistered test suite: {path}")
    prefix = SUITES[suite]
    target = (root / path).resolve()
    directory = (root / prefix).resolve()
    if not target.is_relative_to(directory) or target == directory:
        raise ValueError("test path escapes registered suite")
    return directory, target.relative_to(directory).as_posix()


def suite_args(path):
    # Import tools through its real package path, without adding the repository
    # root (and its unbuilt torch package) to sys.path.
    return ["--import-mode=importlib"] if suite_for(path) == "tools" else []


TOOLS_BOOTSTRAP = r'''
import os, sys, importlib.util, types
_tools_root = os.environ.get("RUN_TEST_TOOLS_ROOT")
if _tools_root:
    import pathlib
    _tools_path = pathlib.Path(_tools_root)
    _init = _tools_path / "__init__.py"
    if "tools" not in sys.modules:
        if _init.is_file():
            _spec = importlib.util.spec_from_file_location("tools", _init, submodule_search_locations=[str(_tools_path)])
            _module = importlib.util.module_from_spec(_spec)
            sys.modules["tools"] = _module
            _spec.loader.exec_module(_module)
        else:
            _module = types.ModuleType("tools")
            _module.__path__ = [str(_tools_path)]
            _module.__package__ = "tools"
            _module.__spec__ = importlib.util.spec_from_loader("tools", loader=None, is_package=True)
            sys.modules["tools"] = _module
'''
