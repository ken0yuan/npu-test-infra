"""Preparation must keep the venv entry point, not its base-Python symlink target."""
import importlib.util
import json
import sys
from pathlib import Path


def test_prepare_preserves_venv_interpreter_symlink(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'prepare.py'
    spec = importlib.util.spec_from_file_location('prepare_venv_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'project'
    (root / 'run-test').mkdir(parents=True)
    (root / 'settings.example.json').write_text(json.dumps({'obs': {'downloader': {}}, 'executor': {}}))
    base = tmp_path / 'base-python'
    base.write_text('base interpreter placeholder')
    entry = root / 'venv/test/bin/python'
    entry.parent.mkdir(parents=True)
    entry.symlink_to(base)
    monkeypatch.setattr(module, '__file__', str(root / 'run-test/prepare.py'))
    monkeypatch.setattr(sys, 'argv', ['prepare.py', '--test-python', str(entry)])
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: calls.append((a, kw)))
    module.main()
    configured = json.loads((root / 'settings.json').read_text())['executor']['test_python_env']
    assert configured == str(entry)
    assert configured != str(base)
    assert calls
