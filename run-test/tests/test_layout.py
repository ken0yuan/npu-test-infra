"""Relocation regressions: configuration and executable resources resolve from any cwd."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_test.cli import _default_config_path
from run_test.prefilter import _infra_filter_script

ROOT = Path(__file__).resolve().parents[2]


def test_project_config_and_filter_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _default_config_path() == ROOT / "settings.json"
    assert _infra_filter_script() == ROOT / "run-test/tools/pr_exclusion_filter.py"
    assert _infra_filter_script().is_file()


def test_cli_wrapper_runs_from_another_directory(tmp_path):
    env = dict(os.environ, RUN_TEST_PYTHON=sys.executable)
    result = subprocess.run(["bash", str(ROOT / "run.sh"), "--help"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "baseline" in result.stdout and "pipeline" in result.stdout


def test_prepare_creates_config_and_databases_at_project_root(tmp_path):
    project = tmp_path / "project"
    (project / "run-test").mkdir(parents=True)
    shutil.copy2(ROOT / "run-test/prepare.py", project / "run-test/prepare.py")
    shutil.copytree(ROOT / "run-test/src", project / "run-test/src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "settings.example.json", project / "settings.example.json")
    args = [sys.executable, str(project / "run-test/prepare.py"), "--device", "cpu"]
    result = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    config_path = project / "settings.json"
    config = json.loads(config_path.read_text())
    assert config["data_root"] == str(project / "data/runtime")
    assert (project / "data/runtime/queue/queue.db").is_file()
    assert (project / "data/runtime/pr_results.db").is_file()
    assert not (project / "run-test/settings.json").exists()
    before = config_path.read_bytes()
    second = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert config_path.read_bytes() == before
