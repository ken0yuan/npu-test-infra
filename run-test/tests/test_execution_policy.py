import subprocess
import sys

import pytest

from run_test.artifacts import inventory, select_wheel, wheel_identity
from run_test.config import ConfigError
from run_test.device_detect import make_device_resolver
from run_test.device_policy import ExecutionDeviceResolver, execution_env
from run_test.runner import PytestRunner


def test_policy_does_not_relabel_cuda_as_native(tmp_path):
    (tmp_path/'test').mkdir()
    (tmp_path/'test/test_x.py').write_text('def test_x():\n    import torch\n    torch.ones(2).cuda()\n')
    resolver = ExecutionDeviceResolver(make_device_resolver(tmp_path))
    assert resolver('test/test_x.py', 'test_x') == []
    assert resolver.diagnostics[-1]['excluded_devices'] == ['cuda']


def test_generic_test_defaults_to_native_channels(tmp_path):
    (tmp_path/'test').mkdir()
    (tmp_path/'test/test_x.py').write_text('def test_x():\n    assert 1 + 1 == 2\n')
    resolver = ExecutionDeviceResolver(make_device_resolver(tmp_path))
    assert resolver('test/test_x.py', 'test_x') == ['cpu', 'accelerator']


def test_manual_cuda_execution_never_spawns(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CUDA request spawned a process')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(ConfigError):
        PytestRunner(sys.executable).run_function('test/test_x.py::test_x','cuda')


def test_cpu_env_disables_cuda_and_external_plugins(monkeypatch):
    monkeypatch.setenv('PYTEST_PLUGINS', 'unwanted_plugin')
    monkeypatch.setenv('PYTEST_ADDOPTS', '-p unwanted_plugin')
    env = execution_env('cpu')
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['PYTORCH_TESTING_DEVICE_ONLY_FOR'] == 'cpu'
    assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
    assert 'PYTEST_PLUGINS' not in env and 'PYTEST_ADDOPTS' not in env


def test_configuration_cannot_enable_cuda():
    from run_test.device_policy import configured_devices
    assert configured_devices({'executor':{'allowed_devices':['cpu']}}) == ['cpu']
    with pytest.raises(ConfigError):
        configured_devices({'executor':{'allowed_devices':['cuda']}})


BUCKET = 'obs://example/7day-cache'
SHA = '0123456789'
NAME = f'{BUCKET}/wheels-{SHA}-npu-abcdef0123-py311-builder-aarch64.tar.gz'


def test_current_obs_name_and_pair():
    listing = {NAME:{}, f'{BUCKET}/pytorch-source-{SHA}.tar.gz':{}}
    assert select_wheel(listing, BUCKET, SHA, {}) == NAME
    assert wheel_identity(NAME)['python_tag'] == '311'
    assert len(inventory(listing, BUCKET, {})['unique_pairs']) == 1
    assert wheel_identity(f'{BUCKET}/wheels-torchnpu-{SHA}-abcdef0123-py311-builder-aarch64.tar.gz') is None


def test_multiple_builds_require_explicit_filter():
    other = NAME.replace('builder', 'cann-a3')
    listing = {NAME:{}, other:{}}
    with pytest.raises(ValueError, match='多个 wheel'):
        select_wheel(listing, BUCKET, SHA, {})
    assert select_wheel(listing, BUCKET, SHA, {'obs':{'wheel_name_contains':['cann-a3']}}) == other
