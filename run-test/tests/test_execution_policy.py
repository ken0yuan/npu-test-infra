import subprocess
import sys

import pytest

from run_test.artifacts import inventory, select_bundle, select_wheel, wheel_identity
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


# 2026-10-22 起构建方把全量 wheels 包拆成两个 -vN 对象（实证 2026-10-26）：
# wheels-pytorch-{sha}-…（torch+torchvision）+ wheels-torchnpu-{npu}-{sha}-…（torch_npu）。
PT_NAME = f'{BUCKET}/wheels-pytorch-{SHA}-py311-torch-npu-builder-aarch64-torch-master-202610200103-e0dcc43e-aarch64-v2.tar.gz'
NPU_NAME = f'{BUCKET}/wheels-torchnpu-abcdef0123-{SHA}-py311-torch-npu-builder-aarch64-torch-master-202610200103-e0dcc43e-aarch64-v2.tar.gz'


def test_split_bundle_identity():
    pt = wheel_identity(PT_NAME)
    assert pt['format'] == 'split-pytorch'
    assert pt['pytorch_short'] == SHA and pt['python_tag'] == '311'
    assert pt['architecture'] == 'aarch64'
    npu = wheel_identity(NPU_NAME)
    assert npu['format'] == 'split-torchnpu'
    assert npu['torch_npu_short'] == 'abcdef0123'
    assert npu['pytorch_short'] == SHA


def test_split_bundle_pairing():
    listing = {PT_NAME:{}, NPU_NAME:{}, f'{BUCKET}/pytorch-source-{SHA}.tar.gz':{}}
    assert select_bundle(listing, BUCKET, SHA, {}) == {'wheels':PT_NAME, 'wheels_npu':NPU_NAME}
    pair = inventory(listing, BUCKET, {})['unique_pairs'][0]
    assert pair['wheels'] == PT_NAME and pair['wheels_npu'] == NPU_NAME


def test_split_bundle_without_npu_object_reports_missing_npu():
    listing = {PT_NAME:{}, f'{BUCKET}/pytorch-source-{SHA}.tar.gz':{}}
    bundle = select_bundle(listing, BUCKET, SHA, {})
    assert bundle == {'wheels':PT_NAME, 'wheels_npu':None}


def test_torchnpu_without_v_suffix_is_still_a_different_product():
    listing = {NPU_NAME.replace('-aarch64-v2.tar.gz','-aarch64.tar.gz'):{},
               f'{BUCKET}/pytorch-source-{SHA}.tar.gz':{}}
    assert inventory(listing, BUCKET, {})['unique_pairs'] == []


def test_multiple_npu_builds_require_explicit_filter():
    other = NPU_NAME.replace('e0dcc43e', '8a315d17')
    listing = {PT_NAME:{}, NPU_NAME:{}, other:{}, f'{BUCKET}/pytorch-source-{SHA}.tar.gz':{}}
    with pytest.raises(ValueError, match='多个 wheel'):
        select_bundle(listing, BUCKET, SHA, {})
