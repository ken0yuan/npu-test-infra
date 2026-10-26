"""Run explicitly in a working torch_npu environment; no numerical benchmarking."""
import pytest
import torch
import torch_npu  # noqa: F401 -- register the installed NPU backend


@pytest.mark.parametrize('size', [2, 3], ids=['size2', 'size3'])
def test_npu_add(size):
    value = torch.zeros(size, device='npu')
    torch.add(value, value)
    torch.npu.synchronize()


def test_npu_fill():
    value = torch.empty(2, device='npu')
    value.fill_(1)
    torch.npu.synchronize()
