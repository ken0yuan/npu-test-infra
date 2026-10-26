"""Portable migration check: real torch, CPU only, no Triton or accelerator."""
import cProfile
import subprocess
import sys
import threading

import pytest
import torch


@pytest.mark.parametrize('value', [1, 2], ids=['one', 'two'])
def test_fill(value):
    torch.empty(2).fill_(value)


def test_native_profile():
    tensor = torch.ones(2)
    profiler = cProfile.Profile()
    profiler.enable(); torch.add(tensor, tensor); profiler.disable(); torch.mul(tensor, tensor)


def test_thread():
    tensor = torch.ones(2)
    thread = threading.Thread(target=lambda: torch.sub(tensor, tensor))
    thread.start()
    thread.join()


def test_child():
    subprocess.run([sys.executable, '-c', 'import torch; torch.neg(torch.ones(2))'], check=True)
