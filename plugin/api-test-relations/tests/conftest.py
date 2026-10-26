import os
from pathlib import Path
import shutil
import sys
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from api_test_relations.cli import PACKAGE, build_identity

# Compile for the test host in a disposable folder, never in the distributed source.
_scratch = tempfile.TemporaryDirectory(prefix='relation-plugin-regression-')
_runtime = Path(_scratch.name)
shutil.copyfile(PACKAGE / 'runtime/relation_identity.c', _runtime / 'relation_identity.c')
shutil.copyfile(PACKAGE / 'runtime/relation_guard.h', _runtime / 'relation_guard.h')
build_identity(_runtime)
sys.path[:0] = [str(_runtime), str(PACKAGE / 'runtime')]


def pytest_collection_modifyitems(items):
    if os.environ.get('API_RELATIONS_REAL_TORCH_TESTS') == '1': return
    for item in items:
        if item.get_closest_marker('integration'):
            item.add_marker(pytest.mark.skip(reason='Real torch/CUDA integration is explicit opt-in'))
