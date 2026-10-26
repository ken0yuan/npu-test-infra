import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest
from api_test_relations.cli import PACKAGE, initialize, read
from api_test_relations.audit import audit


def launch(repo, *args, enabled=True):
    env = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
           'PYTHONPATH': os.pathsep.join([str(repo), str(PACKAGE.parent), str(Path(pytest.__file__).resolve().parent.parent)])}
    for key in list(env):
        if key.startswith('API_RELATION_'): env.pop(key)
    command = [sys.executable, '-m', 'pytest', '-p', 'api_test_relations.pytest_plugin', '-q', 'test']
    result = subprocess.run([*command, *args], cwd=repo, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def evidence(tmp_path):
    repo = tmp_path / 'repo'; (repo / 'test').mkdir(parents=True)
    (repo / 'torch').mkdir()
    (repo / 'torch/__init__.py').write_text('def alpha(): pass\ndef beta(): pass\nalias = alpha\n')
    (repo / 'torch_npu.py').write_text('import torch\ndef alpha(): pass\ntorch.alpha = alpha\ntorch.alias = alpha\n')
    # Editable torch finders can precede PathFinder. Restrict this fixture process
    # to the normal import system so it cannot accidentally import a host build.
    (repo / 'sitecustomize.py').write_text('import sys\nfrom importlib.machinery import BuiltinImporter,FrozenImporter,PathFinder\nsys.meta_path[:]=[BuiltinImporter,FrozenImporter,PathFinder]\n')
    (repo / 'test/test_mapping.py').write_text('''
import subprocess,sys,threading
import pytest
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

@pytest.mark.parametrize('number',[1,2],ids=['one','two'])
def test_alpha(number): torch.alpha()
def test_beta(): torch.beta()
def test_child(): subprocess.run([sys.executable,'-c','import torch; torch.beta()'],check=True)
def test_thread():
    t=threading.Thread(target=torch.alpha); t.start(); t.join()
def test_pool():
    with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn')) as pool:
        pool.submit(torch.beta).result()
''')
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'apis': [{'api_id': 'python:torch.' + name, 'qualified_name': 'torch.' + name,
                                           'language': 'python', 'kind': 'function'} for name in ('alpha', 'alias', 'beta')]}))
    out = tmp_path / 'capture'
    launch(repo, '--api-relations-repo', str(repo), '--api-relations-out', str(out),
           '--api-relations-catalog', str(catalog), '--api-relations-import', 'torch_npu', '--api-relations-label', 'synthetic-no-hardware')
    return repo, out


def test_parameterized_aliases_and_no_cross_test_edges(evidence):
    repo, out = evidence
    with sqlite3.connect(out / 'relations.sqlite') as db:
        actual = {(n, a) for n, a in db.execute('SELECT nodeid,api_id FROM relations WHERE phase="call"')}
        prefix = 'pytorch/test/test_mapping.py::'
        expected = {(prefix + n, 'python:torch.' + a)
                    for n in ('test_alpha[one]', 'test_alpha[two]', 'test_thread') for a in ('alpha', 'alias')}
        expected |= {(prefix + n, 'python:torch.beta') for n in ('test_beta', 'test_child', 'test_pool')}
        assert actual == expected
        assert db.execute('SELECT count(*) FROM gaps').fetchone()[0] == 0
        aliases = db.execute('SELECT aliases FROM relations WHERE api_id="python:torch.alpha" LIMIT 1').fetchone()[0]
        assert json.loads(aliases) == ['python:torch.alias', 'python:torch.alpha']


def test_child_and_pool_have_causal_receipts(evidence):
    _, out = evidence
    with sqlite3.connect(out / 'relations.sqlite') as db:
        tasks = list(db.execute('SELECT nodeid,state FROM process_tasks'))
        assert tasks and all(node.endswith('test_pool') and state == 'finished' for node, state in tasks)
        states = list(db.execute('SELECT state FROM processes'))
        assert states and all(s[0] in ('acknowledged_complete', 'helper_service_observed') for s in states)


def test_export_queries_and_structural_audit(evidence):
    _, out = evidence
    assert read(out / 'progress.json')['state'] == 'capture_finished'
    report = audit(out)
    assert not any(report['structural_issues'].values())
    assert report['proves_zero_omissions'] is False
    env = {**os.environ, 'PYTHONPATH': str(PACKAGE.parent)}
    result = subprocess.run([sys.executable, '-m', 'api_test_relations', 'query', '--out', str(out), '--api', 'torch.beta'], env=env, capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)
    assert len(rows['tests']) == 3
    assert rows['unobserved_means_no_test'] is False
    sources = read(out / 'test-sources.json')
    assert len(sources['pytorch/test/test_mapping.py']['tests']) == 6
    assert len(sources['pytorch/test/test_mapping.py']['sha256']) == 64


def test_existing_evidence_cannot_be_overwritten(evidence):
    repo, out = evidence
    with pytest.raises(RuntimeError, match='empty output'):
        initialize(out, repo, out / 'api-catalog.json', [], '', [])


def test_inactive_plugin_does_not_import_torch(tmp_path):
    (tmp_path / 'test').mkdir()
    (tmp_path / 'test/test_idle.py').write_text('import sys\ndef test_idle(): assert "torch" not in sys.modules\n')
    launch(tmp_path)
    assert not (tmp_path / 'manifest.json').exists()


def test_no_device_policy_or_native_x86_assembly_in_runtime():
    files = list((PACKAGE / 'runtime').rglob('*.py')) + [PACKAGE / 'runtime/relation_identity.c', PACKAGE / 'pytest_plugin.py']
    for file in files:
        code = file.read_text()
        assert 'CUDA_VISIBLE_DEVICES' not in code
        assert 'nvidia-smi' not in code
        assert '__asm__' not in code
