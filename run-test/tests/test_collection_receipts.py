import json
import sys
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from run_test.runner import PytestRunner


def test_concurrent_instances_resolve_family_once(tmp_path, monkeypatch):
    root=tree(tmp_path)
    runner=PytestRunner(sys.executable,timeout_seconds=30)
    runner.begin_task()
    actual=runner._resolve_runtime_family
    calls=[]
    def resolve(*args):
        calls.append(args[0])
        return actual(*args)
    monkeypatch.setattr(runner,'_resolve_runtime_family',resolve)
    def discover(_):
        return runner.discover_runtime_nodes('test/test_dynamic.py::test_values','cpu',source_root=root)
    with ThreadPoolExecutor(8) as executor:
        results=list(executor.map(discover,range(16)))
    assert all(nodes==results[0] and len(nodes)==2 for nodes in results)
    assert len(calls)==1
    runner.begin_task()
    discover(None)
    assert len(calls)==2
    source=root/'test/test_dynamic.py'
    source.write_text(source.read_text().replace('["a", "b"]','["a"]'))
    assert len(discover(None))==1
    assert len(calls)==3


def test_different_execution_environments_collect_concurrently(tmp_path,monkeypatch):
    root=tree(tmp_path)
    runner=PytestRunner(sys.executable,timeout_seconds=30)
    runner.begin_task()
    import run_test.runner as module
    actual=module.subprocess.run
    barrier=threading.Barrier(2)
    def run(*args,**kwargs):
        barrier.wait(timeout=5)
        return actual(*args,**kwargs)
    monkeypatch.setattr(module.subprocess,'run',run)
    def discover(value):
        return runner.discover_runtime_nodes('test/test_dynamic.py::test_values','cpu',source_root=root,
                                             extra_env={'RECEIPT_ENV':value})
    with ThreadPoolExecutor(2) as executor:
        results=list(executor.map(discover,['first','second']))
    assert all(len(nodes)==2 for nodes in results)


def tree(tmp_path):
    root=tmp_path/'source'
    (root/'test').mkdir(parents=True)
    (root/'test/test_dynamic.py').write_text('''import os
from functools import wraps
from unittest import TestCase
import pytest
def original(self):
    pass
class TestGenerated(TestCase):
    pass
setattr(TestGenerated, "test_mode_nearest-exact", wraps(original)(lambda self: None))
@pytest.mark.parametrize("value", ["a", "b"])
def test_values(value):
    assert value
def test_visible():
    assert os.environ.get("RECEIPT_ENV") == "correct"
''')
    return root


def test_file_receipt_reused_for_all_instances_and_source_functions(tmp_path, monkeypatch):
    root=tree(tmp_path)
    runner=PytestRunner(sys.executable,timeout_seconds=30,artifact_root=tmp_path/'artifacts')
    import run_test.runner as module
    actual=module.subprocess.run
    calls=[]
    def run(*args,**kwargs):
        calls.append(args[0])
        return actual(*args,**kwargs)
    run.__module__='subprocess'
    monkeypatch.setattr(module.subprocess,'run',run)
    runner.begin_task()
    tf='test/test_dynamic.py::test_values'
    nodes=runner.discover_runtime_nodes(tf,'cpu',source_root=root)
    outcomes=[runner.run_function(n,'cpu',source_root=root,source_test_function=tf) for n in nodes]
    assert len(calls)==1
    assert all(o.completed for o in outcomes)
    assert len({o.artifact_dir for o in outcomes})==2
    # A new task always recollects, even when paths and file contents match.
    runner.begin_task()
    runner.discover_runtime_nodes(tf,'cpu',source_root=root)
    assert len(calls)==2
    runner.discover_runtime_nodes(tf,'cpu',source_root=root,extra_env={'RECEIPT_ENV':'correct'})
    assert len(calls)==3


def test_hyphen_generated_node_runs_with_stable_source_identity(tmp_path):
    root=tree(tmp_path)
    # The source identity comes from inspect.unwrap, not sanitizing the node.
    p=root/'test/test_dynamic.py'
    p.write_text(p.read_text().replace('def original(self):','def test_original(self):').replace('wraps(original)','wraps(test_original)') + '\n# source declaration for family lookup\n')
    runner=PytestRunner(sys.executable,timeout_seconds=30)
    runner.begin_task()
    tf='test/test_dynamic.py::TestGenerated::test_original'
    nodes=runner.discover_runtime_nodes(tf,'cpu',source_root=root)
    assert nodes==['test/test_dynamic.py::TestGenerated::test_mode_nearest-exact']
    outcome=runner.run_function(nodes[0],'cpu',source_root=root,source_test_function=tf)
    assert outcome.completed and outcome.test_function==tf


def test_stale_node_never_executed_and_changed_source_invalidates_receipt(tmp_path):
    root=tree(tmp_path)
    runner=PytestRunner(sys.executable,timeout_seconds=30,artifact_root=tmp_path/'artifacts')
    runner.begin_task()
    tf='test/test_dynamic.py::test_values'
    nodes=runner.discover_runtime_nodes(tf,'cpu',source_root=root)
    p=root/'test/test_dynamic.py'
    p.write_text(p.read_text().replace('["a", "b"]','["a"]'))
    outcome=runner.run_function(nodes[1],'cpu',source_root=root,source_test_function=tf)
    assert not outcome.completed and outcome.execution_status=='collection_error'
    assert not (Path(outcome.artifact_dir)/'junit.xml').exists()
    assert 'node absent' in outcome.log_text


def test_requirements_use_explicit_device_marker_and_unknown_is_conservative(tmp_path):
    root=tree(tmp_path)
    p=root/'test/test_dynamic.py'
    p.write_text(p.read_text()+'\ndef test_multi(devices): pass\ntest_multi.num_required_devices=4\ndef test_single(device): pass\n')
    runner=PytestRunner(sys.executable,timeout_seconds=30)
    runner.begin_task()
    for name,need in [('test_multi',4),('test_single',1),('test_visible',8)]:
        tf='test/test_dynamic.py::'+name
        # CPU collection carries reflection metadata too, so this test needs no hardware.
        runner.discover_runtime_nodes(tf,'cpu',source_root=root)
        assert (runner._runtime_requirements.get(('cpu',tf)) or 8)==need
    assert runner.required_npu_cards('unknown','accelerator',8)==8
