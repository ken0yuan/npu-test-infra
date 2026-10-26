"""Regression: selected source identities must resolve to real complete pytest nodes."""
import sys
from pathlib import Path

import pytest

from run_test.runner import PytestRunner
from run_test.executor import Executor
from run_test import selection
from run_test.identifiers import Counts
from run_test.record import submit_complete_result
from run_test.db import connect
from run_test.pr_store import get_pr_result
from run_test.baseline_store import baseline_db_path, get_baseline
from run_test.device_detect import make_device_resolver
from run_test.device_policy import ExecutionDeviceResolver
from test_executor import verified_task, FakeRunner

REL = 'test/distributed/_tensor/test_identity.py'
SOURCE = '''from functools import wraps
import pytest
from unittest import TestCase
from sibling import Base

def expand(cls):
    for name, fn in list(vars(cls).items()):
        if not name.startswith('test_') or name.endswith('scalar_cpu'):
            continue
        delattr(cls, name)
        for value in ('cpu', 'other'):
            def make(fn, value):
                @wraps(fn)
                def generated(self):
                    return fn(self, value)
                return generated
            setattr(cls, name + '_kind_' + value, make(fn, value))
    return cls

@expand
class TestMixed(TestCase):
    def test_mix(self, kind):
        assert kind == 'cpu'

@expand
class TestChild(Base):
    def test_sub(self, kind):
        pass

@expand
class TestPrefixes(TestCase):
    def test_sub(self, kind):
        pass
    def test_sub_scalar(self, kind):
        pass
    def test_sub_scalar_cpu(self):
        assert False

class TestPlain:
    @pytest.mark.parametrize('value', [1, 2])
    def test_values(self, value):
        assert value > 0

def test_module():
    pass
'''


@pytest.fixture
def identity_tree(tmp_path):
    root = tmp_path / 'source'
    path = root / REL
    path.parent.mkdir(parents=True)
    # Package layout reproduces the PR #197470 sibling-import failure.
    for d in [root/'test/distributed', path.parent]:
        (d/'__init__.py').write_text('')
    (path.parent/'sibling.py').write_text('from unittest import TestCase\nclass Base(TestCase):\n    def test_sub_scalar(self):\n        assert False\n')
    path.write_text(SOURCE)
    return root


def runner():
    return PytestRunner(sys.executable, timeout_seconds=30)


def test_discovers_all_renamed_instances_even_when_one_contains_cpu(identity_tree):
    tf = REL + '::TestMixed::test_mix'
    assert runner().discover_runtime_nodes(tf, 'cpu', source_root=identity_tree) == [
        tf+'_kind_cpu', tf+'_kind_other']


@pytest.mark.parametrize('owner', ['TestChild', 'TestPrefixes'])
def test_inherited_and_same_prefix_methods_never_mix(identity_tree, owner):
    tf = REL + '::' + owner + '::test_sub'
    nodes = runner().discover_runtime_nodes(tf, 'cpu', source_root=identity_tree)
    assert nodes == [tf+'_kind_cpu', tf+'_kind_other']


@pytest.mark.parametrize('method', ['test_missing', 'test_mix_kind'])
def test_nonexistent_source_method_cannot_borrow_a_device_suffix(identity_tree, method):
    with pytest.raises(RuntimeError, match='found no node'):
        runner().discover_runtime_nodes(REL+'::TestMixed::'+method, 'cpu', source_root=identity_tree)


def test_module_function_does_not_match_class_methods(identity_tree):
    tf=REL+'::test_module'
    assert runner().discover_runtime_nodes(tf,'cpu',source_root=identity_tree)==[tf]


def test_discovery_and_execution_share_sibling_import_environment(identity_tree):
    tf=REL+'::TestChild::test_sub'
    r=runner()
    nodes=r.discover_runtime_nodes(tf,'cpu',source_root=identity_tree)
    for n in nodes:
        outcome=r.run_function(n,'cpu',source_root=identity_tree)
        assert outcome.completed, outcome.log_text
        assert outcome.counts == Counts(1,0,0,0)


def test_selection_executor_record_and_baseline_keep_all_instances(identity_tree, config, queue, paths, monkeypatch):
    tf=REL+'::TestMixed::test_mix'
    report={'p0':[{'path':REL,'test_nodes':['TestMixed.test_mix'],'evidence':[]}],
            'p1':[],'missing_coverage':[],'ci_recommendations':[],'analysis_gaps':[],'notes':[]}
    monkeypatch.setattr(selection,'discover_cases',lambda *a,**kw:[
        'TestMixed.test_mix_kind_cpu','TestMixed.test_mix_kind_other'])
    selection.finalize_plan(identity_tree,report,10,30,'',sys.executable)
    resolver=ExecutionDeviceResolver(make_device_resolver(identity_tree),allowed=['cpu'])
    manifest=selection.plan_to_manifest(report,device_of=resolver)
    assert len(manifest)==1
    assert manifest[0]['test_function']==tf
    assert set(manifest[0]['runtime_nodeids'])=={tf+'_kind_cpu',tf+'_kind_other'}
    task=verified_task(queue,manifest=manifest)
    task=queue.claim_next()
    Path(task.work_dir).mkdir(parents=True,exist_ok=True)
    sub=Executor(config,queue_store=queue,runner=runner())._run(task,identity_tree)
    result=sub.function_results[0]
    assert result.counts==Counts(1,1,0,0)  # The previously omitted failing instance is now visible.
    assert result.coverage_status=='full'
    assert len(result.instance_ids)==2
    submit_complete_result(config,sub)
    submit_complete_result(config,sub)
    with connect(paths.pr_db) as c:
        pr=get_pr_result(c,123456)
        assert pr.counts==Counts(1,1,0,0)
        assert c.execute('select count(*) from pr_results').fetchone()[0]==1
    with connect(baseline_db_path(paths.baselines_root/'test/distributed/_tensor')) as c:
        baseline=get_baseline(c,tf,'cpu')
        assert baseline.counts==Counts(1,1,0,0)
        assert len(baseline.instance_ids)==2
        assert c.execute('select count(*) from baseline').fetchone()[0]==1


def test_explicit_sample_is_partial_and_cannot_establish_baseline(identity_tree,config,queue,paths):
    tf=REL+'::TestMixed::test_mix'
    task=verified_task(queue,manifest=[{'test_function':tf,'device':'cpu','runtime_nodeid':tf+'_kind_cpu'}])
    task=queue.claim_next()
    Path(task.work_dir).mkdir(parents=True,exist_ok=True)
    sub=Executor(config,queue_store=queue,runner=runner())._run(task,identity_tree)
    assert sub.function_results[0].counts==Counts(1,0,0,0)
    assert sub.function_results[0].coverage_status=='partial'
    submit_complete_result(config,sub)
    with connect(baseline_db_path(paths.baselines_root/'test/distributed/_tensor')) as c:
        assert get_baseline(c,tf,'cpu') is None


def test_full_pytest_bracket_family_has_full_coverage(identity_tree,config,queue):
    tf=REL+'::TestPlain::test_values'
    task=verified_task(queue,manifest=[{'test_function':tf,'device':'cpu'}])
    task=queue.claim_next()
    Path(task.work_dir).mkdir(parents=True,exist_ok=True)
    sub=Executor(config,queue_store=queue,runner=runner())._run(task,identity_tree)
    assert sub.function_results[0].counts==Counts(2,0,0,0)
    assert sub.function_results[0].coverage_status=='full'


@pytest.mark.parametrize('explicit', [False,True])
def test_discovery_failure_never_executes_static_or_stale_name(config,queue,paths,explicit):
    tf=REL+'::TestMixed::test_mix'
    class NoRun(FakeRunner):
        def discover_runtime_nodes(self,*a,**kw):
            if explicit:return [tf+'_new_name']
            raise RuntimeError('cannot resolve source family')
        def run_function(self,*a,**kw):
            pytest.fail('Must not execute rejected node')
    manifest={'test_function':tf,'device':'cpu'}
    if explicit:manifest['runtime_nodeid']=tf+'_old_name'
    task=verified_task(queue,manifest=[manifest])
    task=queue.claim_next()
    Path(task.work_dir).mkdir(parents=True,exist_ok=True)
    sub=Executor(config,queue_store=queue,runner=NoRun())._run(task,Path('/unused'))
    result=sub.function_results[0]
    assert result.status=='no_result'
    assert result.execution_status=='collection_error'
    submit_complete_result(config,sub)
    with connect(paths.pr_db) as c:
        assert get_pr_result(c,123456).no_result_count==1
    with connect(baseline_db_path(paths.baselines_root/'test/distributed/_tensor')) as c:
        assert get_baseline(c,tf,'cpu') is None


def test_selection_does_not_assign_imported_base_method_to_child_family(identity_tree, monkeypatch):
    report={'p0':[{'path':REL,'test_nodes':['TestChild.test_sub'],'evidence':[]}],
            'p1':[],'missing_coverage':[],'ci_recommendations':[],'analysis_gaps':[],'notes':[]}
    monkeypatch.setattr(selection,'discover_cases',lambda *a,**kw:[
        'TestChild.test_sub_scalar','TestChild.test_sub_kind_cpu','TestChild.test_sub_kind_other'])
    selection.finalize_plan(identity_tree,report,10,30,'',sys.executable)
    resolver=ExecutionDeviceResolver(make_device_resolver(identity_tree),allowed=['cpu'])
    manifest=selection.plan_to_manifest(report,device_of=resolver)
    assert len(manifest)==1
    assert manifest[0]['test_function']==REL+'::TestChild::test_sub'
    assert set(manifest[0]['runtime_nodeids'])=={
        REL+'::TestChild::test_sub_kind_cpu',REL+'::TestChild::test_sub_kind_other'}
