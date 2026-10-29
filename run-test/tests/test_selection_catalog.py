import json
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from run_test import selection
from run_test.test_catalog import Catalog, order_candidates, sample_instances
from run_test.runner import PytestRunner, NoRuntimeNodes
from run_test.task_queue import validate_manifest
from run_test.device_policy import ExecutionDeviceResolver
from run_test.device_detect import make_device_resolver
from test_executor import FakeRunner
from test_executor_leases import setup


def report_for(tmp_path, text, *, lines=(1,), source='def target(): return 2\n'):
    path = tmp_path/'test/test_mixed.py'; path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    product = tmp_path/'torch/foo.py'; product.parent.mkdir(exist_ok=True); product.write_text(source)
    changed = [selection.ChangedFile('torch/foo.py','M',module='torch.foo',symbols={'target'}),
               selection.ChangedFile('test/test_mixed.py','M',changed_lines=set(lines))]
    tests = selection.analyze_tests(tmp_path, changed[:1])
    return selection.build_report('a'*40,'b'*40,changed,tests,tmp_path,1,5000,6)


def test_evidence_never_leaks_to_unrelated_sibling_under_one_slot(tmp_path):
    report = report_for(tmp_path, 'from torch.foo import target\nclass TestMixed:\n'
        '    def test_a_unrelated(self): assert True\n'
        '    def test_z_target(self): assert target() == 2\n')
    selection.finalize_plan(tmp_path, report, 1, 1, defer_runtime_discovery=True)
    assert report['p0'][0]['test_nodes'] == ['TestMixed.test_z_target']
    unrelated = [e for e in report['deferred_tests'] if e['test_nodes']==['TestMixed.test_a_unrelated']]
    assert all(e['relevance'] >= 2 and not any(ev['type']=='direct_changed_symbol_reference'
                                             for ev in e['evidence']) for e in unrelated)


def test_jit_helpers_map_through_subscript_launch_and_local_wrapper():
    catalog = Catalog('import triton\n@triton.jit\ndef test_kernel(x): return x\n'
        'class TestK:\n    def launch(self): test_kernel[(1,)](1)\n'
        '    def test_result(self): self.launch()\n', 'test/test_k.py')
    assert 'test_kernel' not in catalog.tests
    assert catalog.objects['test_kernel'].kind == 'kernel'
    evidence = catalog.changed_test_evidence({3})['TestK.test_result']
    assert evidence['chain'] == ['TestK.test_result','TestK.launch','test_kernel']
    assert evidence['relevance'] == 1


def test_changed_product_helper_reaches_test_through_public_wrapper(tmp_path):
    report = report_for(tmp_path, 'from torch.foo import public\nclass TestWrap:\n'
        '    def test_result(self): assert public() == 2\n', lines=(),
        source='def target(): return 2\ndef public(): return target()\n')
    entry=report['p0'][0]
    assert entry['groups']==['torch.foo::target']
    assert entry['evidence'][0]['product_chain']==['public','target']


def test_reference_is_weaker_than_actual_call_and_pr_mention_does_not_promote(tmp_path):
    report = report_for(tmp_path, 'from torch.foo import target\nclass TestRefs:\n'
        '    def test_a_reference(self): f = target\n'
        '    def test_z_call(self): assert target() == 2\n', lines=())
    selection.finalize_plan(tmp_path, report, 1, 1, pr_text='test_a_reference is important', defer_runtime_discovery=True)
    assert report['p0'][0]['test_nodes']==['TestRefs.test_z_call']
    assert report['deferred_tests'][0]['relevance']==2


def test_groups_get_strong_representatives_before_redundant_tests():
    def entry(name, group, level=1):return dict(path='test/test_x.py',test_nodes=[name],groups=[group],relevance=level)
    result=order_candidates([entry('test_a','A'),entry('test_b','A'),entry('test_c','B'),entry('test_tiny','smoke',3)])
    assert [e['test_nodes'][0] for e in result[:3]]==['test_a','test_c','test_b']


def test_group_novelty_never_promotes_a_weaker_evidence_tier():
    def entry(name, group, level):return dict(path='test/test_x.py',test_nodes=[name],groups=[group],relevance=level)
    result=order_candidates([entry('test_first','A',1),entry('test_second','A',1),entry('test_weak','B',2)])
    assert [e['test_nodes'][0] for e in result]==['test_first','test_second','test_weak']


def test_no_product_python_symbols_means_no_reference_graph_scan(tmp_path, monkeypatch):
    directory=tmp_path/'test';directory.mkdir()
    (directory/'test_cpu.py').write_text('import torch\ndef test_one(): assert torch.tensor(1).item()==1\n')
    def unexpected(*args, **kwargs):raise AssertionError('No changed Python module exists')
    monkeypatch.setattr(Catalog,'references',unexpected)
    found=selection.analyze_tests(tmp_path,[])
    assert found[0].test_nodes=={'test_one':(2,2)} and found[0].direct_refs==[]


def test_reserve_device_resolution_reuses_source_ast_and_declarations_only_within_one_tree(tmp_path, monkeypatch):
    import ast
    from run_test import runner
    path=tmp_path/'test/test_device.py';path.parent.mkdir()
    path.write_text('class TestCPU:\n    def test_a(self): return "cpu"\n    def test_b(self): return "cpu"\n')
    parse=ast.parse;declared=runner.declared_class_methods;calls=[]
    def tracked_parse(*args,**kwargs):calls.append('parse');return parse(*args,**kwargs)
    def tracked_declared(*args,**kwargs):calls.append('declared');return declared(*args,**kwargs)
    monkeypatch.setattr(ast,'parse',tracked_parse);monkeypatch.setattr(runner,'declared_class_methods',tracked_declared)
    resolver=make_device_resolver(tmp_path)
    assert resolver('test/test_device.py','TestCPU.test_a')==['cpu']
    first=len(calls)
    assert resolver('test/test_device.py','TestCPU.test_b')==['cpu'] and len(calls)==first
    path.write_text('class TestCPU:\n    def test_a(self): return "npu"\n')
    fresh=make_device_resolver(tmp_path)
    assert fresh('test/test_device.py','TestCPU.test_a')==['accelerator'] and len(calls)>first


def test_class_context_does_not_affect_another_class_and_local_alias_shadow_is_not_a_reference():
    catalog=Catalog('from torch.foo import target\n'
        'class TestA:\n    FLAG = 2\n    def test_one(self): pass\n'
        'class TestB:\n    def test_two(self): pass\n','test/test_x.py')
    assert set(catalog.changed_test_evidence({3}))=={'TestA.test_one'}
    shadow=Catalog('from torch.foo import target\nclass TestA:\n'
        '    def test_one(self, target): assert target() == 2\n','test/test_x.py')
    assert not shadow.references('TestA.test_one',{'torch.foo':{'target'}})


def test_added_method_spacing_does_not_become_shared_class_evidence():
    catalog=Catalog('class TestX:\n    def test_old(self): assert True\n\n'
                    '    # new regression\n    def test_new(self): assert True\n','test/test_x.py')
    assert set(catalog.changed_test_evidence({3,4,5}))=={'TestX.test_new'}
    value=Catalog('def test_value():\n    data = """first\n# content\nlast"""\n    assert data\n','test/test_x.py')
    assert value.changed_test_evidence({3})['test_value']['relevance']==0


def test_explicit_regressions_prevent_broad_cpp_path_budget_padding(tmp_path, monkeypatch):
    monkeypatch.setattr(selection,'cpp_test_catalog',lambda root:({},[]))
    report={'p0':[{'path':'test/distributed/test_pybackend.py','test_nodes':['TestOptions.test_none'],'relevance':0}],
            'p1':[],'analysis_gaps':[],'ci_recommendations':[]}
    unrelated=selection.TestInfo(path='test/distributed/test_other_backend.py',test_nodes={'TestOther.test_big':(1,20)})
    selection.add_cpp_candidates(tmp_path,report,['torch/csrc/distributed/c10d/init.cpp'],[unrelated],'',1)
    assert not report['p1'] and report['ci_recommendations']


def test_shared_helper_reference_cache_keeps_each_callers_identity():
    catalog=Catalog('from torch.foo import target\ndef helper(): return target()\n'
        'class TestX:\n    def test_a(self): assert helper()==2\n'
        '    def test_b(self): assert helper()==2\n','test/test_x.py')
    a=catalog.references('TestX.test_a',{'torch.foo':{'target'}})
    b=catalog.references('TestX.test_b',{'torch.foo':{'target'}})
    assert a[0]['chain']==['TestX.test_a','helper']
    assert b[0]['chain']==['TestX.test_b','helper'] and b[0]['test_node']=='TestX.test_b'


def test_local_helper_shadowing_and_fixture_arguments_do_not_create_global_call_edges():
    catalog=Catalog('def helper(): return 1\nclass TestX:\n'
        '    def test_local(self):\n        def helper(): return 2\n        assert helper()==2\n'
        '    def test_argument(self, helper): assert helper()==3\n','test/test_x.py')
    assert [n for n,_ in catalog.reachable('TestX.test_local')]==['TestX.test_local','TestX.test_local.helper']
    assert [n for n,_ in catalog.reachable('TestX.test_argument')]==['TestX.test_argument']


def test_import_smoke_tests_do_not_fill_the_budget_with_redundant_cases(tmp_path):
    report=report_for(tmp_path,'from torch.foo import unrelated\nclass TestX:\n'
        '    def test_a(self): unrelated()\n    def test_b(self): unrelated()\n',lines=())
    assert len(report['p1'])==1 and len(report['redundant_canaries'])==1


def test_fixture_setup_and_nested_helpers_have_distinct_evidence():
    text='class T(unittest.TestCase):\n    def setUp(self): pass\n    def test_one(self):\n        def helper(): return 1\n        assert helper() == 1\n'
    catalog=Catalog(text,'test/test_fixture.py')
    assert set(catalog.tests)=={'T.test_one'}
    assert catalog.changed_test_evidence({2})['T.test_one']['relevance']==2
    assert catalog.changed_test_evidence({4})['T.test_one']['chain']==['T.test_one','T.test_one.helper']


def test_tools_index_collects_and_runs_without_shadowing_installed_torch(tmp_path):
    root=tmp_path/'source';(root/'tools/test').mkdir(parents=True);(root/'tools/linter').mkdir()
    (root/'torch').mkdir();(root/'torch/__init__.py').write_text('raise AssertionError("unbuilt source torch imported")')
    (root/'tools/linter/helper.py').write_text('def answer(): return 42')
    path=root/'tools/test/test_linter.py'
    path.write_text('from tools.linter.helper import answer\nclass TestTool:\n    def test_answer(self): assert answer() == 42\n')
    infos=selection.analyze_tests(root,[])
    assert [i.path for i in infos]==['tools/test/test_linter.py']
    resolver=ExecutionDeviceResolver(make_device_resolver(root))
    assert resolver('tools/test/test_linter.py','TestTool.test_answer')==['cpu']
    runner=PytestRunner(sys.executable,artifact_root=tmp_path/'artifacts',timeout_seconds=20)
    runner.begin_task();tf='tools/test/test_linter.py::TestTool::test_answer'
    assert runner.discover_runtime_nodes(tf,'cpu',source_root=root)==[tf]
    outcome=runner.run_function(tf,'cpu',source_root=root,source_test_function=tf)
    assert outcome.completed and outcome.counts.pass_count==1


def test_pytest_collected_unittest_template_is_not_an_executable_case(tmp_path):
    (tmp_path/'test').mkdir()
    (tmp_path/'test/test_template.py').write_text('import unittest\nclass TestTemplate(unittest.TestCase):\n'
        '    def test_case(self, device, dtype): raise AssertionError("template must not run")\n')
    runner=PytestRunner(sys.executable,artifact_root=tmp_path/'artifacts',timeout_seconds=20);runner.begin_task()
    import pytest
    with pytest.raises(NoRuntimeNodes,match='uninstantiated_unittest_templates=1'):
        runner.discover_runtime_nodes('test/test_template.py::TestTemplate::test_case','cpu',source_root=tmp_path)


def test_backfill_prefers_same_affected_behavior_at_same_evidence_level(config,queue,paths,tmp_path):
    a='test/test_x.py::TestX::test_absent';b='test/test_x.py::TestX::test_other';c='test/test_x.py::TestX::test_same'
    manifest=[candidate(a,0),candidate(b,1),candidate(c,2)]
    manifest[1]['selection']['groups']=['other_behavior']
    class Runner(FakeRunner):
        def discover_runtime_nodes(self,tf,dev,**kwargs):
            if tf==a:raise NoRuntimeNodes('absent')
            return [tf]
    task,executor=setup(config,queue,paths,tmp_path,Runner(),manifest)
    result=executor._run(task,tmp_path)
    assert [r.test_function for r in result.function_results]==[a,c]


def test_invalid_source_is_not_a_silent_empty_pool(tmp_path):
    (tmp_path/'tools/test').mkdir(parents=True)
    (tmp_path/'tools/test/test_broken.py').write_text('def test_broken(:')
    infos=selection.analyze_tests(tmp_path,[])
    report=selection.build_report('a'*40,'b'*40,[],infos,tmp_path,1,5000,6)
    assert not report['catalog']['complete']
    assert report['analysis_gaps'][0]['path']=='tools/test/test_broken.py'


def test_changed_parameters_are_sampled_and_strata_spread():
    nodes=['test/test_x.py::test_x[cpu_float32_'+str(i)+']' for i in range(120)]
    nodes+=['test/test_x.py::test_x[npu_float64_boundary]']
    picked=sample_instances(nodes,100,['float64','boundary'])
    assert picked[0].endswith('[npu_float64_boundary]') and len(picked)==100
    assert picked==sample_instances(reversed(nodes),100,['float64','boundary'])


def candidate(tf, rank, dev='cpu', budget=1):
    return dict(test_function=tf,device=dev,selection=dict(rank=rank,budget=budget,relevance=1,groups=['target'],evidence=[],focus=[]))


def test_reserve_metadata_survives_queue_validation():
    item=candidate('test/test_x.py::TestX::test_x',3)
    assert validate_manifest([item])==[item]


def test_missing_candidate_backfills_while_other_execution_runs(config,queue,paths,tmp_path):
    a='test/test_x.py::TestX::test_absent';b='test/test_x.py::TestX::test_long';c='test/test_x.py::TestX::test_reserve'
    started, replacement=threading.Event(),threading.Event()
    class Runner(FakeRunner):
        def discover_runtime_nodes(self,tf,dev,**kwargs):
            if tf==a:
                assert started.wait(5)
                raise NoRuntimeNodes('successful file collection, source absent')
            return [tf]
        def run_function(self,tf,dev,**kwargs):
            if tf==b:
                started.set();assert replacement.wait(5)
            if tf==c:replacement.set()
            return super().run_function(tf,dev,**kwargs)
    task,executor=setup(config,queue,paths,tmp_path,Runner(),[candidate(a,0,budget=2),candidate(b,1,budget=2),candidate(c,2,budget=2)],workers=4)
    submission=executor._run(task,tmp_path)
    assert replacement.is_set()
    assert {r.test_function for r in submission.function_results}=={a,b,c}
    assert next(r for r in submission.function_results if r.test_function==a).status=='no_result'
    assert sum(r.counts.pass_count for r in submission.function_results if r.counts)==2


def test_environment_failure_does_not_silently_replace_high_priority_candidate(config,queue,paths,tmp_path):
    a='test/test_x.py::TestX::test_a';b='test/test_x.py::TestX::test_b'
    class Runner(FakeRunner):
        def discover_runtime_nodes(self,tf,dev,**kwargs):
            raise RuntimeError('missing import; not an absent source node')
    task,executor=setup(config,queue,paths,tmp_path,Runner(),[candidate(a,0),candidate(b,1)])
    submission=executor._run(task,tmp_path)
    assert [r.test_function for r in submission.function_results]==[a]
    assert submission.function_results[0].status=='no_result'


def test_deterministic_empty_plan_stops_after_one_attempt_with_explanation(config,queue):
    import queue as queue_module
    from types import SimpleNamespace
    from test_pipeline import _pending_analysis_task
    from run_test.selection_pool import SelectionPool
    from run_test.task_staging import task_root
    task=_pending_analysis_task(queue)
    pool=SimpleNamespace(config=config,store=queue,retry_seconds=30,max_failures=3,
                         events=queue_module.SimpleQueue())
    SelectionPool._finish(pool,task,4)
    assert queue.get_task(task.task_id).state=='blocked'
    assert 'needs review' in queue.get_task(task.task_id).error_reason
    assert json.loads((task_root(config,task)/'selection-state.json').read_text())['failures']==1


def test_one_source_with_two_channels_uses_one_budget_and_does_not_double_backfill(config,queue,paths,tmp_path):
    a='test/test_x.py::TestX::test_a';b='test/test_x.py::TestX::test_b';c='test/test_x.py::TestX::test_c'
    class Runner(FakeRunner):
        def discover_runtime_nodes(self,tf,dev,**kwargs):
            if tf==a:raise NoRuntimeNodes('absent')
            return [tf]
    runner=Runner()
    task,executor=setup(config,queue,paths,tmp_path,runner,[candidate(a,0),candidate(a,0,'accelerator'),candidate(b,1),candidate(c,2)])
    result=executor._run(task,tmp_path)
    assert [r.test_function for r in result.function_results].count(b)==1
    assert c not in [r.test_function for r in result.function_results]
