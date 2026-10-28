"""Real collection regressions for helper scopes, copied templates and devices."""
import json
import sys
from pathlib import Path

import pytest

from run_test import selection
from run_test.device_detect import make_device_resolver
from run_test.device_policy import ExecutionDeviceResolver
from run_test.runner import PytestRunner
from run_test.executor import Executor
from test_executor import verified_task, fake_pip


def test_nested_helpers_keep_outer_references_but_never_become_candidates():
    visitor = selection.TestVisitor({'torch.foo': {'changed'}})
    import ast
    visitor.visit(ast.parse('''from torch.foo import changed
def factory():
    def test(self): changed()
    return test
class TestOuter:
    def test_real(self):
        def test_fn(): return changed()
        class Local:
            def test_case(self): return changed()
        test_fn()
def test_top(): pass
'''))
    assert set(visitor.info.test_nodes) == {'TestOuter.test_real', 'test_top'}
    # A reference inside the helper still contributes to its enclosing test.
    assert {ref['test_node'] for ref in visitor.info.direct_refs} == {'TestOuter.test_real'}


def test_deferred_selection_never_imports_torch_or_claims_concrete_cases(tmp_path, monkeypatch):
    path = tmp_path / 'test/test_scope.py'
    path.parent.mkdir()
    path.write_text('class TestScope:\n    def test_real(self): pass\n')
    report = {'p0': [{'path': 'test/test_scope.py', 'test_nodes': ['TestScope.test_real'],
                       'evidence': [], 'score': 250}], 'p1': [], 'analysis_gaps': [], 'notes': []}
    monkeypatch.setattr(selection, 'discover_cases', lambda *a, **kw: pytest.fail('control must not import tests'))
    selection.finalize_plan(tmp_path, report, 15, 1, defer_runtime_discovery=True)
    assert report['selection_granularity'] == 'family'
    assert report['p0'][0]['score'] == 250
    assert not report['analysis_gaps']
    manifest = selection.plan_to_manifest(report)
    assert manifest[0]['execution_mode'] == 'resolve_family'


def test_longest_declared_class_owns_collected_cpu_and_npu_variants(tmp_path):
    path = tmp_path / 'test/test_devices.py'; path.parent.mkdir()
    path.write_text('''class TestNN:
    def test_unrelated(self): pass
class TestNNDeviceType:
    def test_norm(self, device): pass
''')
    resolver = ExecutionDeviceResolver(make_device_resolver(tmp_path))
    assert resolver('test/test_devices.py', 'TestNNDeviceTypeCPU.test_norm_cpu') == ['cpu']
    assert resolver.source_test_function('test/test_devices.py', 'TestNNDeviceTypeCPU.test_norm_cpu') == \
        'test/test_devices.py::TestNNDeviceType::test_norm'
    assert resolver('test/test_devices.py', 'TestNNDeviceTypePRIVATEUSE1.test_norm_npu') == ['accelerator']


@pytest.fixture
def copied_templates(tmp_path):
    root = tmp_path / 'source'; path = root / 'test/test_templates.py'; path.parent.mkdir(parents=True)
    path.write_text('''from functools import wraps
from unittest import TestCase
class OriginalTemplate:
    def test_copied(self): assert self.device == 'cpu'
class UnrelatedTemplate:
    def test_copied(self): assert False
def copy(fn):
    @wraps(fn)
    def wrapper(self): return fn(self)
    return wrapper
class RenamedCpu(TestCase):
    device = 'cpu'
    test_copied_cpu = copy(OriginalTemplate.test_copied)
class RenamedCuda(TestCase):
    device = 'cuda'
    test_copied_cuda = copy(OriginalTemplate.test_copied)
class RenamedMps(TestCase):
    device = 'mps'
    test_copied_mps = copy(OriginalTemplate.test_copied)
class UnrelatedCpu(TestCase):
    device = 'cpu'
    test_copied_cpu = copy(UnrelatedTemplate.test_copied)
''')
    return root


def test_copied_template_uses_exact_origin_and_excludes_other_backends(copied_templates):
    runner = PytestRunner(sys.executable, timeout_seconds=30)
    nodes = runner.discover_runtime_nodes('test/test_templates.py::OriginalTemplate::test_copied',
                                          'cpu', source_root=copied_templates)
    assert nodes == ['test/test_templates.py::RenamedCpu::test_copied_cpu']
    result = runner.run_function(nodes[0], 'cpu', source_root=copied_templates)
    assert result.completed and result.counts.pass_count == 1


def test_execution_artifact_has_separate_device_qualified_names(config, queue, paths, tmp_path, fake_pip):
    root = tmp_path / 'source'; path = root / 'test/test_plain.py'; path.parent.mkdir(parents=True)
    path.write_text('def test_plain(): pass\n')
    tf = 'test/test_plain.py::test_plain'
    class BothChannels(PytestRunner):
        # Exercise executor naming without requiring NPU in a unit test.
        def discover_runtime_nodes(self, test_function, device, **kwargs): return [test_function]
        def run_function(self, test_function, device, **kwargs):
            from run_test.runner import FunctionRunOutcome
            from run_test.identifiers import Counts
            return FunctionRunOutcome(test_function, device, True, Counts(1,0,0,0), 'real fixture result', 0)
    task = verified_task(queue, manifest=[{'test_function': tf, 'device': dev} for dev in ['cpu', 'accelerator']])
    task = queue.claim_next(); Path(task.work_dir).mkdir(parents=True, exist_ok=True)
    Executor(config, queue_store=queue, runner=BothChannels(sys.executable))._run(task, root)
    artifact = json.loads((paths.diagnostics_root / task.task_id / 'execution-manifest.json').read_text())
    assert [item['execution_name'] for item in artifact['items']] == ['cpu::'+tf, 'accelerator::'+tf]
    assert artifact['verified_against_execution']
