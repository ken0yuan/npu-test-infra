"""Bounded execution retains unexecuted instances without creating a baseline."""
import json
from pathlib import Path

from run_test.executor import Executor
from run_test.identifiers import Counts
from run_test.record import submit_complete_result
from run_test.db import connect
from run_test.baseline_store import baseline_db_path, get_baseline
from conftest import TF_A
from test_executor import FakeRunner, verified_task, fake_pip


class ExpandedRunner(FakeRunner):
    def __init__(self, count):
        super().__init__(); self.count = count; self.executed = []

    def discover_runtime_nodes(self, tf, dev, **kwargs):
        return [tf + f'_{i:03d}' for i in range(self.count)]

    def run_function(self, tf, dev, **kwargs):
        self.executed.append((tf, dev))
        return super().run_function(tf, dev, **kwargs)


def test_each_function_device_executes_at_most_100_and_keeps_partial_coverage(config, queue, paths, tmp_path, fake_pip):
    manifest = [{'test_function':TF_A,'device':d} for d in ['cpu','accelerator']]
    task = verified_task(queue, manifest=manifest); task = queue.claim_next()
    Path(task.work_dir).mkdir(parents=True,exist_ok=True)
    runner = ExpandedRunner(151)
    sub = Executor(config,queue_store=queue,runner=runner)._run(task,tmp_path)
    assert len(runner.executed) == 200
    for result in sub.function_results:
        assert result.counts == Counts(100,0,0,0)
        assert result.coverage_status == 'partial'
        assert len(result.instance_ids) == 100
    artifact=json.loads((paths.diagnostics_root/task.task_id/'execution-manifest.json').read_text())
    assert artifact['max_instances_per_function_device'] == 100
    assert len(artifact['items']) == 200
    assert [x['deferred_count'] for x in artifact['bounded_families']] == [51,51]
    assert all(len(x['deferred_nodeids'])==51 for x in artifact['bounded_families'])
    submit_complete_result(config, sub)
    with connect(baseline_db_path(paths.baselines_root/'test/distributed/_tensor')) as db:
        assert get_baseline(db,TF_A,'cpu') is None
        assert get_baseline(db,TF_A,'accelerator') is None


def test_family_at_limit_retains_full_coverage(config, queue, paths, tmp_path):
    task=verified_task(queue,manifest=[{'test_function':TF_A,'device':'cpu'}]);task=queue.claim_next()
    Path(task.work_dir).mkdir(parents=True,exist_ok=True)
    runner=ExpandedRunner(100)
    sub=Executor(config,queue_store=queue,runner=runner)._run(task,tmp_path)
    assert sub.function_results[0].coverage_status=='full'
    assert len(runner.executed)==100
    artifact=json.loads((paths.diagnostics_root/task.task_id/'execution-manifest.json').read_text())
    assert artifact['bounded_families']==[]
