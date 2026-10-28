from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

from run_test.executor import Executor
from run_test.npu_readiness import ProbeResult
from run_test.test_scheduler import CardPool
from run_test.process_lock import lease_handle
from run_test.record import submit_complete_result
from run_test.db import connect
from run_test.baseline_store import baseline_db_path, get_baseline
from run_test.identifiers import Counts
from conftest import TF_A, TF_B
from test_executor import FakeRunner, verified_task


class Ready:
    def __init__(self, pool):self.pool, self.calls = pool, []
    def ensure(self, lease, env):
        assert lease.cards and len(lease.fds) == len(lease.cards)
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, lease.cards))
        for card in lease.cards:
            with lease_handle(self.pool.lock_dir / f"npu-{card}.lock") as fd:
                assert fd is None, "probe ran without the OS card lease"
        self.calls.append(lease.cards)
        return ProbeResult(True)


def setup(config, queue, paths, tmp_path, runner, manifest, *, ids=(2, 4), free=lambda _: True, workers=4):
    config["executor"]["test_max_workers"] = workers
    config["executor"]["test_resource_poll_seconds"] = .01
    task = verified_task(queue, manifest=manifest); task = queue.claim_next()
    Path(task.work_dir).mkdir(parents=True, exist_ok=True)
    executor = Executor(config, queue_store=queue, runner=runner)
    executor.card_pool = CardPool(paths.queue_dir / "cards", ids=ids, is_free=free)
    executor.npu_readiness = Ready(executor.card_pool)
    return task, executor


def test_one_collected_family_releases_all_cards_then_fills_individual_cards(config, queue, paths, tmp_path):
    barrier = threading.Barrier(2)
    class Runner(FakeRunner):
        def required_npu_cards(self, node, device, total):return total if node == TF_A else 1
        def discover_runtime_nodes(self, tf, dev, **kwargs):
            assert kwargs["extra_env"]["ASCEND_RT_VISIBLE_DEVICES"] == "2,4"
            return [tf+"[one]", tf+"[two]"]
        def run_function(self, tf, dev, **kwargs):
            barrier.wait(timeout=5)
            return super().run_function(tf, dev, **kwargs)
    runner = Runner()
    task, executor = setup(config, queue, paths, tmp_path, runner,
                           [{"test_function": TF_A, "device": "accelerator"}])
    submission = executor._run(task, tmp_path)
    assert submission.function_results[0].counts.pass_count == 2
    assert {env["ASCEND_RT_VISIBLE_DEVICES"] for env in runner.extra_envs} == {"2", "4"}
    assert not executor.card_pool.reserved


def test_cpu_actual_execution_precedes_busy_npu_collection(config, queue, paths, tmp_path):
    free, cpu_ran = threading.Event(), threading.Event()
    class Runner(FakeRunner):
        def discover_runtime_nodes(self, tf, dev, **kwargs):
            if dev == "accelerator":assert free.is_set() and cpu_ran.is_set()
            return [tf]
        def run_function(self, tf, dev, **kwargs):
            if dev == "cpu":cpu_ran.set()
            return super().run_function(tf, dev, **kwargs)
    task, executor = setup(config, queue, paths, tmp_path, Runner(),
        [{"test_function": TF_A,"device":"accelerator"}, {"test_function": TF_B,"device":"cpu"}],
        ids=(4,), free=lambda _: free.is_set())
    with ThreadPoolExecutor(1) as service:
        future = service.submit(executor._run, task, tmp_path)
        try:assert cpu_ran.wait(5)
        finally:free.set()
        assert sum(r.counts.pass_count for r in future.result(timeout=10).function_results) == 2


def test_failed_card_is_local_cpu_persists_and_multicard_requirement_is_not_lowered(config, queue, paths, tmp_path):
    manifest = [{"test_function":TF_A,"device":"accelerator","required_npu_cards":1},
                {"test_function":TF_B,"device":"cpu"},
                {"test_function":TF_B,"device":"accelerator","required_npu_cards":2}]
    runner = FakeRunner();task, executor = setup(config,queue,paths,tmp_path,runner,manifest,ids=(4,6))
    class Fails:
        def ensure(self, lease, env):return ProbeResult(False,"TsdOpen 507033","HDC failed",(4,))
    executor.npu_readiness = Fails()
    submission = executor._run(task,tmp_path)
    assert len(runner.extra_envs)==1 and runner.extra_envs[0]["ASCEND_RT_VISIBLE_DEVICES"]==""
    assert all(r.status=="no_result" for r in submission.function_results if r.device=="accelerator")
    submit_complete_result(config,submission)
    with connect(baseline_db_path(paths.baselines_root/'test/distributed/fsdp')) as db:
        assert get_baseline(db,TF_B,'cpu').counts==Counts(1,0,0,0)
        assert get_baseline(db,TF_B,'accelerator') is None


def test_serial_mode_also_reserves_and_binds_nonzero_physical_card(config,queue,paths,tmp_path):
    runner=FakeRunner();task,executor=setup(config,queue,paths,tmp_path,runner,
        [{"test_function":TF_A,"device":"accelerator","required_npu_cards":1}],ids=(6,),workers=1)
    assert executor._run(task,tmp_path).function_results[0].counts.pass_count==1
    assert runner.extra_envs[0]["ASCEND_RT_VISIBLE_DEVICES"]=="6"
    assert executor.npu_readiness.calls and not executor.card_pool.reserved
