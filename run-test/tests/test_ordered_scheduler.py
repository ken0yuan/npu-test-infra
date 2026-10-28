import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from run_test.test_scheduler import CardPool, JobExpansion, TestJob, run_ordered
from run_test.process_lock import executor_lease, lease_handle
from run_test.task_queue import validate_manifest, QueueError


def test_first_npu_waiter_blocks_later_npu_but_admits_cpu(tmp_path):
    pool = CardPool(tmp_path, ids=range(4), is_free=lambda _: True)
    first_started = threading.Event()
    allow_first_finish = threading.Event()
    waiting = threading.Event()
    cpu_started = threading.Event()
    order = []
    events = []
    def run(name, lease):
        order.append(name)
        if name == 'cpu': cpu_started.set()
        if name == 'first':
            first_started.set()
            assert allow_first_finish.wait(10)
        return name
    def event(**data):
        events.append(data)
        if data['state'] == 'resource_wait':
            waiting.set()
    jobs = [TestJob(name, need, lambda lease, name=name: run(name, lease))
            for name, need in [('first', 3), ('head', 2), ('small', 1), ('cpu', 0)]]
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(run_ordered, jobs, pool, max_workers=4, interval=.01, event=event)
        try:
            assert first_started.wait(5) and waiting.wait(5)
            assert cpu_started.wait(5)
            assert order == ['first', 'cpu']
        finally:
            allow_first_finish.set()
        assert future.result(timeout=10) == ['first', 'head', 'small', 'cpu']
    starts = [e['key'] for e in events if e['state'] == 'started']
    assert starts == ['first', 'cpu', 'head', 'small']
    assert not pool.reserved


def test_uses_free_cards_concurrently_and_binds_disjoint_ids(tmp_path):
    pool = CardPool(tmp_path, ids=[2, 4], is_free=lambda _: True)
    barrier = threading.Barrier(2)
    def run(lease):
        barrier.wait(timeout=5)
        return lease.cards
    results = run_ordered([TestJob('a',1,run), TestJob('b',1,run)], pool, max_workers=2, interval=.01)
    assert results == [(2,), (4,)]


def test_external_busy_npu_head_does_not_block_cpu(tmp_path):
    free = threading.Event()
    waiting = threading.Event()
    pool = CardPool(tmp_path, ids=[0], is_free=lambda _: free.is_set())
    starts = []
    cpu_started = threading.Event()
    def event(**data):
        if data['state'] == 'resource_wait': waiting.set()
        if data['state'] == 'started': starts.append(data['key'])
        if data['state'] == 'started' and data['key'] == 'cpu': cpu_started.set()
    with ThreadPoolExecutor(1) as executor:
        result = executor.submit(run_ordered, [TestJob('npu',1,lambda _:1),TestJob('cpu',0,lambda _:2)],
                                 pool, interval=.01, event=event)
        try:
            assert waiting.wait(5)
            assert cpu_started.wait(5)
            assert starts == ['cpu']
        finally:
            free.set()
        assert result.result(timeout=5) == [1,2]


def test_cpu_fills_slots_behind_waiting_npu_with_combined_limit_eight(tmp_path):
    pool = CardPool(tmp_path, ids=range(2), is_free=lambda _: True)
    release = threading.Event()
    filled = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0
    starts = []
    def run(name, lease):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active); starts.append(name)
            if active == 8: filled.set()
        try:
            assert release.wait(10)
            return name
        finally:
            with lock: active -= 1
    needs = [('first',1),('head',2),('later-npu',1)] + [(f'cpu-{i}',0) for i in range(10)]
    jobs = [TestJob(name, need, lambda lease, name=name: run(name, lease)) for name,need in needs]
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(run_ordered, jobs, pool, max_workers=8, interval=.01)
        try:
            assert filled.wait(5)
            assert len(starts) == 8
            assert 'first' in starts and sum(n.startswith('cpu-') for n in starts) == 7
            assert 'head' not in starts and 'later-npu' not in starts
        finally:
            release.set()
        assert future.result(timeout=10) == [name for name,_ in needs]
    assert peak <= 8
    assert starts.index('head') < starts.index('later-npu')
    assert not pool.reserved


def test_partial_reservation_is_released_and_impossible_request_rejected(tmp_path):
    first = CardPool(tmp_path, ids=[0,1], is_free=lambda c: c == 0)
    assert first.acquire(2) is None
    other = CardPool(tmp_path, ids=[0], is_free=lambda _: True)
    lease = other.acquire(1)
    assert lease.cards == (0,)
    other.release(lease)
    with pytest.raises(ValueError): first.acquire(3)


def test_failure_drains_running_jobs_before_releasing_leases(tmp_path):
    pool = CardPool(tmp_path, ids=[0,1], is_free=lambda _: True)
    barrier = threading.Barrier(2)
    def fail(lease):
        barrier.wait(timeout=5)
        raise RuntimeError('test error')
    def finish(lease):
        barrier.wait(timeout=5)
        return True
    with pytest.raises(RuntimeError, match='test error'):
        run_ordered([TestJob('a',1,fail),TestJob('b',1,finish)],pool,interval=.01)
    assert not pool.reserved
    lease = pool.acquire(2)
    assert len(lease.cards) == 2
    pool.release(lease)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX inherited flock')
def test_child_keeps_environment_lease_after_parent_context_closes(tmp_path):
    path = tmp_path/'executor.lock'
    with lease_handle(path) as fd:
        child = subprocess.Popen([sys.executable,'-c','import sys; print("ready",flush=True);sys.stdin.read()'],
                                 pass_fds=(fd,),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        assert child.stdout.readline().strip() == 'ready'
    try:
        with executor_lease(path) as acquired:
            assert not acquired
    finally:
        child.communicate('',timeout=5)
    with executor_lease(path) as acquired:
        assert acquired


def test_resource_manifest_validation():
    row = dict(test_function='test/test_a.py::test_one', device='accelerator', required_npu_cards=4)
    assert validate_manifest([row]) == [row]
    for value in [0,-1,True,'2']:
        with pytest.raises(QueueError): validate_manifest([{**row,'required_npu_cards':value}])


def test_cpu_collection_expands_and_executes_while_npu_collection_waits(tmp_path):
    free = threading.Event()
    cpu_finished = threading.Event()
    waiting = threading.Event()
    pool = CardPool(tmp_path, ids=[4], is_free=lambda _: free.is_set())
    def execute_cpu(_):
        cpu_finished.set()
        return 'cpu result'
    def event(**data):
        if data['state'] == 'resource_wait' and data['phase'] == 'collect':
            waiting.set()
    jobs = [
        TestJob('npu collect', 1, lambda _: JobExpansion([
            TestJob('npu instance', 1, lambda lease: lease.cards)]), phase='collect'),
        TestJob('cpu collect', 0, lambda _: JobExpansion([
            TestJob('cpu instance', 0, execute_cpu)]), phase='collect'),
    ]
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(run_ordered, jobs, pool, max_workers=2, interval=.01, event=event)
        try:
            assert waiting.wait(5) and cpu_finished.wait(5)
            assert not future.done()
        finally:
            free.set()
        result = future.result(timeout=5)
    assert result[0].results == [(4,)]
    assert result[1].results == ['cpu result']


def test_collection_releases_cards_before_expanded_instances_fill_pool(tmp_path):
    pool = CardPool(tmp_path, ids=[2, 4], is_free=lambda _: True)
    barrier = threading.Barrier(2)
    events = []
    def execute(lease):
        barrier.wait(timeout=5)
        return lease.cards
    def collect(lease):
        assert lease.cards == (2, 4)
        return JobExpansion([TestJob('instance-a', 1, execute), TestJob('instance-b', 1, execute)],
                            metadata={'selected': 2})
    def event(**data):
        events.append(data)
        if data['state'] == 'expanded':
            assert not pool.reserved
            # Physical file locks, not just the in-memory set, are released.
            other = CardPool(tmp_path, ids=[2, 4], is_free=lambda _: True)
            lease = other.acquire(2)
            assert lease is not None
            other.release(lease)
    result = run_ordered([TestJob('collect', 2, collect, phase='collect')], pool,
                         max_workers=2, interval=.01, event=event)
    assert result[0].metadata == {'selected': 2}
    assert result[0].results == [(2,), (4,)]
    assert [e['phase'] for e in events if e['state'] == 'started'] == ['collect', 'execute', 'execute']
    assert not pool.reserved


def test_quarantined_card_finishes_unavailable_multicard_job_and_cpu_continues(tmp_path):
    pool = CardPool(tmp_path, ids=[2, 4], is_free=lambda _: True)
    ran = []
    def collect(lease):
        pool.quarantine([2], 'TSD initialization failed on physical card 2')
        return JobExpansion([
            TestJob('needs-two', 2, lambda _: pytest.fail('must not reduce card requirement'),
                    on_unavailable=lambda reason: {'status': 'no_result', 'reason': reason}),
            TestJob('healthy-one', 1, lambda child: child.cards),
            TestJob('cpu', 0, lambda _: ran.append('cpu') or 'cpu result'),
        ])
    result = run_ordered([TestJob('collect', 2, collect, phase='collect')], pool,
                         max_workers=3, interval=.01)
    rejected, healthy, cpu = result[0].results
    assert rejected['status'] == 'no_result' and 'needs 2' in rejected['reason']
    assert 'healthy pool has 1/2' in rejected['reason']
    assert healthy == (4,) and cpu == 'cpu result' and ran == ['cpu']
    assert pool.unavailable_reason(1) is None and not pool.reserved
    assert pool.acquire(2) is None


def test_quarantine_without_handler_is_not_an_infinite_resource_wait(tmp_path):
    pool = CardPool(tmp_path, ids=[4], is_free=lambda _: True)
    pool.quarantine([4], 'device init failed')
    with pytest.raises(ValueError, match='healthy pool has 0/1'):
        run_ordered([TestJob('npu', 1, lambda _: pytest.fail('unavailable'))], pool, interval=.01)
    with pytest.raises(ValueError, match='outside the visible pool'):
        pool.quarantine([8], 'wrong mapping')


def test_expansion_preserves_parent_and_child_order_and_one_cpu_budget(tmp_path):
    pool = CardPool(tmp_path, ids=[])
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    active = peak = 0
    def execute(value):
        def run(_):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait(timeout=5)
                return value
            finally:
                with lock:
                    active -= 1
        return run
    first = JobExpansion([TestJob('a1', 0, execute('a1')), TestJob('a2', 0, execute('a2'))], metadata='first')
    second = JobExpansion([TestJob('b1', 0, execute('b1')), TestJob('b2', 0, execute('b2'))], metadata='second')
    result = run_ordered([
        TestJob('first collect', 0, lambda _: first, phase='collect'),
        TestJob('second collect', 0, lambda _: second, phase='collect'),
    ], pool, max_workers=8, cpu_budget=lambda: 2, interval=.01)
    assert [item.metadata for item in result] == ['first', 'second']
    assert [item.results for item in result] == [['a1', 'a2'], ['b1', 'b2']]
    assert peak == 2


def test_expanded_failure_drains_other_children_before_releasing_card_leases(tmp_path):
    pool = CardPool(tmp_path, ids=[2, 4], is_free=lambda _: True)
    barrier = threading.Barrier(2)
    finished = threading.Event()
    failed_released = threading.Event()
    allow_finish = threading.Event()
    original_release = pool.release
    def release(lease):
        original_release(lease)
        if lease.cards == (2,):
            failed_released.set()
    pool.release = release
    def fail(_):
        barrier.wait(timeout=5)
        raise RuntimeError('expanded failure')
    def finish(_):
        barrier.wait(timeout=5)
        assert allow_finish.wait(10)
        finished.set()
        return True
    collector = TestJob('collect', 2, lambda _: JobExpansion([
        TestJob('fail', 1, fail), TestJob('finish', 1, finish)]), phase='collect')
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(run_ordered, [collector], pool, max_workers=2, interval=.01)
        try:
            assert failed_released.wait(5)
            assert not future.done() and not finished.is_set()
            assert pool.reserved == {4}
            other = CardPool(tmp_path, ids=[2, 4], is_free=lambda _: True)
            assert other.acquire(2) is None
        finally:
            allow_finish.set()
        with pytest.raises(RuntimeError, match='expanded failure'):
            future.result(timeout=5)
    assert finished.is_set() and not pool.reserved
    lease = pool.acquire(2)
    assert lease is not None
    pool.release(lease)


def test_event_failure_releases_newly_acquired_lease(tmp_path):
    pool = CardPool(tmp_path, ids=[2], is_free=lambda _: True)
    def event(**data):
        if data['state'] == 'started':
            raise RuntimeError('event output failed')
    with pytest.raises(RuntimeError, match='event output failed'):
        run_ordered([TestJob('npu', 1, lambda _: True)], pool, event=event)
    assert not pool.reserved
    lease = pool.acquire(1)
    assert lease is not None
    pool.release(lease)
