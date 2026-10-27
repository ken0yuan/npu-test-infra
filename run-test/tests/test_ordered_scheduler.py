import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from run_test.test_scheduler import CardPool, TestJob, run_ordered
from run_test.process_lock import executor_lease, lease_handle
from run_test.task_queue import validate_manifest, QueueError


def test_first_waiter_stops_later_small_and_cpu_jobs(tmp_path):
    pool = CardPool(tmp_path, ids=range(4), is_free=lambda _: True)
    first_started = threading.Event()
    allow_first_finish = threading.Event()
    waiting = threading.Event()
    order = []
    events = []
    def run(name, lease):
        order.append(name)
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
            assert order == ['first']
        finally:
            allow_first_finish.set()
        assert future.result(timeout=10) == ['first', 'head', 'small', 'cpu']
    starts = [e['key'] for e in events if e['state'] == 'started']
    assert starts == ['first', 'head', 'small', 'cpu']
    assert not pool.reserved


def test_uses_free_cards_concurrently_and_binds_disjoint_ids(tmp_path):
    pool = CardPool(tmp_path, ids=[2, 4], is_free=lambda _: True)
    barrier = threading.Barrier(2)
    def run(lease):
        barrier.wait(timeout=5)
        return lease.cards
    results = run_ordered([TestJob('a',1,run), TestJob('b',1,run)], pool, max_workers=2, interval=.01)
    assert results == [(2,), (4,)]


def test_external_busy_head_resumes_without_admitting_cpu(tmp_path):
    free = threading.Event()
    waiting = threading.Event()
    pool = CardPool(tmp_path, ids=[0], is_free=lambda _: free.is_set())
    starts = []
    def event(**data):
        if data['state'] == 'resource_wait': waiting.set()
        if data['state'] == 'started': starts.append(data['key'])
    with ThreadPoolExecutor(1) as executor:
        result = executor.submit(run_ordered, [TestJob('npu',1,lambda _:1),TestJob('cpu',0,lambda _:2)],
                                 pool, interval=.01, event=event)
        try:
            assert waiting.wait(5)
            assert starts == []
        finally:
            free.set()
        assert result.result(timeout=5) == [1,2]


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
