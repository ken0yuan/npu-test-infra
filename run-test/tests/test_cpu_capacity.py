import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from run_test import cpu_capacity as capacity
from run_test.test_scheduler import CardPool, TestJob, run_ordered


@pytest.fixture
def proc(tmp_path, monkeypatch):
    p = tmp_path / 'proc'
    (p / 'self').mkdir(parents=True)
    monkeypatch.setattr(capacity.os, 'cpu_count', lambda: 256)
    monkeypatch.setattr(capacity.os, 'sched_getaffinity', lambda _: set(range(16)), raising=False)
    return p


def test_nested_cgroup_v2_uses_tightest_ancestor_quota(proc, tmp_path):
    mount = tmp_path / 'cgroup'
    child = mount / 'team/job'; child.mkdir(parents=True)
    (proc/'self/cgroup').write_text('0::/team/job\n')
    (proc/'self/mountinfo').write_text(f'1 0 0:1 / {mount} rw - cgroup2 cgroup rw\n')
    (mount/'cpu.max').write_text('max 100000')
    (mount/'team/cpu.max').write_text('350000 100000')
    (child/'cpu.max').write_text('800000 100000')
    assert capacity.cpu_capacity(proc) == {
        'available_cpus':3, 'limits':{'online':256,'affinity':16,'quota':3}}


def test_container_cgroup_v1_mount_root_is_not_appended_twice(proc, tmp_path):
    mount = tmp_path / 'cpu'; mount.mkdir()
    (proc/'self/cgroup').write_text('12:cpu,cpuacct:/docker/example\n')
    (proc/'self/mountinfo').write_text(f'1 0 0:1 /docker/example {mount} rw - cgroup cgroup rw,cpu,cpuacct\n')
    (mount/'cpu.cfs_quota_us').write_text('200000')
    (mount/'cpu.cfs_period_us').write_text('100000')
    assert capacity.cpu_capacity(proc)['available_cpus'] == 2
    (mount/'cpu.cfs_quota_us').write_text('-1')
    assert capacity.cpu_capacity(proc)['available_cpus'] == 16


def test_no_linux_cgroup_or_affinity_falls_back_to_online_cpus(proc, monkeypatch):
    def missing(_): raise OSError('unavailable')
    monkeypatch.setattr(capacity.os, 'sched_getaffinity', missing)
    assert capacity.cpu_capacity(proc)['available_cpus'] == 256
    monkeypatch.setattr(capacity.os, 'cpu_count', lambda: None)
    assert capacity.cpu_capacity(proc)['available_cpus'] == 1


def test_auto_and_explicit_worker_limits_respect_available_cpus(monkeypatch):
    monkeypatch.setattr(capacity, 'cpu_capacity', lambda: {'available_cpus':3,'limits':{'online':256}})
    assert capacity.worker_capacity('auto')['max_workers'] == 3
    assert capacity.worker_capacity(8)['max_workers'] == 3
    assert capacity.worker_capacity(2)['max_workers'] == 2
    for value in [0,-1,True,'bad']:
        with pytest.raises(ValueError): capacity.worker_capacity(value)


def test_explicit_thread_settings_cost_more_budget(monkeypatch):
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        monkeypatch.delenv(name, raising=False)
    assert capacity.thread_budget(capacity.test_thread_env()) == 1
    monkeypatch.setenv('OMP_NUM_THREADS','4')
    env = capacity.test_thread_env({'MKL_NUM_THREADS':'8'})
    assert env['OMP_NUM_THREADS'] == '4'
    assert capacity.thread_budget(env) == 8


def test_dynamic_cpu_budget_grows_admissions_and_counts_npu_host_threads(tmp_path):
    budget = [4]
    release = threading.Event(); two = threading.Event(); four = threading.Event()
    lock = threading.Lock(); active = 0; peak = 0
    def execute(lease):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak,active)
            if active == 2: two.set()
            if active == 4: four.set()
        try:
            assert release.wait(10)
            return lease.cards
        finally:
            with lock: active -= 1
    pool = CardPool(tmp_path, ids=[0], is_free=lambda _: True)
    jobs = [TestJob('npu',1,execute,cpu_threads=2)] + [TestJob(f'cpu-{i}',0,execute,cpu_threads=2) for i in range(5)]
    with ThreadPoolExecutor(1) as executor:
        result = executor.submit(run_ordered,jobs,pool,max_workers=16,cpu_budget=lambda:budget[0],interval=.01)
        try:
            assert two.wait(5)
            assert active == 2
            budget[0] = 8
            assert four.wait(5)
            assert active == 4
            budget[0] = 2
        finally:
            release.set()
        assert result.result(timeout=10) == [(0,),(),(),(),(),()]
    assert peak == 4


def test_cpu_quota_shrink_drains_before_new_admissions(tmp_path):
    budget = [2]; starts=[]; releases=[threading.Event() for _ in range(3)]
    initial=threading.Event(); shrunk=threading.Event(); first_done=threading.Event()
    def execute(i, lease):
        starts.append(i)
        if len(starts)==2: initial.set()
        assert releases[i].wait(10)
        return i
    def event(**data):
        if data['state']=='capacity_changed': shrunk.set()
        if data['state']=='completed' and data['key']=='0': first_done.set()
    jobs=[TestJob(str(i),0,lambda lease,i=i:execute(i,lease)) for i in range(3)]
    with ThreadPoolExecutor(1) as executor:
        future=executor.submit(run_ordered,jobs,CardPool(tmp_path,ids=[]),max_workers=3,
                               cpu_budget=lambda:budget[0],event=event,interval=.01)
        try:
            assert initial.wait(5)
            budget[0]=1
            assert shrunk.wait(5)
            releases[0].set()
            assert first_done.wait(5)
            assert starts==[0,1]
        finally:
            for release in releases: release.set()
        assert future.result(timeout=10)==[0,1,2]
