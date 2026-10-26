"""pipeline 节拍器：三阶段顺序、单任务容错、make_runner 配置语义。"""

from __future__ import annotations

import itertools
import os
import threading
from pathlib import Path
from types import SimpleNamespace

from run_test import executor as executor_mod
from run_test import pipeline
from run_test import prefilter as prefilter_mod
from run_test.cli import build_parser

from conftest import COMMIT_A, COMMIT_NPU

SHA10 = COMMIT_A[:10]
WHEELS_URL = f"obs://pytorch-ci-cache/7day-cache/wheels-{SHA10}.tar.gz"
SOURCE_URL = f"obs://pytorch-ci-cache/7day-cache/pytorch-source-{SHA10}.tar.gz"

_version_counter = itertools.count(1)


def _pending_analysis_task(queue):
    """造一个停在 pending_analysis 的任务（不提交清单）。

    同 URL 不同 version 视为不同对象（与 executor 测试的去重规避一致）。
    """
    ver = f"pipe-v{next(_version_counter)}"
    task, _ = queue.register_candidate(
        WHEELS_URL, SOURCE_URL,
        obs_wheels_version=ver, obs_source_version=ver,
    )
    queue.register_identity_rule("r", "规则", "测试 fixture")
    queue.verify_identity(
        task.task_id, "r",
        {"pr_number": 123456, "pytorch_commit": COMMIT_A,
         "torch_npu_commit": COMMIT_NPU, "binding": "x"},
    )
    queue.prefilter_outcome(task.task_id, True)
    return queue.get_task(task.task_id)


def _fake_prefilter(calls):
    def _run(config, queue=None):
        calls.append("prefilter")
        return {"processed": 3, "kept": 2, "filtered_out": 1, "errors": []}
    return _run



import json
import sys
import time
import pytest
from run_test import selection_pool, task_staging
from run_test.process_lock import executor_lease
from run_test.config import store_paths
from conftest import FakeDownloader, TF_A
from test_executor import FakeRunner, fake_pip, verified_task


def wait_until(predicate, timeout=8):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate(): return
        time.sleep(.05)
    raise AssertionError('fixture condition did not complete')


@pytest.fixture
def pool_factory(config, monkeypatch, tmp_path):
    monkeypatch.setattr(prefilter_mod, 'prefilter_pending_tasks', lambda *a, **k: {'processed':0})
    script=tmp_path/'selection-fixture.py'
    script.write_text("""
import sys,time,json
from pathlib import Path
from run_test.config import load_config
from run_test.task_queue import QueueStore
config=load_config(sys.argv[1]); tid=sys.argv[2]
gate=Path(sys.argv[3]); slow=sys.argv[4]; fail=sys.argv[5]
sys.stderr.write('X'*1100000);sys.stderr.flush()
if tid==slow:
 while not gate.exists(): time.sleep(.05)
if fail=='yes': sys.exit(7)
QueueStore(config).submit_manifest(tid,[{'test_function':sys.argv[6], 'device':'cpu'}])
""",encoding='utf-8')
    pools=[]
    def start(slow='', fail=False):
        gate=tmp_path/'release-selection'
        monkeypatch.setattr(selection_pool,'selection_command',lambda cp,tid:
                            [sys.executable,str(script),str(cp),tid,str(gate),slow,
                             'yes' if fail else 'no',TF_A])
        pool=selection_pool.SelectionPool(config);pools.append(pool)
        return pool,gate
    yield start
    for pool in pools:
        pool.stop.set();pool.thread.join(3)
        for _,proc,stream in list(pool.running.values()):
            # Only processes created by this isolated fixture, never server tasks.
            if proc.poll() is None: proc.terminate()
            proc.wait(timeout=5);stream.close()
        pool.close()


def test_stuck_selection_does_not_block_next_or_new_arrival(config,queue,pool_factory):
    slow=_pending_analysis_task(queue); fast=_pending_analysis_task(queue)
    pool,gate=pool_factory(slow.task_id)
    wait_until(lambda:queue.get_task(fast.task_id).state=='pending_test')
    assert queue.get_task(slow.task_id).state=='pending_analysis'
    assert len(pool.running)<=2
    queue.mark_blocked(fast.task_id,'fixture consumed ready slot')
    new=_pending_analysis_task(queue)
    wait_until(lambda:queue.get_task(new.task_id).state=='pending_test')
    assert not gate.exists()
    assert (task_staging.task_root(config,fast)/'select-one.log').stat().st_size>=1100000
    gate.touch()
    wait_until(lambda:queue.get_task(slow.task_id).state=='pending_test')


def test_prefetch_bound_includes_completed_ready_tasks(config,queue,pool_factory):
    tasks=[_pending_analysis_task(queue) for _ in range(5)]
    pool,_=pool_factory()
    wait_until(lambda:len(queue.list_tasks('pending_test'))==2)
    time.sleep(.5)
    assert len(queue.list_tasks('pending_test'))==2
    assert len(queue.list_tasks('pending_analysis'))==3


def test_surviving_worker_counts_against_ready_plus_prefetch_bound(config,queue,pool_factory):
    surviving=_pending_analysis_task(queue)
    ready=_pending_analysis_task(queue)
    queue.submit_manifest(ready.task_id,[{'test_function':TF_A,'device':'cpu'}])
    waiting=_pending_analysis_task(queue)
    # The OS lock represents a child surviving a previous coordinator process.
    with executor_lease(task_staging.task_root(config,surviving)/'.selection.lock') as held:
        assert held
        pool,_=pool_factory()
        assert pool.started.wait(2)
        time.sleep(.6)
        assert queue.get_task(waiting.task_id).state=='pending_analysis'
        assert len(queue.list_tasks('pending_test'))==1


def test_selection_failure_is_persistent_bounded_retry(config,queue,pool_factory):
    config['selector']={'max_failures':1}
    task=_pending_analysis_task(queue);pool,_=pool_factory(fail=True)
    wait_until(lambda:queue.get_task(task.task_id).state=='blocked')
    state=json.loads((task_staging.task_root(config,task)/'selection-state.json').read_text())
    assert state['failures']==1 and state['last_exit']==7


def test_pipeline_runs_executor_while_selection_still_busy(config,queue,monkeypatch,pool_factory):
    task=_pending_analysis_task(queue);pool,gate=pool_factory(task.task_id)
    wait_until(lambda:bool(pool.running))
    monkeypatch.setattr(selection_pool,'get_pool',lambda c:pool)
    class Executor:
        def __init__(self,*a,**k):self.calls=0
        def run_once(self):
            self.calls+=1
            assert not gate.exists()
            return SimpleNamespace(task_id='ready',outcome='done' if self.calls==1 else 'idle',detail='')
    monkeypatch.setattr(executor_mod,'Executor',Executor)
    started=time.monotonic();report=pipeline.pipeline_once(config)
    assert time.monotonic()-started<2
    assert [r['outcome'] for r in report['executed']]==['done']
    assert task.task_id in report['in_flight']


def test_selection_lock_releases_on_exception_and_busy(config,queue,monkeypatch):
    task=_pending_analysis_task(queue)
    lock=task_staging.task_root(config,task)/'.selection.lock'
    with executor_lease(lock) as acquired:
        assert acquired
        assert pipeline.run_selection(config,queue,task.task_id)==3
    def fail(*a,**k):raise ValueError('fixture failure')
    monkeypatch.setattr(pipeline,'_run_selection_body',fail)
    with pytest.raises(ValueError,match='fixture failure'):
        pipeline.run_selection(config,queue,task.task_id)
    monkeypatch.setattr(pipeline,'_run_selection_body',lambda *a,**k:0)
    assert pipeline.run_selection(config,queue,task.task_id)==0


def test_stage_identity_checksum_and_partial_download(config,queue):
    task=_pending_analysis_task(queue);dl=FakeDownloader()
    path=task_staging.stage_package(config,task,'pytorch-source.tar.gz',dl)
    task_staging.stage_package(config,task,path.name,dl)
    assert len(dl.calls)==1
    path.write_bytes(b'corrupted')
    assert task_staging.verified_package(config,task,path.name) is None
    task_staging.stage_package(config,task,path.name,dl)
    assert len(dl.calls)==2
    from dataclasses import replace
    changed=replace(task,pytorch_commit='b'*40)
    assert task_staging.verified_package(config,changed,path.name) is None
    class Failed:
        def download(self,url,dest):dest.write_bytes(b'partial');raise OSError('network')
    with pytest.raises(OSError):task_staging.stage_package(config,task,'wheels.tar.gz',Failed())
    assert not (path.parent/'wheels.tar.gz.partial').exists()
    assert task_staging.verified_package(config,task,'wheels.tar.gz') is None


def test_executor_reuses_downloads_and_preserves_audit(config,queue,paths,fake_pip):
    task=verified_task(queue);dl=FakeDownloader()
    for name in ['pytorch-source.tar.gz','wheels.tar.gz']:
        task_staging.stage_package(config,task,name,dl)
    audit=task_staging.task_root(config,task)/'test-plan.json';audit.write_text('{}')
    before=list(dl.calls)
    executor=executor_mod.Executor(config,queue_store=queue,downloader=dl,runner=FakeRunner())
    assert executor.run_once().outcome=='done'
    assert dl.calls==before
    assert audit.read_text()=='{}'
    assert not (audit.parent/'source-staging').exists()
    assert not (paths.task_work_root/task.task_id).exists()


def test_cleanup_failure_keeps_cleaning_and_evidence(config,queue,paths,fake_pip,monkeypatch):
    task=verified_task(queue);dl=FakeDownloader()
    task_staging.stage_package(config,task,'pytorch-source.tar.gz',dl)
    original=task_staging.remove_staging
    def fail(*a):raise OSError('fixture cleanup failure')
    monkeypatch.setattr(task_staging,'remove_staging',fail)
    executor=executor_mod.Executor(config,queue_store=queue,downloader=dl,runner=FakeRunner())
    assert executor.run_once().outcome=='cleaning'
    assert queue.get_task(task.task_id).state=='cleaning'
    assert paths.pr_db.exists()
    monkeypatch.setattr(task_staging,'remove_staging',original)
    executor.run_once()
    assert queue.get_task(task.task_id).state=='done'


def test_actual_selection_entry_reads_effective_config_without_network(config,tmp_path):
    import subprocess
    cp=tmp_path/'custom-config.json';task_staging.atomic_json(cp,config)
    cmd=selection_pool.selection_command(cp,'missing-task')
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
    assert result.returncode==2
    assert store_paths(config).queue_db.exists()
    assert '--config' in cmd and str(cp) in cmd


def test_aborted_task_stays_failed_after_cleanup_retry(config,queue,paths,monkeypatch):
    task=verified_task(queue)
    task=queue.claim_next()
    work=Path(task.work_dir);work.mkdir(parents=True)
    (work/'failed-install.txt').write_text('fixture')
    executor=executor_mod.Executor(config,queue_store=queue)
    original=task_staging.remove_staging
    def fail(*args):raise OSError('cleanup interrupted')
    monkeypatch.setattr(task_staging,'remove_staging',fail)
    executor._abort(task,'installation failed')
    assert queue.get_task(task.task_id).state=='cleaning'
    monkeypatch.setattr(task_staging,'remove_staging',original)
    executor.run_once()
    final=queue.get_task(task.task_id)
    assert final.state=='failed'
    assert final.error_reason=='installation failed'
    assert not work.exists()


@pytest.mark.skipif(os.name=='nt',reason='POSIX staging symlink boundary')
def test_staging_rejects_external_symlink(config,queue,tmp_path):
    task=_pending_analysis_task(queue)
    root=task_staging.task_root(config,task);root.mkdir(parents=True)
    external=tmp_path/'outside';external.mkdir()
    sentinel=external/'pytorch-source.tar.gz';sentinel.write_bytes(b'untouched')
    (root/'source-staging').symlink_to(external,target_is_directory=True)
    dl=FakeDownloader()
    with pytest.raises(ValueError,match='outside task'):
        task_staging.stage_package(config,task,'pytorch-source.tar.gz',dl)
    assert not dl.calls and sentinel.read_bytes()==b'untouched'


def test_runner_and_cli_contract(config):
    config['executor']['test_python_env']=None
    assert pipeline.make_runner(config) is None
    assert build_parser().parse_args(['pipeline']).cmd=='pipeline'
    env=pipeline.boot_discovery_env(config)
    assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']=='1'


def test_selection_body_source_reuse_isolation_and_progress(config,queue,monkeypatch):
    import run_test.upstream as upstream
    import run_test.selection as selection
    import run_test.obs as obs
    class Mirror:
        git_dir='/fixture/mirror'
        def __init__(self,**kwargs):pass
        def ensure(self):pass
    monkeypatch.setattr(upstream,'PyTorchMirror',Mirror)
    monkeypatch.setattr(selection,'git',lambda *a,**k:'fixture subject')
    dl=FakeDownloader();monkeypatch.setattr(obs,'make_downloader',lambda cfg:dl)
    captured={}
    def select(*a,**kwargs):captured.update(kwargs);return {}
    monkeypatch.setattr(selection,'select_for_commit',select)
    monkeypatch.setattr(selection,'plan_to_manifest',lambda *a,**k:[{'test_function':TF_A,'device':'cpu'}])
    monkeypatch.setattr(selection,'render_text',lambda *a:'fixture report')
    monkeypatch.setattr(selection,'baseline_coverage',lambda *a:{})
    config['executor']['test_python_env']='/separate-active-test-env/python'
    task=_pending_analysis_task(queue)
    assert pipeline.run_selection(config,queue,task.task_id,no_submit=True)==0
    assert pipeline.run_selection(config,queue,task.task_id)==0
    assert dl.calls.count(SOURCE_URL)==1
    assert dl.calls.count(WHEELS_URL)==1
    assert captured['interpreter']==sys.executable
    assert captured['interpreter']!=config['executor']['test_python_env']
    progress=json.loads((task_staging.task_root(config,task)/'progress.json').read_text())
    assert progress['stage']=='complete' and 'test_selection' in progress['stage_seconds']
    assert queue.get_task(task.task_id).state=='pending_test'
