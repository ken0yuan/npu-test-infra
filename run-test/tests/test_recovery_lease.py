"""Regression: recovery may only run after the live OS executor lease ends."""
import os
from pathlib import Path
import subprocess
import sys
from run_test.executor import Executor
from run_test.process_lock import executor_lease
from conftest import COMMIT_A, COMMIT_NPU, TF_A


def active_task(queue):
    task, _ = queue.register_candidate('obs://fixture/wheels', 'obs://fixture/source')
    queue.register_identity_rule('fixture', 'fixture rule', 'regression')
    queue.verify_identity(task.task_id, 'fixture', {'pr_number':123456,
        'pytorch_commit':COMMIT_A,'torch_npu_commit':COMMIT_NPU,'binding':'fixture'})
    queue.prefilter_outcome(task.task_id, True)
    queue.submit_manifest(task.task_id, [{'test_function':TF_A,'device':'cpu'}])
    return queue.claim_next()


def test_live_executor_cannot_be_recovered_or_replaced(config, queue, paths):
    task=active_task(queue)
    with executor_lease(paths.queue_db.with_suffix('.executor.lock')) as acquired:
        assert acquired
        assert queue.recover_interrupted()==[]
        assert Executor(config,queue_store=queue,downloader=None).run_once().outcome=='idle'
        assert queue.get_task(task.task_id).state=='preparing'
    recovered=queue.recover_interrupted()
    assert [t.task_id for t in recovered]==[task.task_id]
    assert queue.get_task(task.task_id).state=='blocked'


def test_os_releases_lease_after_process_termination(config, queue, paths):
    task=active_task(queue)
    code='''import sys
from run_test.process_lock import executor_lease
with executor_lease(sys.argv[1]) as acquired:
 print("locked" if acquired else "busy",flush=True)
 sys.stdin.read()
'''
    child=subprocess.Popen([sys.executable,'-c',code,str(paths.queue_db.with_suffix('.executor.lock'))],
                           stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='locked'
        assert queue.recover_interrupted()==[]
        assert queue.get_task(task.task_id).state=='preparing'
    finally:
        child.terminate()
        child.communicate(timeout=10)
    assert [t.task_id for t in queue.recover_interrupted()]==[task.task_id]
    assert queue.get_task(task.task_id).state=='blocked'
