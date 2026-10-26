import json
import sqlite3

from run_test.monitoring import issues_for, monitor_once, brief_once


def snapshot_fixture():
    return {'tasks':[{'task_id':'one','pr_number':1,'state':'done','pytorch_commit':'a'*40,
                     'updated_at':'2026-10-24T00:00:00+00:00'}],
            'results':[{'pr_number':1,'pytorch_commit':'a'*40,'pass_count':3,'fail_count':0,
                        'error_count':0,'skip_count':2,'no_result_count':0,
                        'tests':json.dumps([{'test_function':'test/x.py::test_a','device':'cpu',
                                            'counts':{'pass':3,'fail':0,'error':0,'skip':0}},
                                           {'test_function':'test/x.py::test_b','device':'cpu',
                                            'counts':{'pass':0,'fail':0,'error':0,'skip':2}}])}]}


def test_all_skip_function_is_reported_even_when_other_tests_pass():
    issues=issues_for(snapshot_fixture())
    assert issues[0]['reasons']==[{'kind':'function_all_skipped','test_function':'test/x.py::test_b','device':'cpu'}]


def test_missing_results_and_no_result_are_never_success():
    snap=snapshot_fixture();snap['results']=[]
    assert issues_for(snap)[0]['reasons']==[{'kind':'done_without_result'}]
    snap=snapshot_fixture();snap['results'][0]['no_result_count']=1
    assert any(r['kind']=='no_result' for r in issues_for(snap)[0]['reasons'])


def test_newer_pr_version_is_not_misattributed_to_old_task():
    snap=snapshot_fixture();snap['results'][0]['pytorch_commit']='b'*40
    assert issues_for(snap)==[]


def test_monitor_archive_is_idempotent_without_changing_results(tmp_path,monkeypatch):
    snap=snapshot_fixture();snap['time']='2026-10-24T12:00:00+08:00'
    monkeypatch.setattr('run_test.monitoring.snapshot',lambda cfg:snap)
    original=json.dumps(snap,sort_keys=True)
    for _ in range(2):assert monitor_once({},tmp_path)['current_issues']==1
    assert len(list(tmp_path.glob('*.json')))==1
    assert json.dumps(snap,sort_keys=True)==original


def test_empty_real_databases_are_read_only(config,queue,router,paths):
    before=paths.pr_db.read_bytes()
    brief=brief_once(config)
    assert brief['prs']==0 and brief['tasks']=={}
    assert brief['counts']['no_result_count']==0
    assert monitor_once(config,paths.data_root/'monitor')['current_issues']==0
    assert paths.pr_db.read_bytes()==before
