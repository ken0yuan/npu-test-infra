import json
from copy import deepcopy
from pathlib import Path
import pytest
from run_test.issue_ledger import Ledger
from run_test.issue_resolution import load_index, active_counts, annotate_report
from run_test.monitoring import issues_for
from run_test.result_docs import build_report, render_html, feishu_plan


def task(name='old'):
    return dict(task_id=name,pr_number=1,pytorch_commit='a'*40,torch_npu_commit='b'*40,
                state='done',updated_at='2026-10-27',error_reason='')


def record(tests):
    return dict(pr_number=1,pytorch_commit='a'*40,torch_npu_environment={'torch_npu_commit':'b'*40},
                tests=tests,pass_count=sum(t.get('counts',{}).get('pass',0) for t in tests if t.get('counts')),
                fail_count=sum(t.get('counts',{}).get('fail',0) for t in tests if t.get('counts')),
                error_count=0,skip_count=0,no_result_count=sum(t.get('status')=='no_result' for t in tests))


def add(ledger, owner, *, tf='test/test_x.py::test_x', instance='', kind='test.failure'):
    ev,_=ledger.fact({'case':instance},'verified-source')
    ledger.add(owner,phase='execution',kind=kind,text='specific old failure',evidence=[ev],
               test_function=tf,device='cpu' if tf else '',instance=instance,
               extra={'pytest_outcome':'failure'} if kind=='test.failure' else {})
    return ledger.conn.execute('SELECT event_key FROM occurrences ORDER BY rowid DESC').fetchone()[0]


def close(ledger,config,keys):
    proof=Path(config['data_root'])/'repair-proof.json'
    proof.write_text(json.dumps({'fix':'c'*40,'regression':'passed','historical_rerun':False}))
    ledger.resolve_events(keys,'修复已验收，用户要求不重跑历史用例',proof,'c'*40)


def test_specific_archive_does_not_close_siblings_or_reopen_when_old_evidence_is_rescanned(config):
    ledger=Ledger(config)
    first=add(ledger,task('old'),kind='collection_error');second=add(ledger,task('other'),kind='collection_error')
    close(ledger,config,[first]);add(ledger,task('old'),kind='collection_error')
    assert dict(ledger.conn.execute('SELECT event_key,state FROM occurrences'))=={first:'resolved',second:'open'}
    add(ledger,task('recurrence'),kind='collection_error');ledger.render()
    overview=(ledger.docs/'问题总览.md').read_text()
    assert '## 待处理' in overview and '## 已修复' in overview and '已解决归档' in overview
    assert ledger.summary()['open']==2 and ledger.summary()['resolved']==1
    ledger.close()


def test_archive_unknown_event_is_atomic_and_requires_evidence(config):
    ledger=Ledger(config);key=add(ledger,task())
    with pytest.raises(ValueError,match='Unknown occurrence'):close(ledger,config,[key,'unknown'])
    assert ledger.summary()['resolved']==0
    with pytest.raises(FileNotFoundError):ledger.resolve_events([key],'fix',Path(config['data_root'])/'missing','c'*40)
    ledger.close()


def test_partial_case_archive_adjusts_only_closed_failure_and_keeps_raw_results(config):
    ledger=Ledger(config);first=add(ledger,task(),instance='TestX::one');second=add(ledger,task(),instance='TestX::two')
    tests=[dict(test_function='test/test_x.py::test_x',device='cpu',counts=dict(pass_=0))]
    tests[0]['counts']={'pass':0,'fail':2,'error':0,'skip':0};row=record(tests);original=deepcopy(row)
    close(ledger,config,[first]);index=load_index(config)
    assert active_counts(task(),row,index)['fail_count']==1
    assert issues_for({'tasks':[task()],'results':[row]},index)[0]['reasons'][0]['fail']==1
    assert row==original
    changed=task();changed['pytorch_commit']='d'*40
    assert active_counts(changed,row,index)['fail_count']==2
    close(ledger,config,[second]);assert active_counts(task(),row,load_index(config))['fail_count']==0
    assert row==original;ledger.close()


def test_resolved_gap_is_archived_not_passed_and_reports_keep_original_evidence(config):
    ledger=Ledger(config);key=add(ledger,task(),kind='collection_error');close(ledger,config,[key])
    test={'test_function':'test/test_x.py::test_x','device':'cpu','status':'no_result','counts':None,'reason':'old invalid candidate'}
    row=record([test]);raw=deepcopy(row);index=load_index(config)
    assert active_counts(task(),row,index)['no_result_count']==0
    assert not issues_for({'tasks':[task()],'results':[row]},index)
    model=annotate_report(build_report(row),task(),index)
    assert model['metrics']['Passed']==0 and model['metrics']['Failed']==0 and model['metrics']['No Result']==0
    assert model['resolved_test_count']==1 and model['tests'][0]['outcome']=='resolved'
    assert model['tests'][0]['status']=='no_result' and model['tests'][0]['counts'] is None
    assert '已解决' in render_html(model) and '未重跑' in render_html(model)
    assert '已解决' in json.dumps(feishu_plan(model),ensure_ascii=False)
    assert row==raw
    fresh=task('new-run');assert active_counts(fresh,row,index)['no_result_count']==1
    ledger.close()


def test_task_error_can_be_archived_without_changing_queue_state(config):
    ledger=Ledger(config);owner=task();owner['state']='failed';owner['error_reason']='old identifier rejection'
    key=add(ledger,owner,tf='',kind='task.failed');close(ledger,config,[key])
    assert not issues_for({'tasks':[owner],'results':[]},load_index(config))
    assert owner['state']=='failed' and owner['error_reason']=='old identifier rejection'
    ledger.close()


def test_brief_separates_current_counts_from_historical_counts(config,monkeypatch):
    from run_test import monitoring
    ledger=Ledger(config);owner=task();key=add(ledger,owner,kind='collection_error');close(ledger,config,[key])
    row=record([dict(test_function='test/test_x.py::test_x',device='cpu',status='no_result',counts=None)])
    snap={'time':'now','tasks':[dict(owner,enqueued_at='2026-10-29T00:00:00+00:00')],
          'results':[row],'task_states':{'done':1},'upstream_states':{}}
    original=deepcopy(snap);monkeypatch.setattr(monitoring,'snapshot',lambda config:snap)
    result=monitoring.brief_once(config)
    assert result['counts']['no_result_count']==0 and result['historical_counts']['no_result_count']==1
    assert result['counts']['pass_count']==0 and snap==original
    ledger.close()
