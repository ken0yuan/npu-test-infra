import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from run_test.issue_ledger import Ledger, classification, scan, selection_error


def task(state='done', identifier='one'):
    return {'task_id': identifier, 'pr_number': 1, 'pytorch_commit': 'a'*40,
            'torch_npu_commit': 'b'*40, 'state': state, 'error_reason': '', 'updated_at': '2026-10-26'}


def snapshot(config, *, xml='', status=None, collection=None, counts=None, identifier='one'):
    data = Path(config['data_root'])
    directory = data / 'diagnostics' / identifier / 'artifacts' / 'cpu-test_case'
    directory.mkdir(parents=True)
    if xml:
        (directory/'junit.xml').write_text(xml)
    if status:
        (directory/'result.json').write_text(json.dumps({'execution_status': status}))
    if collection:
        (directory/'collection.json').write_text(json.dumps(collection))
    item = {'test_function': 'test/test_case.py::test_example', 'device': 'cpu',
            'artifact_refs': [str(directory)]}
    if counts is not None:
        item['counts'] = counts
    if status:
        item.update(status='no_result', reason=status)
    return {'tasks': [task(identifier=identifier)], 'results': [{
        'pr_number': 1, 'pytorch_commit': 'a'*40,
        'torch_npu_environment': json.dumps({'torch_npu_commit': 'b'*40}),
        'tests': json.dumps([item])}]}


def events(config):
    ledger = Ledger(config)
    try:
        return [dict(row) | {'body': json.loads(row['payload'])} for row in
                ledger.conn.execute('SELECT * FROM occurrences ORDER BY first_seen,event_key')]
    finally:
        ledger.close()


def test_mixed_pass_skip_keeps_exact_reason_and_is_idempotent(config):
    xml = '<testsuite><testcase name="pass"/><testcase name="skip"><skipped message="requires network">original skip reason</skipped></testcase></testsuite>'
    snap = snapshot(config, xml=xml, counts={'pass': 1, 'fail': 0, 'error': 0, 'skip': 1})
    first = scan(config, snap); second = scan(config, snap)
    assert first['new_events'] == 1 and second['new_events'] == 0
    row, = events(config)
    assert row['body']['kind'] == 'test.skipped'
    assert 'original skip reason' in row['body']['detail']
    ev = row['body']['evidence'][0]
    assert hashlib.sha256(Path(ev['snapshot']).read_bytes()).hexdigest() == ev['sha256']
    assert ev['quote'] in xml


def test_collection_missing_dependency_keeps_full_evidence_after_source_deleted(config):
    message = "ModuleNotFoundError: No module named 'hypothesis'"
    snap = snapshot(config, status='collection_error', collection={
        'returncode': 2, 'stderr': 'x'*4000 + message})
    scan(config, snap)
    row, = events(config)
    assert row['body']['family'] == 'dependency.missing_module'
    assert row['body']['missing_modules'] == ['hypothesis']
    assert row['body']['package_suggestions'] == {'hypothesis': 'hypothesis'}
    source = Path(row['body']['evidence'][1]['source'])
    source.unlink()
    assert any(message in Path(ev['snapshot']).read_text() for ev in row['body']['evidence'])


def test_known_signature_is_not_turned_into_unknown_U(config):
    snap = snapshot(config, xml='<testsuite><testcase name="x"><failure message="HCCL function error">trace</failure></testcase></testsuite>',
                    counts={'pass': 0, 'fail': 1, 'error': 0, 'skip': 0})
    report = scan(config, snap)
    assert report['by_namespace'] == {'C': 1}
    assert 'PR 1' not in (Path(report['docs_root'])/'问题总览.md').read_text()
    assert events(config)[0]['body']['assessment'] == 'known_signature_needs_review'


def test_configured_X_classification_is_explicit_and_does_not_change_result(config):
    config['issue_ledger'] = {'file_exclusions': [{'id': 'X-local-test', 'glob': 'test/test_case.py'}]}
    snap = snapshot(config, xml='<testsuite><testcase name="x"><failure>unexpected</failure></testcase></testsuite>')
    before = json.dumps(snap, sort_keys=True)
    assert scan(config, snap)['by_namespace'] == {'X': 1}
    assert json.dumps(snap, sort_keys=True) == before


def test_selection_records_unfinished_tasks_but_budget_deferral_is_not_a_bug(config):
    path = Path(config['data_root'])/'selection/one/test-plan.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'analysis_gaps': [{'path': 'test/x.py', 'reason': 'discovery exited 2'}],
                               'manifest_skipped': [{'reason': 'nested class not supported'}],
                               'deferred_tests': [{'reason': 'exceeds normal budget'}]}))
    report = scan(config, {'tasks': [task('pending_analysis')], 'results': []})
    assert report['by_namespace'] == {'S': 2}
    assert all(e['body']['phase'] == 'selection' for e in events(config))


def test_selection_exception_before_plan_is_preserved(config):
    selection_error(config, task('pending_analysis'), 'OBS download timed out before test plan')
    row, = events(config)
    assert row['body']['namespace'] == 'S'
    assert 'OBS download timed out' in row['body']['detail']


def test_selection_missing_package_is_a_dependency_not_a_selector_defect(config):
    selection_error(config, task('pending_analysis'), "ModuleNotFoundError: No module named 'hypothesis'")
    row, = events(config)
    assert row['body']['namespace'] == 'U'
    assert row['body']['family'] == 'dependency.missing_module'
    assert row['body']['phase'] == 'selection'


@pytest.mark.parametrize('mismatch', ['pytorch_commit', 'torch_npu_commit', 'task_id'])
def test_old_result_or_other_task_artifacts_are_not_misattributed(config, mismatch):
    snap = snapshot(config, xml='<testsuite><testcase name="x"><failure>bad</failure></testcase></testsuite>')
    if mismatch == 'pytorch_commit':
        snap['results'][0]['pytorch_commit'] = 'c'*40
    elif mismatch == 'torch_npu_commit':
        snap['results'][0]['torch_npu_environment'] = json.dumps({'torch_npu_commit': 'c'*40})
    else:
        snap['tasks'][0]['task_id'] = 'other-task'
    scan(config, snap)
    rows = events(config)
    if mismatch == 'task_id':
        assert len(rows) == 1 and rows[0]['body']['kind'] == 'evidence.binding_mismatch'
    else:
        assert rows == []


def test_outside_evidence_cannot_be_read(config, tmp_path):
    outside = tmp_path/'secret.txt'; outside.write_text('never ingest outside data_root')
    ledger = Ledger(config)
    try:
        with pytest.raises(ValueError, match='outside data_root'):
            ledger.evidence_file(outside)
    finally:
        ledger.close()


def test_resolve_requires_evidence_and_archives_original_record(config):
    snap = snapshot(config, status='timeout', collection={'stderr': 'test exceeded its timeout'})
    scan(config, snap)
    ledger = Ledger(config)
    try:
        identifier = ledger.conn.execute('SELECT identifier FROM families').fetchone()[0]
        with pytest.raises(FileNotFoundError):
            ledger.resolve(identifier, 'claimed fix', Path(config['data_root'])/'absent.log')
        proof = Path(config['data_root'])/'verified.log'; proof.write_text('real regression test passed')
        ledger.resolve(identifier, 'timeout fix independently verified', proof)
        ledger.render()
        assert ledger.summary()['open'] == 0
        overview = (ledger.docs/'问题总览.md').read_text()
        assert 'timeout fix independently verified' in overview
        assert '已修复（resolved）' in overview
        # 原始记录归档在账本（DB）内——视图只做摘要，不粘贴证据全文。
        assert any('execution_status' in e['payload'] for e in events(config))
    finally:
        ledger.close()
    assert scan(config, snap)['open'] == 0  # Scanning old evidence never reopens a verified closure.


def test_two_runs_share_family_but_keep_distinct_occurrences(config):
    first = snapshot(config, identifier='one', status='timeout', collection={'stderr': 'timed out'})
    second = snapshot(config, identifier='two', status='timeout', collection={'stderr': 'timed out'})
    scan(config, first); scan(config, second)
    ledger = Ledger(config)
    try:
        assert ledger.conn.execute('SELECT COUNT(*) FROM families').fetchone()[0] == 1
        assert ledger.summary()['occurrences'] == 2
    finally:
        ledger.close()


def test_bad_json_plan_does_not_hide_other_tasks(config):
    path = Path(config['data_root'])/'selection/one/test-plan.json'
    path.parent.mkdir(parents=True); path.write_text('{broken JSON')
    report = scan(config, {'tasks': [task('pending_analysis')], 'results': []})
    assert report['by_namespace'] == {'S': 1}
    assert events(config)[0]['body']['kind'] == 'selection.invalid_plan_json'


def test_old_retry_signature_does_not_reclassify_current_missing_dependency(config):
    current = task('blocked')
    current['error_reason'] = "ModuleNotFoundError: No module named 'hypothesis'"
    current['history'] = 'previous attempt: HCCL function error'
    folder = Path(config['data_root'])/'diagnostics/one'
    folder.mkdir(parents=True)
    (folder/'old.log').write_text('HCCL function error from a different attempt')
    report = scan(config, {'tasks': [current], 'results': []})
    assert report['by_namespace'] == {'U': 1}
    assert events(config)[0]['body']['family'] == 'dependency.missing_module'


def test_unknown_module_does_not_suggest_framework_or_guess_a_package():
    assert classification("ModuleNotFoundError: No module named 'torch_openreg'", 'collection', 'error')[0] == 'C'
    assert classification("ModuleNotFoundError: No module named 'new_private_helper'", 'collection', 'error')[2] == ['new_private_helper']
