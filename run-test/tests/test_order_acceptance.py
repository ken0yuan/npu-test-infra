"""Offline real-Git fetch -> persisted first-parent order -> baseline submission."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from run_test import upstream as up
from run_test.config import store_paths
from run_test.identifiers import Counts, baseline_log_filename
from run_test.record import submit_complete_result
from run_test.result_docs import generate
from conftest import make_submission, TF_A
from test_record_flow import _baseline


@pytest.fixture
def fetched(config, tmp_path):
    work=tmp_path/'origin';work.mkdir()
    def git(*args, date=None):
        env={**os.environ,'GIT_AUTHOR_NAME':'fixture','GIT_AUTHOR_EMAIL':'fixture@example.invalid',
             'GIT_COMMITTER_NAME':'fixture','GIT_COMMITTER_EMAIL':'fixture@example.invalid'}
        if date:env.update(GIT_AUTHOR_DATE=date,GIT_COMMITTER_DATE=date)
        return subprocess.check_output(['git','-c','commit.gpgsign=false',*args],cwd=work,env=env,text=True).strip()
    git('init','-q','-b','main')
    commits=[]
    # Constructed reversed/equal committer clocks; not a production incident.
    for i,hour in enumerate([3,2,2],1):
        (work/'file').write_text(str(i))
        git('add','file');git('commit','-qm',f'change (#{100+i})',date=f'2026-10-26T0{hour}:00:00Z')
        commits.append(git('rev-parse','HEAD'))
    mirror_path=store_paths(config).data_root/'pytorch-main.git'
    mirror_path.parent.mkdir(parents=True,exist_ok=True)
    git('clone','-q','--bare',work.as_uri(),str(mirror_path))
    class FixtureMirror(up.PyTorchMirror):
        # Only source authentication is replaced by a local fixture. fetch/log/
        # snapshot/submit all execute the real production implementation offline.
        def ensure(self):pass
    mirror=FixtureMirror(config=config)
    report=up.run_cycle(config,mirror=mirror,downloader=None,first_run_backfill_hours=48)
    assert report.fetched_tip==commits[-1]
    snapshot=up.load_order_snapshot(config)
    assert snapshot['order']==list(reversed(commits))
    assert snapshot['inversions']
    return mirror,commits,snapshot


def submit(config,router,sha,pr,count,text):
    sub=make_submission(pr_number=pr,pytorch_commit=sha,
                        merged_at='2026-10-26T03:00:00Z',
                        merged_at_provenance='upstream-main-commit-time',
                        items=[(TF_A,'cpu',Counts(count,0,0,0))])
    (Path(sub.pr_log_source_dir)/baseline_log_filename(TF_A,'cpu')).write_text(text)
    return submit_complete_result(config,sub,router)


def test_fetch_snapshot_controls_baseline_and_log_offline(config,router,fetched,monkeypatch):
    mirror,commits,snapshot=fetched
    def no_query(*a):raise AssertionError('Covered commits must not query Git during submit')
    monkeypatch.setattr(up.PyTorchMirror,'merge_relation',no_query)
    submit(config,router,commits[0],101,1,'first')
    submit(config,router,commits[2],103,3,'newest')
    current=_baseline(router,TF_A,'cpu')
    logfile=router.managed_root('test/distributed/_tensor')/current.log_path
    before=logfile.read_bytes()
    report=submit(config,router,commits[1],102,9,'late-arriving-old-result')
    assert _baseline(router,TF_A,'cpu').pytorch_commit==commits[2]
    assert logfile.read_bytes()==before
    assert report.checks['baseline_skipped_merge_order'][0]['reason']=='verified_first_parent_order'
    assert generate(config,102)['pr_number']==102


@pytest.mark.parametrize('damage',['missing','corrupt','wrong_source','duplicate','wrong_tip','digest','uncovered'])
def test_unverified_order_never_uses_timestamp_label(config,router,fetched,monkeypatch,damage):
    mirror,commits,snapshot=fetched
    submit(config,router,commits[2],103,3,'newest')
    path=store_paths(config).data_root/'upstream-order.json'
    candidate=commits[0]
    if damage=='missing':path.unlink()
    elif damage=='corrupt':path.write_text('{bad')
    elif damage=='uncovered':candidate='f'*40
    else:
        bad=deepcopy(snapshot)
        if damage=='wrong_source':bad['source']='https://example.invalid/repo'
        if damage=='duplicate':bad['order'][1]=bad['order'][0]
        if damage=='wrong_tip':bad['checked_tip']='e'*40
        if damage=='digest':bad['order'].reverse()
        # Structural/source rejection is independent of checksum rejection.
        if damage!='digest':bad['sha256']=up._order_digest(bad)
        path.write_text(json.dumps(bad))
    calls=[]
    def unknown(self,a,b):calls.append((a,b));return 'unknown'
    monkeypatch.setattr(up.PyTorchMirror,'merge_relation',unknown)
    submit(config,router,candidate,101,9,'must-not-replace')
    assert calls==[(candidate,commits[2])]
    assert _baseline(router,TF_A,'cpu').pytorch_commit==commits[2]


def test_snapshot_replace_failure_keeps_previous_and_cleans_temp(config,fetched,monkeypatch):
    _,_,snapshot=fetched
    path=store_paths(config).data_root/'upstream-order.json';before=path.read_bytes()
    files=set(path.parent.iterdir())
    def fail(*a):raise OSError('fixture atomic replace failure')
    monkeypatch.setattr(up.os,'replace',fail)
    with pytest.raises(OSError):up.persist_order_snapshot(config,snapshot)
    assert path.read_bytes()==before
    assert set(path.parent.iterdir())==files
