import json
import subprocess
from types import SimpleNamespace

import pytest

from run_test.git_source import checkout_source,source_metadata,SCHEME,MARKER
from run_test.obs import ObsDownloadError,list_artifacts
from run_test.task_queue import QueueStore
from run_test.upstream import PyTorchMirror,run_cycle


def git(path,*args):
    return subprocess.check_output(['git','-c','user.name=test','-c','user.email=test@example.com',*args],cwd=path,text=True).strip()


@pytest.fixture
def git_sources(config,tmp_path):
    upstream=tmp_path/'upstream';upstream.mkdir()
    git(upstream,'init','-q','-b','main')
    (upstream/'test').mkdir();(upstream/'test/test_real.py').write_text('def test_one(): assert True\n')
    git(upstream,'add','.');git(upstream,'commit','-qm','first (#101)')
    first=git(upstream,'rev-parse','HEAD')
    (upstream/'test/test_real.py').write_text('def test_two(): assert True\n')
    # An uninitialized pinned submodule must be visible in the source receipt.
    git(upstream,'add','.')
    git(upstream,'update-index','--add','--cacheinfo','160000,'+'a'*40+',third_party/example')
    git(upstream,'commit','-qm','second (#102)')
    second=git(upstream,'rev-parse','HEAD')
    mirror=PyTorchMirror(repo_url=str(upstream),mirror_dir=tmp_path/'mirror.git');mirror.ensure()
    config['obs']['source']={'provider':'git','git_mirror':str(mirror.git_dir)}
    return config,mirror,first,second


def test_checkout_is_offline_detached_and_pinned_to_full_wheel_commit(git_sources,tmp_path):
    c,m,first,second=git_sources
    task=SimpleNamespace(pytorch_commit=first,obs_source_url=SCHEME+first,obs_source_version=source_metadata(c,first)['etag'])
    # Remote disappearing must not turn source preparation into a network clone.
    origin=tmp_path/'upstream';origin.rename(tmp_path/'offline')
    dest=checkout_source(c,task,tmp_path/'sources/pytorch')
    assert 'test_one' in (dest/'test/test_real.py').read_text()
    assert git(dest,'rev-parse','HEAD')==first
    assert subprocess.run(['git','-C',str(dest),'symbolic-ref','-q','HEAD'],capture_output=True).returncode==1
    receipt=json.loads((dest/MARKER).read_text());assert receipt['archive_used'] is False
    assert checkout_source(c,task,dest)==dest
    (dest/'test/test_real.py').write_text('modified')
    with pytest.raises(ObsDownloadError):checkout_source(c,task,dest)


def test_wrong_commit_or_tree_identity_is_rejected(git_sources,tmp_path):
    c,m,first,second=git_sources
    task=SimpleNamespace(pytorch_commit=second,obs_source_url=SCHEME+first,obs_source_version=source_metadata(c,first)['etag'])
    with pytest.raises(ObsDownloadError,match='complete commit'):checkout_source(c,task,tmp_path/'wrong')
    task.obs_source_url=SCHEME+second
    with pytest.raises(ObsDownloadError,match='tree identity'):checkout_source(c,task,tmp_path/'wrong')
    assert not (tmp_path/'wrong').exists()


def test_submodule_pins_are_recorded_without_fetching_other_branches(git_sources,tmp_path):
    c,m,first,second=git_sources
    meta=source_metadata(c,second)
    task=SimpleNamespace(pytorch_commit=second,obs_source_url=meta['url'],obs_source_version=meta['etag'])
    dest=checkout_source(c,task,tmp_path/'source')
    receipt=json.loads((dest/MARKER).read_text())
    assert receipt['source_scope']=='tracked-tree'
    assert receipt['submodules']==[{'path':'third_party/example','commit':'a'*40,'status':'not_initialized'}]


def test_git_source_admission_does_not_wait_for_obs_source_and_is_idempotent(git_sources):
    c,m,first,second=git_sources
    c['obs']['wheels']={'bucket':'pytorch-torchnpu'}
    wp='obs://pytorch-torchnpu/7day-cache/'
    pt=wp+f'wheels-pytorch-{second[:10]}-py311-builder-aarch64-v3.tar.gz'
    npu=wp+f'wheels-torchnpu-abcdef0123-{second[:10]}-py311-builder-aarch64-v3.tar.gz'
    class WheelsOnly:
        def list(self,prefix):
            assert prefix==wp # Old source namespace must never be queried.
            return {pt:{'etag':'pt'},npu:{'etag':'npu'}}
    q=QueueStore(c);q.set_mirror_tip(m.repo_url,first)
    report=run_cycle(c,queue=q,mirror=m,downloader=WheelsOnly())
    assert len(report.enqueued)==1 and not report.errors
    task=q.get_task(report.enqueued[0]['task_id'])
    assert task.obs_source_url==SCHEME+second and task.pytorch_commit==second
    assert task.obs_source_version==source_metadata(c,second)['etag']
    assert not run_cycle(c,queue=q,mirror=m,downloader=WheelsOnly()).enqueued


def test_git_source_still_requires_matching_npu_bundle(git_sources):
    c,m,first,second=git_sources
    pt=f'obs://pytorch-ci-cache/7day-cache/wheels-pytorch-{second[:10]}-py311-builder-aarch64-v3.tar.gz'
    class Incomplete:
        def list(self,prefix):return {pt:{'etag':'pt'}}
    q=QueueStore(c);q.set_mirror_tip(m.repo_url,first)
    report=run_cycle(c,queue=q,mirror=m,downloader=Incomplete())
    assert not report.enqueued and report.missing_counts['wheels_npu']==1


def test_non_main_and_branch_names_cannot_supply_sources(git_sources,tmp_path):
    c,m,first,second=git_sources
    with pytest.raises(ObsDownloadError):source_metadata(c,'main')
    with pytest.raises(ObsDownloadError):source_metadata(c,'b'*40)
