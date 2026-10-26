"""Isolated legacy E2E: real git, filtering, selection, pytest, queue and SQLite.

Only OBS download, wheel installation and hardware preflight are simulated.
No torch/NPU packages are installed. Does not use any historical database.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'run-test/src')]
from run_test.config import default_config, store_paths
from run_test.task_queue import QueueStore
from run_test.prefilter import prefilter_pending_tasks
from run_test.pipeline import run_selection, make_runner
from run_test.executor import Executor
from run_test.runner import PytestRunner
from run_test.pr_store import get_pr_result, ensure_pr_db
from run_test.baseline_store import baseline_db_path, get_baseline, ensure_baseline_db
from run_test.db import connect


class LocalArtifacts:
    def __init__(self, mapping): self.mapping = mapping
    def download(self, url, dest):
        Path(dest).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(self.mapping[url], dest)


class DemoExecutor(Executor):
    def _install(self, task):
        self._save_diagnostics(task, 'SIMULATED: no packages installed', 'install')
        return ('failed', 'SIMULATED install failure') if self.config['demo_mode'] == 'install_failure' else ('ok', '')
    def _preflight(self, task, source_root):
        self._save_diagnostics(task, 'SIMULATED hardware preflight; CPU fixture only', 'preflight')
        return True, ''


def run_scenario(root, mode):
    root.mkdir(parents=True)
    cfg = default_config()
    cfg.update(data_root=str(root/'runtime'), baseline_dirs=['test/demo'], demo_mode=mode)
    cfg['skill']['ref'] = 'bf9b16e'
    cfg['executor'].update(test_python_env=sys.executable, function_timeout_seconds=1 if mode=='timeout' else 30)
    (root/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    paths = store_paths(cfg)
    ensure_pr_db(paths.data_root)
    ensure_baseline_db(paths.baselines_root/'test/demo')
    repo = root/'fixture-repo'; repo.mkdir()
    env = dict(os.environ, GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='commit.gpgsign', GIT_CONFIG_VALUE_0='false',
               GIT_AUTHOR_NAME='Recovery Fixture', GIT_COMMITTER_NAME='Recovery Fixture',
               GIT_AUTHOR_EMAIL='fixture@example.invalid', GIT_COMMITTER_EMAIL='fixture@example.invalid')
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=repo,env=env,text=True).strip()
    git('init','-q','-b','main')
    (repo/'torch').mkdir(); (repo/'test/demo').mkdir(parents=True)
    source = repo/'torch/foo.py'; source.write_text('def value():\n    return 1\n',encoding='utf-8')
    case = repo/'test/demo/test_value.py'
    template = "import sys, time\nclass TestDemo:\n    def test_value(self):\n        BODY\nif __name__ == '__main__' and '--discover-tests' in sys.argv:\n    print('TestDemo.test_value')\n"
    case.write_text(template.replace('BODY','assert 1 == 1'),encoding='utf-8')
    git('add','.'); git('commit','-qm','fixture base')
    body = {'passed':'assert 2 == 2','failed':'assert 2 == 3','timeout':'time.sleep(10)', 'install_failure':'assert 2 == 2'}[mode]
    source.write_text('def value():\n    return 2\n',encoding='utf-8')
    case.write_text(template.replace('BODY',body),encoding='utf-8')
    git('add','.'); git('commit','-qm','change value (#123456)')
    sha = git('rev-parse','HEAD')
    paths.data_root.mkdir(parents=True,exist_ok=True)
    mirror = paths.data_root/'pytorch-main.git'
    git('clone','--bare','-q',str(repo),str(mirror))
    # Selection validates origin metadata; no fetch is performed in this demo.
    git('--git-dir',str(mirror),'remote','set-url','origin','https://github.com/pytorch/pytorch.git')
    packages=root/'fixtures'; packages.mkdir()
    wheel_dir=packages/'wheels'; wheel_dir.mkdir()
    (wheel_dir/'torch-0.0.0-py3-none-any.whl').write_bytes(b'fixture-not-installable')
    npu_sha='1234567890abcdef1234567890abcdef12345678'
    with zipfile.ZipFile(wheel_dir/'torch_npu-0.0.0-py3-none-any.whl','w') as z:
        z.writestr('torch_npu/version.py',f"git_version = '{npu_sha}'\n")
    wheels=packages/'wheels.tar.gz'; sources=packages/'source.tar.gz'
    with tarfile.open(wheels,'w:gz') as t:
        for p in wheel_dir.iterdir():t.add(p,arcname=p.name)
    with tarfile.open(sources,'w:gz') as t:
        for p in (source,case):t.add(p,arcname='pytorch/'+p.relative_to(repo).as_posix())
    urls=[f'obs://fixture/wheels-{sha[:10]}.tar.gz',f'obs://fixture/pytorch-source-{sha[:10]}.tar.gz']
    downloader=LocalArtifacts(dict(zip(urls,[wheels,sources])))
    queue=QueueStore(cfg)
    task,created=queue.register_candidate(*urls,obs_wheels_version='fixture-v1',obs_source_version='fixture-v1')
    assert created
    duplicate,created=queue.register_candidate(*urls,obs_wheels_version='fixture-v1',obs_source_version='fixture-v1')
    assert not created and duplicate.task_id==task.task_id
    queue.register_identity_rule('fixture','local artifacts bound to generated fixture commit','offline-demo')
    queue.verify_identity(task.task_id,'fixture',{'pr_number':123456,'pytorch_commit':sha,'torch_npu_commit':npu_sha,'binding':'controlled local fixture'})
    prefilter=prefilter_pending_tasks(cfg,queue=queue,mirror_dir=mirror)
    assert prefilter['kept']==1 and not prefilter['errors'], prefilter
    with patch('run_test.obs.make_downloader',return_value=downloader):
        assert run_selection(cfg,queue,task.task_id,device='cpu',python_env=sys.executable)==0
    manifest=queue.get_task(task.task_id).test_manifest
    assert manifest and len(manifest)==1,manifest
    executor=DemoExecutor(cfg,queue_store=queue,downloader=downloader,
                          runner=make_runner(cfg))
    outcome=executor.run_once()
    final=queue.get_task(task.task_id)
    db=connect(paths.pr_db)
    try: result=get_pr_result(db,123456)
    finally: db.close()
    if mode=='install_failure':
        assert outcome.outcome=='failed' and final.state=='failed' and result is None
    else:
        assert outcome.outcome=='done' and result is not None, (outcome,result)
        if mode=='passed':assert result.counts.pass_count==1 and result.no_result_count==0
        if mode=='failed':assert result.counts.fail_count==1 and result.no_result_count==0
        if mode=='timeout':assert result.no_result_count==1 and result.counts.total==0
        baseline=connect(baseline_db_path(paths.baselines_root/'test/demo'))
        try: base=get_baseline(baseline,manifest[0]['test_function'],'cpu')
        finally: baseline.close()
        assert (base is None) == (mode=='timeout'), (mode,base)
    summary={'scenario':mode,'task':dataclasses.asdict(final),'outcome':dataclasses.asdict(outcome),
             'pr_result':dataclasses.asdict(result) if result else None,'fixture_commit':sha,
             'original_pytorch_binary_commit':None,'hardware':'CPU fixture, no NPU validation',
             'simulated_boundaries':['OBS download','wheel install','hardware preflight'],
             'package_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheels,sources)}}
    (root/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return {'scenario':mode,'task_state':final.state,'result_path':str(root/'result.json')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output=args.output.expanduser().resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='acceptance-',dir=args.output))
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'run-test/src')])
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';os.environ['PYTHONUTF8']='1'
    results=[run_scenario(root/mode,mode) for mode in ('passed','failed','timeout','install_failure')]
    (root/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'run':str(root),'scenarios':results},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
