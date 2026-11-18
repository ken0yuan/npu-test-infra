"""Task-owned Git checkouts pinned to the wheel's complete upstream-main commit."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .config import store_paths
from .obs import ObsDownloadError

SCHEME = 'git-source://pytorch/'
MARKER = '.run-test-git-source.json'


def enabled(config):
    return ((config.get('obs') or {}).get('source') or {}).get('provider') == 'git'


def is_git_source(url):
    return isinstance(url, str) and url.startswith(SCHEME)


def _git(repo, *args):
    try:
        proc = subprocess.run(['git', '--git-dir', str(repo), *args],
                              capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ObsDownloadError(f'Git source command failed: {error}') from error
    if proc.returncode:
        raise ObsDownloadError(f'Git source {args[0]} failed: {proc.stderr.strip()[-1000:]}')
    return proc.stdout.strip()


def mirror_path(config):
    configured = ((config.get('obs') or {}).get('source') or {}).get('git_mirror')
    return Path(configured).resolve() if configured else store_paths(config).data_root / 'pytorch-main.git'


def source_metadata(config, revision):
    if not re.fullmatch(r'[0-9a-f]{10}|[0-9a-f]{40}|[0-9a-f]{64}', revision or ''):
        raise ObsDownloadError('Git source requires a commit identity, never a branch name')
    repo = mirror_path(config)
    # No partial/shallow mirror or lazy network download during selection.
    settings = _git(repo, 'config', '--local', '--list')
    if (_git(repo, 'rev-parse', '--is-shallow-repository') != 'false'
            or re.search(r'(?im)^(extensions\.partialclone|remote\..*\.(promisor|partialclonefilter))=', settings)
            or any((repo / 'objects/pack').glob('*.promisor'))):
        raise ObsDownloadError('Git source requires the complete local upstream mirror')
    sha = _git(repo, 'rev-parse', '--verify', revision + '^{commit}')
    if not sha.startswith(revision):
        raise ObsDownloadError('Git source commit mismatch')
    _git(repo, 'merge-base', '--is-ancestor', sha, 'refs/heads/main')
    tree = _git(repo, 'rev-parse', sha + '^{tree}')
    return {'url':SCHEME + sha, 'etag':f'git:{sha}:{tree}:checkout-v1',
            'commit':sha, 'tree':tree, 'provider':'git'}


def checkout_source(config, task, dest):
    """Local shared clone + detached checkout. No tarball or per-task network clone."""
    if not is_git_source(task.obs_source_url):
        raise ObsDownloadError('Task does not identify a Git source checkout')
    sha = task.obs_source_url[len(SCHEME):]
    if sha != task.pytorch_commit or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', sha):
        raise ObsDownloadError('Git source URL must equal the task complete commit')
    metadata = source_metadata(config, sha)
    if task.obs_source_version != metadata['etag']:
        raise ObsDownloadError('Git source tree identity differs from the admitted task')
    dest = Path(dest)
    if dest.is_symlink():
        raise ObsDownloadError('Refusing a symlinked Git source checkout')
    if dest.exists():
        try:
            marker = json.loads((dest / MARKER).read_text())
            if (marker['commit'] == sha and marker['tree'] == metadata['tree']
                    and _git(dest / '.git', 'rev-parse', 'HEAD') == sha):
                _git(dest / '.git', '--work-tree=' + str(dest.resolve()), 'diff', '--quiet', 'HEAD', '--')
                return dest
        except (OSError, ValueError, KeyError):
            pass
        raise ObsDownloadError('Existing Git source checkout is unowned, modified or has another identity')
    dest.parent.mkdir(parents=True, exist_ok=True)
    repo = mirror_path(config)
    with tempfile.TemporaryDirectory(prefix='.git-source-', dir=dest.parent) as temporary:
        staged = Path(temporary) / 'pytorch'
        for command in [
            ['git','clone','--shared','--no-checkout','--no-hardlinks',str(repo),str(staged)],
            ['git','-C',str(staged),'checkout','--detach',sha],
        ]:
            try:
                proc = subprocess.run(command,capture_output=True,text=True,timeout=600)
            except (OSError,subprocess.TimeoutExpired) as error:
                raise ObsDownloadError(f'Git source checkout failed: {error}') from error
            if proc.returncode:
                raise ObsDownloadError(f'Git source checkout failed: {proc.stderr.strip()[-1000:]}')
        if _git(staged / '.git','rev-parse','HEAD') != sha:
            raise ObsDownloadError('Checked out Git source has the wrong commit')
        pins = []
        raw = _git(repo,'ls-tree','-r',sha)
        for line in raw.splitlines():
            if line.startswith('160000 '):
                header,path = line.split('\t',1)
                pins.append({'path':path,'commit':header.split()[2],'status':'not_initialized'})
        # Align with upstream test-collect: tracked source only; do not replace
        # pinned submodules with arbitrary latest branches or hide their state.
        receipt = {**metadata,'submodules':pins,'submodule_policy':'not_initialized',
                   'source_scope':'tracked-tree','archive_used':False}
        (staged / MARKER).write_text(json.dumps(receipt,indent=2)+'\n')
        staged.rename(dest)
    return dest
