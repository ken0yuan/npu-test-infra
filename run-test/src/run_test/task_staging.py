"""Task-owned downloads: atomically completed, identity checked, never installed here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from .config import store_paths


def task_root(config, task):
    root = (store_paths(config).data_root / 'selection').resolve()
    path = (root / task.task_id).resolve()
    if path == root or path.parent != root:
        raise ValueError('Invalid task staging path')
    return path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as stream:
            os.chmod(tmp, 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def identity(task, name):
    attr = {'pytorch-source.tar.gz': 'source', 'wheels.tar.gz': 'wheels',
            'wheels-npu.tar.gz': 'wheels_npu'}[name]
    return {'task_id': task.task_id, 'commit': task.pytorch_commit,
            'url': getattr(task, 'obs_' + attr + '_url'),
            'version': getattr(task, 'obs_' + attr + '_version', None)}


def checksum(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def package_path(config, task, name):
    identity(task, name)  # Only the three supported artifact names are valid.
    root = task_root(config, task)
    staging = root / 'source-staging'
    if staging.is_symlink() or staging.resolve().parent != root:
        raise ValueError('Refusing staging path outside task')
    path = staging / name
    if path.is_symlink() or path.with_suffix(path.suffix + '.json').is_symlink():
        raise ValueError('Refusing symlinked task package')
    return path


def verified_package(config, task, name):
    path = package_path(config, task, name)
    try:
        metadata = json.loads(path.with_suffix(path.suffix + '.json').read_text(encoding='utf-8'))
        if metadata['identity'] == identity(task, name) and path.stat().st_size == metadata['size']:
            if checksum(path) == metadata['sha256']:
                return path
    except (OSError, ValueError, KeyError):
        pass
    return None


def stage_package(config, task, name, downloader):
    old = verified_package(config, task, name)
    if old:
        return old
    if downloader is None:
        raise ValueError('OBS downloader is not configured')
    path = package_path(config, task, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.unlink(missing_ok=True)
    started = time.monotonic()
    try:
        downloader.download(identity(task, name)['url'], temporary)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise ValueError('Download did not produce a nonempty archive')
        os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + '.json'), {
            'identity': identity(task, name), 'sha256': checksum(path),
            'size': path.stat().st_size, 'download_seconds': time.monotonic() - started})
    finally:
        temporary.unlink(missing_ok=True)
    return path


def remove_staging(config, task):
    # Keep test-plan, progress, subprocess log and lock inode for audit/retry.
    target = task_root(config, task) / 'source-staging'
    if target.is_symlink() or target.resolve().parent != task_root(config, task):
        raise ValueError('Refusing staging path outside task')
    if target.exists():
        shutil.rmtree(target)
