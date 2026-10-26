"""Standalone relation utilities. Never select a device or install torch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time

PACKAGE = Path(__file__).resolve().parent
DEFAULT_CATALOG = PACKAGE / 'catalogs/pytorch-api-ledger.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_host():
    if sys.platform != 'linux' or sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 11):
        raise RuntimeError('v0.1 requires Linux + CPython 3.11; other Python versions require a native adapter')


def build_identity(runtime):
    require_host()
    runtime = Path(runtime)
    target = runtime / ('_relation_identity' + sysconfig.get_config_var('EXT_SUFFIX'))
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    subprocess.run([*compiler, '-shared', '-fPIC', '-O2', '-Wall', '-Werror',
                    '-I' + sysconfig.get_path('include'), str(runtime / 'relation_identity.c'),
                    '-o', str(target)], check=True)
    return target


def initialize(root, repo, catalog, exclude, label, imports):
    require_host()
    root, repo, catalog = root.resolve(), repo.resolve(), catalog.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Use an empty output directory for each pytest invocation; existing evidence is never overwritten')
    if not repo.is_dir(): raise ValueError('Repository root does not exist')
    data = read(catalog)
    ids = [row['api_id'] for row in data['apis']]
    for row in data['apis']:
        for key in ('qualified_name', 'language', 'kind'): row[key]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate api_id in catalog')
    root.mkdir(parents=True, exist_ok=True)
    runtime = root / 'runtime'
    shutil.copytree(PACKAGE / 'runtime', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so'))
    build_identity(runtime)
    shutil.copyfile(catalog, root / 'api-catalog.json')
    (root / 'shards').mkdir()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    manifest = {'schema': 'api_test_relations_plugin/1', 'package_version': '0.1.2', 'capture_pid': os.getpid(), 'created_at': time.time(), 'repo': str(repo),
                'commit': commit, 'python': sys.executable, 'python_version': sys.version,
                'platform': platform.platform(), 'machine': platform.machine(), 'backend_label': label,
                'catalog': str(root / 'api-catalog.json'), 'catalog_count': len(ids),
                'catalog_sha256': digest(root / 'api-catalog.json'), 'excluded_apis': sorted(set(exclude)),
                'requested_imports': imports, 'test_outcomes_stored': False,
                'package_files': {p.name: digest(p) for p in PACKAGE.glob('*.py')},
                'collector_files': {str(p): digest(p) for p in runtime.rglob('*') if p.is_file()},
                'mapping_semantics': 'Observed callable identities in setup/call/teardown; includes transitive calls and aliases. Not assertion or native-kernel coverage.'}
    save(root / 'manifest.json', manifest)
    from .storage import connect
    db = connect(root)
    db.executemany('INSERT INTO apis VALUES (?,?,?,?)', [(a['api_id'], a['qualified_name'], a['language'], a['kind']) for a in data['apis']])
    db.execute('INSERT INTO jobs VALUES (?,?,?,?)', ('pytest', 'pytorch/', 'collecting', str(root / 'shards')))
    db.commit(); db.close()
    save(root / 'progress.json', {'state': 'initialized', 'updated_at': time.time()})
    return manifest


def summarize(root, state):
    import sqlite3
    with sqlite3.connect((root / 'relations.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        counts = {name: db.execute('SELECT count(*) FROM ' + name).fetchone()[0] for name in ('tests', 'relations', 'gaps')}
        counts['apis_observed'] = db.execute('SELECT count(distinct api_id) FROM relations').fetchone()[0]
    result = {'state': state, 'updated_at': time.time(), **counts, 'proves_zero_omissions': False}
    save(root / 'progress.json', result)
    return result


def finalize(root, timeout=120):
    from .storage import connect, merge, drain_children
    from .recovery import merge_recovering
    root = root.resolve()
    remaining = drain_children(root / 'shards', timeout=timeout)
    db = connect(root)
    try:
        complete = merge_recovering(merge, db, 'pytest', root / 'shards', final=True)
        state = 'capture_finished' if complete and not remaining else 'capture_incomplete'
        if remaining:
            db.execute('INSERT OR IGNORE INTO gaps VALUES (?,?,?,?)', ('pytest', '', 'session', 'children_still_active:' + ','.join(map(str, sorted(remaining)))))
        db.execute('UPDATE jobs SET state=? WHERE module=?', (state, 'pytest')); db.commit()
    finally: db.close()
    return summarize(root, state)


def doctor():
    require_host()
    import tempfile
    with tempfile.TemporaryDirectory(prefix='api-relations-doctor-') as folder:
        runtime = Path(folder)
        shutil.copyfile(PACKAGE / 'runtime/relation_identity.c', runtime / 'relation_identity.c')
        shutil.copyfile(PACKAGE / 'runtime/relation_guard.h', runtime / 'relation_guard.h')
        build_identity(runtime)
        subprocess.run([sys.executable, '-c', 'import _relation_identity; assert callable(_relation_identity.frame_function)'], cwd=folder, check=True)
    return {'host': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
            'native_adapter': 'built_and_imported', 'torch_imported': False, 'npu_execution_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Utilities for the opt-in API ↔ pytest relation plugin')
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('doctor', help='Build/import host native adapter without importing torch')
    q = commands.add_parser('query'); q.add_argument('--api', required=True); q.add_argument('--json', type=Path)
    commands.add_parser('status'); commands.add_parser('audit')
    s = commands.add_parser('settle', help='After pytest and its children exit, rebuild raw evidence into a new directory')
    s.add_argument('--destination', type=Path, required=True)
    for name, command in commands.choices.items():
        if name != 'doctor': command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'doctor': result = doctor()
    elif args.action == 'settle':
        from .settle import settle
        result = settle(args.out, args.destination)
    elif args.action == 'query':
        from .storage import query
        query(args); return
    elif args.action == 'status': result = read(args.out / 'progress.json')
    else:
        from .audit import audit
        result = audit(args.out.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
