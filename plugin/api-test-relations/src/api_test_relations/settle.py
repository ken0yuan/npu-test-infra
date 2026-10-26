"""Rebuild a post-exit summary in a NEW directory, preserving the initial run."""
from pathlib import Path
import shutil
from .cli import digest, finalize, read, save
from .storage import connect, active_children


def settle(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError('Destination must be outside the original evidence directory')
    if destination.exists():
        raise ValueError('Destination must not exist; original and derived evidence are never overwritten')
    manifest = read(source / 'manifest.json')
    owners = {int(p.name.split('-', 1)[0].split('.', 1)[0])
              for p in (source / 'shards').glob('*.sqlite') if p.name.split('-', 1)[0].split('.', 1)[0].isdigit()}
    if manifest.get('capture_pid'): owners.add(manifest['capture_pid'])
    # Conservative about PID reuse; never merge a possibly live producer.
    live = sorted(pid for pid in owners if Path('/proc', str(pid)).exists())
    live = sorted(set(live) | active_children(source / 'shards'))
    if live: raise RuntimeError('Capture producers must exit before settling: ' + str(live))
    # Reconstruct from raw evidence instead of importing provisional child states.
    snapshot = {}
    destination.mkdir(parents=True)
    for name in ('manifest.json', 'api-catalog.json', 'test-sources.json', 'environment.json', 'shards', 'runtime'):
        path = source / name
        if not path.exists(): continue
        target = destination / name
        if path.is_dir(): shutil.copytree(path, target)
        else: shutil.copy2(path, target)
        for copied in ([target] if target.is_file() else target.rglob('*')):
            if copied.is_file(): snapshot[str(copied.relative_to(destination))] = digest(copied)
    manifest['catalog'] = str(destination / 'api-catalog.json')
    manifest['derived_from'] = str(source)
    save(destination / 'manifest.json', manifest)
    save(destination / 'settlement.json', {'source': str(source), 'source_manifest_sha256': digest(source / 'manifest.json'),
         'source_progress': read(source / 'progress.json'), 'copied_file_hashes': snapshot,
         'operation': 'post-exit raw evidence rebuild', 'finalizer_version': '0.1.2', 'original_evidence_modified': False})
    db = connect(destination)
    try:
        catalog = read(destination / 'api-catalog.json')
        db.executemany('INSERT INTO apis VALUES (?,?,?,?)', [(a['api_id'], a['qualified_name'], a['language'], a['kind']) for a in catalog['apis']])
        db.execute('INSERT INTO jobs VALUES (?,?,?,?)', ('pytest', 'pytorch/', 'settling', str(destination / 'shards')))
        db.commit()
    finally: db.close()
    return finalize(destination, 0)
