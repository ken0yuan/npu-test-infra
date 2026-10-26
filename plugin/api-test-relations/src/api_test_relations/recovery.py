"""Recover only hot-journal shards whose owner PID is absent; preserve original evidence."""
import hashlib,json,shutil,sqlite3,time
from pathlib import Path

def proc(pid):
    return Path("/proc",str(pid)).exists()

def recover_closed_shards(folder):
    """Let SQLite recover hot journals only after their owning process has exited.

    Preserve byte-for-byte incident copies first. Never recover an active writer,
    discard a journal, bypass locking, or treat other SQLite failures as empty data.
    """
    repaired=[]
    for path in sorted(Path(folder).glob('*.sqlite')):
        conn=None
        try:
            conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=2)
            conn.execute('SELECT name FROM sqlite_master').fetchall()
            continue
        except sqlite3.OperationalError as exc:
            if getattr(exc,'sqlite_errorname',None)!='SQLITE_READONLY_ROLLBACK':raise
        finally:
            if conn:conn.close()
        owner_pid=int(path.stem.split('-',1)[0])
        if proc(owner_pid):
            raise RuntimeError('Refusing recovery while shard owner PID exists: '+str(path))
        archive=Path(folder)/'sqlite-recovery'/str(time.time_ns())
        original=archive/'original';probe=archive/'probe'
        original.mkdir(parents=True);probe.mkdir()
        hashes={}
        for suffix in ('','-journal','-wal','-shm'):
            item=Path(str(path)+suffix)
            if item.exists():
                shutil.copy2(item,original/item.name)
                shutil.copy2(item,probe/item.name)
                hashes[item.name]=hashlib.sha256(item.read_bytes()).hexdigest()
        evidence={'source':str(path),'owner_pid':owner_pid,'at':time.time(),'original_sha256':hashes,'state':'snapshotted'}
        (archive/'recovery.json').write_text(json.dumps(evidence,indent=2))
        def restore(target):
            db=sqlite3.connect(target,timeout=2)
            try:
                tables=db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
                integrity=db.execute('PRAGMA quick_check').fetchone()[0]
                if integrity!='ok':raise RuntimeError('Recovery integrity check failed: '+str(target))
                return {'tables':[row[0] for row in tables],'dump':list(db.iterdump())}
            finally:db.close()
        expected=restore(probe/path.name)
        # The process is still absent and evidence bytes have not changed.
        if proc(owner_pid):raise RuntimeError('Shard PID appeared during recovery')
        for name,digest in hashes.items():
            if hashlib.sha256((path.parent/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Shard changed during recovery: '+str(path))
        actual=restore(path)
        if actual!=expected:raise RuntimeError('Recovered original differs from recovery copy')
        evidence.update(state='recovered',tables=actual['tables'],recovered_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (archive/'recovery.json').write_text(json.dumps(evidence,indent=2))
        repaired.append(str(archive/'recovery.json'))
    return repaired

def merge_recovering(merge,db,job,folder,final=False):
    try:return merge(db,job,folder,final=final)
    except sqlite3.OperationalError as exc:
        if getattr(exc,'sqlite_errorname',None)!='SQLITE_READONLY_ROLLBACK':raise
        if not recover_closed_shards(folder):raise
        return merge(db,job,folder,final=final)
