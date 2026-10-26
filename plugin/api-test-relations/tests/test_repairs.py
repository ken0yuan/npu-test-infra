import cProfile
import _lsprof
import inspect
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace
import pytest
from test_relation_collector import catalog, collecting
from relation_collector import Registry
from api_test_relations.recovery import recover_closed_shards, merge_recovering


@pytest.mark.parametrize('factory',[cProfile.Profile,_lsprof.Profiler])
def test_existing_native_profiler_same_line_and_stats(factory):
    profiler=factory()  # Construct before collector install, no class patching.
    def public():pass
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('native[param]','call'))
        profiler.enable();public();profiler.disable();public()
        c.leave(token)
    assert hits==[(('native[param]','call'),('python:torch.public',))]*2
    assert not gaps
    assert sum(e.callcount for e in profiler.getstats() if e.code is public.__code__)==1


def test_native_profiler_already_enabled_is_forwarded_and_restored():
    profiler=cProfile.Profile()
    def public():pass
    profiler.enable()
    try:
        with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
            assert sys.getprofile() is profiler
            token=c.enter(('preexisting','call'));public();c.leave(token)
        assert sys.getprofile() is profiler
        public()
    finally:profiler.disable()
    assert len(hits)==1 and not gaps
    assert sum(e.callcount for e in profiler.getstats() if e.code is public.__code__)==2


def test_profiler_context_manager_and_exceptions_keep_capture():
    def public():raise ValueError('workload')
    profiler=cProfile.Profile()
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('exception','call'))
        with pytest.raises(ValueError,match='workload'):
            with profiler:public()
        with pytest.raises(ValueError):public()
        c.leave(token)
    assert len(hits)==2 and not gaps


@pytest.mark.parametrize('opcode',[False,True])
def test_preexisting_trace_client_keeps_events_and_flags(opcode):
    def public():
        value=1
        return value+1
    events=[]
    def trace(frame,event,arg):
        if frame.f_code is public.__code__:
            if event=='call':frame.f_trace_opcodes=opcode
            events.append(event)
        return trace
    sys.settrace(trace)
    try:public()
    finally:sys.settrace(None)
    expected=list(events);events.clear()
    sys.settrace(trace)
    try:
        with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
            assert sys.gettrace() is trace
            token=c.enter(('trace','call'));public();c.leave(token)
        assert sys.gettrace() is trace
    finally:sys.settrace(None)
    assert events==expected
    assert len(hits)==1 and not gaps


def test_trace_install_remove_during_capture_and_raw_bypass_is_visible():
    raw_trace=sys.settrace
    def public():pass
    events=[]
    def trace(frame,event,arg):
        if frame.f_code is public.__code__:events.append(event)
        return trace
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('trace-changes','call'))
        sys.settrace(trace);public();sys.settrace(None);public()
        raw_trace(None)  # Arbitrary native/raw loss of BOTH channels is a boundary.
        c.leave(token)
    assert len(hits)==2
    assert events==['call','line','return']
    assert any(reason=='trace_guard_bypassed' for _,reason in gaps)


def test_native_profiler_in_new_thread_keeps_original_owner():
    def public():pass
    def worker():
        p=cProfile.Profile();p.enable();public();p.disable();public()
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('thread-profiler','call'))
        thread=threading.Thread(target=worker);thread.start();thread.join()
        c.leave(token)
    assert hits==[(('thread-profiler','call'),('python:torch.public',))]*2
    assert not gaps


def test_registry_concurrent_refresh_has_no_partial_publication():
    root=ModuleType('torch');root.child=ModuleType('child')
    def public():pass
    root.child.public=public
    registry=Registry(catalog(('child.public','function')),root)
    errors=[];barrier=threading.Barrier(6)
    def refresh():
        try:
            barrier.wait()
            for _ in range(50):
                registry.refresh(force=True)
                with registry._lock:
                    assert registry.python[public]=={'python:torch.child.public'}
                    assert len(registry.namespace_objects)==len(registry.versions)
        except BaseException as exc:errors.append(exc)
    threads=[threading.Thread(target=refresh) for _ in range(6)]
    for t in threads:t.start()
    for t in threads:t.join(timeout=20)
    assert not errors and not any(t.is_alive() for t in threads)


def test_failed_refresh_preserves_old_snapshot_and_phase_records_gap(monkeypatch):
    def public():pass
    def replacement():pass
    root=SimpleNamespace(public=public)
    with collecting(root,catalog(('public','function'))) as (c,hits,gaps):
        old=c.registry.python
        root.public=replacement
        original=Registry._refresh
        def broken(stage,force=False):
            original(stage,True)
            raise RuntimeError('injected refresh failure')
        monkeypatch.setattr(Registry,'_refresh',broken)
        token=c.enter(('rollback','call'))
        assert c.registry.python is old and replacement not in old
        c.leave(token)
        monkeypatch.setattr(Registry,'_refresh',original)
    assert any('injected refresh failure' in reason for _,reason in gaps)


def test_rebind_during_scan_remains_dirty_until_new_callable_resolved(monkeypatch):
    from _relation_identity import namespaces_changed
    root=ModuleType('torch')
    def public():pass
    def replacement():pass
    root.public=public
    registry=Registry(catalog(('public','function')),root)
    original=inspect.getattr_static
    def read_and_rebind(obj,name):
        value=original(obj,name)
        if obj is root and name=='public':root.public=replacement
        return value
    with monkeypatch.context() as patch:
        patch.setattr(inspect,'getattr_static',read_and_rebind)
        registry.refresh(force=True)
    assert namespaces_changed(registry.namespace_objects,registry.versions)
    registry.refresh()
    assert registry.python[replacement]=={'python:torch.public'}


def test_concurrent_refresh_with_growing_namespaces_and_observed_calls():
    root=ModuleType('torch');errors=[]
    def public():pass
    cat=catalog(*[(f'child{i}.public','function') for i in range(12)])
    barrier=threading.Barrier(4)
    with collecting(root,cat) as (c,hits,gaps):
        token=c.enter(('growing','call'))
        def worker(start):
            try:
                barrier.wait()
                for i in range(start,12,4):
                    child=ModuleType(f'child{i}');child.public=public
                    setattr(root,f'child{i}',child)
                    c.registry.refresh(force=True)
                    public()
            except BaseException as exc:errors.append(exc)
        threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
        for t in threads:t.start()
        for t in threads:t.join(timeout=20)
        c.leave(token)
    assert not errors and not any(t.is_alive() for t in threads) and not gaps
    assert len(hits)==12 and all(owner==('growing','call') for owner,ids in hits)


def test_profiler_python_client_exception_is_not_swallowed():
    def public():pass
    def client(frame,event,arg):
        if frame.f_code is public.__code__ and event=='call':raise LookupError('client failed')
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('client-error','call'))
        sys.setprofile(client)
        with pytest.raises(LookupError,match='client failed'):public()
        sys.setprofile(None);public();c.leave(token)
    assert len(hits)==2


def test_trace_client_can_disable_itself_before_native_profiler_start():
    def public():pass
    def runner():
        p=cProfile.Profile();p.enable();public();p.disable();public()
    def trace(frame,event,arg):
        if frame.f_code is runner.__code__ and event=='line':sys.settrace(None)
        return trace
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('self-disabling-trace','call'))
        sys.settrace(trace);runner();c.leave(token)
    assert len(hits)==2 and not gaps


def hot_shard(folder,pid=2147483000):
    path=folder/(str(pid)+'-incident.sqlite')
    code="""import sqlite3,sys,os
d=sqlite3.connect(sys.argv[1]);d.execute('PRAGMA journal_mode=DELETE');d.execute('PRAGMA cache_size=1')
d.execute('CREATE TABLE evidence(value BLOB)');d.execute('INSERT INTO evidence VALUES (?)',(b'committed',));d.commit()
d.execute('BEGIN IMMEDIATE');d.execute('UPDATE evidence SET value=?',(b'x'*1000000,));os._exit(0)
"""
    subprocess.run([sys.executable,'-c',code,str(path)],check=True)
    return path


def test_recovery_preserves_evidence_and_committed_rows(tmp_path):
    path=hot_shard(tmp_path);before=path.read_bytes();journal=Path(str(path)+'-journal').read_bytes()
    logs=recover_closed_shards(tmp_path)
    assert len(logs)==1
    original=Path(logs[0]).parent/'original'
    assert (original/path.name).read_bytes()==before
    assert (original/(path.name+'-journal')).read_bytes()==journal
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        assert db.execute('SELECT value FROM evidence').fetchall()==[(b'committed',)]
    assert recover_closed_shards(tmp_path)==[]


def test_recovery_refuses_live_owner_and_unrelated_corruption(tmp_path):
    path=hot_shard(tmp_path,os.getpid());before=path.read_bytes()
    with pytest.raises(RuntimeError,match='owner PID exists'):recover_closed_shards(tmp_path)
    assert path.read_bytes()==before
    other=tmp_path/'other';other.mkdir();(other/'2147483000.sqlite').write_bytes(b'not a database'*100)
    with pytest.raises(sqlite3.DatabaseError):recover_closed_shards(other)


def test_merge_retry_restores_committed_data(tmp_path):
    path=hot_shard(tmp_path);calls=[]
    def merge(db,job,folder,final=False):
        calls.append(job);conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        try:return conn.execute('SELECT value FROM evidence').fetchone()[0]
        finally:conn.close()
    assert merge_recovering(merge,None,'test',tmp_path,final=True)==b'committed'
    assert calls==['test','test']


def test_finalize_recovers_real_relation_shard_and_publishes_mapping(tmp_path):
    from api_test_relations.cli import finalize
    from api_test_relations.storage import connect
    folder=tmp_path/'shards';folder.mkdir();path=folder/'2147483000.sqlite'
    code="""import sqlite3,sys,os
d=sqlite3.connect(sys.argv[1]);d.execute('PRAGMA journal_mode=DELETE');d.execute('PRAGMA cache_size=1')
d.executescript('CREATE TABLE meta(key TEXT,value TEXT); CREATE TABLE tests(nodeid TEXT,file TEXT); CREATE TABLE relations(nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,site TEXT);')
d.execute('INSERT INTO meta VALUES (?,?)',('session_finished','true'))
d.execute('INSERT INTO tests VALUES (?,?)',('pytorch/test/test_one.py::test[param]','pytorch/test/test_one.py'))
d.execute('INSERT INTO relations VALUES (?,?,?,?,?)',('pytorch/test/test_one.py::test[param]','call','api-one','[]','["pytorch/test/test_one.py", 3]'));d.commit()
d.execute('BEGIN IMMEDIATE');d.execute('UPDATE relations SET site=?',('x'*1000000,));os._exit(0)
"""
    subprocess.run([sys.executable,'-c',code,str(path)],check=True)
    db=connect(tmp_path)
    db.execute('INSERT INTO apis VALUES (?,?,?,?)',('api-one','torch.public','python','function'))
    db.execute('INSERT INTO jobs VALUES (?,?,?,?)',('pytest','pytorch/','collecting',str(folder)))
    db.commit();db.close()
    result=finalize(tmp_path,0)
    assert result['state']=='capture_finished' and result['relations']==1
    with sqlite3.connect(tmp_path/'relations.sqlite') as db:
        assert db.execute('SELECT site FROM relations').fetchone()[0]=='["pytorch/test/test_one.py", 3]'
    assert len(list((folder/'sqlite-recovery').glob('*/recovery.json')))==1
