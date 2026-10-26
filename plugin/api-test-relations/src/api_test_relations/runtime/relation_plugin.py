"""pytest API relations only: no test outcomes, assertion text, or timing results."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import pytest
from relation_collector import Collector,Registry


def repo_path(path,repo):
    try:return 'pytorch/'+Path(path).resolve().relative_to(repo).as_posix()
    except (ValueError,OSError):return str(path)


class RelationPlugin:
    def __init__(self,child_context=None):
        import torch
        self.repo=Path(os.environ['API_RELATION_REPO']).resolve()
        self.pid=os.getpid();self.main_thread=threading.get_ident()
        folder=Path(os.environ['API_RELATION_SHARDS']);folder.mkdir(parents=True,exist_ok=True)
        suffix='-'+child_context['token'] if child_context else ''
        self.db=sqlite3.connect(folder/(str(self.pid)+suffix+'.sqlite'))
        self.db.executescript('''
PRAGMA journal_mode=WAL;
CREATE TABLE tests(nodeid TEXT PRIMARY KEY,file TEXT);
CREATE TABLE phases(nodeid TEXT,phase TEXT,state TEXT,PRIMARY KEY(nodeid,phase));
CREATE TABLE relations(nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,site TEXT,PRIMARY KEY(nodeid,phase,api_id));
CREATE TABLE gaps(nodeid TEXT,phase TEXT,reason TEXT,PRIMARY KEY(nodeid,phase,reason));
CREATE TABLE registry(api_id TEXT PRIMARY KEY,state TEXT);
CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE processes(token TEXT PRIMARY KEY,nodeid TEXT,phase TEXT,pid INTEGER,state TEXT);
''')
        cat=json.loads(Path(os.environ['API_RELATION_CATALOG']).read_text())
        self.registry=Registry(cat,torch,excluded=json.loads(os.environ.get('API_RELATION_EXCLUDE','[]')))
        self.pending={};self.gaps=set();self.processes={};self.lock=threading.RLock()
        self.paths={};self.tokens={};self.collector=Collector(self.registry,self.record,self.gap)
        self.db.execute('INSERT INTO meta VALUES (?,?)',('collector','exact_identity_relations_v4'))
        self.child_context=child_context
        self.db.execute('INSERT INTO meta VALUES (?,?)',('role','child' if child_context else 'root'))
        if child_context:
            self.db.execute('INSERT INTO meta VALUES (?,?)',('parent_context',json.dumps(child_context)))
            node=child_context['nodeid'];phase=child_context['phase']
            self.db.execute('INSERT INTO tests VALUES (?,?)',(node,node.split('::')[0]))
            self.db.execute('INSERT INTO phases VALUES (?,?,?)',(node,phase,'entered'))
        self.db.commit();self.collector.install()
        from relation_process import ProcessBridge
        self.bridge=ProcessBridge(self)
        if child_context:self.child_token=self.collector.enter((node,phase))

    def node(self,item):
        _,sep,suffix=item.nodeid.partition('::')
        return repo_path(item.path,self.repo)+(sep+suffix if sep else '')

    def gap(self,owner,reason):
        if reason=='child_process_not_instrumented:subprocess.Popen' and getattr(getattr(getattr(self,'bridge',None),'local',None),'launch_requested',False):
            return  # Verified by process edges and child receipts, not a blanket gap.
        if reason in ('child_process_not_instrumented:os.fork','child_process_not_instrumented:os.posix_spawn','child_process_not_instrumented:os.exec') and getattr(getattr(getattr(self,'bridge',None),'local',None),'launch_requested',False):return
        if reason=='child_process_not_instrumented:os.fork' and getattr(getattr(getattr(self,'bridge',None),'local',None),'fork_requested',False):return
        with self.lock:self.gaps.add((*owner,reason))

    def record(self,owner,ids,site):
        with self.lock:
            for aid in ids:
                # Only the first evidence site is needed to prove a relation.
                self.pending.setdefault((*owner,aid),(ids,site))

    def flush(self):
        if os.getpid()!=self.pid or threading.get_ident()!=self.main_thread:return
        with self.lock:
            pending=self.pending;self.pending={}
            gaps=self.gaps;self.gaps=set()
            processes=[(token,*row) for token,row in self.processes.items()]
        rows=[]
        for (node,phase,aid),(aliases,(file,line)) in pending.items():
            if file not in self.paths:self.paths[file]=repo_path(file,self.repo)
            rows.append((node,phase,aid,json.dumps(aliases),json.dumps([self.paths[file],line])))
        self.db.executemany('INSERT OR IGNORE INTO relations VALUES (?,?,?,?,?)',rows)
        self.db.executemany('INSERT OR IGNORE INTO gaps VALUES (?,?,?)',gaps)
        self.db.executemany('INSERT OR REPLACE INTO registry VALUES (?,?)',self.registry.status.items())
        self.db.executemany('INSERT OR REPLACE INTO processes VALUES (?,?,?,?,?)',processes)
        self.db.commit()

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self,session,config,items):
        for item in items:
            if not Path(item.path).resolve().is_relative_to(self.repo):
                raise pytest.UsageError('Captured tests must be inside --api-relations-repo: '+str(item.path))
            self.db.execute('INSERT OR IGNORE INTO tests VALUES (?,?)',(self.node(item),repo_path(item.path,self.repo)))
        self.registry.refresh();self.flush()

    def pytest_collectreport(self,report):
        if report.failed:
            self.gap((str(report.nodeid),'collection'),'collection_unavailable:'+str(report.longrepr)[-1200:])
            self.flush()

    def begin(self,item,phase):
        node=self.node(item)
        self.db.execute('INSERT OR REPLACE INTO phases VALUES (?,?,?)',(node,phase,'entered'));self.db.commit()
        self.tokens[phase]=self.collector.enter((node,phase))

    def end(self,item,phase):
        self.collector.leave(self.tokens.pop(phase))
        self.flush()
        self.db.execute('INSERT OR REPLACE INTO phases VALUES (?,?,?)',(self.node(item),phase,'closed'));self.db.commit()

    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_runtest_setup(self,item):
        self.begin(item,'setup')
        try:yield
        finally:self.end(item,'setup')

    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_runtest_call(self,item):
        self.begin(item,'call')
        try:yield
        finally:self.end(item,'call')

    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_runtest_teardown(self,item):
        self.begin(item,'teardown')
        try:yield
        finally:self.end(item,'teardown')

    def pytest_sessionfinish(self,session,exitstatus):
        self.bridge.close();self.collector.close();self.flush()
        # Execution completion, NOT pytest result or exit status.
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('session_finished','true'));self.db.commit()

    def finish_child(self):
        self.collector.leave(self.child_token)
        self.bridge.close();self.collector.close();self.flush()
        self.db.execute("UPDATE phases SET state='closed'")
        self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('session_finished','true'));self.db.commit()


def pytest_configure(config):
    if os.environ.get('API_RELATION_SHARDS') and not os.environ.get('API_RELATION_PARENT'):
        config.pluginmanager.register(RelationPlugin(),'api-relations-v4')
