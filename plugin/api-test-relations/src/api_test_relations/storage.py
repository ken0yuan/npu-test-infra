"""Evidence store extracted from relations-v4r2; no scheduler or device policy."""
import json,sqlite3,time
from pathlib import Path
from .cli import save

def connect(root):
    db=sqlite3.connect(root/'relations.sqlite',timeout=60)
    db.executescript('''
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS apis(api_id TEXT PRIMARY KEY,name TEXT,language TEXT,kind TEXT);
CREATE TABLE IF NOT EXISTS jobs(module TEXT PRIMARY KEY,file TEXT,state TEXT,attempt TEXT);
CREATE TABLE IF NOT EXISTS tests(job TEXT,nodeid TEXT,file TEXT,PRIMARY KEY(job,nodeid));
CREATE TABLE IF NOT EXISTS phases(job TEXT,nodeid TEXT,phase TEXT,state TEXT,PRIMARY KEY(job,nodeid,phase));
CREATE TABLE IF NOT EXISTS relations(job TEXT,nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,site TEXT,PRIMARY KEY(job,nodeid,phase,api_id));
CREATE INDEX IF NOT EXISTS by_api ON relations(api_id,nodeid);
CREATE TABLE IF NOT EXISTS gaps(job TEXT,nodeid TEXT,phase TEXT,reason TEXT,PRIMARY KEY(job,nodeid,phase,reason));
CREATE TABLE IF NOT EXISTS registry(job TEXT,api_id TEXT,state TEXT,PRIMARY KEY(job,api_id));
CREATE TABLE IF NOT EXISTS processes(job TEXT,token TEXT,nodeid TEXT,phase TEXT,pid INTEGER,state TEXT,PRIMARY KEY(job,token));
CREATE TABLE IF NOT EXISTS process_details(job TEXT,token TEXT,detail TEXT,PRIMARY KEY(job,token));
CREATE TABLE IF NOT EXISTS process_tasks(job TEXT,token TEXT,nodeid TEXT,phase TEXT,state TEXT,detail TEXT,PRIMARY KEY(job,token));
CREATE TABLE IF NOT EXISTS retained(job TEXT,nodeid TEXT,source_root TEXT,source_snapshot TEXT,source_attempt TEXT,source_version TEXT,relation_rows INTEGER,PRIMARY KEY(job,nodeid));
''')
    return db

def merge(db,job,folder,final=False):
    finished=False;shards=sorted(folder.glob('*.sqlite'))
    for shard in shards:
        source=sqlite3.connect(shard.as_uri()+'?mode=ro',uri=True)
        try:
            source.execute('BEGIN')
            tables={r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'meta' not in tables:
                if final:
                    db.execute('INSERT OR IGNORE INTO gaps VALUES (?,?,?,?)', (job, '', 'session', 'shard_without_metadata:' + shard.name))
                continue
            meta=dict(source.execute('SELECT key,value FROM meta'))
            role=meta.get('role','root')
            for table in ('tests','phases','relations','gaps','registry','processes'):
                if table not in tables or (table=='phases' and role=='child'):continue
                rows=list(source.execute('SELECT * FROM '+table))
                if table=='relations':
                    for node,phase,aid,aliases,site in rows:
                        if not node.startswith('pytorch/'):raise ValueError('invalid test path: '+node)
                        if not db.execute('SELECT 1 FROM tests WHERE job=? AND nodeid=?',(job,node)).fetchone():raise ValueError('owner missing')
                        if not db.execute('SELECT 1 FROM apis WHERE api_id=?',(aid,)).fetchone():raise ValueError('API outside ledger')
                if rows:db.executemany('INSERT OR REPLACE INTO '+table+' VALUES ('+','.join('?' for _ in range(len(rows[0])+1))+')',[(job,*row) for row in rows])
            flag=source.execute("SELECT value FROM meta WHERE key='session_finished'").fetchone()
            finished=finished or bool(role=='root' and flag and flag[0]=='true')
        finally:source.close()
    launches={}
    for file in (folder/'launches').glob('*.json'):
        data=json.loads(file.read_text());launches[data['token']]=data
        if not db.execute('SELECT 1 FROM tests WHERE job=? AND nodeid=?',(job,data['nodeid'])).fetchone():raise ValueError('process owner missing')
        db.execute('INSERT OR REPLACE INTO processes VALUES (?,?,?,?,?,?)',(job,data['token'],data['nodeid'],data['phase'],data['pid'],'spawned'))
        db.execute('INSERT OR REPLACE INTO process_details VALUES (?,?,?)',(job,data['token'],json.dumps(data)))
    receipts={}
    for file in (folder/'receipts').glob('*.json'):
        data=json.loads(file.read_text());receipts.setdefault(data['token'],[]).append(data)
        if data.get('transport')=='fork':
            db.execute('INSERT OR REPLACE INTO processes VALUES (?,?,?,?,?,?)',
                (job,data['token'],data['nodeid'],data['phase'],data['pid'],data['state']))
    for token,node,phase,pid in list(db.execute('SELECT token,nodeid,phase,pid FROM processes WHERE job=?',(job,))):
        observations=receipts.get(token,[])
        if any(r['nodeid']!=node or r['phase']!=phase for r in observations):raise ValueError('child receipt owner mismatch')
        states={r['state'] for r in observations}
        detail=launches.get(token,{})
        if 'bootstrap_error' in states:state='bootstrap_error'
        elif detail.get('service') and states and states<= {'service_started','service_finished'}:state='helper_service_observed'
        elif states and states<= {'finished','finished_without_torch','exec_transferred'} and all(r['state']!='exec_transferred' or receipts.get((r.get('detail') or {}).get('successor')) for r in observations):state='acknowledged_complete'
        elif states:state='child_active'
        elif detail.get('kind')=='launch_failed':state='not_started'
        elif final and detail.get('kind')=='external_executable':state='external_uninstrumented'
        else:state='unacknowledged' if final else 'awaiting_receipt'
        db.execute('UPDATE processes SET state=? WHERE job=? AND token=?',(state,job,token))
        if final and state not in ('acknowledged_complete','helper_service_observed','not_started'):
            db.execute('INSERT OR IGNORE INTO gaps VALUES (?,?,?,?)',(job,node,phase,'child_capture_'+state+':'+token))
    task_receipts={}
    for file in (folder/'task-receipts').glob('*.json'):
        data=json.loads(file.read_text());task_receipts.setdefault(data['token'],[]).append(data)
    for file in (folder/'tasks').glob('*.json'):
        data=json.loads(file.read_text());observations=task_receipts.get(data['token'],[])
        if any((r['nodeid'],r['phase'])!=(data['nodeid'],data['phase']) for r in observations):raise ValueError('task receipt owner mismatch')
        state='finished' if observations and all(r['state']=='finished' for r in observations) else 'unconfirmed'
        db.execute('INSERT OR REPLACE INTO process_tasks VALUES (?,?,?,?,?,?)',(job,data['token'],data['nodeid'],data['phase'],state,json.dumps({'submission':data,'receipts':observations})))
        if final and state!='finished':db.execute('INSERT OR IGNORE INTO gaps VALUES (?,?,?,?)',(job,data['nodeid'],data['phase'],'process_task_unconfirmed:'+data['token']))
    db.commit();return finished

def active_children(folder):
    active=set()
    for section in ('launches','receipts'):
        for file in (folder/section).glob('*.json'):
            data=json.loads(file.read_text())
            if data.get('service') or not data.get('pid'):continue
            if section=='receipts' and data.get('state') in ('finished','finished_without_torch','exec_transferred','bootstrap_error'):continue
            try:
                stat=Path('/proc',str(data['pid']),'stat').read_text().rpartition(')')[2].split()
                if stat[0]=='Z' or (data.get('start_ticks') and stat[19]!=data['start_ticks']):continue
                active.add(data['pid'])
            except (OSError,IndexError):pass
    return active

def drain_children(folder,timeout=120,on_tick=None):
    deadline=time.monotonic()+timeout;last_tick=time.monotonic()
    while active_children(folder) and time.monotonic()<deadline:
        time.sleep(.25)
        if on_tick and time.monotonic()-last_tick>=30:on_tick();last_tick=time.monotonic()
    return active_children(folder)

def query(args):
    root=args.out.resolve();published=(root/'storage.json').exists()
    path=root/('relations-published.sqlite' if published else 'relations.sqlite')
    db=sqlite3.connect(path.as_uri()+'?mode=ro'+('&immutable=1' if published else ''),uri=True)
    ids=[r[0] for r in db.execute('SELECT api_id FROM apis WHERE name=? OR api_id=?',(args.api,args.api))]
    tests={}
    for aid in ids:
        for job,node,phase,aliases,site in db.execute('SELECT job,nodeid,phase,aliases,site FROM relations WHERE api_id=? ORDER BY nodeid',(aid,)):
            entry=tests.setdefault((job,node),{'nodeid':node,'file':node.split('::')[0],'evidence':[],'capture_gaps':[]})
            entry['evidence'].append({'api_id':aid,'phase':phase,'same_callable_aliases':json.loads(aliases),'first_site':json.loads(site)})
    for (job,node),entry in tests.items():
        entry['capture_gaps']=[r[0] for r in db.execute('SELECT reason FROM gaps WHERE job=? AND (nodeid=? OR nodeid=\'\')',(job,node))]
        prior=db.execute('SELECT source_root,source_version,source_attempt FROM retained WHERE job=? AND nodeid=?',(job,node)).fetchone()
        entry['evidence_origin']={'root':prior[0],'version':prior[1],'attempt':prior[2],'retained':True} if prior else {'root':str(root),'version':root.name,'retained':False}
    result={'api':args.api,'catalog_matches':ids,'tests':list(tests.values()),'unobserved_means_no_test':False,
        'binding_states':{aid:dict(db.execute('SELECT state,count(*) FROM registry WHERE api_id=? GROUP BY state',(aid,))) for aid in ids}}
    if args.json:save(args.json,result)
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
