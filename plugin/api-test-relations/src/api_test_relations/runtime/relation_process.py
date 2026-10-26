"""Causal process propagation, explicit Python entry and durable launch evidence."""
import atexit,builtins,importlib,inspect,json,os,subprocess,sys,threading,uuid
from pathlib import Path
from relation_context import OWNER
from relation_launch import prepare
DIRECTORY=Path(__file__).resolve().parent
ACTIVE=None
BOOT_PID=None
BOOT_CONTEXT=None
EARLY_BRIDGE=None
IMPORT_RESTORE=None
TASK_CONTEXT=None
SETTINGS=('API_RELATION_REPO','API_RELATION_CATALOG','API_RELATION_SHARDS','API_RELATION_EXCLUDE')

def document(section,name,data):
    folder=Path(os.environ['API_RELATION_SHARDS'])/section;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.json');temp=path.with_suffix('.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(data));temp.replace(path)

def receipt(context,state,detail=None):
    document('receipts',context['token']+'-'+str(os.getpid()),{**context,'pid':os.getpid(),'state':state,'detail':detail})

def current_owner():
    owner=OWNER.get()
    if owner is None and BOOT_PID==os.getpid() and BOOT_CONTEXT and not BOOT_CONTEXT.get('service'):
        owner=(BOOT_CONTEXT['nodeid'],BOOT_CONTEXT['phase'])
    return owner

def context_for(owner,transport):
    frame=inspect.currentframe().f_back
    while frame and (Path(frame.f_code.co_filename).parent==DIRECTORY or frame.f_code.co_filename.endswith('/subprocess.py')):frame=frame.f_back
    return {'token':uuid.uuid4().hex,'nodeid':owner[0],'phase':owner[1],'parent_pid':os.getpid(),
            'parent_token':BOOT_CONTEXT['token'] if BOOT_PID==os.getpid() and BOOT_CONTEXT else None,
            'transport':transport,'site':[frame.f_code.co_filename,frame.f_lineno] if frame else None,
            'settings':{key:os.environ[key] for key in SETTINGS if key in os.environ}}

def launch(context,pid,detail):
    try:start_ticks=Path('/proc',str(pid),'stat').read_text().rpartition(')')[2].split()[19]
    except (OSError,IndexError):start_ticks=None
    document('launches',context['token'],{**context,'pid':pid,'start_ticks':start_ticks,**detail})

def environment(context,base,explicit=False):
    env=dict(base);env.update(context['settings']);env['API_RELATION_PARENT']=json.dumps(context)
    if not explicit:env['PYTHONPATH']=str(DIRECTORY/'bootstrap')+os.pathsep+str(DIRECTORY)+os.pathsep+env.get('PYTHONPATH','')
    return env

def register_task(plugin,context):
    node=context['nodeid']
    plugin.db.execute('INSERT OR IGNORE INTO tests VALUES (?,?)',(node,node.split('::')[0]));plugin.db.commit()

def run_owned(context,function,args,kwargs):
    global TASK_CONTEXT
    old=TASK_CONTEXT;TASK_CONTEXT=context
    token=OWNER.set((context['nodeid'],context['phase']))
    if ACTIVE is not None:register_task(ACTIVE,context)
    document('task-receipts',context['token']+'-'+str(os.getpid()),{**context,'pid':os.getpid(),'state':'started'})
    try:return function(*args,**kwargs)
    finally:
        if ACTIVE is not None:ACTIVE.flush()
        document('task-receipts',context['token']+'-'+str(os.getpid()),{**context,'pid':os.getpid(),'state':'finished'})
        OWNER.reset(token);TASK_CONTEXT=old

def mapped_owned(context,function,*args):return run_owned(context,function,args,{})

def task_context():
    owner=current_owner()
    if owner is None:return None
    context=context_for(owner,'process_pool_task');document('tasks',context['token'],context);return context

class ProcessBridge:
    def __init__(self,plugin=None):
        self.plugin=plugin;self.pid=os.getpid();self.enabled=True;self.local=threading.local();self.patches=[]
        original=subprocess.Popen.__init__;signature=inspect.signature(original)
        def popen(process,*args,**kwargs):
            owner=current_owner()
            if owner is None or os.getpid()!=self.pid:return original(process,*args,**kwargs)
            context=context_for(owner,'subprocess');bound=signature.bind(process,*args,**kwargs);values=bound.arguments
            base=values.get('env');base=os.environ if base is None else base
            command,detail=prepare(values['args'],context,base,values.get('executable'),values.get('shell',False),values.get('cwd'))
            values['args']=command;values['env']=environment(context,base,detail['kind']=='python_explicit')
            self.local.launch_requested=True
            try:original(*bound.args,**bound.kwargs)
            except BaseException:
                launch(context,None,{**detail,'kind':'launch_failed'});raise
            finally:self.local.launch_requested=False
            launch(context,process.pid,detail)
        self.patch(subprocess.Popen,'__init__',popen)
        self.install_multiprocessing()
        self.install_pools()
        if hasattr(os,'register_at_fork'):
            os.register_at_fork(after_in_child=self.after_fork)
            original_fork=os.fork
            def fork():
                owner=current_owner();context=getattr(self.local,'mp_context',None)
                if context is None and owner is not None:context=context_for(owner,'fork')
                self.local.fork_context=context;self.local.fork_requested=True
                try:pid=original_fork()
                finally:self.local.fork_requested=False
                if pid>0 and context:launch(context,pid,{'kind':'python_fork','argv':[],'executable':sys.executable})
                return pid
            self.patch(os,'fork',fork)
        for name in ('posix_spawn','posix_spawnp'):
            if not hasattr(os,name):continue
            original_spawn=getattr(os,name)
            def spawn(path,argv,env,*args,_original=original_spawn,_name=name,**kwargs):
                owner=current_owner()
                if owner is None or getattr(self.local,'launch_requested',False):return _original(path,argv,env,*args,**kwargs)
                context=context_for(owner,_name);command,detail=prepare(argv,context,env,path)
                self.local.launch_requested=True
                try:pid=_original(path,command,environment(context,env,detail['kind']=='python_explicit'),*args,**kwargs)
                finally:self.local.launch_requested=False
                launch(context,pid,detail);return pid
            self.patch(os,name,spawn)
        original_execve=os.execve
        def execve(path,argv,env):
            owner=current_owner()
            if owner is None or isinstance(path,int):return original_execve(path,argv,env)
            context=context_for(owner,'exec');command,detail=prepare(argv,context,env,path)
            launch(context,os.getpid(),detail)
            plugin=ACTIVE or self.plugin
            if plugin is not None:plugin.flush()
            old=BOOT_CONTEXT if BOOT_PID==os.getpid() else None
            if old:receipt(old,'exec_transferred',{'successor':context['token']})
            self.local.launch_requested=True
            try:return original_execve(path,command,environment(context,env,detail['kind']=='python_explicit'))
            except BaseException:
                launch(context,None,{**detail,'kind':'launch_failed'})
                if old:receipt(old,'instrumented' if ACTIVE else 'python_started')
                raise
            finally:self.local.launch_requested=False
        self.patch(os,'execve',execve)
        original_execv=os.execv
        def execv(path,argv):
            if current_owner() is None:return original_execv(path,argv)
            return execve(path,argv,os.environ)
        self.patch(os,'execv',execv)

    def patch(self,obj,name,replacement):
        self.patches.append((obj,name,getattr(obj,name),replacement));setattr(obj,name,replacement)

    def install_pools(self):
        import concurrent.futures.process as futures
        import multiprocessing.pool as pool
        from functools import partial
        original_submit=futures.ProcessPoolExecutor.submit
        def submit(executor,fn,*args,**kwargs):
            context=task_context()
            if context:return original_submit(executor,run_owned,context,fn,args,kwargs)
            return original_submit(executor,fn,*args,**kwargs)
        self.patch(futures.ProcessPoolExecutor,'submit',submit)
        original_apply=pool.Pool.apply_async
        def apply_async(instance,func,args=(),kwds=None,callback=None,error_callback=None):
            context=task_context();kwds={} if kwds is None else kwds
            if context:return original_apply(instance,run_owned,(context,func,args,kwds),{},callback,error_callback)
            return original_apply(instance,func,args,kwds,callback,error_callback)
        self.patch(pool.Pool,'apply_async',apply_async)
        original_map=pool.Pool._map_async
        def map_async(instance,func,iterable,mapper,chunksize=None,callback=None,error_callback=None):
            context=task_context()
            if context:func=partial(mapped_owned,context,func)
            return original_map(instance,func,iterable,mapper,chunksize,callback,error_callback)
        self.patch(pool.Pool,'_map_async',map_async)
        for name in ('imap','imap_unordered'):
            original=getattr(pool.Pool,name)
            def imap(instance,func,iterable,chunksize=1,_original=original):
                context=task_context()
                if context:func=partial(mapped_owned,context,func)
                return _original(instance,func,iterable,chunksize)
            self.patch(pool.Pool,name,imap)

    def install_multiprocessing(self):
        import multiprocessing.process as process
        import multiprocessing.spawn as spawn
        import multiprocessing.util as util
        original_start=process.BaseProcess.start
        def start(instance):
            owner=current_owner()
            if owner is None:return original_start(instance)
            context=context_for(owner,'multiprocessing');self.local.mp_context=context
            try:result=original_start(instance)
            finally:self.local.mp_context=None
            path=Path(os.environ['API_RELATION_SHARDS'])/'launches'/(context['token']+'.json')
            if not path.exists():launch(context,instance.pid,{'kind':'python_multiprocessing','argv':[],'executable':sys.executable})
            return result
        self.patch(process.BaseProcess,'start',start)
        original_data=spawn.get_preparation_data
        def get_data(name):
            data=original_data(name);context=getattr(self.local,'mp_context',None)
            if context:data['_api_relation_context']=context
            return data
        self.patch(spawn,'get_preparation_data',get_data)
        original_prepare=spawn.prepare
        def prepare_child(data):
            context=data.get('_api_relation_context')
            if context:
                if os.getpid()!=self.pid and spawn.prepare is prepare_child:spawn.prepare=original_prepare
                bootstrap(context)
            return original_prepare(data)
        self.prepare_child=prepare_child;self.patch(spawn,'prepare',prepare_child)
        original_spawnv=util.spawnv_passfds
        def spawnv(path,args,passfds):
            owner=current_owner()
            if owner is None:return original_spawnv(path,args,passfds)
            words=[os.fsdecode(x) for x in args]
            service=any('multiprocessing.resource_tracker' in x or 'multiprocessing.forkserver' in x for x in words)
            context=None if service else getattr(self.local,'mp_context',None)
            context=context or context_for(owner,'multiprocessing_spawnv')
            if service:context={**context,'service':True}
            command,detail=prepare(words,context,os.environ,path)
            self.local.launch_requested=True
            try:pid=original_spawnv(path,command,passfds)
            finally:self.local.launch_requested=False
            launch(context,pid,detail);return pid
        self.patch(util,'spawnv_passfds',spawnv)

    def after_fork(self):
        global ACTIVE,BOOT_PID,BOOT_CONTEXT,EARLY_BRIDGE,IMPORT_RESTORE,TASK_CONTEXT
        if not self.enabled or os.getpid()==self.pid:return
        context=getattr(self.local,'fork_context',None)
        if ACTIVE is not None:ACTIVE.collector.close()
        elif self.plugin is not None:self.plugin.collector.close()
        self.close()
        if IMPORT_RESTORE:builtins.__import__,importlib.import_module=IMPORT_RESTORE
        ACTIVE=None;BOOT_PID=None;BOOT_CONTEXT=None;EARLY_BRIDGE=None;IMPORT_RESTORE=None;TASK_CONTEXT=None
        if context:bootstrap(context)
        else:
            import multiprocessing.spawn as spawn
            spawn.prepare=self.prepare_child

    def close(self):
        self.enabled=False
        for obj,name,original,replacement in reversed(self.patches):
            if getattr(obj,name)==replacement:setattr(obj,name,original)

def start_child(context):
    global ACTIVE,EARLY_BRIDGE
    if ACTIVE is not None and ACTIVE.pid==os.getpid():return
    try:
        if EARLY_BRIDGE:EARLY_BRIDGE.close();EARLY_BRIDGE=None
        sys.path.insert(0,str(DIRECTORY))
        try:
            from relation_plugin import RelationPlugin
            plugin=RelationPlugin(child_context=context)
        finally:sys.path.remove(str(DIRECTORY))
        ACTIVE=plugin
        if TASK_CONTEXT:
            register_task(plugin,TASK_CONTEXT);OWNER.set((TASK_CONTEXT['nodeid'],TASK_CONTEXT['phase']))
        receipt(context,'instrumented')
    except Exception as exc:receipt(context,'bootstrap_error',type(exc).__name__+': '+str(exc)[:500])

def bootstrap(context=None):
    global BOOT_PID,BOOT_CONTEXT,EARLY_BRIDGE,IMPORT_RESTORE
    if context is None:
        raw=os.environ.get('API_RELATION_PARENT')
        if not raw:return
        context=json.loads(raw)
    if BOOT_PID==os.getpid() and BOOT_CONTEXT and BOOT_CONTEXT['token']==context['token']:return
    if BOOT_PID==os.getpid() and BOOT_CONTEXT:raise RuntimeError('Refusing to silently reuse a process across different test owners')
    BOOT_PID=os.getpid();BOOT_CONTEXT=context
    for key,value in context.get('settings',{}).items():os.environ[key]=value
    os.environ['API_RELATION_PARENT']=json.dumps(context)
    receipt(context,'service_started' if context.get('service') else 'python_started')
    EARLY_BRIDGE=ProcessBridge()
    original_import=builtins.__import__;original_module=importlib.import_module
    IMPORT_RESTORE=(original_import,original_module);busy=False
    def activate():
        nonlocal busy
        if busy or ACTIVE is not None or context.get('service'):return
        torch=sys.modules.get('torch')
        if torch is None or getattr(getattr(torch,'__spec__',None),'_initializing',False):return
        busy=True;builtins.__import__=original_import;importlib.import_module=original_module
        try:start_child(context)
        finally:busy=False
    def importing(*args,**kwargs):
        result=original_import(*args,**kwargs);activate();return result
    def importing_module(*args,**kwargs):
        result=original_module(*args,**kwargs);activate();return result
    builtins.__import__=importing;importlib.import_module=importing_module
    finished=False;pid=os.getpid()
    def finish():
        nonlocal finished
        if finished or os.getpid()!=pid:return
        if ACTIVE is not None:ACTIVE.finish_child();receipt(context,'finished')
        elif context.get('service'):receipt(context,'service_finished')
        elif 'torch' not in sys.modules:receipt(context,'finished_without_torch')
        if EARLY_BRIDGE:EARLY_BRIDGE.close()
        finished=True
    atexit.register(finish);original_exit=os._exit
    def exit_with_receipt(code):
        try:finish()
        finally:original_exit(code)
    os._exit=exit_with_receipt
    activate()
