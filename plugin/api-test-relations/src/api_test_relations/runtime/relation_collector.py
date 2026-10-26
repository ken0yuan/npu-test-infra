"""Exact callable identity, independent profiling clients, causal thread ownership.

No PyTorch names, source offsets or test-specific exceptions are hard-coded.
"""
import _thread
import concurrent.futures
import contextvars
import inspect
import os
import sys
import threading
import types
from _relation_identity import frame_function, method_definition, namespace_version, namespaces_changed, set_profile, current_profile, install_guard, remove_guard, profile_client, trace_client, guard_intact, guard_available

from relation_context import OWNER


class Registry:
    def __init__(self, catalog, root, excluded=()):
        self.root=root;self.catalog=catalog
        self.excluded=set(excluded)
        self.python={};self.cfunc={};self.descriptor={};self.status={};self.keepalive=[]
        self.code=set();self.names={};self.seen=set();self.namespaces={};self.versions=()
        self._lock=threading.RLock();self._refreshing=False
        self.refresh()

    def refresh(self,force=False):
        # Build privately; a failed refresh cannot publish half a registry.
        with self._lock:
            if self._refreshing:return
            if not force and self.versions and self.module_count==len(sys.modules) and not namespaces_changed(self.namespace_objects,self.versions):return
            self._refreshing=True
            try:
                stage=object.__new__(Registry);stage.__dict__.update(self.__dict__)
                for name in ('python','cfunc'):
                    setattr(stage,name,{key:set(value) for key,value in getattr(self,name).items()})
                stage.descriptor={key:list(value) for key,value in self.descriptor.items()}
                for name in ('status','names','namespaces'):setattr(stage,name,dict(getattr(self,name)))
                stage.code=set(self.code);stage.seen=set(self.seen);stage.keepalive=list(self.keepalive)
                stage._refresh(force)
                self.__dict__.update(stage.__dict__)
            finally:self._refreshing=False

    def _refresh(self,force=False):
        versions=tuple(namespace_version(n) for n in self.namespaces.values())
        if not force and self.versions and versions==self.versions and self.module_count==len(sys.modules):return
        observed={id(n):version for n,version in zip(self.namespaces.values(),versions)}
        module_count=len(sys.modules)
        # Historical callable objects remain known for aliases captured before rebinding.
        for row in self.catalog['apis']:
            aid=row['api_id'];name=row['qualified_name'];parts=name.split('.')
            self.names[aid]=name
            if aid in self.excluded or name in self.excluded:
                self.status[aid]='excluded_by_request';continue
            if row.get('language')!='python':
                self.status[aid]='native_symbol_not_instrumented';continue
            if row['kind'] not in ('function','method','class'):
                self.status[aid]='non_callable_entry';continue
            if parts[0]!='torch':
                self.status[aid]='outside_namespace';continue
            value=self.root;owner=None
            try:
                for part in parts[1:]:
                    # isinstance may read a proxy's __class__ and run user code.
                    # These built-in base types have no custom instance/subclass hooks.
                    if issubclass(type(value),(types.ModuleType,type)):
                        self.namespaces[id(value)]=value
                        observed.setdefault(id(value),namespace_version(value))
                    owner=value;value=inspect.getattr_static(value,part)
                if issubclass(type(value),classmethod):value=classmethod.__dict__['__func__'].__get__(value)
                elif issubclass(type(value),staticmethod):value=staticmethod.__dict__['__func__'].__get__(value)
            except (AttributeError,TypeError):
                self.status[aid]='unresolved_binding';continue
            if issubclass(type(value),type):
                # __init__ can be called manually; it is not proof of a class call.
                self.status[aid]='class_call_requires_additional_instrumentation';continue
            if type(value) is types.FunctionType:
                self.python.setdefault(value,set()).add(aid);self.code.add(value.__code__)
            elif type(value) is types.BuiltinFunctionType:
                self.cfunc.setdefault(id(value),set()).add(aid)
            elif method_definition(value) is not None:
                key=method_definition(value)
                entries=self.descriptor.setdefault(key,[])
                pair=(aid,owner if issubclass(type(owner),type) else None)
                if pair not in entries:entries.append(pair)
            else:
                self.status[aid]='call_protocol_not_observable';continue
            if id(value) not in self.seen:
                self.keepalive.append(value);self.seen.add(id(value))
            self.status[aid]='resolved_exact_callable'
        self.module_count=module_count
        self.namespace_objects=tuple(self.namespaces.values())
        # If user code rebinds during traversal, retain pre-read versions so
        # the next event retries instead of treating a stale map as current.
        self.versions=tuple(observed[id(n)] for n in self.namespace_objects)

    def match(self,frame,event,arg):
        with self._lock:return self._match(frame,event,arg)

    def _match(self,frame,event,arg):
        if event=='call':
            return tuple(sorted(self.python.get(frame_function(frame),())))
        if event!='c_call':return ()
        direct=self.cfunc.get(id(arg))
        if direct:return tuple(sorted(direct))
        key=method_definition(arg)
        entries=self.descriptor.get(key)
        if not entries:return ()
        receiver=getattr(arg,'__self__',None)
        # Native descriptor receivers require actual inheritance, not a proxy's
        # advertised __class__ or arbitrary metaclass __instancecheck__ behavior.
        ancestry=type.__getattribute__(type(receiver),'__mro__')
        class_ancestry=type.__getattribute__(receiver,'__mro__') if issubclass(type(receiver),type) else ()
        return tuple(sorted(aid for aid,owner in entries if owner is None or
            any(base is owner for base in ancestry) or any(base is owner for base in class_ancestry)))


class Collector:
    def __init__(self,registry,on_relation,on_gap):
        self.registry=registry;self.on_relation=on_relation;self.on_gap=on_gap
        self.pid=os.getpid();self.local=threading.local();self.enabled=False
        self.raw_set=sys.setprofile;self.original_get=sys.getprofile;self.raw_get=current_profile
        self.raw_trace=sys.settrace;self.original_gettrace=sys.gettrace
        torch=sys.modules.get('torch')
        frame_api=getattr(getattr(getattr(torch,'_C',None),'_dynamo',None),'eval_frame',None)
        self.evaluation_guard=getattr(frame_api,'set_eval_frame',None)
        self.old_start=_thread.start_new_thread;self.old_thread_start=threading._start_new_thread
        self.old_submit=concurrent.futures.ThreadPoolExecutor.submit
        self.dispatch=self._dispatch

    def gap(self,reason):
        owner=OWNER.get()
        if owner is not None and os.getpid()==self.pid:self.on_gap(owner,reason)

    def _audit(self,event,args):
        if not self.enabled or os.getpid()!=self.pid:return
        if event=='sys.setprofile' and not getattr(self.local,'internal',False):
            if not guard_available():self.gap('profile_bypassed_multiplexer')
        elif event=='sys.settrace' and not getattr(self.local,'internal',False):
            self.gap('trace_guard_bypassed')
        elif event in ('subprocess.Popen','os.fork','os.posix_spawn','os.exec','os.system'):
            self.gap('child_process_not_instrumented:'+event)

    def _set_raw(self,callback):
        previous=getattr(self.local,'internal',False);self.local.internal=True
        try:
            if callback==self.dispatch:install_guard(callback,self.evaluation_guard)
            else:
                remove_guard()
                if callback is None or callable(callback):self.raw_set(callback)
        finally:self.local.internal=previous

    def _dispatch(self,frame,event,arg):
        if os.getpid()!=self.pid:return
        if not self.enabled:
            return
        owner=OWNER.get()
        if owner is not None and event in ('call','c_call'):
            try:
                if len(sys.modules)!=self.registry.module_count or namespaces_changed(self.registry.namespace_objects,self.registry.versions):
                    self.registry.refresh()
                ids=self.registry.match(frame,event,arg)
                if ids:
                    caller=frame.f_back if event=='call' else frame
                    site=(caller.f_code.co_filename,caller.f_lineno) if caller else ('unknown',0)
                    self.on_relation(owner,ids,site)
            except Exception as exc:
                # Observer errors must not fail the workload or silently disable profiling.
                self.gap('collector_exception:'+type(exc).__name__+':'+str(exc)[:500])
    def setprofile(self,callback):
        if callback is not None and not callable(callback):raise TypeError('profile callback must be callable')
        self._change_client(self.raw_set,callback)

    def getprofile(self):return profile_client()

    def settrace(self,callback):self._change_client(self.raw_trace,callback)

    def gettrace(self):return trace_client()

    def _change_client(self,setter,callback):
        previous=getattr(self.local,'internal',False);self.local.internal=True
        try:
            remove_guard()
            try:setter(callback)
            finally:install_guard(self.dispatch,self.evaluation_guard)
        finally:self.local.internal=previous

    def _run_context(self,context,fn,args,kwargs):
        previous=self.raw_get()
        self._set_raw(self.dispatch)
        try:return context.run(fn,*args,**kwargs)
        finally:
            if previous!=self.dispatch:self._set_raw(previous)

    def start_thread(self,fn,args,kwargs=None):
        context=contextvars.copy_context()
        return self.old_start(self._run_context,(context,fn,args,kwargs or {}))

    def install(self):
        if self.enabled:return
        self.enabled=True
        sys.addaudithook(self._audit)
        sys.setprofile=self.setprofile;sys.getprofile=self.getprofile
        sys.settrace=self.settrace;sys.gettrace=self.gettrace
        _thread.start_new_thread=self.start_thread;threading._start_new_thread=self.start_thread
        collector=self
        def submit(pool,fn,/,*args,**kwargs):
            # Submission owns a task even when the worker existed before this test.
            context=contextvars.copy_context()
            return collector.old_submit(pool,collector._run_context,context,fn,args,kwargs)
        concurrent.futures.ThreadPoolExecutor.submit=submit
        self.submit_wrapper=submit
        self._set_raw(self.dispatch)

    def enter(self,owner):
        token=OWNER.set(owner)
        try:self.registry.refresh()
        except Exception as exc:self.gap('collector_exception:'+type(exc).__name__+':'+str(exc)[:500])
        if self.raw_get()!=self.dispatch or not guard_intact():
            self.gap('profile_hook_lost_before_phase');self._set_raw(self.dispatch)
        if sys.setprofile!=self.setprofile or sys.getprofile!=self.getprofile:
            self.gap('profile_interface_replaced')
        return token

    def leave(self,token):
        if self.raw_get()!=self.dispatch or not guard_intact():self.gap('profile_hook_lost_during_phase')
        OWNER.reset(token)

    def close(self):
        self.enabled=False
        previous=getattr(self.local,'internal',False);self.local.internal=True
        try:remove_guard()
        finally:self.local.internal=previous
        if sys.setprofile==self.setprofile:sys.setprofile=self.raw_set
        if sys.getprofile==self.getprofile:sys.getprofile=self.original_get
        if sys.settrace==self.settrace:sys.settrace=self.raw_trace
        if sys.gettrace==self.gettrace:sys.gettrace=self.original_gettrace
        if _thread.start_new_thread==self.start_thread:_thread.start_new_thread=self.old_start
        if threading._start_new_thread==self.start_thread:threading._start_new_thread=self.old_thread_start
        if concurrent.futures.ThreadPoolExecutor.submit==self.submit_wrapper:
            concurrent.futures.ThreadPoolExecutor.submit=self.old_submit
