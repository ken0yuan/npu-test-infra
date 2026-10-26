import concurrent.futures
from contextlib import contextmanager
import sys
import threading
from types import SimpleNamespace,FunctionType,ModuleType
import pytest
from relation_collector import Collector,Registry


def catalog(*items):
    return {'apis':[{'api_id':'python:torch.'+name,'qualified_name':'torch.'+name,
                     'language':'python','kind':kind} for name,kind in items]}


@contextmanager
def collecting(root,cat):
    hits=[];gaps=[]
    c=Collector(Registry(cat,root),lambda owner,ids,site:hits.append((owner,ids)),lambda owner,reason:gaps.append((owner,reason)))
    c.install()
    try:yield c,hits,gaps
    finally:c.close()


def test_different_function_with_same_bytecode_is_not_an_api():
    def public(x=1):return x
    foreign=FunctionType(public.__code__,public.__globals__,'foreign',(2,))
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('test_a','call'))
        foreign();public();c.leave(token)
    assert hits==[(('test_a','call'),('python:torch.public',))]
    assert gaps==[]


def test_profile_disable_restore_preserves_api_observation_and_client_semantics():
    def public():pass
    events=[]
    def client(frame,event,arg):
        if frame.f_code is public.__code__ and event=='call':events.append('call')
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('test_a','call'))
        sys.setprofile(client);assert sys.getprofile() is client
        public()
        old=sys.getprofile();sys.setprofile(None);assert sys.getprofile() is None
        public();sys.setprofile(old);public();sys.setprofile(None)
        c.leave(token)
    assert len(hits)==3 and events==['call','call'] and gaps==[]


def test_thread_keeps_originating_test_when_next_test_has_started():
    def public():pass
    ready=threading.Event();release=threading.Event()
    def worker():
        ready.set();release.wait();public()
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        first=c.enter(('test_first[param]','call'))
        thread=threading.Thread(target=worker);thread.start();ready.wait();c.leave(first)
        second=c.enter(('test_second[param]','call'))
        release.set();thread.join();c.leave(second)
    assert hits==[(('test_first[param]','call'),('python:torch.public',))]
    assert gaps==[]


def test_preexisting_pool_uses_submission_owner_not_worker_creation_owner():
    def public():pass
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        pool.submit(lambda:None).result()
        with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
            for test in ('test_a','test_b'):
                token=c.enter((test,'call'));pool.submit(public).result();c.leave(token)
    assert hits==[(('test_a','call'),('python:torch.public',)),(('test_b','call'),('python:torch.public',))]
    assert gaps==[]


def test_saved_raw_setprofile_cannot_hide_calls_between_opcodes():
    original=sys.setprofile
    def public():pass
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('test_a','call'));saved=c.raw_get()
        original(None);public();original(saved);public();c.leave(token)
    assert hits==[(('test_a','call'),('python:torch.public',))]*2
    assert not gaps


def test_manual_init_does_not_invent_class_api_call():
    class A:
        def __init__(self):pass
    with collecting(SimpleNamespace(A=A),catalog(('A','class'))) as (c,hits,gaps):
        token=c.enter(('test_a','call'));obj=object.__new__(A);A.__init__(obj);c.leave(token)
    assert hits==[]
    assert c.registry.status['python:torch.A']=='class_call_requires_additional_instrumentation'


def test_builtin_receiver_and_saved_descriptor_remain_exact_after_rebinding():
    class SpecialList(list):pass
    value=SpecialList();saved=value.append
    with collecting(SimpleNamespace(SpecialList=SpecialList),catalog(('SpecialList.append','method'))) as (c,hits,gaps):
        token=c.enter(('test_a','call'));[].append(1)
        SpecialList.append=lambda self,value:None
        saved(2);c.leave(token)
    assert hits==[(('test_a','call'),('python:torch.SpecialList.append',))]


def test_exact_aliases_and_rebinding_are_preserved():
    def public():pass
    def replacement():pass
    root=SimpleNamespace(public=public,alias=public)
    with collecting(root,catalog(('public','function'),('alias','function'))) as (c,hits,gaps):
        token=c.enter(('test_a','call'));public();c.leave(token)
        root.public=replacement
        token=c.enter(('test_b','call'));replacement();c.leave(token)
    assert hits==[(('test_a','call'),('python:torch.alias','python:torch.public')),
                  (('test_b','call'),('python:torch.public',))]


@pytest.mark.integration
def test_real_cuda_api_identity_and_thread_calls():
    import torch
    cat=catalog(('Tensor.fill_','method'),('Tensor.clone','method'),('add','function'))
    x=torch.empty(1,device='cuda');scalar=torch.tensor(4,device='cuda')
    with collecting(torch,cat) as (c,hits,gaps):
        token=c.enter(('test_cuda','call'))
        x.fill_(2);x.fill_(scalar);x.clone();torch.add(x,x)
        worker=threading.Thread(target=lambda:x.fill_(3));worker.start();worker.join()
        c.leave(token)
    assert [ids for owner,ids in hits]==[('python:torch.Tensor.fill_',)]*2+[
        ('python:torch.Tensor.clone',),('python:torch.add',),('python:torch.Tensor.fill_',)]
    assert gaps==[]


def test_rebinding_mid_phase_resolves_new_function_without_source_names():
    def first():pass
    def second():pass
    root=ModuleType('torch');root.public=first
    with collecting(root,catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('test_a','call'))
        root.public=second
        root.public()
        c.leave(token)
    assert hits==[(('test_a','call'),('python:torch.public',))]


def test_explicit_api_exclusion_preserves_catalog_and_other_bindings():
    def public():pass
    cat=catalog(('public','function'),('alias','function'))
    registry=Registry(cat,SimpleNamespace(public=public,alias=public),excluded=['torch.public'])
    assert registry.status['python:torch.public']=='excluded_by_request'
    assert registry.python[public]=={'python:torch.alias'}
    assert len(registry.status)==2
