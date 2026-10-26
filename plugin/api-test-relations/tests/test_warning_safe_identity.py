import json,warnings
from pathlib import Path
from types import ModuleType
import pytest
from relation_collector import Registry
from test_relation_collector import catalog,collecting


def test_hostile_class_attribute_does_not_interrupt_api_capture():
    touched=[]
    class Proxy:
        def __getattribute__(self,name):
            touched.append(name)
            warnings.warn('proxy access must not run in registry',FutureWarning)
            return object.__getattribute__(self,name)
    def public():pass
    root=ModuleType('torch');root.opaque=Proxy();root.public=public
    cat=catalog(('opaque','function'),('public','function'))
    with warnings.catch_warnings():
        warnings.simplefilter('error',FutureWarning)
        with collecting(root,cat) as (collector,hits,gaps):
            owner=collector.enter(('test_safe[param]','call'))
            root.opaque=Proxy()  # Force namespace refresh during the active phase.
            public();collector.leave(owner)
        # The observer must not globally hide warnings emitted by user code.
        with pytest.raises(FutureWarning):warnings.warn('workload warning',FutureWarning)
    assert touched==[]
    assert hits==[(('test_safe[param]','call'),('python:torch.public',))]
    assert gaps==[]


def test_class_and_static_methods_still_resolve():
    class Methods:
        @classmethod
        def class_call(cls):pass
        @staticmethod
        def static_call():pass
    root=ModuleType('torch');root.Methods=Methods
    with collecting(root,catalog(('Methods.class_call','method'),('Methods.static_call','method'))) as (c,hits,gaps):
        token=c.enter(('methods','call'));Methods.class_call();Methods.static_call();c.leave(token)
    assert [ids for _,ids in hits]==[('python:torch.Methods.class_call',),('python:torch.Methods.static_call',)]
    assert not gaps


@pytest.mark.integration
def test_actual_frozen_ledger_refresh_with_warnings_as_errors():
    import torch
    file=Path(__file__).resolve().parents[1]/'src/api_test_relations/catalogs/pytorch-api-ledger.json'
    if not file.exists():pytest.skip('frozen integration ledger not present')
    with warnings.catch_warnings():
        warnings.simplefilter('error',FutureWarning)
        registry=Registry(json.loads(file.read_text()),torch,excluded=['torch.compile'])
        registry.refresh(force=True)
    assert len(registry.status)==10362
    assert any(registry.names[aid]=='torch.add' for ids in registry.cfunc.values() for aid in ids)
