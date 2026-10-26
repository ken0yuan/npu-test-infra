"""Explicit CPython entry; retain interpreter flags and restore user argv/path."""
import os,sys,runpy,types
from pathlib import Path

def main(payload):
    directory=str(Path(__file__).resolve().parent)
    sys.path.insert(0,directory)
    try:
        from relation_process import bootstrap
    finally:
        sys.path.remove(directory)
    context=payload['context']
    for key,value in context['settings'].items():os.environ[key]=value
    os.environ['API_RELATION_PARENT']=__import__('json').dumps(context)
    sys.argv=payload['argv']
    sys.orig_argv=payload['original']
    mode=payload['mode'];target=payload['target']
    if not sys.flags.safe_path:
        sys.path[0]=str(Path(target).resolve().parent) if mode=='script' else (os.getcwd() if mode=='module' else '')
    bootstrap(context)
    if mode=='module':
        runpy._run_module_as_main(target,alter_argv=True)
    elif mode=='script':
        runpy.run_path(target,run_name='__main__')
    else:
        module=types.ModuleType('__main__');module.__builtins__=__builtins__
        if mode=='stdin':module.__file__='<stdin>'
        sys.modules['__main__']=module
        source=sys.stdin.read() if mode=='stdin' else target
        exec(compile(source,'<stdin>' if mode=='stdin' else '<string>','exec'),module.__dict__)

if 'payload' in globals():main(payload)
