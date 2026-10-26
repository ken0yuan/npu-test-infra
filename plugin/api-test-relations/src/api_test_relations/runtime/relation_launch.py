"""Parse Python command lines without deleting isolation/site flags."""
import json,os,re,shutil,sys
from pathlib import Path

def executable_path(executable,env,cwd=None):
    value=os.fsdecode(executable)
    if '/' in value:
        return str((Path(cwd or os.getcwd())/value).resolve())
    return shutil.which(value,path=env.get('PATH',os.defpath)) or value

def prepare(argv,context,env,executable=None,shell=False,cwd=None):
    original=[os.fsdecode(x) for x in argv] if isinstance(argv,(list,tuple)) else [os.fsdecode(argv)]
    resolved=executable_path(executable or original[0],env,cwd)
    detail={'argv':original,'executable':resolved,'shell':bool(shell),'cwd':str(cwd or os.getcwd()),'kind':'external_executable'}
    if shell:
        detail['kind']='shell';return argv,detail
    looks_python=resolved==str(Path(sys.executable).resolve()) or re.fullmatch(r'python(?:\d+(?:\.\d+)*)?',Path(resolved).name)
    if not looks_python:return argv,detail
    detail['kind']='python_unwrapped'
    options=[];i=1;mode='stdin';target='';userargs=['-']
    while i<len(original):
        word=original[i]
        if word=='--':
            i+=1
            if i<len(original):mode='script';target=original[i];userargs=original[i:]
            break
        if word=='-':userargs=original[i:];break
        if word.startswith('-c') or word.startswith('-m'):
            mode='code' if word[1]=='c' else 'module'
            if len(word)>2:target=word[2:];i+=1
            elif i+1<len(original):target=original[i+1];i+=2
            else:return argv,detail
            userargs=['-c' if mode=='code' else target,*original[i:]];break
        if not word.startswith('-'):
            mode='script';target=word;userargs=original[i:];break
        if word in ('-W','-X','--check-hash-based-pycs'):
            if i+1>=len(original):return argv,detail
            options+=original[i:i+2];i+=2;continue
        if word.startswith(('-W','-X')) or re.fullmatch(r'-[bBdEIOPqRsSuvx]+',word):
            options.append(word);i+=1;continue
        return argv,detail
    # Interactive sessions and -x source transformations need a separate adapter.
    if any('x' in option for option in options if option.startswith('-') and not option.startswith(('-X','-W'))):return argv,detail
    data={'context':context,'mode':mode,'target':target,'argv':userargs,'original':original}
    entry=str(Path(__file__).with_name('relation_child_entry.py'))
    code='import runpy;runpy.run_path('+repr(entry)+',init_globals={"payload":'+repr(data)+'})'
    detail['kind']='python_explicit';detail['python_mode']=mode;detail['python_flags']=options
    return [original[0],*options,'-c',code,*userargs[1:]],detail
