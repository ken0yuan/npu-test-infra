"""Watchdog owns four worker loops: cycle, pipeline, bug_monitor and brief."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT/'data/service'
OPS.mkdir(parents=True, exist_ok=True)
(ROOT/'data/logs').mkdir(parents=True, exist_ok=True)
PID = OPS/'service.json'
WORKERS = OPS/'workers.json'


def specs():
    return {
        'cycle': (['bash',str(ROOT/'run.sh'),'cycle','--loop'], OPS/'cycle.log'),
        'pipeline': (['bash',str(ROOT/'run.sh'),'pipeline','--loop'], OPS/'pipeline.log'),
        'bug_monitor': ([sys.executable,str(ROOT/'run-test/services/bug_monitor.py'),'--loop'],OPS/'bug_monitor.log'),
        'brief': (['bash',str(ROOT/'run-test/services/poll-brief-loop.sh')],ROOT/'data/logs/简报.log'),
    }


def worker_status():
    try: return json.loads(WORKERS.read_text())
    except (OSError, ValueError): return {}


def start_ticks(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError): return None


def alive():
    try:
        state = json.loads(PID.read_text())
        process = Path(f"/proc/{state['pid']}")
        ticks = (process/'stat').read_text().rsplit(')', 1)[1].split()[19]
        cmd = (process/'cmdline').read_bytes()
        return state['pid'] if ticks == state['start_ticks'] and str(Path(__file__).resolve()).encode() in cmd else None
    except (OSError, ValueError, KeyError):
        return None


def run():
    lock = (OPS/'service.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 1
    pid = os.getpid()
    ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    PID.write_text(json.dumps({'pid':pid,'start_ticks':ticks}))
    children = {}
    restarts = {}
    next_start = {}
    stopping = False
    def stop(*args):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            for name,(command,path) in specs().items():
                child = children.get(name)
                if child is not None and child.poll() is not None:
                    # Retire remaining descendants from this worker's process group.
                    try: os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    print(f'{name} exited {child.returncode}; restart in 5s',flush=True)
                    children.pop(name)
                    next_start[name]=time.monotonic()+5
                if name not in children and time.monotonic() >= next_start.get(name,0):
                    with path.open('a') as log:
                        children[name]=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    restarts[name]=restarts.get(name,-1)+1
                    print(f'{name} started pid={children[name].pid}',flush=True)
            state={name:{'pid':c.pid,'returncode':c.poll(),'restarts':restarts[name],
                         'start_ticks':start_ticks(c.pid) if c.poll() is None else None}
                   for name,c in children.items()}
            temp=WORKERS.with_suffix('.tmp');temp.write_text(json.dumps(state));temp.replace(WORKERS)
            time.sleep(1)
        return 0
    finally:
        # Each subprocess group belongs to this supervisor; no global pkill.
        for c in children.values():
            try: os.killpg(c.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        for c in children.values():
            try: c.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try: os.killpg(c.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                c.wait()
        PID.unlink(missing_ok=True)
        WORKERS.unlink(missing_ok=True)


def main():
    command = sys.argv[1] if len(sys.argv)>1 else 'status'
    if command == '_run':
        return run()
    pid = alive()
    if command == 'status':
        print(f'watchdog running pid={pid}' if pid else 'stopped')
        if pid:
            workers=worker_status()
            for name in specs():
                info=workers.get(name,{})
                try:
                    ticks=Path(f"/proc/{info['pid']}/stat").read_text().rsplit(')',1)[1].split()[19]
                    valid=ticks==info.get('start_ticks') and info.get('returncode') is None
                except (OSError,KeyError): valid=False
                print(f"{name}: {'running' if valid else 'restarting/missing'} pid={info.get('pid')} restarts={info.get('restarts',0)}")
        return 0
    if command == 'stop':
        if pid:
            os.kill(pid,signal.SIGTERM)
            for _ in range(40):
                if not alive(): break
                time.sleep(1)
            if alive():
                print('仍在停止中，请查看 data/service/*.log'); return 1
        print('stopped'); return 0
    if command == 'start':
        if pid:
            print(f'already running pid={pid}'); return 0
        with (OPS/'supervisor.log').open('a') as log:
            child = subprocess.Popen(['bash',str(ROOT/'run-test/services/watchdog.sh')],cwd=ROOT,
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(1)
        if not alive():
            print('启动失败，查看 data/service/supervisor.log'); return 1
        print(f'started pid={child.pid}; logs: {OPS}'); return 0
    print('usage: start.sh [start|stop|status]'); return 2


if __name__ == '__main__':
    raise SystemExit(main())
