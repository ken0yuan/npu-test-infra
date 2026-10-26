"""Exercise supervisor ownership and start/stop without upstream or package installation."""
import shutil
import subprocess
import sys
import json
import os
import signal
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != 'linux', reason='WSL/Linux supervisor uses /proc and flock')
def test_supervisor_start_status_duplicate_stop(tmp_path):
    source = Path(__file__).resolve().parents[1]/'services/service.py'
    service = tmp_path/'run-test/services/service.py'
    service.parent.mkdir(parents=True)
    shutil.copy2(source, service)
    shutil.copy2(source.with_name('watchdog.sh'),service.with_name('watchdog.sh'))
    service.with_name('bug_monitor.py').write_text('import time\nwhile True: time.sleep(1)\n')
    service.with_name('poll-brief-loop.sh').write_text('#!/usr/bin/env bash\nwhile true; do sleep 1; done\n')
    (tmp_path/'run.sh').write_text('#!/usr/bin/env bash\nwhile true; do sleep 1; done\n')
    def call(command):
        return subprocess.run([sys.executable,str(service),command],capture_output=True,text=True,timeout=45)
    try:
        started = call('start')
        assert started.returncode == 0, started.stdout+started.stderr
        assert 'running pid=' in call('status').stdout
        assert 'already running' in call('start').stdout
        workers=tmp_path/'data/service/workers.json'
        info=json.loads(workers.read_text())
        assert set(info)=={'cycle','pipeline','bug_monitor','brief'}
        old=info['brief']['pid']
        os.kill(old,signal.SIGKILL)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            current=json.loads(workers.read_text()).get('brief',{})
            if current.get('pid') and current['pid']!=old:break
            time.sleep(0.2)
        assert current.get('pid') and current['pid']!=old
        assert current['restarts']==1
    finally:
        stopped = call('stop')
    assert stopped.returncode == 0
    assert call('status').stdout.strip() == 'stopped'
    assert not (tmp_path/'data/service/service.json').exists()
