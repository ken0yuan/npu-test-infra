"""Bounded asynchronous selection; executor never joins a selection process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from .config import store_paths
from .process_lock import executor_lease
from .task_queue import QueueStore, STATE_PENDING_ANALYSIS, STATE_PENDING_TEST
from .task_staging import atomic_json, task_root, remove_staging

_POOLS = {}
_GUARD = threading.Lock()


def selection_command(config_path, task_id):
    return [sys.executable, '-m', 'run_test.pipeline', 'select-one',
            '--task', task_id, '--config', str(config_path)]


class SelectionPool:
    def __init__(self, config):
        self.config = config
        self.paths = store_paths(config)
        self.store = QueueStore(config)
        settings = config.get('selector') or {}
        self.capacity = max(1, int(settings.get('max_in_flight', 2)))
        self.prefetch_limit = max(self.capacity, int(settings.get('prefetch_tasks', 2)))
        self.max_failures = max(1, int(settings.get('max_failures', 3)))
        self.retry_seconds = max(1., float(settings.get('retry_seconds', 30)))
        self.running = {}
        self.events = queue.SimpleQueue()
        self.stop = threading.Event()
        self.started = threading.Event()
        self.config_path = self.paths.data_root / 'selection' / 'effective-config.json'
        # Effective values, not a possibly unrelated cwd/settings.json.
        atomic_json(self.config_path, config)
        self.last_prefilter = {}
        self.next_prefilter = 0
        self.thread = threading.Thread(target=self._run, name='selection-coordinator', daemon=True)
        self.thread.start()

    def _finish(self, task, rc):
        path = task_root(self.config, task) / 'selection-state.json'
        try:
            previous = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            previous = {}
        current = self.store.get_task(task.task_id)
        if rc == 0 and current and current.state == STATE_PENDING_ANALYSIS:
            rc = 2  # An empty successful process is not a submitted test plan.
        if rc == 3:
            atomic_json(path, {**previous, 'retry_at': time.time() + 1, 'last_exit': rc})
        elif rc == 0:
            atomic_json(path, {'failures': 0, 'last_exit': 0, 'finished_at': time.time()})
            self.events.put(('selected', task.task_id))
        elif rc != 3:
            failures = previous.get('failures', 0) + 1
            message = f'{task.task_id}: selection exited {rc}; see select-one.log'
            atomic_json(path, {'failures': failures, 'last_exit': rc,
                               'retry_at': time.time() + self.retry_seconds,
                               'finished_at': time.time()})
            if failures >= self.max_failures:
                current = self.store.get_task(task.task_id)
                if current and current.state == STATE_PENDING_ANALYSIS:
                    self.store.mark_blocked(task.task_id, message)
            # Failed analysis has no submitted result evidence in source-staging.
            # Logs and plan remain. Do not accumulate packages from failed candidates.
            remove_staging(self.config, task)
            self.events.put(('select_errors', message))

    def tick(self):
        for tid, (task, proc, stream) in list(self.running.items()):
            rc = proc.poll()
            if rc is not None:
                del self.running[tid]
                stream.close()
                self._finish(task, rc)
        if time.monotonic() >= self.next_prefilter:
            self.next_prefilter = time.monotonic() + 5
            from .prefilter import prefilter_pending_tasks
            try:
                result = prefilter_pending_tasks(self.config, queue=self.store)
                self.last_prefilter = result
                for error in result.get('errors', []):
                    self.events.put(('select_errors', str(error)))
            except Exception as error:
                self.events.put(('select_errors', 'prefilter: ' + str(error)))
        ready = len(self.store.list_tasks(STATE_PENDING_TEST))
        allowance = min(self.capacity - len(self.running), self.prefetch_limit - ready - len(self.running))
        for task in self.store.list_tasks(STATE_PENDING_ANALYSIS):
            if allowance <= 0 or self.stop.is_set():
                break
            if task.task_id in self.running:
                continue
            directory = task_root(self.config, task)
            try:
                state = json.loads((directory / 'selection-state.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                state = {}
            if state.get('retry_at', 0) > time.time():
                continue
            # A surviving worker from an earlier parent still owns this inode.
            with executor_lease(directory / '.selection.lock') as available:
                if not available:
                    continue
            directory.mkdir(parents=True, exist_ok=True)
            stream = (directory / 'select-one.log').open('ab', buffering=0)
            env = dict(os.environ)
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
            try:
                proc = subprocess.Popen(selection_command(self.config_path, task.task_id),
                                        stdout=stream, stderr=stream, env=env,
                                        start_new_session=True)
            except Exception:
                stream.close()
                raise
            self.running[task.task_id] = (task, proc, stream)
            allowance -= 1

    def _run(self):
        while not self.stop.is_set():
            try:
                # Serialize candidate scheduling across multiple CLI callers.
                with executor_lease(self.paths.queue_db.with_suffix('.selector.lock')) as acquired:
                    if acquired:
                        self.tick()
            except Exception as error:
                self.events.put(('select_errors', str(error)))
            finally:
                self.started.set()
            self.stop.wait(.2)

    def collect(self, report):
        report['prefilter'] = self.last_prefilter
        while True:
            try:
                key, value = self.events.get_nowait()
            except queue.Empty:
                break
            report[key].append(value)
        report['in_flight'] = list(self.running)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        # Never terminate an active selection or hardware test on caller exit.
        for _, proc, stream in self.running.values():
            stream.close()  # Child owns its own descriptor; writing can continue.


def get_pool(config):
    key = str(store_paths(config).queue_db.resolve())
    with _GUARD:
        pool = _POOLS.get(key)
        if pool is None or not pool.thread.is_alive():
            pool = _POOLS[key] = SelectionPool(config)
    return pool
