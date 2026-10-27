"""Ordered NPU admission with CPU work filling unused execution slots.

Running jobs are never stopped to make room. Unknown NPU requirements take the
whole device pool; a guess of one card can silently turn multi-device tests into
skips. The executor still owns the shared Python environment for the entire PR.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import time

from . import npu_cards
from .process_lock import lease_handle


@dataclass(frozen=True)
class TestJob:
    __test__ = False
    key: str
    cards: int
    run: object
    cpu_threads: int = 1


class CardLease:
    def __init__(self, cards, stack=None, fds=()):
        self.cards = tuple(cards)
        self.stack = stack or ExitStack()
        self.fds = tuple(fds)

    def close(self):
        self.stack.close()


class CardPool:
    def __init__(self, lock_dir: Path, ids=None, is_free=None):
        self.lock_dir = Path(lock_dir)
        self.ids = tuple(npu_cards.npu_ids() if ids is None else ids)
        self.is_free = is_free or npu_cards.card_free
        self.reserved = set()

    def acquire(self, count):
        if count < 0 or count > len(self.ids):
            raise ValueError(f"test needs {count} NPU cards; visible pool has {len(self.ids)}")
        if count == 0:
            return CardLease(())
        stack = ExitStack()
        chosen = []
        fds = []
        try:
            for card in self.ids:
                if card in self.reserved:
                    continue
                candidate = ExitStack()
                fd = candidate.enter_context(lease_handle(self.lock_dir / f"npu-{card}.lock"))
                try:
                    free = fd is not None and self.is_free(card)
                except BaseException:
                    candidate.close()
                    raise
                if free:
                    stack.enter_context(candidate)
                    chosen.append(card)
                    fds.append(fd)
                    if len(chosen) == count:
                        self.reserved.update(chosen)
                        return CardLease(chosen, stack, fds)
                else:
                    candidate.close()
            stack.close()
            return None
        except BaseException:
            stack.close()
            raise

    def release(self, lease):
        self.reserved.difference_update(lease.cards)
        lease.close()


def run_ordered(jobs, pool, *, max_workers=8, interval=1.0, event=None, cpu_budget=None):
    """Return results in input order. No timeout turns resource wait into failure."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    event = event or (lambda **data: None)
    jobs = list(jobs)
    if any(job.cpu_threads < 1 for job in jobs):
        raise ValueError("job cpu_threads must be positive")
    results = [None] * len(jobs)
    pending = deque(enumerate(jobs))
    running = {}
    waiting_key = None
    previous_budget = None
    # A bounded pool limits CPU/memory use as well as NPU occupancy.
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pytest") as executor:
        try:
            while pending or running:
                for future in list(running):
                    if not future.done():
                        continue
                    index, lease = running.pop(future)
                    pool.release(lease)
                    results[index] = future.result()
                    event(state="completed", key=jobs[index].key, cards=list(lease.cards))
                blocked_npu = False
                budget = max(1, int(cpu_budget())) if cpu_budget else max_workers
                if previous_budget is not None and previous_budget != budget:
                    event(state="capacity_changed", cpu_threads=budget, running=len(running))
                previous_budget = budget
                while pending and len(running) < max_workers:
                    index, job = pending[0]
                    used = sum(jobs[i].cpu_threads for i, _ in running.values())
                    # A test with an explicit larger thread budget may run alone.
                    # Shrinking quotas stop admissions; running tests are drained.
                    if running and used + job.cpu_threads > budget:
                        break
                    # CPU jobs need no device probe or card lease. Waiting for
                    # the first NPU job blocks later NPU jobs, but not CPUs.
                    lease = (None if blocked_npu else pool.acquire(job.cards)) if job.cards else CardLease(())
                    offset = 0
                    if lease is None:
                        blocked_npu = True
                        if waiting_key != job.key:
                            event(state="resource_wait", key=job.key, required_cards=job.cards,
                                  running=len(running))
                            waiting_key = job.key
                        offset = next((i for i, (_, candidate) in enumerate(pending)
                                       if candidate.cards == 0), None)
                        if offset is None:
                            break
                        index, job = pending[offset]
                        if running and used + job.cpu_threads > budget:
                            break
                        lease = CardLease(())
                    else:
                        waiting_key = None
                    event(state="started", key=job.key, cards=list(lease.cards))
                    try:
                        future = executor.submit(job.run, lease)
                    except BaseException:
                        pool.release(lease)
                        raise
                    running[future] = (index, lease)
                    del pending[offset]
                if running:
                    wait(running, timeout=interval, return_when=FIRST_COMPLETED)
                elif pending:
                    time.sleep(interval)
        finally:
            # Drain workers before releasing cards, even when another job failed.
            if running:
                wait(running)
                for _, lease in running.values():
                    pool.release(lease)
    return results
