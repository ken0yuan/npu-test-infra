"""Ordered NPU admission with CPU work filling unused execution slots.

Running jobs are never stopped to make room. Unknown NPU requirements take the
whole device pool; a guess of one card can silently turn multi-device tests into
skips. The executor still owns the shared Python environment for the entire PR.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
import threading
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
    phase: str = "execute"
    on_unavailable: object = None


@dataclass
class JobExpansion:
    """A collection result whose children re-enter the same admission queue."""

    jobs: object
    metadata: object = None
    results: list = field(init=False)

    def __post_init__(self):
        self.jobs = list(self.jobs)
        self.results = [None] * len(self.jobs)


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
        self._health_lock = threading.Lock()
        self._quarantined = {}

    def quarantine(self, cards, reason):
        """Exclude cards with direct failure evidence for this pool's lifetime."""
        cards = tuple(cards)
        if any(card not in self.ids for card in cards):
            raise ValueError("cannot quarantine a card outside the visible pool")
        with self._health_lock:
            for card in cards:
                self._quarantined.setdefault(card, str(reason).strip() or "NPU initialization failed")

    def unavailable_reason(self, count):
        if count < 0 or count > len(self.ids):
            raise ValueError(f"test needs {count} NPU cards; visible pool has {len(self.ids)}")
        with self._health_lock:
            failed = dict(self._quarantined)
        healthy = len(self.ids) - len(failed)
        if count <= healthy:
            return None
        details = "; ".join(f"{card}: {failed[card]}" for card in self.ids if card in failed)
        return (f"test needs {count} NPU cards; healthy pool has {healthy}/{len(self.ids)}; "
                f"quarantined cards: {details}")

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
                with self._health_lock:
                    quarantined = card in self._quarantined
                if card in self.reserved or quarantined:
                    continue
                candidate = ExitStack()
                fd = candidate.enter_context(lease_handle(self.lock_dir / f"npu-{card}.lock"))
                try:
                    free = fd is not None and self.is_free(card)
                    with self._health_lock:
                        free = free and card not in self._quarantined
                except BaseException:
                    candidate.close()
                    raise
                if free:
                    stack.enter_context(candidate)
                    chosen.append(card)
                    fds.append(fd)
                    if len(chosen) == count:
                        with self._health_lock:
                            healthy = not any(card in self._quarantined for card in chosen)
                            if healthy:
                                self.reserved.update(chosen)
                        if not healthy:
                            stack.close()
                            return None
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
    def validate(items):
        if any(job.cpu_threads < 1 for job in items):
            raise ValueError("job cpu_threads must be positive")
    validate(jobs)
    results = [None] * len(jobs)
    # A child writes into its parent's expansion, keeping both levels ordered
    # without introducing a second executor or another CPU budget.
    pending = deque((job, results, index) for index, job in enumerate(jobs))
    running = {}
    waiting_key = None
    previous_budget = None
    def complete(job, target, index, outcome):
        target[index] = outcome
        if isinstance(outcome, JobExpansion):
            validate(outcome.jobs)
            outcome.results[:] = [None] * len(outcome.jobs)
            children = [(child, outcome.results, child_index)
                        for child_index, child in enumerate(outcome.jobs)]
            event(state="expanded", key=job.key, phase=job.phase, children=len(children))
            return children
        return []
    # A bounded pool limits CPU/memory use as well as NPU occupancy.
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pytest") as executor:
        try:
            while pending or running:
                expanded = []
                for future in list(running):
                    if not future.done():
                        continue
                    job, target, index, lease = running.pop(future)
                    pool.release(lease)
                    outcome = future.result()
                    event(state="completed", key=job.key, phase=job.phase, cards=list(lease.cards))
                    expanded.extend(complete(job, target, index, outcome))
                # If several collectors finish in one poll, preserve their
                # admission order as well as each collector's child order.
                pending.extendleft(reversed(expanded))
                blocked_npu = False
                budget = max(1, int(cpu_budget())) if cpu_budget else max_workers
                if previous_budget is not None and previous_budget != budget:
                    event(state="capacity_changed", phase="scheduler", cpu_threads=budget, running=len(running))
                previous_budget = budget
                while pending and len(running) < max_workers:
                    job, target, index = pending[0]
                    used = sum(active.cpu_threads for active, _, _, _ in running.values())
                    # A test with an explicit larger thread budget may run alone.
                    # Shrinking quotas stop admissions; running tests are drained.
                    if running and used + job.cpu_threads > budget:
                        break
                    try:
                        reason = pool.unavailable_reason(job.cards) if job.cards else None
                    except ValueError as error:
                        if job.on_unavailable is None:
                            raise
                        reason = str(error)
                    if reason:
                        if job.on_unavailable is None:
                            raise ValueError(reason)
                        pending.popleft()
                        outcome = job.on_unavailable(reason)
                        event(state="unavailable", key=job.key, phase=job.phase,
                              required_cards=job.cards, reason=reason, running=len(running))
                        pending.extendleft(reversed(complete(job, target, index, outcome)))
                        waiting_key = None
                        continue
                    # CPU jobs need no device probe or card lease. Waiting for
                    # the first NPU job blocks later NPU jobs, but not CPUs.
                    lease = (None if blocked_npu else pool.acquire(job.cards)) if job.cards else CardLease(())
                    offset = 0
                    if lease is None:
                        blocked_npu = True
                        if waiting_key != job.key:
                            event(state="resource_wait", key=job.key, required_cards=job.cards,
                                  phase=job.phase, running=len(running))
                            waiting_key = job.key
                        offset = next((i for i, (candidate, _, _) in enumerate(pending)
                                       if candidate.cards == 0), None)
                        if offset is None:
                            break
                        job, target, index = pending[offset]
                        if running and used + job.cpu_threads > budget:
                            break
                        lease = CardLease(())
                    else:
                        waiting_key = None
                    try:
                        event(state="started", key=job.key, phase=job.phase, cards=list(lease.cards))
                        future = executor.submit(job.run, lease)
                    except BaseException:
                        pool.release(lease)
                        raise
                    running[future] = (job, target, index, lease)
                    del pending[offset]
                if running:
                    wait(running, timeout=interval, return_when=FIRST_COMPLETED)
                elif pending:
                    time.sleep(interval)
        finally:
            # Drain workers before releasing cards, even when another job failed.
            if running:
                wait(running)
                for _, _, _, lease in running.values():
                    pool.release(lease)
    return results
