"""OS-owned executor lease; released on process exit, without stale PID guesses."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path


@contextmanager
def lease_handle(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
        try:
            yield handle.fileno() if acquired else None
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                # On POSIX close releases the open-file-description lock. Do
                # not LOCK_UN: a pytest child may still hold an inherited fd
                # after its executor died, protecting its wheel environment.


@contextmanager
def executor_lease(path):
    with lease_handle(path) as fd:
        yield fd is not None
