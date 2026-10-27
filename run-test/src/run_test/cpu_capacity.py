"""CPU capacity available to this process, including Linux container limits."""
import os
from pathlib import Path, PurePosixPath


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def cpu_capacity(proc=Path('/proc')):
    detected = max(1, os.cpu_count() or 1)
    limits = {'online': detected}
    try:
        limits['affinity'] = len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        pass
    groups = {}
    for line in _read(proc / 'self/cgroup').splitlines():
        _, controllers, path = line.split(':', 2)
        for controller in controllers.split(','):
            groups[controller] = PurePosixPath(path)
    quotas = []
    for line in _read(proc / 'self/mountinfo').splitlines():
        left, sep, right = line.partition(' - ')
        fields, mount = left.split(), right.split()
        if not sep or len(fields) < 5 or len(mount) < 3:
            continue
        if mount[0] == 'cgroup2':
            group, version = groups.get(''), 2
        elif mount[0] == 'cgroup' and 'cpu' in mount[2].split(','):
            group, version = groups.get('cpu'), 1
        else:
            continue
        if group is None:
            continue
        unescape = lambda s: s.replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\')
        root, point = PurePosixPath(unescape(fields[3])), Path(unescape(fields[4]))
        try:
            relative = group.relative_to(root)
        except ValueError:
            # Cgroup namespaces may expose membership relative to a mounted root.
            if group != PurePosixPath('/'):
                continue
            relative = PurePosixPath('.')
        if '..' in relative.parts:
            continue
        current = point / str(relative)
        while True:
            try:
                if version == 2:
                    quota, period = _read(current / 'cpu.max').split()
                    quota, period = (0 if quota == 'max' else int(quota)), int(period)
                else:
                    quota = int(_read(current / 'cpu.cfs_quota_us'))
                    period = int(_read(current / 'cpu.cfs_period_us'))
                if quota > 0 and period > 0:
                    quotas.append(max(1, quota // period))
            except ValueError:
                pass
            if current == point:
                break
            current = current.parent
    if quotas:
        limits['quota'] = min(quotas)
    return {'available_cpus': max(1, min(limits.values())), 'limits': limits}


def worker_capacity(configured):
    snapshot = cpu_capacity()
    if configured == 'auto':
        workers = snapshot['available_cpus']
    else:
        if isinstance(configured, bool):
            raise ValueError('test_max_workers must be auto or a positive integer')
        workers = int(configured)
        if workers < 1:
            raise ValueError('test_max_workers must be auto or a positive integer')
        workers = min(workers, snapshot['available_cpus'])
    return {**snapshot, 'configured': configured, 'max_workers': workers}


def test_thread_env(extra=None):
    """Bound default BLAS/OpenMP parallelism while honoring explicit settings."""
    env = dict(extra or {})
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(variable, os.environ.get(variable, '1'))
    return env


def thread_budget(env):
    values = [1]
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        for value in env.get(variable, '1').split(','):
            try:
                values.append(max(1, int(value)))
            except ValueError:
                pass
    return max(values)
