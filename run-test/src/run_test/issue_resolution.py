"""Resolved-history overlay; original test results and baselines stay immutable."""
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import json
import sqlite3
from .config import store_paths


def binding(task, test=None):
    return (task['task_id'], task.get('pytorch_commit'), task.get('torch_npu_commit'),
            (test or {}).get('test_function', ''), (test or {}).get('device', ''))


class ResolutionIndex:
    def __init__(self, rows=()):
        self.events = defaultdict(list)
        for row in rows:
            body = json.loads(row['payload'])
            self.events[binding(body, body)].append((row['state'], body, row.get('note') or '已有修复证据，按用户要求归档，不重跑历史用例'))

    def task(self, task):
        rows = [row for row in self.events[binding(task)] if row[1]['kind'] in ('task.failed', 'task.blocked')]
        return bool(rows) and all(row[0] == 'resolved' for row in rows)

    def test(self, task, test):
        rows = self.events[binding(task, test)]
        closed = [row for row in rows if row[0] == 'resolved']
        if not closed:
            return None
        c = test.get('counts') or {}
        failures = {row[1]['instance'] for row in closed if row[1].get('pytest_outcome') == 'failure'}
        errors = {row[1]['instance'] for row in closed if row[1].get('pytest_outcome') == 'error'}
        rf, re = min(len(failures), c.get('fail', 0)), min(len(errors), c.get('error', 0))
        # Missing artifacts / collection failures are whole function gaps. A
        # partial closure must not hide another unresolved cause on that item.
        nr = int(test.get('status') == 'no_result' and len(closed) == len(rows))
        if not (rf or re or nr):
            return None
        complete = len(closed) == len(rows) and (nr or
                    rf == c.get('fail', 0) and re == c.get('error', 0))
        return dict(state='resolved' if complete else 'partially_resolved',
                    resolved_fail=rf, resolved_error=re, resolved_no_result=nr,
                    notes=sorted({row[2] for row in closed}), historical_rerun=False)


@lru_cache(maxsize=4)
def _read(path, signature):
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'occurrences' not in tables:
            return ResolutionIndex()
        if 'occurrence_resolutions' in tables:
            rows = conn.execute('SELECT o.state,o.payload,r.note FROM occurrences o LEFT JOIN occurrence_resolutions r ON r.event_key=o.event_key').fetchall()
        else:
            rows = conn.execute('SELECT state,payload,NULL AS note FROM occurrences').fetchall()
        return ResolutionIndex([dict(row) for row in rows])


def load_index(config):
    if not config.get('data_root'):
        return ResolutionIndex()
    path = store_paths(config).data_root.resolve()/'issue-ledger/issues.db'
    if not path.is_file():
        return ResolutionIndex()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30) as conn:
        versioned=conn.execute("SELECT 1 FROM sqlite_master WHERE name='resolution_state'").fetchone()
        revision=conn.execute('SELECT revision FROM resolution_state WHERE id=1').fetchone()[0] if versioned else None
    files = [path, path.with_name(path.name+'-wal')]
    signature = tuple((p.stat().st_mtime_ns,p.stat().st_size) if p.exists() else None for p in files)
    return _read(path, (revision,signature))


def effective_test(test, resolution):
    value = deepcopy(test)
    if resolution and value.get('counts') is not None:
        for key in ('fail','error'):
            value['counts'][key] -= resolution['resolved_'+key]
    return value


def matching_task(tasks, record):
    env = record.get('torch_npu_environment') or {}
    env = json.loads(env) if isinstance(env,str) else env
    candidates = [task for task in tasks if task['state']=='done' and
                  task['pr_number']==record['pr_number'] and task['pytorch_commit']==record['pytorch_commit'] and
                  task.get('torch_npu_commit')==env.get('torch_npu_commit')]
    return max(candidates,key=lambda task:task['updated_at']) if candidates else None


def active_counts(task, record, index):
    result = {k:record.get(k,0) for k in ('pass_count','fail_count','error_count','skip_count','no_result_count')}
    env=record.get('torch_npu_environment') or {}
    env=json.loads(env) if isinstance(env,str) else env
    if task and (task['pr_number']!=record['pr_number'] or task.get('pytorch_commit')!=record['pytorch_commit'] or
                 task.get('torch_npu_commit')!=env.get('torch_npu_commit')):
        return result
    if task:
        tests = record['tests'];tests=json.loads(tests) if isinstance(tests,str) else tests
        for test in tests:
            resolution=index.test(task,test)
            if resolution:
                for key in ('fail','error','no_result'):
                    result[key+'_count'] -= resolution['resolved_'+key]
    return result


def annotate_report(model, task, index):
    if not task:
        return model
    changes = [(test,index.test(task,test)) for test in model['tests']]
    if not any(resolution for _,resolution in changes):
        return model
    model['historical_instance_counts'] = dict(model['instance_counts'])
    model['historical_metrics'] = dict(model['metrics'])
    model['resolved_test_count'] = 0
    for test,resolution in changes:
        if not resolution:
            continue
        test['resolution'] = resolution
        if resolution['state']=='resolved':
            test['historical_outcome'] = test['outcome'];test['outcome']='resolved'
            model['resolved_test_count'] += 1
        else:
            from .result_docs import outcome
            test['outcome'] = outcome(effective_test(test,resolution))
        test['comparison_text'] += '；已解决归档基于修复验收，未重跑历史用例'
        for key in ('fail','error'):
            model['instance_counts'][key] -= resolution['resolved_'+key]
    c=Counter(test['outcome'] for test in model['tests'])
    model['metrics'].update(Total=len(model['tests'])-c['resolved'],Passed=c['pass'],Skipped=c['skip'],
                            Failed=c['fail'],Error=c['error'],**{'No Result':c['no_result']})
    model['metric_note'] += ' 当前表排除已解决归档项；归档不等于测试通过，原始结果与日志保留。'
    return model
