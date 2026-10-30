"""Rebuilt local monitoring from documented responsibilities; never changes task results."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .config import load_config, store_paths


def read_rows(path, table):
    if not path.is_file():
        raise FileNotFoundError(f'数据库未就绪: {path}')
    conn = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(f'SELECT * FROM {table}')]
    finally:
        conn.close()


def snapshot(config):
    paths = store_paths(config)
    tasks = read_rows(paths.queue_db, 'tasks')
    records = read_rows(paths.pr_db, 'pr_results')
    upstream = read_rows(paths.queue_db, 'upstream_commits')
    from .upstream_health import read_health, health_warning
    health = read_health(config)
    mirror = read_rows(paths.queue_db, 'mirror_state')
    warning = health_warning(config, health, mirror[0]['last_fetch_at'] if mirror else None)
    return {'time':datetime.now().astimezone().isoformat(), 'tasks':tasks, 'results':records,
            'upstream_health':health, 'upstream_warning':warning,
            'task_states':dict(Counter(t['state'] for t in tasks)),
            'upstream_states':dict(Counter(r['enqueue_state'] for r in upstream))}


def issues_for(snap, resolutions=None):
    results = {r['pr_number']:r for r in snap['results']}
    issues = []
    for task in snap['tasks']:
        if task['state'] not in ('done','failed','blocked'):
            continue
        reasons = []
        record = results.get(task['pr_number'])
        if task['state'] in ('failed','blocked'):
            if resolutions is None or not resolutions.task(task):
                reasons.append({'kind':task['state'], 'detail':task.get('error_reason')})
        elif record is None:
            reasons.append({'kind':'done_without_result'})
        elif record['pytorch_commit'] == task['pytorch_commit']:
            historical_record=record
            if resolutions is not None:
                from .issue_resolution import active_counts
                record = {**record,**active_counts(task,record,resolutions)}
            # PR results overwrite older versions; a different commit is not evidence of corruption.
            if record.get('no_result_count',0):
                reasons.append({'kind':'no_result', 'count':record['no_result_count']})
            if record['fail_count'] or record['error_count']:
                reasons.append({'kind':'test_failure','fail':record['fail_count'],'error':record['error_count']})
            tests = json.loads(record['tests']) if isinstance(record['tests'],str) else record['tests']
            for test in tests:
                counts = test.get('counts') or {}
                if counts.get('skip',0) and not any(counts.get(k,0) for k in ('pass','fail','error')):
                    reasons.append({'kind':'function_all_skipped','test_function':test.get('test_function'),'device':test.get('device')})
            if not tests or not any(historical_record.get(k,0) for k in ('pass_count','fail_count','error_count','skip_count','no_result_count')):
                reasons.append({'kind':'empty_result'})
        if reasons:
            issues.append({'task_id':task['task_id'],'pr_number':task['pr_number'],
                'pytorch_commit':task['pytorch_commit'],'task_updated_at':task['updated_at'],
                'reasons':reasons,'task_state':task['state']})
    return issues


def monitor_once(config, output):
    snap = snapshot(config)
    output.mkdir(parents=True, exist_ok=True)
    from .issue_resolution import load_index
    issues = issues_for(snap, load_index(config))
    for issue in issues:
        body = json.dumps(issue,ensure_ascii=False,sort_keys=True)
        name = hashlib.sha256(body.encode()).hexdigest()+'.json'
        path = output/name
        if not path.exists():
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps({'observed_at':snap['time'],**issue},ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(path)
    message = {'time':snap['time'],'tasks_checked':len(snap['tasks']),'current_issues':len(issues),'archive':str(output)}
    message['upstream_health'] = snap.get('upstream_health', {})
    if snap.get('upstream_warning'): message['upstream_warning'] = snap['upstream_warning']
    if config.get('data_root'):
        from .issue_ledger import scan
        message['ledger'] = scan(config, snap)
        if (config.get('reports') or {}).get('enabled', False):
            from .result_docs import scan as scan_reports
            try:
                message['reports'] = scan_reports(config)
                if ((config.get('reports') or {}).get('feishu') or {}).get('enabled', False):
                    from .feishu_publish import scan as publish_reports
                    message['feishu'] = publish_reports(config)
            except Exception as error:
                # A report failure must not change execution or ledger results.
                message['reports'] = {'error': str(error)}
    return message


def brief_once(config):
    snap = snapshot(config)
    totals = {k:sum(r.get(k,0) for r in snap['results']) for k in ('pass_count','fail_count','error_count','skip_count','no_result_count')}
    from .issue_resolution import load_index, active_counts, matching_task
    index=load_index(config)
    current=dict.fromkeys(totals,0)
    for record in snap['results']:
        effective=active_counts(matching_task(snap['tasks'],record),record,index)
        for key in current:current[key]+=effective[key]
    task_states=dict(snap['task_states']);archived_tasks=0
    for task in snap['tasks']:
        if task['state'] in ('failed','blocked') and index.task(task):
            task_states[task['state']]-=1;archived_tasks+=1
    if archived_tasks:task_states['resolved_archived']=archived_tasks
    message = {'time':snap['time'],'tasks':task_states,'upstream':snap['upstream_states'],
               'prs':len(snap['results']),'counts':current,'note':'当前计数排除已解决归档错误；归档不是通过，原始结果保留；done 仅表示落库完成'}
    message['upstream_health'] = snap.get('upstream_health', {})
    if snap.get('upstream_warning'): message['upstream_warning'] = snap['upstream_warning']
    if current!=totals or archived_tasks:
        message.update(historical_counts=totals, historical_tasks=snap['task_states'],
                       resolved_counts={key:totals[key]-current[key] for key in totals})
    if snap['tasks']:
        last = max(t['enqueued_at'] for t in snap['tasks'])
        when = datetime.fromisoformat(last)
        if when.tzinfo is None: when=when.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc)-when).total_seconds()/3600
        if hours >= 2: message['queue_warning']=f'{hours:.1f} 小时无新任务入队，请核对 cycle 和 OBS'
    else:
        message['queue_warning']='暂无入队任务，请查看 cycle 的首次同步或失败日志'
    return message


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['bugs','brief'])
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[3]/'settings.json'))
    parser.add_argument('--loop',action='store_true')
    parser.add_argument('--interval',type=float,default=300)
    args=parser.parse_args(argv)
    if args.interval <= 0: parser.error('--interval must be positive')
    while True:
        try:
            cfg=load_config(args.config)
            message=monitor_once(cfg,store_paths(cfg).data_root/'bug-monitor') if args.mode=='bugs' else brief_once(cfg)
            print(json.dumps(message,ensure_ascii=False),flush=True)
        except Exception as error:
            print(json.dumps({'time':datetime.now().astimezone().isoformat(),'monitor_error':str(error)},ensure_ascii=False),flush=True)
            if not args.loop: return 1
        if not args.loop:return 0
        time.sleep(args.interval)


if __name__=='__main__':
    raise SystemExit(main())
