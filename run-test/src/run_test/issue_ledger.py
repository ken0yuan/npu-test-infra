"""Evidence-backed U/S ledgers; immutable occurrences, explicit verified closure.

This module observes queue/results/artifacts. It never changes test outcomes,
baselines, manifests, exclusions, or installs packages while monitoring.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET

from .config import store_paths
from .triage import CATEGORY_OTHER, classify_text

MODULE_PACKAGES = {
    'hypothesis': 'hypothesis', 'parameterized': 'parameterized',
    'expecttest': 'expecttest', 'pytest': 'pytest', 'xdist': 'pytest-xdist',
    'numpy': 'numpy==1.26.4', 'scipy': 'scipy', 'pandas': 'pandas',
    'psutil': 'psutil', 'yaml': 'PyYAML', 'PIL': 'pillow',
    'requests': 'requests', 'dateutil': 'python-dateutil', 'zstandard': 'zstandard',
    'pydot': 'pydot', 'z3': 'z3-solver==4.15.4.0', 'sympy': 'sympy',
    'networkx': 'networkx', 'packaging': 'packaging', 'typing_extensions': 'typing_extensions',
    'filelock': 'filelock', 'fsspec': 'fsspec', 'jinja2': 'jinja2',
}
MISSING_MODULE = re.compile(r"(?:ModuleNotFoundError|ImportError): No module named ['\"]([^'\"]+)['\"]")
SCHEMA = """
CREATE TABLE IF NOT EXISTS families (
 id INTEGER PRIMARY KEY, identifier TEXT UNIQUE, namespace TEXT NOT NULL,
 family_key TEXT NOT NULL UNIQUE, kind TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS occurrences (
 event_key TEXT PRIMARY KEY, family_id INTEGER NOT NULL REFERENCES families(id),
 first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'open',
 payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS resolutions (
 id INTEGER PRIMARY KEY, family_id INTEGER NOT NULL REFERENCES families(id),
 verified_at TEXT NOT NULL, note TEXT NOT NULL, evidence TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS occurrence_resolutions (
 event_key TEXT PRIMARY KEY REFERENCES occurrences(event_key), resolved_at TEXT NOT NULL,
 note TEXT NOT NULL, evidence TEXT NOT NULL, fix_commit TEXT NOT NULL,
 basis TEXT NOT NULL DEFAULT 'repair_verified_no_historical_rerun');
CREATE TABLE IF NOT EXISTS resolution_state (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL);
INSERT OR IGNORE INTO resolution_state(id,revision) VALUES(1,0);
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def decoded(value, default):
    if isinstance(value, str):
        return json.loads(value)
    return value if value is not None else default


def classification(text, phase, kind, test_file='', exclusions=()):
    """Full evidence is classified before display excerpts are shortened."""
    modules = sorted(set(MISSING_MODULE.findall(text)))
    for rule in exclusions:
        if test_file and fnmatch.fnmatchcase(test_file, rule['glob']):
            return 'X', rule['id'], modules
    category = classify_text(text)
    if category != CATEGORY_OTHER:
        return 'C', category, modules
    if modules:
        return 'U', 'dependency.missing_module', modules
    if re.search(r'cannot open shared object file|GLIBC_[\d.]+.*not found|undefined symbol:', text):
        return 'U', 'dependency.native_library_or_abi', modules
    if 'compiled using NumPy 1.x' in text:
        return 'U', 'dependency.numpy_abi', modules
    if phase == 'selection':
        return 'S', kind, modules
    if kind == 'timeout':
        return 'U', 'execution.timeout', modules
    if kind == 'function_all_skipped':
        return 'U', 'execution.all_skipped', modules
    if phase == 'collection':
        return 'U', 'collection.error', modules
    if re.search(r'URLError|HTTPError|ConnectionError|Connection refused|Temporary failure in name resolution', text):
        return 'U', 'environment.network', modules
    return 'U', kind, modules


class Ledger:
    def __init__(self, config):
        self.config = config
        self.data = store_paths(config).data_root.resolve()
        self.root = self.data / 'issue-ledger'
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.root / 'evidence'
        self.evidence_dir.mkdir(exist_ok=True, mode=0o700)
        self.docs = Path((config.get('issue_ledger') or {}).get('docs_root') or self.root / 'views')
        self.db = self.root / 'issues.db'
        self.conn = sqlite3.connect(self.db, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.executescript(SCHEMA)
        self.new_events = 0

    def close(self):
        self.conn.close()

    def evidence_bytes(self, raw, source, hint=''):
        digest = hashlib.sha256(raw).hexdigest()
        target = self.evidence_dir / (digest + '.txt')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError('Immutable evidence checksum mismatch')
        else:
            with tempfile.NamedTemporaryFile(dir=self.evidence_dir, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
            os.replace(temporary, target)
        text = raw.decode('utf-8', errors='replace')
        lines = text.splitlines()
        index = next((i for i, line in enumerate(lines) if hint and hint in line), None)
        if index is None:
            index = next((i for i, line in enumerate(lines) if re.search(
                r'ModuleNotFoundError|ImportError|Error|FAIL|skipped|reason|timed out', line)), 0)
        excerpt = '\n'.join(lines[max(0, index - 1):index + 5])
        column = 1
        if len(excerpt) > 2400:
            position = excerpt.find(hint) if hint else -1
            start = max(0, position - 250)
            excerpt = excerpt[start:start + 2400]
            column = start + 1
        return {'source': str(source), 'snapshot': str(target), 'sha256': digest,
                'line_start': max(0, index - 1) + 1, 'quote': excerpt,
                'excerpt_column_start': column,
                'decode_replacements': '\ufffd' in text}, text

    def evidence_file(self, path, hint=''):
        path = Path(path).resolve()
        if not path.is_relative_to(self.data):
            raise ValueError(f'Evidence outside data_root: {path}')
        return self.evidence_bytes(path.read_bytes(), path, hint)

    def fact(self, body, source, hint=''):
        return self.evidence_bytes(json.dumps(body, ensure_ascii=False, indent=2,
                                              sort_keys=True).encode(), source, hint)

    def add(self, task, *, phase, kind, text, evidence, test_function='', device='',
            instance='', extra=None):
        if not evidence:
            raise ValueError('An occurrence must have preserved evidence')
        namespace, family, modules = classification(text, phase, kind,
            test_function.split('::')[0], (self.config.get('issue_ledger') or {}).get('file_exclusions', []))
        payload = {'task_id': task['task_id'], 'pr_number': task.get('pr_number'),
            'pytorch_commit': task.get('pytorch_commit'), 'torch_npu_commit': task.get('torch_npu_commit'),
            'phase': phase, 'kind': kind, 'namespace': namespace, 'family': family,
            'test_function': test_function, 'device': device, 'instance': instance,
            'missing_modules': modules,
            'package_suggestions': {m: MODULE_PACKAGES.get(m.split('.')[0]) for m in modules},
            'assessment': 'known_signature_needs_review' if namespace == 'C' else
                          'configured_file_exclusion' if namespace == 'X' else 'needs_review',
            'evidence': evidence, **(extra or {})}
        identity = {k: payload[k] for k in ('task_id', 'pytorch_commit', 'torch_npu_commit',
            'phase', 'kind', 'test_function', 'device', 'instance', 'family')}
        identity['evidence'] = sorted({e['sha256'] for e in evidence})
        event_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        timestamp = now()
        self.conn.execute('BEGIN IMMEDIATE')
        try:
            self.conn.execute('INSERT OR IGNORE INTO families(namespace,family_key,kind) VALUES(?,?,?)',
                              (namespace, namespace + ':' + family, family))
            row = self.conn.execute('SELECT * FROM families WHERE family_key=?',
                                    (namespace + ':' + family,)).fetchone()
            identifier = row['identifier'] or (family if namespace in ('C', 'X') else
                                               f'{namespace}-new-{row["id"]:06d}')
            self.conn.execute('UPDATE families SET identifier=? WHERE id=?', (identifier, row['id']))
            created = self.conn.execute('INSERT OR IGNORE INTO occurrences '
                '(event_key,family_id,first_seen,last_seen,payload) VALUES(?,?,?,?,?)',
                (event_key, row['id'], timestamp, timestamp, json.dumps(payload, ensure_ascii=False))).rowcount
            self.conn.execute('UPDATE occurrences SET last_seen=? WHERE event_key=?', (timestamp, event_key))
            if created:
                self.conn.execute('UPDATE resolution_state SET revision=revision+1 WHERE id=1')
            self.conn.commit()
            self.new_events += created
        except Exception:
            self.conn.rollback()
            raise

    def selection(self, task):
        plan = self.data / 'selection' / task['task_id'] / 'test-plan.json'
        if not plan.is_file():
            return
        evidence, text = self.evidence_file(plan)
        try:
            report = json.loads(text)
        except ValueError:
            self.add(task, phase='selection', kind='selection.invalid_plan_json',
                     text=text, evidence=[evidence])
            return
        # Deferred by the normal budget is informational, never automatically a bug.
        for field in ('analysis_gaps', 'unresolved_tests', 'missing_coverage', 'manifest_skipped'):
            for index, gap in enumerate(report.get(field) or []):
                self.add(task, phase='selection', kind='selection.' + field,
                    text=json.dumps(gap, ensure_ascii=False), evidence=[evidence],
                    instance=f'{field}/{index}', extra={'detail': gap, 'source_field': field})

    def test(self, task, test, record_evidence):
        tf, dev = test.get('test_function', ''), test.get('device', '')
        counts = test.get('counts') or {}
        artifacts = test.get('artifact_refs') or ([test['artifact_ref']] if test.get('artifact_ref') else [])
        bound = []
        for ref in artifacts:
            path = Path(ref).resolve()
            if not path.is_relative_to(self.data / 'diagnostics' / task['task_id']):
                continue  # Never attribute another run's evidence to this task.
            bound.append(path)
        if artifacts and not bound:
            self.add(task, phase='record', kind='evidence.binding_mismatch', text='Artifact task binding mismatch',
                     evidence=[record_evidence], test_function=tf, device=dev)
            return
        found = False
        for directory in bound:
            junit = directory / 'junit.xml'
            if junit.is_file():
                ev, raw = self.evidence_file(junit)
                try:
                    tree = ET.fromstring(raw)
                except ET.ParseError:
                    self.add(task, phase='record', kind='evidence.invalid_junit', text=raw,
                             evidence=[ev], test_function=tf, device=dev)
                else:
                    for case in tree.iter('testcase'):
                        for element in case:
                            if element.tag not in ('error', 'failure', 'skipped'):
                                continue
                            failure = (element.get('message', '') + '\n' + (element.text or '')).strip()
                            evidence, _ = self.evidence_file(junit, element.get('message', ''))
                            self.add(task, phase='execution',
                                kind=('function_all_skipped' if not any(counts.get(k, 0) for k in ('pass', 'fail', 'error'))
                                      else 'test.skipped') if element.tag == 'skipped' else 'test.' + element.tag,
                                text=failure, evidence=[evidence], test_function=tf, device=dev,
                                instance=case.get('classname', '') + '::' + case.get('name', ''),
                                extra={'pytest_outcome': element.tag, 'detail': failure})
                            found = True
            status_file = directory / 'result.json'
            try:
                status = json.loads(status_file.read_text()) if status_file.is_file() else {}
            except ValueError:
                ev, text = self.evidence_file(status_file)
                self.add(task, phase='record', kind='evidence.invalid_result_json', text=text,
                         evidence=[ev], test_function=tf, device=dev)
                found = True
                continue
            if status.get('execution_status') in (None, 'completed'):
                continue
            evidence, texts = [], []
            for name in ('result.json', 'collection.json', 'stderr.log', 'stdout.log'):
                p = directory / name
                if p.is_file():
                    ev, text = self.evidence_file(p)
                    evidence.append(ev); texts.append(text)
            kind = status['execution_status']
            self.add(task, phase='collection' if kind == 'collection_error' else 'execution',
                kind=kind, text='\n'.join(texts), evidence=evidence,
                test_function=tf, device=dev, extra={'execution_status': kind})
            found = True
        suspect = (test.get('status') == 'no_result' or counts.get('fail', 0) or counts.get('error', 0) or
                   (counts.get('skip', 0) and not any(counts.get(k, 0) for k in ('pass', 'fail', 'error'))))
        if suspect and not found:
            text = test.get('reason') or 'Result flags an issue, but raw testcase evidence is missing'
            self.add(task, phase='record', kind='evidence.missing_test_artifacts', text=text,
                evidence=[record_evidence], test_function=tf, device=dev,
                extra={'detail': text, 'source_status': test.get('status'), 'counts': counts})

    def observe(self, snap):
        records = {r['pr_number']: r for r in snap['results']}
        for task in snap['tasks']:
            self.selection(task)  # Selection failures count even while a task is pending.
            if task['state'] in ('blocked', 'failed'):
                reason = task.get('error_reason') or ''
                ev, _ = self.fact(task, f'queue.db/tasks/{task["task_id"]}', reason)
                text = reason
                evidence = [ev]
                diagnostics = self.data / 'diagnostics' / task['task_id']
                logs = sorted(diagnostics.glob('*.log')) if diagnostics.is_dir() else []
                # Previous retry logs and task.history are context, not the current error.
                # Only expand classification with a diagnostic quoting this current reason.
                for path in sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True):
                    item, body = self.evidence_file(path)
                    if any(len(line.strip()) >= 20 and line.strip() in body for line in reason.splitlines()):
                        evidence.append(item); text += '\n' + body
                        break
                self.add(task, phase='preparation', kind='task.' + task['state'],
                         text=reason + '\n' + text, evidence=evidence, extra={'detail': reason})
            if task['state'] != 'done':
                continue
            record = records.get(task.get('pr_number'))
            if record is None:
                ev, text = self.fact(task, f'queue.db/tasks/{task["task_id"]}')
                self.add(task, phase='record', kind='record.done_without_result', text=text, evidence=[ev])
                continue
            if record['pytorch_commit'] != task.get('pytorch_commit'):
                continue  # The PR table retains its latest result, not every task.
            environment = decoded(record.get('torch_npu_environment'), {})
            if (task.get('torch_npu_commit') and environment.get('torch_npu_commit') and
                    task['torch_npu_commit'] != environment['torch_npu_commit']):
                continue
            ev, _ = self.fact(record, f'pr_results.db/pr_results/{record["pr_number"]}')
            try:
                tests = decoded(record.get('tests'), [])
            except ValueError:
                self.add(task, phase='record', kind='record.invalid_tests_json',
                         text=record['tests'], evidence=[ev])
                continue
            for test in tests:
                self.test(task, test, ev)

    def resolve(self, identifier, note, evidence_path):
        if not note.strip():
            raise ValueError('A verified repair note is required')
        evidence, _ = self.evidence_file(evidence_path)
        with self.conn:
            row = self.conn.execute('SELECT id FROM families WHERE identifier=?', (identifier,)).fetchone()
            if row is None:
                raise ValueError('Unknown ledger family')
            self.conn.execute('INSERT INTO resolutions(family_id,verified_at,note,evidence) VALUES(?,?,?,?)',
                              (row['id'], now(), note, json.dumps(evidence, ensure_ascii=False)))
            self.conn.execute("UPDATE occurrences SET state='resolved' WHERE family_id=?", (row['id'],))
            self.conn.execute('UPDATE resolution_state SET revision=revision+1 WHERE id=1')

    def summary(self):
        counts = dict(self.conn.execute('SELECT namespace,COUNT(*) FROM occurrences o JOIN families f '
                                       'ON f.id=o.family_id GROUP BY namespace').fetchall())
        return {'db': str(self.db), 'docs_root': str(self.docs), 'new_events': self.new_events,
                'occurrences': sum(counts.values()), 'by_namespace': counts,
                'open': self.conn.execute("SELECT COUNT(*) FROM occurrences WHERE state='open'").fetchone()[0],
                'resolved': self.conn.execute("SELECT COUNT(*) FROM occurrences WHERE state='resolved'").fetchone()[0]}

    def resolve_events(self, event_keys, note, evidence_path, fix_commit):
        """Archive reviewed exact occurrences; never close a whole broad family."""
        keys = sorted(set(event_keys))
        if not keys or not note.strip() or not re.fullmatch(r'[0-9a-f]{40}', fix_commit):
            raise ValueError('Exact events, repair note and full fix commit are required')
        evidence, _ = self.evidence_file(evidence_path)
        with self.conn:
            self.conn.execute('BEGIN IMMEDIATE')
            existing = {row[0] for row in self.conn.execute('SELECT event_key FROM occurrences')}
            if set(keys)-existing:
                raise ValueError('Unknown occurrence; no archive changes applied')
            for key in keys:
                self.conn.execute('INSERT OR IGNORE INTO occurrence_resolutions '
                    '(event_key,resolved_at,note,evidence,fix_commit) VALUES(?,?,?,?,?)',
                    (key,now(),note,json.dumps(evidence,ensure_ascii=False),fix_commit))
                self.conn.execute("UPDATE occurrences SET state='resolved' WHERE event_key=?",(key,))
            self.conn.execute('UPDATE resolution_state SET revision=revision+1 WHERE id=1')
        return len(keys)

    def render(self):
        """Four reproducible views; historical legacy IDs are never fabricated."""
        self.conn.execute('BEGIN IMMEDIATE')
        try:
            self._render()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _render(self):
        """单一台账文件（用户 2026-10-26 决策）：一个问题一节，DB 为账本，
        markdown 只做人读摘要——出现次数/涉及 PR 聚合，证据留指针不贴全文。"""
        self.docs.mkdir(parents=True, exist_ok=True)
        open_lines, resolved_lines = [], []
        rows = self.conn.execute(
            'SELECT f.identifier,f.namespace,f.kind,o.* FROM occurrences o JOIN families f '
            'ON f.id=o.family_id ORDER BY o.first_seen,o.event_key').fetchall()
        families = {}
        for row in rows:
            if row['namespace'] in ('C', 'X'):
                continue
            fam = families.setdefault((row['identifier'], row['state']), {
                'namespace': row['namespace'], 'kind': row['kind'],
                'state': row['state'], 'count': 0, 'first': row['first_seen'],
                'last': row['first_seen'], 'prs': [], 'tests': [], 'latest': row,
            })
            fam['count'] += 1
            fam['last'] = max(fam['last'], row['last_seen'])
            body = json.loads(row['payload'])
            pr = body.get('pr_number')
            if pr and pr not in fam['prs']:
                fam['prs'].append(pr)
            tf = body.get('test_function')
            if tf and tf not in fam['tests']:
                fam['tests'].append(tf)
            fam['latest'] = row  # 行序按 first_seen，最后一条即最近
        for (identifier, state), fam in sorted(families.items()):
            latest = fam['latest']
            body = json.loads(latest['payload'])
            head = (f"### {identifier} · {fam['kind']}"
                    + (f" · PR {'/'.join(str(p) for p in fam['prs'][:5])}"
                       + (f" 等{len(fam['prs'])}个" if len(fam['prs']) > 5 else '')
                       if fam['prs'] else ''))
            lines = [head,
                f"- 状态：{fam['state']}；出现：{fam['count']} 次"
                f"（{fam['first'][:19]} → {fam['last'][:19]}）",
                f"- 涉及 PR：{len(fam['prs'])} 个；测试函数：{len(fam['tests'])} 个"
                + (f"；最近：{fam['tests'][-1][:80]}" if fam['tests'] else ''),
                f"- 最近任务：`{body.get('task_id')}`；设备：`{body.get('device')}`；阶段：`{body.get('phase')}`",
                '- 结论：已解决归档；基于修复证据，未要求重跑历史用例。' if state == 'resolved'
                else '- 结论：待核对因果，不自动判 PR 回归。']
            if body.get('missing_modules'):
                lines.append('- 缺失模块：' + ', '.join(body['missing_modules']))
            ev = body['evidence'][0] if body.get('evidence') else None
            if ev:
                lines.append(f"- 证据（最近）：`{ev['snapshot']}`；SHA256：`{ev['sha256']}`"
                             f"（全部快照按需查库，不在此展开）")
            if fam['state'] == 'resolved':
                resolved_lines.append('\n'.join(lines))
            else:
                open_lines.append('\n'.join(lines))
        for row in self.conn.execute('SELECT f.identifier,f.kind,r.* FROM resolutions r '
                                     'JOIN families f ON f.id=r.family_id ORDER BY r.id'):
            ev = json.loads(row['evidence'])
            resolved_lines.append('\n'.join([
                f"### {row['identifier']} · {row['kind']} · 已核验修复",
                f"- 核验时间：{row['verified_at']}；说明：{row['note']}",
                f"- 验证证据：`{ev['source']}`；快照：`{ev['snapshot']}`；SHA256：`{ev['sha256']}`"]))
        for row in self.conn.execute('SELECT r.fix_commit,r.note,r.evidence,COUNT(*) n FROM occurrence_resolutions r GROUP BY r.fix_commit,r.note,r.evidence'):
            ev=json.loads(row['evidence'])
            resolved_lines.append('\n'.join([f"### 已解决归档 · {row['n']} 条历史事件",
                f"- 修复提交：`{row['fix_commit']}`；说明：{row['note']}",
                '- 依据：修复已验收，用户要求无需重跑历史用例；保留原始结果和日志。',
                f"- 修复证据：`{ev['source']}`；SHA256：`{ev['sha256']}`"]))
        sections = []
        if open_lines:
            sections.append('## 待处理（open）\n\n' + '\n\n'.join(open_lines))
        if resolved_lines:
            sections.append('## 已修复（resolved）\n\n' + '\n\n'.join(resolved_lines))
        text = ('# 问题总览\n\n'
                '自动生成自 issues.db（一个问题一节；出现次数/涉及 PR 聚合，'
                '证据快照按需查库）。新部署编号使用 U-new/S-new，不冒用历史编号；'
                'skip/no_result 不算通过，不自动判 PR 回归。\n'
                + ('\n\n'.join(sections) + '\n' if sections else '\n（当前无问题记录）\n'))
        path = self.docs / '问题总览.md'
        if not path.exists() or path.read_text(encoding='utf-8') != text:
            with tempfile.NamedTemporaryFile(dir=self.docs, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text.encode('utf-8'))
            os.replace(temporary, path)


def scan(config, snap=None):
    if snap is None:
        from .monitoring import snapshot
        snap = snapshot(config)
    ledger = Ledger(config)
    try:
        ledger.observe(snap)
        ledger.render()
        return ledger.summary()
    finally:
        ledger.close()


def selection_error(config, task, message):
    """Capture errors that happen before a test-plan.json can be written."""
    ledger = Ledger(config)
    try:
        task = task if isinstance(task, dict) else task.__dict__
        evidence, text = ledger.fact({'task_id': task['task_id'], 'message': message},
                                    'pipeline.run_selection/exception', message)
        ledger.add(task, phase='selection', kind='selection.stage_error', text=text,
                   evidence=[evidence], extra={'detail': message})
        ledger.render()
    finally:
        ledger.close()
