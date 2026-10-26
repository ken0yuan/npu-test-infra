"""Generate evidence-backed reports in the layout recovered from Feishu samples.

Pure rendering and local materialization only: no credentials, network or changes
to test results/baselines. feishu-plan.json is an ordered publisher input, not a
claim that a document has been uploaded.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

from .config import store_paths
from .identifiers import baseline_log_filename

VERSION = 'npu-test-report/1'
COUNT_KEYS = ('pass', 'fail', 'error', 'skip')
HEADERS = ('Total', 'Passed', 'Skipped', 'No Result', 'Failed', 'Error')
STATES = ('pass', 'skip', 'no_result', 'fail', 'error')
ICONS = dict(zip(STATES, ('✅', '⏭️', '❔', '❌', '⛔')))
ANSI = re.compile(r'\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]')
COUNT_NOTE = ('上表按测试项（函数 × 设备）计数，每项只归一类：Error > Failed > '
              'No Result > Skipped > Passed。部分实例跳过的测试项归入 Skipped，'
              '只有全部实例通过才归入 Passed；实例计数另列。')


class ReportError(ValueError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def counts(value):
    if not isinstance(value, dict) or set(value) != set(COUNT_KEYS):
        raise ReportError('计数必须含 pass/fail/error/skip 四项')
    if any(type(value[k]) is not int or value[k] < 0 for k in COUNT_KEYS):
        raise ReportError('计数必须是非负整数')
    return {k: value[k] for k in COUNT_KEYS}


def count_text(value):
    if value is None:
        return '未取得完整结果'
    return '过{pass} 跳{skip} 败{fail} 错{error}'.format(**value)


def decoded(value):
    return json.loads(value) if isinstance(value, str) else deepcopy(value)


def clean_log(text):
    # Rendering removes terminal escapes; the evidence digest still refers to
    # original bytes. Never interpret HTML/Markdown from a test subprocess.
    return ANSI.sub('', text).replace('\x00', '')


def outcome(test):
    if test.get('status') == 'no_result':
        return 'no_result'
    c = test['counts']
    if c['error']:
        return 'error'
    if c['fail']:
        return 'fail'
    if not sum(c.values()):
        return 'no_result'
    if c['skip']:
        return 'skip'
    return 'pass'


def comparison_text(test):
    comp = test.get('comparison')
    if test['outcome'] == 'no_result':
        return '无完整结果，基线不动（' + (test.get('reason') or '零实例/未取得结果') + '）'
    if not comp:
        return '缺少本次基线快照，无法判定前后变化'
    action = comp.get('baseline_action')
    old = comp.get('before')
    if action == 'incomparable':
        return '本次 ' + count_text(test['counts']) + '；覆盖或环境不可比，基线不动'
    if action == 'created' and old is None:
        return '基线无记录 → 新建（' + count_text(test['counts']) + '）'
    if action == 'overwritten' and old:
        before = counts(old['counts'])
        result = '基线 ' + count_text(before) + ' → 本次 ' + count_text(test['counts'])
        if before == test['counts']:
            result += '（计数无变化；不据此判定失败根因相同）'
        return result
    return '基线动作未确认，未推断改善或回归'


def build_report(row, archive=None, logs=None):
    """Validated row + the captured before/after archive + exact log mapping."""
    row = deepcopy(row)
    if type(row.get('pr_number')) is not int or row['pr_number'] <= 0:
        raise ReportError('PR 编号非法')
    tests = decoded(row['tests'])
    env = decoded(row['torch_npu_environment'])
    if not isinstance(env, dict) or not env.get('torch_npu_commit'):
        raise ReportError('缺少 torch-npu 提交身份')
    if not isinstance(tests, list) or not tests:
        raise ReportError('没有测试结果，不生成空成功报告')
    comparisons = {}
    if archive is not None:
        if archive.get('pr_number') != row['pr_number']:
            raise ReportError('基线快照的 PR 与结果不符')
        for item in archive.get('comparisons', []):
            key = (item['test_function'], item['device'])
            if key in comparisons:
                raise ReportError('基线快照含重复测试项')
            comparisons[key] = item
    warnings, details, seen = [], [], set()
    totals = dict.fromkeys(COUNT_KEYS, 0)
    declared_no_result = 0
    for source in tests:
        test = deepcopy(source)
        key = (test.get('test_function'), test.get('device'))
        if not all(isinstance(x, str) and x for x in key) or key in seen:
            raise ReportError('测试项身份缺失或重复')
        seen.add(key)
        if test.get('status') == 'no_result':
            if test.get('counts') is not None:
                raise ReportError('no_result 不得携带完整计数')
            test['counts'] = None
            declared_no_result += 1
        else:
            test['counts'] = counts(test.get('counts'))
            for k in COUNT_KEYS:
                totals[k] += test['counts'][k]
        if test.get('observed_counts') is not None:
            test['observed_counts'] = counts(test['observed_counts'])
        comp = comparisons.get(key)
        if comp is not None:
            after = comp.get('after') or {}
            if (after.get('pytorch_commit') != row['pytorch_commit'] or
                    after.get('torch_npu_environment') != env or
                    after.get('counts') != test['counts'] or
                    (after.get('status') == 'no_result') != (test.get('status') == 'no_result')):
                raise ReportError('基线快照与本次提交/环境/计数不符，拒绝混读旧结果')
            if (set(after.get('instance_ids') or []) != set(test.get('instance_ids') or []) or
                    after.get('environment_fingerprint') != test.get('environment_fingerprint')):
                raise ReportError('基线快照与本次执行实例/环境指纹不符')
        test['comparison'] = deepcopy(comp)
        test['outcome'] = outcome(test)
        if test['counts'] is not None and sum(test['counts'].values()) == 0:
            warnings.append(f'{key[0]} [{key[1]}] 四计数为零，展示为未取得结果')
        evidence = (logs or {}).get(key)
        test['log'] = deepcopy(evidence) if evidence else {
            'text': '（无日志：本次对应日志缺失，不能据此推断通过）',
            'source': None, 'sha256': None, 'missing': True}
        if not evidence or evidence.get('missing'):
            warnings.append(f'{key[0]} [{key[1]}] 本次日志缺失或无法绑定')
        test['comparison_text'] = comparison_text(test)
        details.append(test)
    if archive is not None and seen != set(comparisons):
        raise ReportError('基线快照测试集合与结果不一致')
    if any(totals[k] != row.get(k + '_count') for k in COUNT_KEYS):
        raise ReportError('PR 汇总与逐项实例计数不一致')
    if declared_no_result != row.get('no_result_count', 0):
        raise ReportError('PR no_result 汇总与逐项结果不一致')
    if archive is None:
        warnings.append('未找到本次基线快照；没有读取当前基线来冒充历史对照')
    classified = Counter(t['outcome'] for t in details)
    metrics = dict(zip(HEADERS, [len(details), *(classified[s] for s in STATES)]))
    return {'schema': VERSION, 'title': f"PR#{row['pr_number']}",
            'pr_number': row['pr_number'], 'pr_url': f"https://github.com/pytorch/pytorch/pull/{row['pr_number']}",
            'pytorch_commit': row['pytorch_commit'], 'torch_npu_environment': env,
            'merged_at': row.get('merged_at'), 'metrics': metrics,
            'metric_note': COUNT_NOTE, 'instance_counts': totals,
            'declared_no_result_count': declared_no_result,
            'warnings': warnings, 'tests': details}


def _read_row(config, pr):
    path = store_paths(config).pr_db.resolve()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM pr_results WHERE pr_number=?', (pr,)).fetchone()
        if row is None:
            raise ReportError(f'PR#{pr} 尚无落库结果')
        return dict(row)


def _inside(path, root):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ReportError('证据路径越出对应 PR 日志目录')
    return path


def load_report(config, pr):
    root = store_paths(config).data_root.resolve()
    row = _read_row(config, pr)
    folder = _inside(root / row['log_path'], root / 'pr_logs')
    archive_path = _inside(folder / 'baseline-diff.json', folder)
    raw_archive = archive_path.read_bytes() if archive_path.is_file() else None
    archive = json.loads(raw_archive) if raw_archive is not None else None
    logs, observed = {}, {}
    # Without a matching historical archive the mutable PR directory cannot
    # safely prove which run its logs belong to. Show missing evidence instead.
    if archive is not None:
        for test in decoded(row['tests']):
            key = (test['test_function'], test['device'])
            path = _inside(folder / baseline_log_filename(*key), folder)
            if path.is_file():
                raw = path.read_bytes()
                observed[path] = digest(raw)
                logs[key] = {'text': clean_log(raw.decode('utf-8', errors='replace')),
                             'source': str(path), 'sha256': digest(raw), 'missing': False}
    model = build_report(row, archive, logs)
    if _read_row(config, pr) != row:
        raise ReportError('读取期间 PR 结果已更新，请重试')
    if raw_archive is not None and archive_path.read_bytes() != raw_archive:
        raise ReportError('读取期间基线快照已更新，请重试')
    if any(not p.is_file() or digest(p.read_bytes()) != sha for p, sha in observed.items()):
        raise ReportError('读取期间 PR 日志已更新，请重试')
    return model


def environment_line(model):
    return ('环境: pytorch ' + model['pytorch_commit'][:10] + ' · torch-npu ' +
            model['torch_npu_environment']['torch_npu_commit'][:12])


def detail_title(test):
    path, _, name = test['test_function'].partition('::')
    return (f"{ICONS[test['outcome']]} {name or path} — {path} · "
            f"{test['device']} · {test['outcome']}")


def detail_notes(test):
    notes = ['实例计数：' + count_text(test['counts'])]
    if test.get('reason'):
        notes.append('原因：' + test['reason'])
    if test.get('observed_counts') is not None:
        notes.append('中断前观察计数（不计入完整结果）：' + count_text(test['observed_counts']))
    if test.get('execution_status'):
        notes.append('执行状态：' + test['execution_status'])
    if test.get('coverage_status'):
        notes.append('覆盖状态：' + test['coverage_status'])
    if test.get('triage'):
        notes.append('已知签名（仍需复核）：' + ', '.join(test['triage']))
    if test['log'].get('sha256'):
        notes.append('日志 SHA256：' + test['log']['sha256'])
    return notes


def render_html(model):
    esc = lambda value: html.escape(str(value), quote=True)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>' + esc(model['title']) + '</title>',
             '<style>body{max-width:1080px;margin:40px auto;padding:0 24px;font:15px/1.65 system-ui;color:#20252b}',
             'table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #dce1e7;padding:12px;text-align:center}',
             'th{background:#f5f7fa}td{font-size:24px}summary{cursor:pointer;font-weight:600;overflow-wrap:anywhere}',
             'details{border-top:1px solid #e5e7eb;padding:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:16px}',
             '.note{font-size:13px;color:#59636f}.warning{background:#fff8dc;padding:12px}li{overflow-wrap:anywhere}</style>',
             '<h2><a href="' + esc(model['pr_url']) + '">' + esc(model['title']) + '</a></h2>',
             '<p>' + esc(environment_line(model)) + '</p>']
    if model['merged_at']:
        parts.append('<p>Merged at: ' + esc(model['merged_at']) + '</p>')
    parts += ['<table><tr>' + ''.join('<th>' + h + '</th>' for h in HEADERS) + '</tr><tr>',
              ''.join('<td>' + str(model['metrics'][h]) + '</td>' for h in HEADERS), '</tr></table>',
              '<p class="note">' + esc(model['metric_note']) + '</p>',
              '<p>执行实例：' + esc(count_text(model['instance_counts'])) + '</p>']
    if model['warnings']:
        parts.append('<div class="warning">证据提示<ul>' + ''.join('<li>' + esc(w) + '</li>' for w in model['warnings']) + '</ul></div>')
    parts.append('<h3>前后对比（基线 → 本次）</h3><ul>')
    for test in model['tests']:
        parts.append('<li>' + esc(test['test_function'] + ' [' + test['device'] + ']：' + test['comparison_text']) + '</li>')
    parts.append('</ul>')
    for test in model['tests']:
        parts += ['<details><summary>' + esc(detail_title(test)) + '</summary>',
                  '<p>' + '<br>'.join(esc(n) for n in detail_notes(test)) + '</p>',
                  '<pre>' + esc(test['log']['text']) + '</pre></details>']
    return ''.join(parts) + '</html>\n'


BLOCK_TEXT_LIMIT = 100_000  # Feishu Docs limits FAQ: 100,000 characters per block.
TEXT_ELEMENT_LIMIT = 1000  # Internal runs within one block; not visible log chunks.


def text_units(text):
    # Count supplementary Unicode characters conservatively as two UTF-16 units.
    return len(text.encode('utf-16-le')) // 2


def text_block(kind, text, *, bold=False, folded=False, link=None):
    types = {'text': 2, 'heading3': 5, 'heading4': 6, 'code': 14}
    style = {'bold': bold}
    if link:
        style['link'] = {'url': link}
    # Internal text runs do not create visible code blocks.
    elements = [{'text_run': {'content': text[i:i+TEXT_ELEMENT_LIMIT], 'text_element_style': style}}
                for i in range(0, len(text), TEXT_ELEMENT_LIMIT)] or [{'text_run': {'content': ''}}]
    block_style = {'language': 1, 'wrap': True} if kind == 'code' else {'align': 1, 'folded': folded}
    return {'block_type': types[kind], kind: {'elements': elements, 'style': block_style}}


def _blocks_size(blocks):
    """块列表的（块数, 文本字符量）——文档级预算的计算单位。"""
    field = {2: 'text', 5: 'heading3', 6: 'heading4', 14: 'code'}
    n_blocks = len(blocks)
    n_chars = 0
    for child in blocks:
        f = field.get(child['block_type'])
        if f:
            for e in child[f].get('elements', []):
                n_chars += text_units(e.get('text_run', {}).get('content', ''))
    return n_blocks, n_chars


def _plan_op_size(op):
    """一个计划操作的（块数, 文本字符量）。"""
    if op['op'] == 'table':
        cells = op.get('cell_children', [])
        blocks = 1 + 2 * len(cells) + sum(len(cell) for cell in cells)
        chars = sum(_blocks_size(cell)[1] for cell in cells)
        return blocks, chars
    return _blocks_size(op.get('body', {}).get('children', []))


DEFAULT_DOC_BUDGET = {'doc_max_blocks': 50, 'doc_max_chars': 500_000}


def feishu_plan(model, budget=None):
    """Pack complete content into bounded documents; split text only at capacity.

    Counts include the document root, table cells and navigation. A normal log
    stays in one code block. The block cap is separate from document budgets.
    """
    limits = {**DEFAULT_DOC_BUDGET, 'block_max_chars': BLOCK_TEXT_LIMIT, **(budget or {})}
    for key, minimum in [('doc_max_blocks', 42), ('doc_max_chars', 2048),
                         ('block_max_chars', 1)]:
        value = limits[key]
        if type(value) is not int or value < minimum:
            raise ReportError(f'{key} must be an integer >= {minimum}')
    if limits['block_max_chars'] > BLOCK_TEXT_LIMIT:
        raise ReportError('block_max_chars exceeds the documented block limit')
    max_blocks, max_chars = limits['doc_max_blocks'], limits['doc_max_chars']
    stream = []

    def add(block, context=None):
        stream.append(({'op': 'append', 'parent': 'document',
                        'body': {'index': -1, 'children': [block]}}, context))

    add(text_block('heading3', f"PR #{model['pr_number']}", bold=True, link=model['pr_url']))
    add(text_block('text', environment_line(model)))
    if model['merged_at']:
        add(text_block('text', 'Merged at: ' + model['merged_at']))
    stream.append(({'op': 'table', 'parent': 'document',
        'body': {'index': -1, 'children': [{'block_type': 31,
            'table': {'property': {'row_size': 2, 'column_size': 6}}}]},
        'cell_children': [[text_block('text', value, bold=i < 6)] for i, value in enumerate(
            list(HEADERS) + [str(model['metrics'][h]) for h in HEADERS])],
        'cell_order': 'row-major'}, None))
    add(text_block('text', model['metric_note']))
    add(text_block('text', '执行实例：' + count_text(model['instance_counts'])))
    for warning in model['warnings']:
        add(text_block('text', '证据提示：' + warning))
    add(text_block('heading4', '前后对比（基线 → 本次）', bold=True))
    for test in model['tests']:
        add(text_block('text', f"{ICONS[test['outcome']]} {test['test_function']} [{test['device']}]：{test['comparison_text']}"))
    for test in model['tests']:
        label = detail_title(test)
        add(text_block('heading4', label, bold=True, folded=True), label)
        for note in detail_notes(test):
            add(text_block('text', note), label)
        add(text_block('code', test['log']['text']), label)

    def split_block(block, capacity):
        field = {2: 'text', 5: 'heading3', 6: 'heading4', 14: 'code'}[block['block_type']]
        pieces, elements, used = [], [], 0
        def finish():
            result = deepcopy(block)
            result[field]['elements'] = list(elements)
            pieces.append(result)
        for element in block[field]['elements']:
            text = element.get('text_run', {}).get('content', '')
            if not text:
                elements.append(deepcopy(element))
            while text:
                if used == capacity:
                    finish(); elements, used = [], 0
                take = min(len(text), capacity - used)
                while take and text_units(text[:take]) > capacity - used:
                    take -= max(1, (text_units(text[:take]) - (capacity - used) + 1) // 2)
                if not take:
                    if not used:
                        raise ReportError('Block capacity cannot hold a Unicode character')
                    finish(); elements, used = [], 0
                    continue
                fragment = deepcopy(element)
                fragment['text_run']['content'], text = text[:take], text[take:]
                elements.append(fragment); used += text_units(fragment['text_run']['content'])
        if elements:
            finish()
        return pieces

    def pack(toc_reserve):
        parts = [{'part_index': 0, 'role': 'main', 'title_suffix': '',
                  'over_budget': False, 'operations': []}]
        # Root + up to three navigation blocks. TOC uses heading + one link block.
        used_blocks, used_chars = 4, toc_reserve
        for operation, label in stream:
            context = text_block('heading4', (label or model['title']) + '（续）', bold=True)
            context_chars = _blocks_size([context])[1]
            capacity = min(limits['block_max_chars'], max_chars - max(toc_reserve, 64) - context_chars)
            if capacity < 1:
                raise ReportError('Document budget is too small for continuation context')
            if operation['op'] == 'table':
                fragments = [operation]
            else:
                fragments = [{'op': 'append', 'parent': 'document',
                              'body': {'index': -1, 'children': [block]}}
                             for block in split_block(operation['body']['children'][0], capacity)]
            for fragment in fragments:
                n, chars = _plan_op_size(fragment)
                if used_blocks + n > max_blocks or used_chars + chars > max_chars:
                    index = len(parts)
                    parts.append({'part_index': index, 'role': 'continuation',
                                  'title_suffix': f'-{index}', 'over_budget': False,
                                  'operations': [{'op': 'append', 'parent': 'document',
                                                  'body': {'index': -1, 'children': [context]}}]})
                    used_blocks, used_chars = 5, 64 + context_chars
                if used_blocks + n > max_blocks or used_chars + chars > max_chars:
                    raise ReportError('Document budget cannot hold an indivisible report component')
                parts[-1]['operations'].append(fragment)
                used_blocks += n; used_chars += chars
        return parts

    reserve = 256
    for _ in range(12):
        parts = pack(reserve)
        titles = [model['title'] + p['title_suffix'] for p in parts[1:]]
        required = 32 + sum(text_units(t) + 1 for t in titles)
        if required <= reserve:
            break
        reserve = required
    else:
        raise ReportError('Document navigation budget did not converge')
    if titles:
        parts[0]['operations'].append({'op': 'toc', 'titles': titles})
    return {'schema': 'feishu-report-plan/3', 'title': model['title'],
            'status': 'local_unpublished', 'part_count': len(parts), 'parts': parts,
            'budget': limits, 'operations': [op for part in parts for op in part['operations']]}


def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', encoding='utf-8', newline='\n', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(config, pr):
    # Serialize CLI/monitor rendering so an older worker cannot overwrite the
    # pointer installed by a newer worker. This DB is separate from result data.
    root = store_paths(config).data_root / 'reports'
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / 'render-lock.db', timeout=30)
    try:
        conn.execute('BEGIN IMMEDIATE')
        result = _generate_locked(config, pr)
        conn.commit()
        return result
    finally:
        conn.close()


def _generate_locked(config, pr):
    model = load_report(config, pr)
    fingerprint = digest(encoded(model).encode())
    root = store_paths(config).data_root / 'reports' / f'PR_{pr}'
    # Content-addressed versions preserve reruns; current.json changes last.
    version = root / fingerprint
    fs = {**(config.get('feishu') or {}),
          **((config.get('reports') or {}).get('feishu') or {})}
    budget = {k: fs[k] for k in ('doc_max_blocks', 'doc_max_chars', 'block_max_chars') if k in fs} or None
    outputs = {'report.json': encoded(model), 'report.html': render_html(model),
               'feishu-plan.json': encoded(feishu_plan(model, budget))}
    changed = False
    for name, content in outputs.items():
        target = version / name
        if not target.is_file() or target.read_bytes() != content.encode('utf-8'):
            _atomic(target, content)
            changed = True
    pointer = {'schema': VERSION, 'pr_number': pr, 'fingerprint': fingerprint,
               'version': str(version), 'html': str(version / 'report.html'),
               'files': {name: digest(content.encode()) for name, content in outputs.items()}}
    if digest(encoded(load_report(config, pr)).encode()) != fingerprint:
        raise ReportError('生成期间 PR 证据已更新，保留版本但不切换当前报告；请重试')
    current = root / 'current.json'
    if not current.exists() or current.read_text(encoding='utf-8') != encoded(pointer):
        _atomic(current, encoded(pointer))
        changed = True
    return {'pr_number': pr, 'changed': changed, **pointer}


def scan(config):
    path = store_paths(config).pr_db.resolve()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
        prs = [r[0] for r in conn.execute('SELECT pr_number FROM pr_results ORDER BY pr_number')]
    results, errors = [], []
    for pr in prs:
        try:
            results.append(generate(config, pr))
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
            errors.append({'pr_number': pr, 'error': str(error)})
    return {'checked': len(prs), 'generated': sum(r['changed'] for r in results),
            'unchanged': sum(not r['changed'] for r in results), 'errors': errors}
