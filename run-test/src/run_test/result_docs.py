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


TEXT_ELEMENT_LIMIT = 45_000  # 本租户历史成功样本实测（90k 字符块/2 元素）；官方值核实前以此为准


def text_block(kind, text, *, bold=False, folded=False, link=None):
    types = {'text': 2, 'heading3': 5, 'heading4': 6, 'code': 14}
    style = {'bold': bold}
    if link:
        style['link'] = {'url': link}
    # Split text runs and long code blocks without dropping any characters.
    elements = [{'text_run': {'content': text[i:i+1000], 'text_element_style': style}}
                for i in range(0, len(text), 1000)] or [{'text_run': {'content': ''}}]
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
                n_chars += len(e.get('text_run', {}).get('content', ''))
    return n_blocks, n_chars


def _plan_op_size(op):
    """一个计划操作的（块数, 文本字符量）。"""
    if op['op'] == 'table':
        cells = op.get('cell_children', [])
        blocks = 1 + sum(len(cell) for cell in cells)
        chars = sum(_blocks_size(cell)[1] for cell in cells)
        return blocks, chars
    return _blocks_size(op.get('body', {}).get('children', []))


DEFAULT_DOC_BUDGET = {'doc_max_blocks': 50, 'doc_max_chars': 500_000}


def feishu_plan(model, budget=None):
    """v2：按文档级预算拆主篇与续篇（用户 2026-10-26 澄清——拆分粒度是
    文档，不是把 8000 字符分块做成产品行为；块内 8000 切分只是文本元素
    内部细节，拼接零丢失，由发布器逐篇回读校验）。

    budget 缺省用保守值：低于本租户历史成功样本（60 块、636,744 字符/篇；
    单块实测 ~90k 字符）。正式默认待官方限制核实后经 settings 固化；
    单测试节超预算时独立成篇并标记 over_budget（保持该测试连续）。
    """
    budget = dict(budget or DEFAULT_DOC_BUDGET)
    max_blocks = max(int(budget.get('doc_max_blocks') or 50), 1)
    max_chars = max(int(budget.get('doc_max_chars') or 500_000), 1)
    operations = []

    def append(blocks):
        for i in range(0, len(blocks), 40):
            operations.append({'op': 'append', 'parent': 'document',
                               'body': {'index': -1, 'children': blocks[i:i+40]}})

    append([text_block('heading3', f"PR #{model['pr_number']}", bold=True, link=model['pr_url']),
            text_block('text', environment_line(model))])
    if model['merged_at']:
        append([text_block('text', 'Merged at: ' + model['merged_at'])])
    operations.append({'op': 'table', 'parent': 'document',
        'body': {'index': -1, 'children': [{'block_type': 31,
            'table': {'property': {'row_size': 2, 'column_size': 6}}}]},
        'cell_children': [[text_block('text', value, bold=i < 6)] for i, value in enumerate(
            list(HEADERS) + [str(model['metrics'][h]) for h in HEADERS])],
        'cell_order': 'row-major; resolve returned table.cells before appending cell_children'})
    append([text_block('text', model['metric_note']),
            text_block('text', '执行实例：' + count_text(model['instance_counts']))])
    for warning in model['warnings']:
        append([text_block('text', '证据提示：' + warning)])
    header_size = sum(_plan_op_size(op)[0] for op in operations)
    header_chars = sum(_plan_op_size(op)[1] for op in operations)
    header_ops = operations[:]
    operations = []

    def _ops_size(ops):
        blocks = chars = 0
        for op in ops:
            b, c = _plan_op_size(op)
            blocks += b
            chars += c
        return blocks, chars

    # 逻辑节：逐测试整节移动（一个测试不跨篇，保证连续性）。
    # 块内元素按 TEXT_ELEMENT_LIMIT 切分（真实元素上限的租户实证值），
    # 一个日志=一个连续块，拼接零丢失——8k 不再是产品级分块。
    def _continuous_block(kind, text):
        elements = [{'text_run': {'content': text[i:i + TEXT_ELEMENT_LIMIT]}}
                    for i in range(0, len(text), TEXT_ELEMENT_LIMIT)] or [{'text_run': {'content': ''}}]
        style = {'language': 1, 'wrap': True} if kind == 'code' else {'align': 1, 'folded': True}
        block_type = {'text': 2, 'code': 14}[kind]
        return {'block_type': block_type, kind: {'elements': elements, 'style': style}}

    test_sections = []
    for test in model['tests']:
        section = [text_block('heading4', detail_title(test), bold=True, folded=True)]
        for note in detail_notes(test):
            section.append(_continuous_block('text', note))
        section.append(_continuous_block('code', test['log']['text']))
        test_sections.append((detail_title(test), section))

    parts = [{'role': 'main', 'title_suffix': '', 'ops': list(header_ops),
              'over': header_size > max_blocks or header_chars > max_chars}]

    def _part_append(part, blocks_list):
        # 节内容以一个 append 操作进入（publisher 端仍按 40 块批量上传）。
        part['ops'].append({'op': 'append', 'parent': 'document',
                            'body': {'index': -1, 'children': blocks_list}})

    _part_append(parts[0], [text_block('heading4', '前后对比（基线 → 本次）', bold=True)])
    for test in model['tests']:
        _part_append(parts[0], [text_block('text', f"{ICONS[test['outcome']]} {test['test_function']} [{test['device']}]：{test['comparison_text']}")])
    cur_blocks, cur_chars = _ops_size(parts[0]['ops'])
    for label, section in test_sections:
        blocks, chars = _blocks_size(section)
        # 常规路径：整节放入当前篇（预算内）。
        if (cur_blocks + blocks <= max_blocks and cur_chars + chars <= max_chars) \
                and not (blocks > max_blocks or chars > max_chars):
            _part_append(parts[-1], section)
            cur_blocks += blocks
            cur_chars += chars
            continue
        # 超预算节（或当前篇放不下且节本身超预算）：把节的块序列按预算
        # 分组拆到一篇或多篇续篇（块为原子，全文零丢失），每篇带上下文头
        # 与续篇标记，满足"单一超长测试跨篇"。
        groups, cur, cb, cc = [], [], 0, 0
        for block in section:
            b, c = _blocks_size([block])
            if cur and (cb + b > max_blocks or cc + c > max_chars):
                groups.append(cur)
                cur, cb, cc = [], 0, 0
            cur.append(block)
            cb += b
            cc += c
        if cur:
            groups.append(cur)
        total = len(groups)
        for gi, group in enumerate(groups):
            if gi > 0 or cur_blocks + _blocks_size(group)[0] > max_blocks \
                    or cur_blocks + _blocks_size(group)[1] > max_chars:
                parts.append({'role': 'continuation',
                              'title_suffix': f'-{len(parts)}', 'ops': [], 'over': False})
                cur_blocks, cur_chars = 0, 0
            head = ([text_block('heading4', f'{label}（续 {gi + 1}/{total}）', bold=True, folded=True)]
                    if gi > 0 else [])
            _part_append(parts[-1], head + group)
            cur_blocks, cur_chars = _blocks_size(group)
    if len(parts) > 1:
        parts[0]['ops'].append({'op': 'toc',
                                'titles': [model['title'] + p['title_suffix'] for p in parts[1:]]})
    parts_out = [{'part_index': i, 'role': ('main' if i == 0 else 'continuation'),
                  'title_suffix': p['title_suffix'], 'over_budget': p['over'],
                  'operations': p['ops']} for i, p in enumerate(parts)]
    import sys as _sys
    _sizes = [_blocks_size([b for op in p['operations'] if op.get('body')
                            for b in op['body'].get('children', [])]) for p in parts_out]
    print(f"[DEBUG] max_chars={max_chars} max_blocks={max_blocks} parts={len(parts_out)} "
          f"sizes={_sizes} test_log_len={[len(t['log']['text']) for t in model['tests']]}",
          flush=True, file=_sys.stderr)

    # 扁平 operations = 全部篇依序拼接：无丢失校验与旧消费方兼容。
    flattened = [op for p in parts_out for op in p['operations']]
    return {'schema': 'feishu-report-plan/2', 'title': model['title'],
            'status': 'local_unpublished', 'part_count': len(parts_out),
            'parts': parts_out, 'operations': flattened}


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
    fs = (config.get('feishu') or {})
    budget = {k: fs[k] for k in ('doc_max_blocks', 'doc_max_chars') if k in fs} or None
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
