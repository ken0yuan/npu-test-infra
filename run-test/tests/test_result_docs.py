"""Report truthfulness, replay, evidence binding, and recovered Feishu layout."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from run_test import result_docs as docs
from run_test.config import store_paths
from run_test.identifiers import Counts, baseline_log_filename
from run_test.record import submit_complete_result
from conftest import COMMIT_A, COMMIT_B, COMMIT_NPU, make_submission

class _FakeMirror:
    """线性 A→B 推进替身；commit_times 供 backfill 解析测试注入。"""
    commit_times: dict = {}

    def __init__(self, config=None):
        pass

    def merge_relation(self, a, b):
        if a == b:
            return 'equal'
        order = [COMMIT_A, COMMIT_B]
        if a in order and b in order:
            return 'a_earlier' if order.index(a) < order.index(b) else 'a_later'
        return 'unknown'

    def commit_timestamp(self, sha):
        return self.commit_times.get(sha)


@pytest.fixture(autouse=True)
def _deterministic_merge_order(monkeypatch):
    """结果文档测试统一假定线性推进（提交按 A→B 变新）；合入序防护语义
    本身由 test_record_flow 覆盖。真实镜像对假 commit 会返回 unknown 而
    保守跳过刷新，导致报告与基线指纹不一致——这里替换为确定性替身。"""
    import run_test.upstream as up
    _FakeMirror.commit_times = {}
    monkeypatch.setattr(up, 'PyTorchMirror', _FakeMirror)




def row_for(items):
    totals = {k: sum((t.get('counts') or {}).get(k, 0) for t in items) for k in docs.COUNT_KEYS}
    return {'pr_number': 77, 'pytorch_commit': COMMIT_A,
            'torch_npu_environment': {'torch_npu_commit': COMMIT_NPU},
            'tests': deepcopy(items), **{k + '_count': v for k, v in totals.items()},
            'no_result_count': sum(t.get('status') == 'no_result' for t in items)}


def item(name='test_a', device='cpu', **kwargs):
    return {'test_function': 'test/test_sample.py::TestSample::' + name, 'device': device,
            'counts': {'pass': 1, 'skip': 0, 'fail': 0, 'error': 0}, **kwargs}


def archive_for(row, *, action='created', old=None):
    return {'pr_number': row['pr_number'], 'comparisons': [
        {'test_function': t['test_function'], 'device': t['device'], 'baseline_action': action,
         'before': old, 'after': {'counts': t.get('counts'), 'status': t.get('status'),
          'pytorch_commit': row['pytorch_commit'], 'torch_npu_environment': row['torch_npu_environment']}}
        for t in row['tests']]}


def text_of(block):
    return ''.join(e.get('text_run', {}).get('content', '') for body in block.values()
                   if isinstance(body, dict) for e in body.get('elements', []))


def appended(plan):
    return [b for op in plan['operations'] if op['op'] == 'append' for b in op['body']['children']]


def test_function_counts_and_parameterized_instances_are_distinct():
    tests = [item(counts={'pass': 12, 'skip': 2, 'fail': 0, 'error': 0}),
             item(device='accelerator', counts={'pass': 2, 'skip': 1, 'fail': 1, 'error': 1}),
             item('test_none', counts=None, status='no_result', reason='timeout',
                  observed_counts={'pass': 4, 'skip': 0, 'fail': 0, 'error': 0})]
    m = docs.build_report(row_for(tests))
    assert m['metrics'] == {'Total': 3, 'Passed': 0, 'Skipped': 1, 'No Result': 1, 'Failed': 0, 'Error': 1}
    assert m['instance_counts'] == {'pass': 14, 'skip': 3, 'fail': 1, 'error': 1}
    assert '中断前观察计数' in docs.render_html(m)
    assert '部分实例跳过' in m['metric_note']


def test_all_no_result_does_not_become_success_or_refresh_baseline():
    row = row_for([item(counts=None, status='no_result', reason='collection_error')])
    m = docs.build_report(row, archive_for(row, action='no_result'))
    assert m['metrics']['No Result'] == m['metrics']['Total'] == 1
    assert m['metrics']['Passed'] == 0
    assert '基线不动' in m['tests'][0]['comparison_text']


@pytest.mark.parametrize('node,target', [
    ('test/test_reductions.py::TestReductionsOnCPU::test_cumsum_integer_upcast', 'CPU'),
    ('test/test_meta.py::TestMetaCore::test_empty_quantized', 'Meta'),
    ('test/test_set_default_mobile_cpu_allocator.py::TestSetDefaultMobileCPUAllocator::test_exception', 'CPU'),
    ('test/test_ops.py::TestCommonPRIVATEUSE1::test_compare_cpu_npu_float32', 'NPU'),
])
def test_report_separates_declared_targets_from_accelerator_environment(node, target):
    t = item(device='accelerator', test_function=node)
    before = deepcopy(t)
    model = docs.build_report(row_for([t]))
    label = docs.device_display(model['tests'][0])
    assert '目标 ' + target in label and '通道 accelerator' in label
    assert model['tests'][0]['device'] == 'accelerator' and t == before
    assert label in docs.detail_title(model['tests'][0])
    assert label in docs.render_html(model)
    assert label in ' '.join(text_of(b) for b in appended(docs.feishu_plan(model)))
    assert model['metrics']['Passed'] == 1


def test_accelerator_environment_alone_does_not_prove_npu_execution():
    t = item(device='accelerator', environment_fingerprint={'actual_backend':'npu'})
    model = docs.build_report(row_for([t]))
    assert docs.device_display(model['tests'][0]) == '设备目标未核实（通道 accelerator）'
    assert '不能据此认定实际使用了 NPU' in model['metric_note']


def test_nccl_unavailable_is_not_reported_as_hccl_coverage():
    t = item(device='accelerator', counts=None, status='no_result',
             reason='runtime discovery exited 3: c10d NCCL not available, skipping tests')
    model = docs.build_report(row_for([t]))
    test = model['tests'][0]
    assert 'NCCL 专用，不适用（未执行' in docs.device_display(test)
    assert '未执行 HCCL 测试' in ' '.join(docs.detail_notes(test))
    assert model['metrics']['No Result'] == 1 and model['metrics']['Passed'] == 0


@pytest.mark.parametrize('mutation', ['duplicate', 'wrong_sum', 'negative', 'boolean', 'missing_counts', 'no_result_counts'])
def test_corrupt_results_are_refused(mutation):
    row = row_for([item()])
    if mutation == 'duplicate': row['tests'].append(item())
    if mutation == 'wrong_sum': row['pass_count'] = 99
    if mutation == 'negative': row['tests'][0]['counts']['pass'] = -1
    if mutation == 'boolean': row['tests'][0]['counts']['pass'] = True
    if mutation == 'missing_counts': row['tests'][0].pop('counts')
    if mutation == 'no_result_counts': row['tests'][0]['status'] = 'no_result'
    with pytest.raises(docs.ReportError): docs.build_report(row)


@pytest.mark.parametrize('mismatch', ['pr', 'commit', 'npu', 'counts', 'device', 'duplicate'])
def test_snapshot_identity_must_match(mismatch):
    row = row_for([item()]); archive = archive_for(row)
    entry = archive['comparisons'][0]
    if mismatch == 'pr': archive['pr_number'] = 78
    if mismatch == 'commit': entry['after']['pytorch_commit'] = COMMIT_B
    if mismatch == 'npu': entry['after']['torch_npu_environment'] = {'torch_npu_commit': COMMIT_B}
    if mismatch == 'counts': entry['after']['counts'] = {'pass': 4, 'fail': 0, 'error': 0, 'skip': 0}
    if mismatch == 'device': entry['device'] = 'accelerator'
    if mismatch == 'duplicate': archive['comparisons'].append(deepcopy(entry))
    with pytest.raises(docs.ReportError): docs.build_report(row, archive)


def test_missing_and_incomparable_baseline_never_claim_improvement():
    row = row_for([item()])
    absent = docs.build_report(row)['tests'][0]
    assert '无法判定' in absent['comparison_text']
    incomparable = docs.build_report(row, archive_for(row, action='incomparable'))['tests'][0]
    assert '不可比' in incomparable['comparison_text']
    assert '新建' not in incomparable['comparison_text']


def test_equal_counts_do_not_prove_same_failure():
    row = row_for([item(counts={'pass': 0, 'fail': 1, 'error': 0, 'skip': 0}, triage=['C04'])])
    model = docs.build_report(row, archive_for(row, action='overwritten', old={'counts': row['tests'][0]['counts']}))
    assert '不据此判定失败根因相同' in model['tests'][0]['comparison_text']
    assert model['metrics']['Failed'] == 1


def test_long_logs_are_lossless_and_html_is_escaped():
    row = row_for([item()]); raw = ('\x1b[31m<script>alert(1)</script>中文\x1b[0m\n' * 2500)
    cleaned = docs.clean_log(raw)
    key = ('test/test_sample.py::TestSample::test_a', 'cpu')
    m = docs.build_report(row, logs={key: {'text': cleaned, 'source': '/log', 'sha256': docs.digest(raw.encode()), 'missing': False}})
    output = docs.render_html(m)
    assert '<script>' not in output and '&lt;script&gt;' in output
    assert '\x1b' not in output
    plan = docs.feishu_plan(m)
    code = [b for b in appended(plan) if b['block_type'] == 14]
    # 连续块契约：一个日志=一个代码块，元素内切分拼接零丢失（用户
    # 2026-10-26 澄清——8k 不再是产品级分块）。
    assert len(code) == 1
    assert ''.join(e['text_run']['content']
                   for e in code[0]['code']['elements']) == cleaned
    assert all(len(e['text_run']['content'])
               <= docs.TEXT_ELEMENT_LIMIT for e in code[0]['code']['elements'])
    assert 'local_unpublished' == plan['status']
    assert 'block_id' not in json.dumps(plan)


def test_zero_instances_and_missing_logs_stay_explicit():
    row = row_for([item(counts={'pass': 0, 'fail': 0, 'error': 0, 'skip': 0})])
    model = docs.build_report(row)
    assert model['metrics']['No Result'] == 1
    assert model['declared_no_result_count'] == 0
    assert model['tests'][0]['log']['missing'] is True
    assert any('四计数为零' in w for w in model['warnings'])


def _real_submission(config, router, paths, **kwargs):
    sub = make_submission(**kwargs)
    for fr in sub.function_results:
        (Path(sub.pr_log_source_dir) / baseline_log_filename(fr.test_function, fr.device)).write_text(fr.log_content, encoding='utf-8')
    submit_complete_result(config, sub, router)
    return sub


def test_generation_is_read_only_idempotent_and_keeps_old_versions(config, router, paths):
    sub = _real_submission(config, router, paths)
    db_before = paths.pr_db.read_bytes()
    first = docs.generate(config, sub.pr_number)
    html = Path(first['html']); before = html.stat().st_mtime_ns
    second = docs.generate(config, sub.pr_number)
    assert first['changed'] and not second['changed']
    assert html.stat().st_mtime_ns == before and paths.pr_db.read_bytes() == db_before
    _real_submission(config, router, paths, pytorch_commit=COMMIT_B)
    third = docs.generate(config, sub.pr_number)
    assert third['fingerprint'] != first['fingerprint'] and html.exists()
    assert json.loads((html.parent.parent / 'current.json').read_text())['fingerprint'] == third['fingerprint']


def test_loader_does_not_consult_newer_baseline(config, router, paths):
    sub = _real_submission(config, router, paths)
    # A subsequent PR refreshes the live baseline, but the old report remains first-baseline.
    _real_submission(config, router, paths, pr_number=555, pytorch_commit=COMMIT_B)
    model = docs.load_report(config, sub.pr_number)
    assert all('新建' in t['comparison_text'] for t in model['tests'])


def test_rerun_during_render_cannot_replace_current_report(config, router, paths, monkeypatch):
    sub = _real_submission(config, router, paths)
    first = docs.generate(config, sub.pr_number)
    original = docs.render_html
    def race(model):
        _real_submission(config, router, paths, pytorch_commit=COMMIT_B)
        return original(model)
    monkeypatch.setattr(docs, 'render_html', race)
    with pytest.raises(docs.ReportError, match='生成期间'):
        docs.generate(config, sub.pr_number)
    current = Path(first['version']).parent / 'current.json'
    assert json.loads(current.read_text())['fingerprint'] == first['fingerprint']


def test_instance_identity_is_checked_even_when_counts_match():
    row = row_for([item(instance_ids=['case[param_a]'])])
    archive = archive_for(row)
    archive['comparisons'][0]['after']['instance_ids'] = ['case[param_b]']
    with pytest.raises(docs.ReportError, match='执行实例'):
        docs.build_report(row, archive)


def test_loader_refuses_outside_paths(config, router, paths):
    sub = _real_submission(config, router, paths)
    with sqlite3.connect(paths.pr_db) as conn:
        conn.execute('UPDATE pr_results SET log_path=?', ('../outside',))
    with pytest.raises(docs.ReportError, match='越出'): docs.load_report(config, sub.pr_number)


def test_scan_failure_is_per_pr_and_does_not_change_results(config, router, paths):
    _real_submission(config, router, paths)
    sub = _real_submission(config, router, paths, pr_number=555)
    file = paths.pr_logs_root / 'PR_555' / 'baseline-diff.json'
    bad = json.loads(file.read_text()); bad['pr_number'] = 999
    file.write_text(json.dumps(bad))
    report = docs.scan(config)
    assert report['checked'] == 2 and report['generated'] == 1
    assert report['errors'][0]['pr_number'] == sub.pr_number


def test_recovered_feishu_layout_contract():
    fixture = json.loads((Path(__file__).parent / 'fixtures/feishu-report-contract.json').read_text(encoding='utf-8'))
    for sample in fixture['samples']:
        row = row_for(sample['tests']); row['pr_number'] = sample['pr_number']
        model = docs.build_report(row)
        assert model['metrics'] == sample['metrics']
        plan = docs.feishu_plan(model)
        blocks = appended(plan)
        assert blocks[0]['block_type'] == sample['pr_heading_type']
        assert blocks[0]['heading3']['elements'][0]['text_run']['text_element_style']['link']['url'] == model['pr_url']
        table = next(op for op in plan['operations'] if op['op'] == 'table')
        assert table['body']['children'][0]['table']['property'] == {'row_size': 2, 'column_size': 6}
        assert [text_of(c[0]) for c in table['cell_children']] == list(docs.HEADERS) + [str(sample['metrics'][h]) for h in docs.HEADERS]
        details = [b for b in blocks if b['block_type'] == 6 and b['heading4']['style']['folded']]
        assert len(details) == sample['detail_count']
        assert all(b['heading4']['style']['folded'] for b in details)


def test_cli_exposes_local_render():
    from run_test.cli import build_parser
    args = build_parser().parse_args(['report', 'render', '--pr', '191432'])
    assert args.pr == 191432 and args.report_cmd == 'render'


def test_backfill_skips_reported_pr_without_fingerprint_drift(config, router, paths, monkeypatch):
    """已有报告的 PR：默认回填跳过（needs_refresh），DB 不动、指纹不变、
    不产生重传；--refresh-reports 受控回填并再生成（阻塞项 5）。"""
    import run_test.upstream as up
    from run_test.record import backfill_merged_at
    _real_submission(config, router, paths)
    first = docs.generate(config, 123456)
    _FakeMirror.commit_times = {COMMIT_A: "2026-10-26T01:00:00+00:00"}
    stats = backfill_merged_at(config)
    assert stats["needs_refresh"] == [123456] and stats["pr_backfilled"] == 0
    second = docs.generate(config, 123456)
    assert second["fingerprint"] == first["fingerprint"]  # 零指纹漂移
    stats = backfill_merged_at(config, refresh_reports=True)
    assert stats["pr_backfilled"] == 1 and stats["reports_refreshed"] == [123456]
    third = docs.generate(config, 123456)
    assert third["fingerprint"] != first["fingerprint"]  # 受控刷新生效


def test_newer_then_older_submission_stays_renderable(config, router, paths):
    """较新 PR 先落基线，较旧 PR 提交被防护跳过：基线行与日志不动，
    旧 PR 的报告仍可渲染发布（档案元数据完整，不误判快照不符）。"""
    _real_submission(config, router, paths, pr_number=555, pytorch_commit=COMMIT_B,
                     merged_at="2026-10-26T05:00:00Z",
                     merged_at_provenance="upstream-main-commit-time")
    baseline_before = _baseline_bytes(config)
    _real_submission(config, router, paths, pr_number=556, pytorch_commit=COMMIT_A,
                     merged_at="2026-10-26T01:00:00Z",
                     merged_at_provenance="upstream-main-commit-time")
    assert _baseline_bytes(config) == baseline_before  # 行与日志字节级不变
    result = docs.generate(config, 556)  # 旧 PR 仍可渲染（此前抛 ReportError）
    assert result["fingerprint"]
    archive = json.loads((store_paths(config).pr_logs_root / "PR_556" /
                          "baseline-diff.json").read_text(encoding="utf-8"))
    skipped = [c for c in archive["comparisons"] if c["baseline_action"] == "merge_order_skipped"]
    assert skipped and skipped[0]["after"]["instance_ids"]  # 元数据完整


def _baseline_bytes(config):
    from run_test.baseline_store import baseline_db_path, BaselineRouter
    from run_test.db import connect
    router = BaselineRouter(config.get("baseline_dirs", []), store_paths(config).baselines_root)
    rows = []
    for d in router.dirs:
        conn = connect(baseline_db_path(router.managed_root(d)))
        try:
            rows.extend(conn.execute("SELECT * FROM baseline ORDER BY test_function").fetchall())
        finally:
            conn.close()
    return rows
