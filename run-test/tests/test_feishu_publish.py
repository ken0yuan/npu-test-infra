from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import io
from pathlib import Path
from types import SimpleNamespace
import urllib.parse
import urllib.error

import pytest

from run_test.feishu_publish import Feishu, Publisher, PublishError, UncertainWrite, credentials, block_text
from run_test import result_docs
from run_test.config import store_paths
from run_test import result_docs as result_docs_module
from run_test.result_docs import build_report, feishu_plan
from run_test.identifiers import Counts
from test_record_flow import FakeMirror
from conftest import make_submission
from test_result_docs import row_for, item


class FakeFeishu:
    """Stateful API substitute: folders, real block IDs, auto-created table cells."""
    def __init__(self):
        self.entries = {'target': []}
        self.documents = {}
        self.serial = 0
        self.mutations = 0
        self.lose_response = False
        self.fail_before = False
        self.moves = []
        self.child_batches = []
        self.server_time = datetime(2026, 10, 28, 4, tzinfo=timezone.utc)
        self.capacity = None

    def trusted_utc_now(self):
        if self.server_time is None:
            raise PublishError('飞书 HTTPS 响应缺少有效 Date，无法确定北京时间上传日期；未使用服务器时间')
        return self.server_time

    def check_capacity(self, folder):
        if self.capacity is not None and len(self.entries[folder]) >= self.capacity:
            raise PublishError('folder capacity reached')

    def ident(self):
        self.serial += 1
        return 'id' + str(self.serial)

    def files(self, folder): return deepcopy(self.entries[folder])

    def blocks(self, doc): return deepcopy(list(self.documents[doc].values()))

    def children(self, doc, parent):
        return [deepcopy(self.documents[doc][b]) for b in self.documents[doc][parent].get('children', [])]

    def call(self, method, url, body=None):
        self.mutations += 1
        path = urllib.parse.urlsplit(url).path
        if path.endswith('/create_folder'):
            self.check_capacity(body['folder_token'])
            token = self.ident();self.entries[token] = []
            self.entries[body['folder_token']].append({'token': token, 'name': body['name'], 'type': 'folder'})
            return {'token': token}
        if path == '/docx/v1/documents':
            self.check_capacity(body['folder_token'])
            doc = self.ident();self.documents[doc] = {doc: {'block_id': doc, 'block_type': 1, 'children': []}}
            self.entries[body['folder_token']].append({'token': doc, 'name': body['title'], 'type': 'docx'})
            return {'document': {'document_id': doc}}
        if path.endswith('/move'):
            doc = path.split('/')[-2]
            self.check_capacity(body['folder_token'])
            for entries in self.entries.values():
                found = next((x for x in entries if x['token'] == doc), None)
                if found:
                    entries.remove(found);break
            else: raise PublishError('missing file')
            self.entries[body['folder_token']].append(found)
            self.moves.append((doc, body['folder_token']))
            return {}
        if path.endswith('/children'):
            self.child_batches.append(len(body['children']))
            if self.fail_before:
                self.fail_before = False
                raise PublishError('known rejected write')
            doc, parent = path.split('/')[4], path.split('/')[6]
            nodes = self.documents[doc];created = []
            for source in body['children']:
                b = deepcopy(source);b['block_id'] = self.ident();b['parent_id'] = parent
                if b['block_type'] == 31:
                    b['table']['cells'] = []
                    for _ in range(b['table']['property']['row_size'] * b['table']['property']['column_size']):
                        cell, empty = self.ident(), self.ident()
                        b['table']['cells'].append(cell)
                        nodes[cell] = {'block_id': cell, 'block_type': 32, 'children': [empty]}
                        nodes[empty] = {'block_id': empty, 'block_type': 2, 'text': {'elements': []}}
                nodes[b['block_id']] = b;nodes[parent].setdefault('children', []).append(b['block_id']);created.append(deepcopy(b))
            if self.lose_response:
                self.lose_response = False
                raise UncertainWrite('response lost after server committed')
            return {'children': created}
        raise AssertionError(path)


def inputs():
    model = build_report(row_for([item(), item('test_missing', counts=None, status='no_result', reason='collection_error')]))
    return {'pr_number': 77, 'fingerprint': 'a' * 64}, feishu_plan(model)


def test_end_to_end_verify_promote_and_preserve_old_document(tmp_path):
    api = FakeFeishu();_prepare_destination(api).append({'token': 'legacy', 'name': 'PR#77', 'type': 'docx'})
    p, plan = inputs();worker = Publisher(api, 'target', tmp_path)
    receipt = worker.publish(p, plan)
    assert receipt['verification']['content_verified'] and receipt['archived'] == ['legacy']
    assert {f['token'] for f in api.files(receipt['destination_folder_token']) if f['type'] == 'docx'} == {p['document_id'] for p in receipt['parts']}
    assert any(f['token'] == 'legacy' for entries in api.entries.values() for f in entries)
    before = api.mutations
    again = Publisher(api, 'target', tmp_path).publish(p, plan)
    assert not again['changed'] and api.mutations == before


def test_timeout_after_append_is_reconciled_without_duplicate(tmp_path):
    api = FakeFeishu();p, plan = inputs();api.lose_response = True
    with pytest.raises(UncertainWrite): Publisher(api, 'target', tmp_path).publish(p, plan)
    receipt = Publisher(api, 'target', tmp_path).publish(p, plan)
    assert all(p['verification']['content_verified'] for p in receipt['parts'])
    assert len(api.documents) == plan['part_count']
    clean = FakeFeishu()
    reference = Publisher(clean, 'target', tmp_path / 'reference').publish(p, plan)
    assert [p['verification'] for p in receipt['parts']] == [p['verification'] for p in reference['parts']]


def test_known_rejection_can_retry_without_abandoning_staging(tmp_path):
    api = FakeFeishu();p, plan = inputs();api.fail_before = True
    with pytest.raises(PublishError): Publisher(api, 'target', tmp_path).publish(p, plan)
    receipt = Publisher(api, 'target', tmp_path).publish(p, plan)
    assert receipt['verification']['content_verified']


def test_ambiguous_create_does_not_create_twice(tmp_path):
    api = FakeFeishu();p, plan = inputs();worker = Publisher(api, 'target', tmp_path)
    original = api.call
    def uncertain(method, path, body=None):
        if path == '/docx/v1/documents':
            # No evidence of whether this write was accepted: no blind retry.
            raise UncertainWrite('timeout')
        return original(method, path, body)
    api.call = uncertain
    with pytest.raises(UncertainWrite):worker.publish(p, plan)
    before = api.mutations
    with pytest.raises(UncertainWrite):Publisher(api, 'target', tmp_path).publish(p, plan)
    assert api.mutations == before


def test_unknown_write_no_remote_evidence_requires_review(tmp_path):
    api = FakeFeishu();p, plan = inputs();worker = Publisher(api, 'target', tmp_path)
    api.lose_response = True
    with pytest.raises(UncertainWrite):worker.publish(p, plan)
    doc = worker.state['document_id']
    api.documents[doc][doc]['children'] = []
    before = api.mutations
    with pytest.raises(UncertainWrite):Publisher(api, 'target', tmp_path).publish(p, plan)
    assert api.mutations == before


def test_stale_result_never_replaces_old_report(tmp_path):
    api = FakeFeishu();p, plan = inputs()
    api.entries['target'].append({'token': 'old', 'name': 'PR#77', 'type': 'docx'})
    with pytest.raises(PublishError, match='本地结果已更新'):
        Publisher(api, 'target', tmp_path).publish(p, plan, still_current=lambda: False)
    assert not api.moves
    assert any(f['token'] == 'old' for f in api.files('target'))


def test_remote_text_corruption_fails_before_promotion(tmp_path):
    api = FakeFeishu();p, plan = inputs();original = api.blocks
    def corrupted(doc):
        blocks = original(doc)
        next(b for b in blocks if b['block_type'] == 5)['heading3']['elements'][0]['text_run']['content'] = 'wrong PR'
        return blocks
    api.blocks = corrupted
    with pytest.raises(PublishError, match='回读'):
        Publisher(api, 'target', tmp_path).publish(p, plan)
    assert not api.moves


def test_configuration_is_selective_and_does_not_echo_secret(tmp_path):
    path = tmp_path / '.env'
    path.write_text('FEISHU_APP_ID=example\nFEISHU_APP_SECRET=never-print\nFEISHU_TEST_FOLDER_TOKEN=target\nFEISHU_DOMAIN=tenant.feishu.cn\nUNRELATED_SECRET=ignored\n')
    result = credentials(path)
    assert 'UNRELATED_SECRET' not in result and result['FEISHU_DOMAIN'] == 'https://tenant.feishu.cn'
    path.write_text(path.read_text().replace('tenant.feishu.cn', 'evil.example'))
    with pytest.raises(PublishError) as e: credentials(path)
    assert 'never-print' not in str(e.value)


# ---------------------------------------------------------------------------
# 多文档拆分与完整版本集发布（用户 2026-10-26）
# ---------------------------------------------------------------------------

def _v2_plan(pr=77, parts=2, title='PR#77'):
    """构造 v2 多篇计划：每篇一个 append 操作（含可校验文本）。"""
    def ops(tag, size):
        return [{'op': 'append', 'parent': 'document',
                 'body': {'index': -1,
                          'children': [result_docs.text_block(
                              'text', f'{tag}-' * 10 + f'[{size}]')]}}]
    parts_list = [{'part_index': 0, 'role': 'main', 'title_suffix': '',
                   'operations': ops('main', 0) + ([{'op': 'toc',
                                                     'titles': [f'{title}-1', f'{title}-2'][:parts - 1]}]
                                                    if parts > 1 else [])}]
    for i in range(1, parts):
        parts_list.append({'part_index': i, 'role': 'continuation',
                           'title_suffix': f'-{i}', 'operations': ops(f'cont{i}', i)})
    return {'schema': 'feishu-report-plan/2', 'title': title,
            'status': 'local_unpublished', 'part_count': parts, 'parts': parts_list,
            'operations': [op for p in parts_list for op in p['operations']]}


def _doc_names(api, folder):
    return sorted(f['name'] for f in api.files(folder) if f['type'] == 'docx')


def _destination(api, pr=77, date='2026-10-28'):
    day = next(f['token'] for f in api.files('target') if f['name'] == date)
    return next(f['token'] for f in api.files(day) if f['name'] == f'PR#{pr}')


def _prepare_destination(api, pr=77, date='2026-10-28'):
    """Existing dated reports, independent of the publisher being tested."""
    day, folder = 'day-' + date, f'pr-{date}-{pr}'
    if day not in api.entries:
        api.entries['target'].append({'token': day, 'name': date, 'type': 'folder'})
        api.entries[day] = []
    api.entries[day].append({'token': folder, 'name': f'PR#{pr}', 'type': 'folder'})
    api.entries[folder] = []
    return api.entries[folder]


def test_multi_part_publish_verifies_links_and_archives_old_continuations(tmp_path):
    api = FakeFeishu()
    old = _prepare_destination(api)
    old.append({'token': 'legacy-main', 'name': 'PR#77', 'type': 'docx'})
    old.append({'token': 'legacy-cont', 'name': 'PR#77-1', 'type': 'docx'})
    p, plan = {'pr_number': 77, 'fingerprint': 'b' * 64}, _v2_plan(parts=2)
    receipt = Publisher(api, 'target', tmp_path, domain='https://x').publish(p, plan)
    names = _doc_names(api, receipt['destination_folder_token'])
    assert names == ['PR#77', 'PR#77-1']  # 新版本集完整替换（含旧续篇归档）
    assert receipt['changed'] and len(receipt['parts']) == 2
    assert all(v['content_verified'] for v in
               [receipt['verification']] + [p_['verification'] for p_ in receipt['parts']])
    assert sorted(receipt['archived']) == ['legacy-cont', 'legacy-main']
    # 主篇 TOC 含续篇链接
    main_doc = receipt['document_id']
    toc_blocks = [b for b in api.blocks(main_doc)
                  if b.get('text', {}).get('elements')
                  and any(e.get('text_run', {}).get('text_element_style', {}).get('link')
                          for e in b['text']['elements'])]
    assert toc_blocks  # 续篇链接已写入主篇


def test_multi_part_idempotent_rescan(tmp_path):
    api = FakeFeishu()
    p, plan = {'pr_number': 77, 'fingerprint': 'c' * 64}, _v2_plan(parts=2)
    first = Publisher(api, 'target', tmp_path).publish(p, plan)
    assert first['changed']
    before = api.mutations
    again = Publisher(api, 'target', tmp_path).publish(p, plan)
    assert not again['changed'] and api.mutations == before  # 幂等：零 API 调用


def test_part_count_change_archives_old_continuation(tmp_path):
    api = FakeFeishu()
    p1 = {'pr_number': 77, 'fingerprint': 'd' * 64}
    Publisher(api, 'target', tmp_path).publish(p1, _v2_plan(parts=2))
    p2 = {'pr_number': 77, 'fingerprint': 'e' * 64}
    Publisher(api, 'target', tmp_path).publish(p2, _v2_plan(parts=1))
    assert _doc_names(api, _destination(api)) == ['PR#77']  # 多篇变单篇：旧续篇一并归档


def test_incomplete_publish_stays_staging_then_completes(tmp_path):
    api = FakeFeishu()
    p, plan = {'pr_number': 77, 'fingerprint': 'f' * 64}, _v2_plan(parts=2)
    api.lose_response = True  # 续篇 append 中断
    with pytest.raises(UncertainWrite):
        Publisher(api, 'target', tmp_path).publish(p, plan)
    state = json.loads((tmp_path / 'target' / 'PR_77' / (p['fingerprint'] + '.json')).read_text())
    assert state['status'] == 'staging'  # 未完整发布绝不记 published
    receipt = Publisher(api, 'target', tmp_path).publish(p, plan)  # 断点续传
    assert receipt['changed'] and len(receipt['parts']) == 2
    assert _doc_names(api, _destination(api)) == ['PR#77', 'PR#77-1']


def test_big_log_planner_to_publisher_split_without_loss(tmp_path):
    """真实规划→发布链路：700k 字符日志按 500k 文档预算拆篇，跨篇全文
    零丢失、续篇导航齐备、传输批量 ≤40 块。"""
    import run_test.result_docs as rd
    log = ('L中\U0001f680\n' * 175_000) + 'END'
    model = {
        'pr_number': 1, 'pr_url': 'http://x/1', 'title': 'PR#1',
        'pytorch_commit': 'a' * 40,
        'torch_npu_environment': {'location': 'x', 'torch_npu_commit': 'b' * 40},
        'merged_at': '2026-10-26T01:00:00Z',
        'metrics': {h: 0 for h in rd.HEADERS},
        'metric_note': 'n', 'instance_counts': {'pass': 0, 'skip': 0, 'fail': 0, 'error': 0},
        'warnings': [],
        'tests': [{'outcome': 'pass', 'test_function': 't/big.py::test_huge',
                   'device': 'cpu', 'comparison_text': 'c',
                   'counts': {'pass': 1, 'skip': 0, 'fail': 0, 'error': 0}, 'log': {'text': log}}],
    }
    plan = rd.feishu_plan(model, {'doc_max_blocks': 50, 'doc_max_chars': 500_000})
    assert plan['part_count'] >= 2  # 超预算测试拆到多篇
    # 跨篇零丢失：所有 code 元素拼接 == 本地日志全文
    code_text = ''.join(e['text_run']['content']
                        for p in plan['parts'] for op in p['operations']
                        if op.get('body') for b in op['body']['children']
                        if b['block_type'] == 14
                        for e in b['code']['elements'])
    assert code_text == log
    # 传输批量 ≤40
    assert all(len(op['body']['children']) <= 40
               for p in plan['parts'] for op in p['operations'] if op.get('body'))
    api = FakeFeishu()
    pointer = {'pr_number': 1, 'fingerprint': 'a' * 64}
    receipt = Publisher(api, 'target', tmp_path, domain='https://x').publish(pointer, plan)
    assert receipt['changed'] and len(receipt['parts']) == plan['part_count']
    # 发布端传输同样 ≤40 块/请求（FakeFeishu 记录的 children POST）
    assert max(api.child_batches) <= 40
    for part in receipt['parts']:
        actual = api.blocks(part['document_id'])
        assert len(actual) <= 50
        assert sum(rd.text_units(block_text(b)) for b in actual) <= 500_000
        assert all(rd.text_units(block_text(b)) <= rd.BLOCK_TEXT_LIMIT for b in actual)
    # 跨篇回读拼接零丢失（含续篇导航块外的代码内容）
    joined = ''
    for part in receipt['parts']:
        for b in api.blocks(part['document_id']):
            if b['block_type'] == 14:
                joined += ''.join(e['text_run']['content'] for e in b['code']['elements'])
    assert joined == log
    # 续篇含返回主篇链接
    cont = receipt['parts'][1]
    nav = [b for b in api.blocks(cont['document_id'])
           if b.get('text', {}).get('elements')
           and any(e.get('text_run', {}).get('text_element_style', {}).get('link', {}).get('url', '').endswith('/docx/' + receipt['document_id'])
                   for e in b['text']['elements'])]
    assert nav


def test_generate_uses_configured_budget(config, router, paths, monkeypatch):
    """generate 把 settings.feishu 预算传给规划器（阻塞项 4）。"""
    import run_test.upstream as up
    from run_test.record import submit_complete_result
    monkeypatch.setattr(up, 'PyTorchMirror', FakeMirror)
    log = 'L' * 60_000
    from conftest import TF_A
    sub = make_submission(items=[(TF_A, "cpu", Counts(1, 0, 0, 0))])
    sub.function_results[0].log_content = log
    # 基线日志必须预置在 PR 日志源目录（_real_submission 同款）：
    # load_report 从 pr_logs/PR_n/ 读取，缺失时模型日志为占位符。
    from run_test.identifiers import baseline_log_filename
    for fr in sub.function_results:
        (Path(sub.pr_log_source_dir) /
         baseline_log_filename(fr.test_function, fr.device)).write_text(log, encoding='utf-8')
    submit_complete_result(config, sub, router)
    cfg = {**config, 'reports': {**config.get('reports', {}), 'feishu': {'doc_max_chars': 20_000, 'doc_max_blocks': 50}}}
    result_docs_module.generate(cfg, 123456)
    plan = json.loads(next((store_paths(cfg).data_root / 'reports')
                           .glob('PR_123456/*/feishu-plan.json')).read_text(encoding='utf-8'))
    assert plan['part_count'] >= 2  # 20k 预算下 60k 日志必然拆篇
    assert plan['budget']['doc_max_chars'] == 20_000
    all_ops = [op for p in plan['parts'] for op in p['operations']]
    code_blocks = [b for op in all_ops if op.get('body')
                   for b in op['body']['children'] if b['block_type'] == 14]
    assert code_blocks  # 连续块存在（一篇或跨篇）


def _model_with_logs(size=20000, count=1):
    model = build_report(row_for([item('test_' + str(i)) for i in range(count)]))
    for test in model['tests']:
        test['log']['text'] = ('中\U0001f680ab\n' * (size // 5 + 1))[:size]
    return model


def test_normal_log_has_one_visible_code_block():
    model = _model_with_logs()
    plan = feishu_plan(model)
    code = [b for op in plan['operations'] for b in op.get('body', {}).get('children', [])
            if b['block_type'] == 14]
    assert len(code) == 1 and block_text(code[0]) == model['tests'][0]['log']['text']


def test_old_published_receipt_does_not_republish_on_upgrade(tmp_path):
    api = FakeFeishu(); pointer, plan = inputs()
    legacy = {'title': plan['title'], 'operations': [{'op': 'append', 'body': {
        'index': -1, 'children': [result_docs.text_block('text', 'original published report')]}}]}
    before = Publisher(api, 'target', tmp_path).publish(pointer, legacy)
    mutations = api.mutations
    after = Publisher(api, 'target', tmp_path).publish(pointer, plan)
    assert not after['changed'] and after['document_id'] == before['document_id']
    assert api.mutations == mutations
    assert not list(tmp_path.rglob('*.superseded-*.json'))


def test_old_staging_lost_response_is_isolated_during_upgrade(tmp_path):
    api = FakeFeishu(); pointer, plan = inputs()
    legacy = {'title': plan['title'], 'operations': [{'op': 'append', 'body': {
        'index': -1, 'children': [result_docs.text_block('text', 'old staging content')]}}]}
    api.lose_response = True
    with pytest.raises(UncertainWrite):
        Publisher(api, 'target', tmp_path).publish(pointer, legacy)
    old = deepcopy(api.documents)
    receipt = Publisher(api, 'target', tmp_path).publish(pointer, plan)
    assert all(p['verification']['content_verified'] for p in receipt['parts'])
    assert set(old).isdisjoint(p['document_id'] for p in receipt['parts'])
    assert all(api.documents[doc] == contents for doc, contents in old.items())
    assert len(list(tmp_path.rglob('*.superseded-*.json'))) == 1
    mutations = api.mutations
    assert not Publisher(api, 'target', tmp_path).publish(pointer, plan)['changed']
    assert api.mutations == mutations


def test_changed_staging_budget_does_not_reuse_partial_documents(tmp_path):
    api = FakeFeishu(); model = _model_with_logs(60000)
    pointer = {'pr_number': 77, 'fingerprint': '1' * 64}
    first = feishu_plan(model)
    api.lose_response = True
    with pytest.raises(UncertainWrite):
        Publisher(api, 'target', tmp_path).publish(pointer, first)
    old = deepcopy(api.documents)
    receipt = Publisher(api, 'target', tmp_path).publish(pointer, feishu_plan(model, {'doc_max_chars': 20000}))
    assert set(old).isdisjoint(p['document_id'] for p in receipt['parts'])
    assert all(api.documents[doc] == contents for doc, contents in old.items())


def test_partial_promotion_keeps_old_set_until_all_parts_are_promoted(tmp_path):
    api = FakeFeishu(); _prepare_destination(api).append({'token': 'old', 'name': 'PR#77', 'type': 'docx'})
    pointer, plan = {'pr_number': 77, 'fingerprint': '2' * 64}, _v2_plan(parts=3)
    original = api.call
    def reject_second_move(method, url, body=None):
        if url.endswith('/move') and len(api.moves) == 1:
            raise PublishError('second move rejected')
        return original(method, url, body)
    api.call = reject_second_move
    with pytest.raises(PublishError, match='second move'):
        Publisher(api, 'target', tmp_path).publish(pointer, plan)
    assert any(f['token'] == 'old' for f in api.files(_destination(api)))
    state = json.loads((tmp_path/'target/PR_77'/ (pointer['fingerprint']+'.json')).read_text(encoding='utf-8'))
    assert state['status'] == 'staging'
    api.call = original
    receipt = Publisher(api, 'target', tmp_path).publish(pointer, plan)
    assert receipt['archived'] == ['old'] and len(receipt['parts']) == 3


@pytest.mark.parametrize('count,budget', [(70, {'doc_max_blocks': 50, 'doc_max_chars': 20000}),
                                         (2, {'doc_max_blocks': 42, 'doc_max_chars': 2048})])
def test_actual_complete_documents_respect_budgets(tmp_path, count, budget):
    model = _model_with_logs(3000, count)
    plan = feishu_plan(model, budget)
    api = FakeFeishu()
    receipt = Publisher(api, 'target', tmp_path).publish({'pr_number': 77, 'fingerprint': '3'*64}, plan)
    recovered = []
    for part in receipt['parts']:
        blocks = api.blocks(part['document_id'])
        assert len(blocks) <= budget['doc_max_blocks']
        assert sum(result_docs.text_units(block_text(b)) for b in blocks) <= budget['doc_max_chars']
        recovered.extend(block_text(b) for b in blocks if b['block_type'] == 14)
    assert ''.join(recovered) == ''.join(t['log']['text'] for t in model['tests'])
    assert max(api.child_batches) <= 40


@pytest.mark.parametrize('budget', [{'doc_max_blocks': 1}, {'doc_max_chars': 100},
                                  {'block_max_chars': 100001}, {'doc_max_chars': True}])
def test_invalid_document_capacity_fails_before_any_publish(budget):
    with pytest.raises(result_docs.ReportError):
        feishu_plan(_model_with_logs(), budget)


@pytest.mark.parametrize('stamp,day', [
    ('2026-10-28T15:59:59+00:00', '2026-10-28'),
    ('2026-10-28T16:00:00+00:00', '2026-10-29'),
    ('2026-12-31T16:00:00+00:00', '2027-01-01'),
])
def test_upload_date_uses_trusted_utc_plus_eight(tmp_path, stamp, day):
    api = FakeFeishu(); api.server_time = datetime.fromisoformat(stamp)
    receipt = Publisher(api, 'target', tmp_path).publish(
        {'pr_number': 77, 'fingerprint': '4' * 64}, _v2_plan(parts=1))
    assert receipt['upload_date'] == day and receipt['timezone'] == 'Asia/Shanghai'
    assert receipt['publication_started_at_utc'] == stamp
    assert _destination(api, date=day) == receipt['destination_folder_token']
    assert _doc_names(api, receipt['destination_folder_token']) == ['PR#77']


def test_upload_date_survives_midnight_and_missing_clock_on_retry(tmp_path):
    api = FakeFeishu(); api.server_time = datetime(2026, 10, 28, 15, 59, 59, tzinfo=timezone.utc)
    api.lose_response = True
    pointer, plan = {'pr_number': 77, 'fingerprint': '5' * 64}, _v2_plan(parts=2)
    with pytest.raises(UncertainWrite):
        Publisher(api, 'target', tmp_path).publish(pointer, plan)
    # No wall-clock fallback is needed: the journal has pinned the original date.
    api.server_time = None
    receipt = Publisher(api, 'target', tmp_path).publish(pointer, plan)
    assert receipt['upload_date'] == '2026-10-28'
    assert len(receipt['parts']) == 2
    assert [f['name'] for f in api.files('target') if f['type'] == 'folder'] == [
        '2026-10-28', '报告上传暂存']


def test_missing_platform_time_stops_before_cloud_writes(tmp_path):
    api = FakeFeishu(); api.server_time = None
    with pytest.raises(PublishError, match='未使用服务器时间'):
        Publisher(api, 'target', tmp_path).publish(
            {'pr_number': 77, 'fingerprint': '6' * 64}, _v2_plan(parts=1))
    assert api.mutations == 0 and api.files('target') == []


@pytest.mark.parametrize('date_header', [None, 'invalid', 'Wed, 28 Oct 2026 16:00:00'])
def test_http_clock_rejects_missing_invalid_or_naive_date(monkeypatch, date_header):
    response = io.BytesIO(b'{"data": {}}')
    response.headers = {} if date_header is None else {'Date': date_header}
    api = Feishu({}); api.opener = SimpleNamespace(open=lambda *args, **kwargs: response)
    monkeypatch.setattr('run_test.feishu_publish.time.sleep', lambda _: None)
    api.call('GET', '/drive/v1/files', auth=False)
    with pytest.raises(PublishError, match='未使用服务器时间'):
        api.trusted_utc_now()


def test_http_clock_advances_only_by_monotonic_time(monkeypatch):
    elapsed = [10.0]
    monkeypatch.setattr('run_test.feishu_publish.time.monotonic', lambda: elapsed[0])
    monkeypatch.setattr('run_test.feishu_publish.time.sleep', lambda _: None)
    response = io.BytesIO(b'{"data": {}}')
    response.headers = {'Date': 'Wed, 28 Oct 2026 15:59:59 GMT'}
    api = Feishu({}); api.opener = SimpleNamespace(open=lambda *args, **kwargs: response)
    api.call('GET', '/drive/v1/files', auth=False)
    elapsed[0] += 2
    assert api.trusted_utc_now() == datetime(2026, 10, 28, 16, 0, 1, tzinfo=timezone.utc)


def test_reports_share_date_but_keep_pr_versions_isolated(tmp_path):
    api = FakeFeishu()
    first = Publisher(api, 'target', tmp_path).publish(
        {'pr_number': 77, 'fingerprint': '7' * 64}, _v2_plan(parts=2))
    other = Publisher(api, 'target', tmp_path).publish(
        {'pr_number': 78, 'fingerprint': '8' * 64}, _v2_plan(pr=78, title='PR#78', parts=1))
    updated = Publisher(api, 'target', tmp_path).publish(
        {'pr_number': 77, 'fingerprint': '9' * 64}, _v2_plan(parts=1))
    assert first['date_folder_token'] == other['date_folder_token']
    assert first['destination_folder_token'] != other['destination_folder_token']
    assert set(updated['archived']) == {p['document_id'] for p in first['parts']}
    assert _doc_names(api, other['destination_folder_token']) == ['PR#78']
    assert _doc_names(api, updated['destination_folder_token']) == ['PR#77']


def test_next_day_keeps_previous_day_complete_version(tmp_path):
    api = FakeFeishu()
    old = Publisher(api, 'target', tmp_path).publish(
        {'pr_number': 77, 'fingerprint': 'a1' * 32}, _v2_plan(parts=2))
    api.server_time += timedelta(days=1)
    new = Publisher(api, 'target', tmp_path).publish(
        {'pr_number': 77, 'fingerprint': 'b1' * 32}, _v2_plan(parts=1))
    assert new['archived'] == []
    assert old['date_folder_token'] != new['date_folder_token']
    assert _doc_names(api, old['destination_folder_token']) == ['PR#77', 'PR#77-1']


def test_legacy_published_receipt_needs_no_clock_or_cloud_access(tmp_path):
    pointer, plan = {'pr_number': 77, 'fingerprint': 'c1' * 32}, _v2_plan(parts=1)
    path = tmp_path / 'target' / 'PR_77' / (pointer['fingerprint'] + '.json')
    result_docs._atomic(path, json.dumps({'status': 'published', 'receipt': {
        'document_id': 'legacy-doc', 'pr_number': 77, 'fingerprint': pointer['fingerprint']}}))
    class NoCloud:
        def __getattr__(self, name):
            raise AssertionError('old published receipt must not access cloud')
    receipt = Publisher(NoCloud(), 'target', tmp_path).publish(pointer, plan)
    assert not receipt['changed'] and receipt['document_id'] == 'legacy-doc'


def test_old_partial_promoted_set_resumes_into_dated_folder_without_reupload(tmp_path):
    api = FakeFeishu()
    pointer, plan = {'pr_number': 77, 'fingerprint': 'd1' * 32}, _v2_plan(parts=3)
    first = Publisher(api, 'target', tmp_path).publish(pointer, plan)
    path = tmp_path / 'target' / 'PR_77' / (pointer['fingerprint'] + '.json')
    state = json.loads(path.read_text(encoding='utf-8'))
    docs = [p['document_id'] for p in first['parts']]
    original_doc_content = deepcopy(api.documents)
    for doc in docs[:2]:
        api.call('POST', f'/drive/v1/files/{doc}/move', {'folder_token': 'target'})
    api.call('POST', f'/drive/v1/files/{docs[2]}/move', {'folder_token': state['version_folder']})
    for key in list(state):
        if key.startswith(('publication_', 'upload_', 'date_folder_', 'destination_folder_')) or key == 'timezone':
            state.pop(key)
    state.update(status='staging', **{'promoted-0': True, 'promoted-1': True})
    state.pop('receipt', None)
    # Legacy journals did not bind the promotion receipt to a destination.
    for key in list(state):
        if key.startswith('promoted-') and key.endswith('_destination'):
            state.pop(key)
    state['promoted-1_pending'] = True
    result_docs._atomic(path, json.dumps(state))
    receipt = Publisher(api, 'target', tmp_path).publish(pointer, plan)
    assert {p['document_id'] for p in receipt['parts']} == set(docs)
    assert api.documents == original_doc_content
    assert _doc_names(api, 'target') == []
    assert _doc_names(api, receipt['destination_folder_token']) == ['PR#77', 'PR#77-1', 'PR#77-2']
    final = json.loads(path.read_text(encoding='utf-8'))
    assert not any(key.endswith('_pending') for key in final)


def test_pending_move_to_different_unknown_destination_is_not_retried(tmp_path):
    api = FakeFeishu(); api.entries['old-destination'] = []
    worker = Publisher(api, 'target', tmp_path)
    worker.path = tmp_path / 'journal.json'
    worker.state = {'promoted-0_pending': True, 'promoted-0_destination': 'old-destination'}
    with pytest.raises(UncertainWrite, match='原移动结果'):
        worker.move('promoted-0', 'unknown-doc', 'target')
    assert api.mutations == 0


def test_many_parts_do_not_consume_the_date_folders_capacity(tmp_path):
    api = FakeFeishu(); _prepare_destination(api, pr=78)
    day = next(f['token'] for f in api.files('target') if f['name'] == '2026-10-28')
    api.entries[day].extend({'token': f'other-{n}', 'name': f'PR#{1000+n}', 'type': 'folder'}
                           for n in range(1498))
    api.capacity = 1500
    receipt = Publisher(api, 'target', tmp_path).publish(
        {'pr_number': 77, 'fingerprint': 'e1' * 32}, _v2_plan(parts=3))
    assert len(api.files(day)) == 1500
    assert len(receipt['parts']) == 3
    assert _doc_names(api, receipt['destination_folder_token']) == ['PR#77', 'PR#77-1', 'PR#77-2']


@pytest.mark.parametrize('committed', [False, True])
def test_uncertain_promotion_recovers_without_reupload_or_duplicate(tmp_path, committed):
    api = FakeFeishu(); original = api.call; failed = False
    def flaky(method, url, body=None):
        nonlocal failed
        if url.endswith('/move') and len(api.moves) == 22 and not failed:
            failed = True
            if committed:
                original(method, url, body)
            raise UncertainWrite('HTTP 500')
        return original(method, url, body)
    api.call = flaky
    pointer, plan = {'pr_number': 77, 'fingerprint': 'f1' * 32}, _v2_plan(parts=30)
    with pytest.raises(UncertainWrite):
        Publisher(api, 'target', tmp_path).publish(pointer, plan)
    original_content = deepcopy(api.documents)
    receipt = Publisher(api, 'target', tmp_path).publish(pointer, plan)
    assert len(receipt['parts']) == len(api.documents) == len(api.moves) == 30
    assert api.documents == original_content
    assert {p['document_id'] for p in receipt['parts']} == {
        f['token'] for f in api.files(receipt['destination_folder_token'])}
    state = json.loads((tmp_path / 'target/PR_77' / (pointer['fingerprint'] + '.json')).read_text())
    assert state['status'] == 'published' and not any(k.endswith('_pending') for k in state)
    assert state.get('promoted-22_reconciled_retries', 0) == (0 if committed else 1)


def test_pending_move_without_positive_source_evidence_stays_pending(tmp_path):
    api = FakeFeishu(); api.entries['source'] = []
    worker = Publisher(api, 'target', tmp_path); worker.path = tmp_path / 'state.json'
    worker.state = {'move_pending': True, 'move_destination': 'target'}
    with pytest.raises(UncertainWrite, match='源目录未找到'):
        worker.move('move', 'missing-doc', 'target', source='source')
    assert not api.mutations and worker.state['move_pending']


@pytest.mark.parametrize('status', ['pending', 'processing', 'success', 'unexpected', 'fail'])
def test_pending_async_move_requires_failed_task_and_source_evidence(tmp_path, status):
    api = FakeFeishu(); api.entries['source'] = [{'token': 'doc', 'type': 'docx', 'name': 'report'}]
    original = api.call
    def call(method, url, body=None):
        if url.startswith('/drive/v1/files/task_check?'):
            assert method == 'GET' and urllib.parse.parse_qs(urllib.parse.urlsplit(url).query) == {'task_id': ['task-1']}
            return {'status': status}
        return original(method, url, body)
    api.call = call
    worker = Publisher(api, 'target', tmp_path); worker.path = tmp_path / 'state.json'
    worker.state = {'move_pending': True, 'move_destination': 'target', 'move_task_id': 'task-1'}
    if status == 'fail':
        worker.move('move', 'doc', 'target', source='source')
        assert api.moves == [('doc', 'target')] and worker.state['move']
    else:
        with pytest.raises(UncertainWrite, match='任务尚未确认失败'):
            worker.move('move', 'doc', 'target', source='source')
        assert not api.moves


def test_move_task_id_is_saved_before_waiting_and_reconciled_by_membership(tmp_path, monkeypatch):
    api = FakeFeishu(); api.entries['source'] = [{'token': 'doc', 'type': 'docx', 'name': 'report'}]
    api.call = lambda *args, **kwargs: {'task_id': 'task-1'}
    monkeypatch.setattr('run_test.feishu_publish.time.sleep', lambda _: None)
    worker = Publisher(api, 'target', tmp_path); worker.path = tmp_path / 'state.json'
    with pytest.raises(UncertainWrite, match='移动请求已提交'):
        worker.move('move', 'doc', 'target', source='source')
    saved = json.loads(worker.path.read_text())
    assert saved['move_task_id'] == 'task-1' and saved['move_pending']
    api.entries['target'] = api.entries.pop('source')
    api.call = lambda *args, **kwargs: pytest.fail('membership already confirms success')
    worker.move('move', 'doc', 'target', source='source')
    assert worker.state['move'] and 'move_pending' not in worker.state


@pytest.mark.parametrize('status,body,expected', [
    (400, b'{"code":1061045,"msg":"secret-response"}', 'code=1061045'),
    (500, b'{"code":1061001,"msg":"secret-response"}', 'code=1061001'),
    (500, b'{"code":"secret-response"}', ''),
    (500, b'secret-response', ''),
])
def test_http_error_keeps_only_numeric_business_code(monkeypatch, status, body, expected):
    api = Feishu({})
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError('https://open.feishu.cn', status, 'secret-header', {}, io.BytesIO(body))
    api.opener = SimpleNamespace(open=fail)
    monkeypatch.setattr('run_test.feishu_publish.time.sleep', lambda _: None)
    error_type = UncertainWrite if status == 500 else PublishError
    with pytest.raises(error_type) as caught:
        api.call('POST', '/drive/v1/files/doc/move', auth=False)
    assert f'HTTP {status}' in str(caught.value) and expected in str(caught.value)
    assert 'secret' not in str(caught.value)
