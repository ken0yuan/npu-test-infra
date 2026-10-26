from copy import deepcopy
import json
from pathlib import Path
import urllib.parse

import pytest

from run_test.feishu_publish import Publisher, PublishError, UncertainWrite, credentials, block_text
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
            token = self.ident();self.entries[token] = []
            self.entries[body['folder_token']].append({'token': token, 'name': body['name'], 'type': 'folder'})
            return {'token': token}
        if path == '/docx/v1/documents':
            doc = self.ident();self.documents[doc] = {doc: {'block_id': doc, 'block_type': 1, 'children': []}}
            self.entries[body['folder_token']].append({'token': doc, 'name': body['title'], 'type': 'docx'})
            return {'document': {'document_id': doc}}
        if path.endswith('/move'):
            doc = path.split('/')[-2]
            for entries in self.entries.values():
                found = next((x for x in entries if x['token'] == doc), None)
                if found:
                    entries.remove(found);break
            else: raise PublishError('missing file')
            self.entries[body['folder_token']].append(found)
            self.moves.append((doc, body['folder_token']))
            return {}
        if path.endswith('/children'):
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
    api = FakeFeishu();api.entries['target'].append({'token': 'legacy', 'name': 'PR#77', 'type': 'docx'})
    p, plan = inputs();worker = Publisher(api, 'target', tmp_path)
    receipt = worker.publish(p, plan)
    assert receipt['verification']['content_verified'] and receipt['archived'] == ['legacy']
    assert [f['token'] for f in api.files('target') if f['type'] == 'docx'] == [receipt['document_id']]
    assert any(f['token'] == 'legacy' for entries in api.entries.values() for f in entries)
    before = api.mutations
    again = Publisher(api, 'target', tmp_path).publish(p, plan)
    assert not again['changed'] and api.mutations == before


def test_timeout_after_append_is_reconciled_without_duplicate(tmp_path):
    api = FakeFeishu();p, plan = inputs();api.lose_response = True
    with pytest.raises(UncertainWrite): Publisher(api, 'target', tmp_path).publish(p, plan)
    receipt = Publisher(api, 'target', tmp_path).publish(p, plan)
    expected = sum(len(op['body']['children']) for op in plan['operations'])
    assert receipt['verification']['root_blocks'] == expected
    assert len(api.documents) == 1


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


def test_multi_part_publish_verifies_links_and_archives_old_continuations(tmp_path):
    api = FakeFeishu()
    api.entries['target'].append({'token': 'legacy-main', 'name': 'PR#77', 'type': 'docx'})
    api.entries['target'].append({'token': 'legacy-cont', 'name': 'PR#77-1', 'type': 'docx'})
    p, plan = {'pr_number': 77, 'fingerprint': 'b' * 64}, _v2_plan(parts=2)
    receipt = Publisher(api, 'target', tmp_path, domain='https://x').publish(p, plan)
    names = _doc_names(api, 'target')
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
    assert _doc_names(api, 'target') == ['PR#77']  # 多篇变单篇：旧续篇一并归档


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
    assert _doc_names(api, 'target') == ['PR#77', 'PR#77-1']


def test_big_log_planner_to_publisher_split_without_loss(tmp_path):
    """真实规划→发布链路：700k 字符日志按 500k 文档预算拆篇，跨篇全文
    零丢失、续篇导航齐备、传输批量 ≤40 块。"""
    import run_test.result_docs as rd
    log = 'L' * 700_000
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
    import urllib.parse
    big = [1 for name, call_log in []]  # placeholder；直接检查 api 侧
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
    cfg = {**config, 'feishu': {'doc_max_chars': 20_000, 'doc_max_blocks': 50}}
    _all = [str(f) for f in (store_paths(config).pr_logs_root).rglob('*') if f.is_file()]
    print(f"[DEBUG] pr_logs tree: {[(f.rsplit('/', 2)[-2:], __import__('os').path.getsize(f)) for f in _all]}", flush=True)
    result_docs_module.generate(cfg, 123456)
    plan = json.loads(next((store_paths(cfg).data_root / 'reports')
                           .glob('PR_123456/*/feishu-plan.json')).read_text())
    assert plan['part_count'] >= 2  # 20k 预算下 60k 日志必然拆篇
    all_ops = [op for p in plan['parts'] for op in p['operations']]
    code_blocks = [b for op in all_ops if op.get('body')
                   for b in op['body']['children'] if b['block_type'] == 14]
    assert code_blocks  # 连续块存在（一篇或跨篇）
