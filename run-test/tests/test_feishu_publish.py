from copy import deepcopy
import json
from pathlib import Path
import urllib.parse

import pytest

from run_test.feishu_publish import Publisher, PublishError, UncertainWrite, credentials, block_text
from run_test.result_docs import build_report, feishu_plan
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
