"""Resumable Feishu publication: stage, verify, promote, archive old documents.

No credentials in receipts/errors. Unknown write outcomes are reconciled by
reading the remote state; a write is never blindly retried after a timeout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from . import result_docs
from .config import store_paths


class PublishError(RuntimeError):
    pass


class UncertainWrite(PublishError):
    pass


def credentials(path):
    allowed = {'FEISHU_APP_ID', 'FEISHU_APP_SECRET', 'FEISHU_DOMAIN', 'FEISHU_TEST_FOLDER_TOKEN'}
    result = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        key, sep, value = line.strip().removeprefix('export ').partition('=')
        if sep and key.strip() in allowed:
            value = value.strip()
            if value[:1] in ('"', "'") and value[-1:] == value[:1]:
                value = value[1:-1]
            else:
                value = value.split(' #', 1)[0].rstrip()
            result[key.strip()] = value
    if any(not result.get(k) for k in allowed):
        raise PublishError('飞书专用认证配置不完整（未输出字段值）')
    domain = result['FEISHU_DOMAIN'].strip().rstrip('/')
    if not domain.startswith('https://'):
        domain = 'https://' + domain
    parsed = urllib.parse.urlsplit(domain)
    if parsed.scheme != 'https' or not (parsed.hostname or '').endswith('.feishu.cn') or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
        raise PublishError('FEISHU_DOMAIN 必须为飞书租户 HTTPS 域名')
    result['FEISHU_DOMAIN'] = domain
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise PublishError('飞书 API 意外重定向，已停止')


class Feishu:
    def __init__(self, cfg):
        self.cfg = cfg
        self.token = None
        self.expires = 0
        self.last_request = 0
        self.opener = urllib.request.build_opener(NoRedirect)

    def call(self, method, path, body=None, *, auth=True):
        if auth and time.monotonic() >= self.expires:
            data = self.call('POST', '/auth/v3/tenant_access_token/internal', {
                'app_id': self.cfg['FEISHU_APP_ID'], 'app_secret': self.cfg['FEISHU_APP_SECRET']}, auth=False)
            self.token = data['tenant_access_token']
            self.expires = time.monotonic() + data.get('expire', 7200) - 60
        for attempt in range(4):
            time.sleep(max(0, .4 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            headers = {'Content-Type': 'application/json; charset=utf-8'}
            if auth:
                headers['Authorization'] = 'Bearer ' + self.token
            req = urllib.request.Request('https://open.feishu.cn/open-apis' + path,
                data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
                method=method, headers=headers)
            try:
                with self.opener.open(req, timeout=40) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as error:
                if error.code == 429 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                cls = UncertainWrite if method != 'GET' and error.code >= 500 else PublishError
                raise cls(f'飞书 HTTP {error.code}（响应体及认证头不写日志）') from None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                if method == 'GET' and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise UncertainWrite('飞书请求结果不确定，须读取远端状态核对') from None
            code = result.get('code', 0)
            if code == 99991400 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            if code:
                raise PublishError(f'飞书业务错误 code={code}（原始响应不写日志）')
            return result.get('data', result)
        raise PublishError('飞书请求重试耗尽')

    def files(self, folder):
        return self._pages('/drive/v1/files', {'folder_token': folder, 'page_size': 200}, 'files', 'next_page_token')

    def blocks(self, doc):
        return self._pages(f'/docx/v1/documents/{doc}/blocks', {'page_size': 500}, 'items', 'page_token')

    def children(self, doc, parent):
        return self._pages(f'/docx/v1/documents/{doc}/blocks/{parent}/children', {'page_size': 500}, 'items', 'page_token')

    def _pages(self, path, params, field, cursor):
        found, seen = [], set()
        while True:
            data = self.call('GET', path + '?' + urllib.parse.urlencode(params))
            found.extend(data.get(field, []))
            if not data.get('has_more'):
                return found
            page = data.get(cursor)
            if not page or page in seen:
                raise PublishError('飞书分页游标异常')
            seen.add(page)
            params['page_token'] = page


def block_text(block):
    return ''.join(e.get('text_run', {}).get('content', '') for v in block.values()
                   if isinstance(v, dict) for e in v.get('elements', [])).replace('\r\n', '\n')


def shape(block):
    kind = block['block_type']
    if kind == 31:
        prop = block['table']['property']
        return kind, prop['row_size'], prop['column_size']
    field = {2: 'text', 5: 'heading3', 6: 'heading4', 14: 'code'}.get(kind)
    if field is None:
        return kind,
    value = block[field]
    style = value.get('style', {})
    links = [e.get('text_run', {}).get('text_element_style', {}).get('link', {}).get('url')
             for e in value.get('elements', [])]
    return kind, block_text(block), bool(style.get('folded', False)), tuple(x for x in links if x)


class Publisher:
    def __init__(self, client, folder, root):
        self.api, self.folder, self.root = client, folder, Path(root)
        self.state = {}
        self.path = None

    def save(self):
        result_docs._atomic(self.path, result_docs.encoded(self.state))

    def unique_object(self, key, parent, name, kind):
        """A dedicated, deterministic folder enables recovery without duplicate creation."""
        if self.state.get(key):
            return self.state[key]
        found = [f for f in self.api.files(parent) if f['name'] == name and f['type'] == kind]
        if len(found) > 1:
            raise PublishError('发现重复的发布暂存对象，保留现场等待核对')
        if found:
            token = found[0]['token']
        else:
            if self.state.get(key + '_pending'):
                raise UncertainWrite('先前创建结果仍不确定，未重复创建；请核对暂存目录')
            self.state[key + '_pending'] = True
            self.save()
            try:
                if kind == 'folder':
                    data = self.api.call('POST', '/drive/v1/files/create_folder', {'name': name, 'folder_token': parent})
                    token = data['token']
                else:
                    data = self.api.call('POST', '/docx/v1/documents', {'title': name, 'folder_token': parent})
                    token = data['document']['document_id']
            except UncertainWrite:
                raise
            except PublishError:
                self.state.pop(key + '_pending', None)
                self.save()
                raise
        self.state[key] = token
        self.state.pop(key + '_pending', None)
        self.save()
        return token

    def append(self, key, doc, parent, blocks):
        steps = self.state.setdefault('steps', {})
        step = steps.get(key)
        if step and step.get('done'):
            return step['result']
        actual = self.api.children(doc, parent)
        if step:
            before = step['before']
            if [b['block_id'] for b in actual[:len(before)]] != before:
                raise PublishError('上传期间远端内容被修改，停止自动续写')
            added = actual[len(before):]
            if len(added) == len(blocks) and [shape(b) for b in added] == [shape(b) for b in blocks]:
                step.update(done=True, result=added)
                self.save()
                return added
            raise UncertainWrite('先前写入未能确认，未重复追加内容；保留暂存文档')
        step = steps[key] = {'before': [b['block_id'] for b in actual], 'client_token': str(uuid.uuid4())}
        self.save()
        path = f'/docx/v1/documents/{doc}/blocks/{parent}/children?' + urllib.parse.urlencode({'client_token': step['client_token']})
        try:
            data = self.api.call('POST', path, {'index': -1, 'children': blocks})
        except UncertainWrite:
            raise
        except PublishError:
            del steps[key]
            self.save()
            raise
        step.update(done=True, result=data['children'])
        self.save()
        return data['children']

    def verify(self, doc, plan):
        blocks = self.api.blocks(doc)
        lookup = {b['block_id']: b for b in blocks}
        root = lookup.get(doc)
        if root is None:
            raise PublishError('远端文档根块缺失')
        actual = [lookup[i] for i in root.get('children', [])]
        expected = [b for op in plan['operations'] for b in op['body']['children']]
        if [shape(b) for b in actual] != [shape(b) for b in expected]:
            raise PublishError('远端正文/折叠/链接回读与本地报告不一致')
        tables = [b for b in actual if b['block_type'] == 31]
        planned = [op for op in plan['operations'] if op['op'] == 'table']
        for table, op in zip(tables, planned):
            cells = table['table']['cells']
            if len(cells) != len(op['cell_children']):
                raise PublishError('远端表格单元格数不符')
            for cell, desired in zip(cells, op['cell_children']):
                content = [lookup[i] for i in lookup[cell].get('children', []) if block_text(lookup[i])]
                if [shape(b) for b in content] != [shape(b) for b in desired]:
                    raise PublishError('远端表格计数与报告不一致')
        return {'blocks': len(blocks), 'root_blocks': len(actual), 'tables': len(tables), 'content_verified': True}

    def move(self, key, doc, destination):
        # Moving is idempotent after checking membership, including ambiguous HTTP outcomes.
        if any(f['token'] == doc for f in self.api.files(destination)):
            self.state[key] = True
            self.save()
            return
        if self.state.get(key + '_pending'):
            raise UncertainWrite('先前移动仍未确认，保留两个目录现场')
        self.state[key + '_pending'] = True
        self.save()
        try:
            data = self.api.call('POST', f'/drive/v1/files/{doc}/move', {'type': 'docx', 'folder_token': destination})
        except UncertainWrite:
            raise
        except PublishError:
            self.state.pop(key + '_pending', None)
            self.save()
            raise
        # File movement can be asynchronous. Membership, not HTTP 200, confirms completion.
        for _ in range(6):
            if any(f['token'] == doc for f in self.api.files(destination)):
                self.state[key] = True
                self.state.pop(key + '_pending', None)
                self.save()
                return
            time.sleep(1)
        raise UncertainWrite('移动请求已提交，等待远端目录确认')

    def publish(self, pointer, plan, *, still_current=lambda: True):
        pr, fingerprint = pointer['pr_number'], pointer['fingerprint']
        self.path = self.root / self.folder / f'PR_{pr}' / (fingerprint + '.json')
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding='utf-8'))
        else:
            self.state = {'pr_number': pr, 'fingerprint': fingerprint, 'folder': self.folder, 'status': 'staging'}
            self.save()
        if self.state['status'] == 'published':
            return {**self.state['receipt'], 'changed': False}
        staging = self.unique_object('staging', self.folder, '报告上传暂存', 'folder')
        version = self.unique_object('version_folder', staging, f'PR_{pr}_{fingerprint}', 'folder')
        doc = self.unique_object('document_id', version, plan['title'], 'docx')
        for i, operation in enumerate(plan['operations']):
            if operation['op'] not in ('append', 'table'):
                raise PublishError('未知报告发布操作')
            created = self.append(f'root-{i}', doc, doc, operation['body']['children'])
            if operation['op'] == 'table':
                table = created[0]
                cells = table.get('table', {}).get('cells')
                if not cells:
                    table = next(b for b in self.api.blocks(doc) if b['block_id'] == table['block_id'])
                    cells = table['table']['cells']
                if len(cells) != len(operation['cell_children']):
                    raise PublishError('建表返回单元格数量异常')
                for j, (cell, children) in enumerate(zip(cells, operation['cell_children'])):
                    self.append(f'cell-{i}-{j}', doc, cell, children)
        verification = self.verify(doc, plan)
        if not still_current():
            raise PublishError('本地结果已更新；已验证的旧版本留在暂存，未替换当前文档')
        old_docs = [f for f in self.api.files(self.folder) if f['type'] == 'docx' and f['name'] == plan['title'] and f['token'] != doc]
        # Keep old content until the complete replacement has been verified and promoted.
        self.move('promoted', doc, self.folder)
        if old_docs:
            history = self.unique_object('history', self.folder, '历史测试报告', 'folder')
            for old in old_docs:
                self.move('archived-' + old['token'], old['token'], history)
        matches = [f for f in self.api.files(self.folder) if f['type'] == 'docx' and f['name'] == plan['title']]
        if len(matches) != 1 or matches[0]['token'] != doc:
            raise PublishError('发布后目录对账未通过，未标记完成')
        receipt = {'pr_number': pr, 'document_id': doc, 'fingerprint': fingerprint,
                   'verification': verification, 'archived': [f['token'] for f in old_docs], 'changed': True}
        self.state.update(status='published', receipt=receipt)
        self.save()
        return receipt


def publish(config, pr):
    settings = (config.get('reports') or {}).get('feishu') or {}
    if not settings.get('env_file'):
        raise PublishError('请配置 reports.feishu.env_file')
    cfg = credentials(settings['env_file'])
    pointer = result_docs.generate(config, pr)
    version = Path(pointer['version'])
    raw = (version / 'feishu-plan.json').read_bytes()
    if result_docs.digest(raw) != pointer['files']['feishu-plan.json']:
        raise PublishError('待发布内容校验失败')
    plan = json.loads(raw)
    root = store_paths(config).data_root / 'feishu-publish'
    root.mkdir(exist_ok=True)
    conn = sqlite3.connect(root / 'publish-lock.db', timeout=1)
    try:
        conn.execute('BEGIN IMMEDIATE')
        worker = Publisher(Feishu(cfg), cfg['FEISHU_TEST_FOLDER_TOKEN'], root)
        def still_current():
            return result_docs.digest(result_docs.encoded(result_docs.load_report(config, pr)).encode()) == pointer['fingerprint']
        receipt = worker.publish(pointer, plan, still_current=still_current)
        receipt['url'] = cfg['FEISHU_DOMAIN'] + '/docx/' + receipt['document_id']
        conn.commit()
        return receipt
    finally:
        conn.close()


def scan(config):
    with sqlite3.connect(store_paths(config).pr_db.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        prs = [r[0] for r in conn.execute('SELECT pr_number FROM pr_results ORDER BY pr_number')]
    results, errors = [], []
    for pr in prs:
        try:
            results.append(publish(config, pr))
        except (PublishError, OSError, ValueError, KeyError, sqlite3.Error) as error:
            errors.append({'pr_number': pr, 'error': str(error)})
    return {'published': sum(r['changed'] for r in results), 'unchanged': sum(not r['changed'] for r in results),
            'results': results, 'errors': errors}
