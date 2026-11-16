from contextlib import contextmanager
import io
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from run_test.obs import ObsDownloadError, make_downloader, list_artifacts
from run_test.obs_public import PublicObsDownloader


BUCKET = 'pytorch-torchnpu'
PREFIX = 'obs://' + BUCKET + '/7day-cache/'


def public():
    return PublicObsDownloader('https://obs.cn-southwest-2.myhuaweicloud.com', BUCKET)


class Response(io.BytesIO):
    def __init__(self, raw=b'', headers=None, status=200):
        super().__init__(raw)
        self.headers = headers or {}
        self.status = status


def page(keys, truncated=False, marker=None, name=BUCKET):
    objects = ''.join(f'<Contents><Key>{key}</Key><LastModified>2026-11-16T06:28:13Z</LastModified><ETag>"tag"</ETag><Size>3</Size></Contents>' for key in keys)
    next_marker = f'<NextMarker>{marker}</NextMarker>' if marker is not None else ''
    return (f'<ListBucketResult xmlns="http://obs.myhuaweicloud.com/doc/2015-06-30/"><Name>{name}</Name><Prefix>7day-cache/</Prefix><IsTruncated>{str(truncated).lower()}</IsTruncated>{objects}{next_marker}</ListBucketResult>').encode()


@pytest.mark.parametrize('explicit_marker', [True, False])
def test_all_pages_are_combined_with_exact_namespace(monkeypatch, explicit_marker):
    obj = public()
    seen = []
    @contextmanager
    def response(url, *args, **kwargs):
        params = parse_qs(urlsplit(url).query)
        seen.append(params)
        if len(seen) == 1:
            yield Response(page(['7day-cache/a'], True, '7day-cache/a' if explicit_marker else None))
        else:
            assert params['marker'] == ['7day-cache/a']
            yield Response(page(['7day-cache/b']))
    monkeypatch.setattr(obj, 'response', response)
    listing = obj.list(PREFIX)
    assert set(listing) == {PREFIX + 'a', PREFIX + 'b'}
    assert listing[PREFIX + 'a']['etag'] == 'tag'
    assert seen[0]['prefix'] == ['7day-cache/']


@pytest.mark.parametrize('body', [page([], True), page(['outside/a']), page([], name='other-bucket'), b'not xml'])
def test_invalid_or_truncated_listing_is_not_returned(monkeypatch, body):
    obj = public()
    @contextmanager
    def response(*args, **kwargs):
        yield Response(body)
    monkeypatch.setattr(obj, 'response', response)
    with pytest.raises(ObsDownloadError): obj.list(PREFIX)


def test_stat_distinguishes_absent_object_from_access_failure(monkeypatch):
    obj = public()
    for code in [404, 403]:
        @contextmanager
        def response(*args, **kwargs):
            raise HTTPError('https://example', code, 'error', {}, None)
            yield
        monkeypatch.setattr(obj, 'response', response)
        if code == 404: assert obj.stat(PREFIX + 'a') is None
        else:
            with pytest.raises(ObsDownloadError, match='403'): obj.stat(PREFIX + 'a')


def test_head_uses_object_version_and_remote_time(monkeypatch):
    obj = public()
    @contextmanager
    def response(url, method='GET', **kwargs):
        assert method == 'HEAD'
        yield Response(headers={'ETag':'"tag"', 'Content-Length':'3', 'Last-Modified':'Mon, 16 Nov 2026 06:28:13 GMT'})
    monkeypatch.setattr(obj, 'response', response)
    assert obj.stat(PREFIX+'a') == {'etag':'tag', 'size':3, 'last_modified':'2026-11-16T06:28:13+00:00'}


def test_atomic_download_keeps_old_file_when_body_is_short(monkeypatch, tmp_path):
    obj = public()
    target = tmp_path/'package.tar.gz'
    target.write_bytes(b'old content')
    @contextmanager
    def response(*args, **kwargs):
        yield Response(b'new', {'Content-Length':'100'})
    monkeypatch.setattr(obj, 'response', response)
    monkeypatch.setattr('run_test.obs_public.time.sleep', lambda _:None)
    with pytest.raises(ObsDownloadError, match='length mismatch'): obj.download(PREFIX+'a', target)
    assert target.read_bytes() == b'old content'
    assert list(tmp_path.iterdir()) == [target]
    @contextmanager
    def complete(*args, **kwargs):
        yield Response(b'new', {'Content-Length':'3'})
    monkeypatch.setattr(obj, 'response', complete)
    assert obj.download(PREFIX+'a', target).size == 3
    assert target.read_bytes() == b'new'


def test_routing_preserves_legacy_downloads_without_sending_credentials_to_public_bucket(monkeypatch, tmp_path):
    cfg={'bucket':'pytorch-ci-cache', 'downloader':{'type':'obsutil'},
         'source':{'bucket':'pytorch-ci-cache'},
         'wheels':{'bucket':BUCKET, 'downloader':{'type':'public_https','endpoint':'obs.cn-southwest-2.myhuaweicloud.com'}}}
    router=make_downloader(cfg)
    seen=[]
    monkeypatch.setattr(router.routes['pytorch-ci-cache'], 'download', lambda url,dest:seen.append(('private',url)))
    monkeypatch.setattr(router.routes[BUCKET], 'download', lambda url,dest:seen.append(('public',url)))
    for url in [PREFIX+'wheel', 'obs://pytorch-ci-cache/7day-cache/source', 'obs://pytorch-ci-cache/7day-cache/old-wheel']:
        router.download(url,tmp_path/'a')
    assert [x[0] for x in seen] == ['public','private','private']
    with pytest.raises(ObsDownloadError): router.download('obs://other-bucket/a',tmp_path/'a')


def test_public_requests_have_no_authentication_or_environment_proxy(monkeypatch):
    import run_test.obs_public as module
    seen=[]
    class Opener:
        def open(self, request, timeout):
            assert not request.has_header('Authorization')
            assert not request.has_header('Proxy-Authorization')
            return Response(b'ok')
    def build(*handlers):
        seen.extend(handlers)
        return Opener()
    monkeypatch.setattr(module,'build_opener',build)
    with public().response('https://pytorch-torchnpu.obs.cn-southwest-2.myhuaweicloud.com/a') as response:
        assert response.read() == b'ok'
    assert seen[0].proxies == {}
    assert seen[1].redirect_request(None,None,None,None,None,'https://other') is None


def test_one_bucket_listing_failure_does_not_publish_a_partial_snapshot():
    class Downloader:
        def list(self, url):
            if 'pytorch-ci-cache' in url: raise ObsDownloadError('source unavailable')
            return {PREFIX+'a':{}}
    cfg={'bucket':'pytorch-ci-cache','wheels':{'bucket':BUCKET}}
    with pytest.raises(ObsDownloadError,match='pytorch-ci-cache'): list_artifacts(cfg,Downloader())


@pytest.mark.parametrize('url', ['obs://other-bucket/a', PREFIX+'../a', PREFIX+'a?x=y', 'https://example/a'])
def test_public_url_scope_is_enforced(url):
    with pytest.raises(ObsDownloadError): public().key(url)


@pytest.mark.parametrize('endpoint', ['http://obs.cn-southwest-2.myhuaweicloud.com','https://other.example','https://user:secret@obs.cn-southwest-2.myhuaweicloud.com'])
def test_invalid_public_endpoint_is_rejected(endpoint):
    with pytest.raises(ValueError): PublicObsDownloader(endpoint,BUCKET)
