import socket
import socketserver
import threading

import pytest

from run_test.obs_transport import ObsRegionalProxy
from run_test.obs import make_downloader


def client(proxy):
    port = int(proxy.rsplit(':', 1)[1])
    connection = socket.create_connection(('127.0.0.1', port), timeout=2)
    connection.settimeout(2)
    return connection


def test_only_expected_bucket_can_open_a_tunnel(monkeypatch):
    route = ObsRegionalProxy('https://obs.ap-southeast-1.myhuaweicloud.com', 'test-bucket')
    def forbidden(): raise AssertionError('unapproved destination dialed')
    monkeypatch.setattr(route, '_connect', forbidden)
    with route as proxy:
        for request in [b'GET / HTTP/1.1', b'CONNECT other.example:443 HTTP/1.1']:
            with client(proxy) as connection:
                connection.sendall(request + b'\r\n\r\n')
                assert b'403 Forbidden' in connection.recv(1000)
    assert not route.thread.is_alive()


def test_tunnel_transfers_bytes_and_closes_active_connections(monkeypatch):
    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            while True:
                data = self.request.recv(4096)
                if not data: return
                self.request.sendall(data)
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Echo)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    route = ObsRegionalProxy('https://obs.ap-southeast-1.myhuaweicloud.com', 'test-bucket')
    monkeypatch.setattr(route, '_connect', lambda: socket.create_connection(server.server_address, timeout=2))
    try:
        with route as proxy:
            connection = client(proxy)
            connection.sendall(('CONNECT '+route.authority+' HTTP/1.1\r\n\r\n').encode())
            assert b'200 Connection Established' in connection.recv(1000)
            connection.sendall(b'opaque TLS bytes');assert connection.recv(1000) == b'opaque TLS bytes'
        assert connection.recv(1000) == b''
        connection.close()
        with pytest.raises(OSError): client(proxy)
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


@pytest.mark.parametrize('endpoint', ['http://obs.ap-southeast-1.myhuaweicloud.com',
    'https://untrusted.example', 'https://user:secret@obs.ap-southeast-1.myhuaweicloud.com',
    'https://obs.ap-southeast-1.myhuaweicloud.com/path'])
def test_unsafe_endpoint_is_rejected(endpoint):
    with pytest.raises(ValueError): ObsRegionalProxy(endpoint, 'test-bucket')


def test_downloader_route_is_opt_in_and_scoped_to_one_command(monkeypatch):
    seen = []
    def run(command, **kwargs):
        seen.append(command)
        if any(x.startswith('-proxy-url=') for x in command):
            with client(command[-1].split('=', 1)[1]): pass
        return type('Result', (), {'returncode':0,'stdout':'','stderr':''})()
    monkeypatch.setattr('run_test.obs.subprocess.run', run)
    normal = make_downloader({'bucket':'test-bucket','downloader':{'type':'obsutil'}})
    normal._run(['ls', 'obs://test-bucket'], 1)
    assert seen[-1] == ['obsutil','ls','obs://test-bucket']
    routed = make_downloader({'bucket':'test-bucket','downloader':{'type':'obsutil',
        'regional_proxy_endpoint':'https://obs.ap-southeast-1.myhuaweicloud.com'}})
    routed._run(['ls','obs://test-bucket'], 1)
    with pytest.raises(OSError): client(seen[-1][-1].split('=',1)[1])
