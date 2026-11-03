"""Per-command CONNECT route for one OBS bucket; TLS stays with obsutil."""
import select
import socket
import socketserver
import threading
from urllib.parse import urlsplit
import re


class ObsRegionalProxy:
    def __init__(self, endpoint, bucket):
        parsed = urlsplit(endpoint)
        if (parsed.scheme != 'https' or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in ('', '/')
                or parsed.port not in (None, 443)
                or not re.fullmatch(r'obs\.[a-z0-9-]+\.myhuaweicloud\.com', parsed.hostname or '')
                or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}[a-z0-9]', bucket or '')):
            raise ValueError('OBS regional proxy requires an official HTTPS regional endpoint and one bucket')
        self.endpoint = parsed.hostname
        self.authority = bucket + '.' + self.endpoint + ':443'
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.connections = set()

    def _connect(self):
        return socket.create_connection((self.endpoint, 443), timeout=10)

    def __enter__(self):
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                remote = None
                try:
                    self.request.settimeout(10)
                    with owner.lock:
                        if owner.stop.is_set(): return
                        owner.connections.add(self.request)
                    head = b''
                    while b'\r\n\r\n' not in head:
                        chunk = self.request.recv(4096)
                        if not chunk: return
                        head += chunk
                        if len(head) > 8192: return
                    headers, _, extra = head.partition(b'\r\n\r\n')
                    request = headers.split(b'\r\n', 1)[0].decode('ascii').split()
                    if len(request) != 3 or request[0] != 'CONNECT' or request[1] != owner.authority:
                        self.request.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
                        return
                    remote = owner._connect()
                    remote.settimeout(10)
                    with owner.lock:
                        if owner.stop.is_set(): return
                        owner.connections.add(remote)
                    self.request.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                    if extra: remote.sendall(extra)
                    # No TLS interception or request/header logging. obsutil
                    # verifies the original bucket hostname end to end.
                    while not owner.stop.is_set():
                        readable, _, _ = select.select([self.request, remote], [], [], .2)
                        for source in readable:
                            data = source.recv(65536)
                            if not data: return
                            (remote if source is self.request else self.request).sendall(data)
                except (OSError, ValueError, UnicodeError):
                    return
                finally:
                    with owner.lock:
                        owner.connections.discard(self.request)
                        if remote is not None: owner.connections.discard(remote)
                    if remote is not None: remote.close()

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = False

        self.server = Server(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': .1}, daemon=True)
        self.thread.start()
        return 'http://127.0.0.1:' + str(self.server.server_address[1])

    def __exit__(self, *_):
        self.stop.set()
        self.server.shutdown()
        with self.lock:
            connections = tuple(self.connections)
        for connection in connections:
            try: connection.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            connection.close()
        self.server.server_close()
        self.thread.join(timeout=2)
