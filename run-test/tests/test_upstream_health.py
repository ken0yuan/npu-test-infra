from datetime import datetime, timedelta, timezone
import json

import pytest

from run_test.upstream import PyTorchMirror, UpstreamError
from run_test.upstream_health import health_warning, read_health, record_health


def mirror_probe(config, tmp_path, monkeypatch, outcomes):
    mirror = PyTorchMirror(config=config, mirror_dir=tmp_path / 'mirror.git')
    monkeypatch.setattr(mirror, 'ensure', lambda: None)
    calls, sleeps = [], []
    monkeypatch.setattr('run_test.upstream.time.sleep', sleeps.append)
    def git(path, args, **kwargs):
        if args[0] == 'rev-parse': return 'a' * 40
        calls.append(kwargs)
        result = outcomes.pop(0)
        if isinstance(result, Exception): raise result
        return result
    monkeypatch.setattr('run_test.upstream._git', git)
    return mirror, calls, sleeps


def test_transient_failure_recovers_in_same_round(config, tmp_path, monkeypatch):
    record_health(config, last_success_at='2026-10-29T00:00:00+00:00')
    mirror, calls, sleeps = mirror_probe(config, tmp_path, monkeypatch,
        [UpstreamError('Failed to connect to ghfast.top'), ''])
    assert mirror.fetch() == 'a' * 40
    assert len(calls) == 2 and sleeps == [5]
    health = read_health(config)
    assert health['status'] == 'ok' and health['consecutive_failures'] == 0
    assert health['last_success_at'] != '2026-10-29T00:00:00+00:00'


def test_exhaustion_preserves_success_identity_and_marks_unknown(config, tmp_path, monkeypatch):
    record_health(config, last_success_at='2026-10-29T00:00:00+00:00', fetched_tip='b'*40,
                  consecutive_failures=2)
    mirror, calls, sleeps = mirror_probe(config, tmp_path, monkeypatch,
        [UpstreamError('Connection timed out')] * 3)
    with pytest.raises(UpstreamError): mirror.fetch()
    health = read_health(config)
    assert len(calls) == 3 and sleeps == [5, 10]
    assert health['status'] == 'failed' and health['consecutive_failures'] == 3
    assert health['fetched_tip'] == 'b' * 40
    assert health['last_success_at'] == '2026-10-29T00:00:00+00:00'
    assert '未知' in health_warning(config, health)


def test_permanent_error_does_not_retry(config, tmp_path, monkeypatch):
    mirror, calls, sleeps = mirror_probe(config, tmp_path, monkeypatch,
        [UpstreamError('could not read Username: Authentication failed')])
    with pytest.raises(UpstreamError): mirror.fetch()
    assert len(calls) == 1 and sleeps == []
    assert read_health(config)['status'] == 'failed'


def test_failure_warns_even_with_recent_success_and_unchanged_queue(config):
    now = datetime.now(timezone.utc)
    warning = health_warning(config, {'status':'failed', 'error':'network down',
                            'last_success_at':now.isoformat()}, now=now)
    assert 'network down' in warning and '未知' in warning


def test_quiet_main_is_healthy_but_stale_fetch_is_unknown(config):
    now = datetime.now(timezone.utc)
    assert health_warning(config, {}, now.isoformat(), now=now) is None
    assert '过期' in health_warning(config, {}, (now-timedelta(hours=2)).isoformat(), now=now)


def test_corrupt_health_is_visible(config):
    from run_test.config import store_paths
    path = store_paths(config).data_root / 'upstream-fetch-health.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('broken')
    assert read_health(config)['status'] == 'unreadable'
    assert health_warning(config, read_health(config))


def test_both_monitors_warn_independently_of_queue_age(config, queue, router, paths):
    from run_test.monitoring import brief_once, monitor_once
    record_health(config, status='failed', error='Failed to connect')
    before = paths.queue_db.read_bytes()
    assert 'Failed to connect' in brief_once(config)['upstream_warning']
    assert 'Failed to connect' in monitor_once(config, paths.data_root/'monitor')['upstream_warning']
    assert paths.queue_db.read_bytes() == before
