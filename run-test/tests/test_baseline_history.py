"""Exercise real SQLite WAL snapshots and Linux atomic tree replacement."""
import json
from pathlib import Path

import pytest

from run_test.baseline_history import (
    BaselineHistoryError, snapshot_baselines, rollback_baselines, list_snapshots,
)
from run_test.baseline_rollforward import rollforward_baselines
from run_test.config import store_paths
from run_test.process_lock import executor_lease
from test_rollforward import _seed_pr, _seed_baseline, _counts, read_baseline
from conftest import TF_A, TF_B, COMMIT_A, COMMIT_B
from run_test.baseline_store import baseline_log_filename


def _log(config, tf=TF_A):
    from run_test.baseline_store import BaselineRouter
    paths = store_paths(config)
    return BaselineRouter(config['baseline_dirs'], paths.baselines_root).route(tf).root / 'logs' / baseline_log_filename(tf, 'cpu')


def test_refresh_rollback_and_undo_restore_database_and_logs(config):
    paths = store_paths(config)
    _seed_baseline(config, paths, TF_A, 'cpu', _counts(1))
    _seed_baseline(config, paths, TF_B, 'cpu', _counts(7))
    old_log = _log(config).read_bytes()
    _seed_pr(config, 11, [(TF_A, 'cpu', _counts(2))], merged_at='2026-10-02T00:00:00Z', with_archive=True)
    _seed_pr(config, 12, [(TF_A, 'cpu', _counts(3))], merged_at='2026-10-01T00:00:00Z', with_archive=True, commit=COMMIT_B)
    report = rollforward_baselines(config, [11, 12])
    assert report['applied'] is True
    assert [r['pr_number'] for r in report['prs']] == [12, 11]
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 2
    assert _log(config).read_text() == 'PR 11\n'
    result = rollback_baselines(config, report['before_snapshot'])
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 1
    assert _log(config).read_bytes() == old_log
    assert read_baseline(config, TF_B, 'cpu').counts.pass_count == 7
    rollback_baselines(config, result['undo_snapshot'])
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 2
    assert _log(config).read_text() == 'PR 11\n'


def test_empty_baseline_bootstrap_and_rollback(config):
    paths = store_paths(config)
    _seed_pr(config, 1, [(TF_A, 'cpu', _counts(2))], merged_at='2026-10-01T00:00:00Z')
    assert not paths.baselines_root.exists()
    report = rollforward_baselines(config)  # --all reads current results under lease
    assert report['applied']
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 2
    rollback_baselines(config, report['before_snapshot'])
    assert list(paths.baselines_root.rglob('baseline.db')) == []


def test_corrupt_snapshot_rejected_before_live_changes(config):
    paths = store_paths(config)
    _seed_baseline(config, paths, TF_A, 'cpu', _counts(1))
    snap = snapshot_baselines(config)
    payload = paths.data_root / 'baseline-history' / snap['id'] / 'baselines'
    next(payload.rglob('*.log')).write_text('tampered')
    with pytest.raises(BaselineHistoryError, match='checksum'):
        rollback_baselines(config, snap['id'])
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 1
    assert len(list_snapshots(config)) == 1
    with pytest.raises(BaselineHistoryError, match='invalid snapshot'):
        rollback_baselines(config, '../baselines')


def test_executor_lease_blocks_all_maintenance(config):
    paths = store_paths(config)
    with executor_lease(paths.queue_db.with_suffix('.executor.lock')) as acquired:
        assert acquired
        with pytest.raises(BaselineHistoryError, match='busy'):
            snapshot_baselines(config)
        with pytest.raises(BaselineHistoryError, match='busy'):
            rollforward_baselines(config, [123])
        with pytest.raises(BaselineHistoryError, match='busy'):
            rollback_baselines(config, 'bad')
    assert list_snapshots(config) == []


def test_partial_refresh_failure_restores_entire_batch(config):
    paths = store_paths(config)
    _seed_baseline(config, paths, TF_A, 'cpu', _counts(1))
    old_log = _log(config).read_bytes()
    _seed_pr(config, 1, [(TF_A, 'cpu', _counts(2))], merged_at='2026-10-01T00:00:00Z', with_archive=True)
    folder = _seed_pr(config, 2, [(TF_A, 'cpu', _counts(3))], merged_at='2026-10-02T00:00:00Z', with_archive=True)
    (folder / baseline_log_filename(TF_A, 'cpu')).unlink()
    report = rollforward_baselines(config, [2, 1])
    assert not report['applied']
    assert report['errors']
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 1
    assert _log(config).read_bytes() == old_log
    assert json.loads(Path(report['index_file']).read_text())['applied'] is False


def test_reconstructed_replay_preserves_pr_results_and_archives(config):
    import sqlite3
    paths = store_paths(config)
    folder = _seed_pr(config, 1, [(TF_A, 'cpu', _counts(2))], merged_at='2026-10-01T00:00:00Z')
    with sqlite3.connect(paths.pr_db) as conn:
        before = conn.execute('SELECT * FROM pr_results').fetchall()
    logs = {p.name: p.read_bytes() for p in folder.iterdir()}
    report = rollforward_baselines(config, [1])
    assert report['applied']
    with sqlite3.connect(paths.pr_db) as conn:
        assert conn.execute('SELECT * FROM pr_results').fetchall() == before
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == logs


def test_sqlite_backup_includes_live_wal_and_removes_new_keys(config):
    from run_test.db import connect
    paths = store_paths(config)
    _seed_baseline(config, paths, TF_A, 'cpu', _counts(1))
    db = next(paths.baselines_root.rglob('baseline.db'))
    conn = connect(db)
    try:
        conn.execute('UPDATE baseline SET pass_count=9')
        assert Path(str(db) + '-wal').is_file()
        snap = snapshot_baselines(config)
    finally:
        conn.close()
    _seed_baseline(config, paths, TF_A, 'cpu', _counts(2))
    _seed_baseline(config, paths, TF_B, 'cpu', _counts(3))
    rollback_baselines(config, snap['id'])
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 9
    assert not _log(config, TF_B).exists()


def test_same_merge_time_order_is_deterministic(config):
    for pr in (2, 1):
        _seed_pr(config, pr, [(TF_A, 'cpu', _counts(pr))], merged_at='2026-10-01T00:00:00Z', with_archive=True)
    report = rollforward_baselines(config, [2, 1])
    assert [r['pr_number'] for r in report['prs']] == [1, 2]
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 2


def test_symlink_snapshot_rejected(config, tmp_path):
    root = store_paths(config).baselines_root
    root.mkdir(parents=True)
    (root / 'outside').symlink_to(tmp_path / 'secret')
    with pytest.raises(BaselineHistoryError, match='symlink'):
        snapshot_baselines(config)


def test_configuration_mismatch_rejected(config):
    snap = snapshot_baselines(config)
    config['baseline_dirs'] = ['test']
    with pytest.raises(BaselineHistoryError, match='baseline_dirs changed'):
        rollback_baselines(config, snap['id'])


def test_process_death_during_replay_leaves_live_tree_unchanged(config):
    import subprocess
    import sys
    paths = store_paths(config)
    _seed_baseline(config, paths, TF_A, 'cpu', _counts(1))
    original_log = _log(config).read_bytes()
    _seed_pr(config, 1, [(TF_A, 'cpu', _counts(2))], merged_at='2026-10-01T00:00:00Z', with_archive=True)
    script = '''import json, os, sys
import run_test.baseline_rollforward as rf
original = rf._process_pr
def crash(*a, **kw):
    original(*a, **kw)
    os._exit(27)
rf._process_pr = crash
rf.rollforward_baselines(json.loads(sys.argv[1]), [1])
'''
    child = subprocess.run([sys.executable, '-c', script, json.dumps(config)], capture_output=True, timeout=15)
    assert child.returncode == 27, child.stderr
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 1
    assert _log(config).read_bytes() == original_log
    assert len(list_snapshots(config)) == 1


def test_failure_after_publish_restores_before_snapshot(config, monkeypatch):
    import run_test.baseline_history as history
    paths = store_paths(config)
    _seed_baseline(config, paths, TF_A, 'cpu', _counts(1))
    _seed_pr(config, 1, [(TF_A, 'cpu', _counts(2))], merged_at='2026-10-01T00:00:00Z', with_archive=True)
    original = history._snapshot
    def fail_after(config, reason, details=None):
        if reason == 'after-rollforward':
            raise OSError('disk full')
        return original(config, reason, details)
    monkeypatch.setattr(history, '_snapshot', fail_after)
    with pytest.raises(OSError, match='disk full'):
        rollforward_baselines(config, [1])
    assert read_baseline(config, TF_A, 'cpu').counts.pass_count == 1
    assert _log(config).read_text() == 'PASSED x\n'


def test_partial_and_missing_results_do_not_create_baselines(config):
    folder = _seed_pr(config, 1, [(TF_A, 'cpu', _counts(2)), (TF_B, 'cpu', _counts(3))], merged_at='2026-10-01T00:00:00Z', with_archive=True)
    archive = folder / 'baseline-diff.json'
    data = json.loads(archive.read_text())
    data['comparisons'][0]['after']['coverage_status'] = 'partial'
    data['comparisons'][1]['after']['counts'] = None
    archive.write_text(json.dumps(data))
    report = rollforward_baselines(config, [1])
    assert report['applied']
    assert report['prs'][0]['baseline_created'] == 0
    assert read_baseline(config, TF_A, 'cpu') is None
    assert read_baseline(config, TF_B, 'cpu') is None
