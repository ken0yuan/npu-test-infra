"""Persistent baseline checkpoints. Public mutations share the executor lease.

Snapshots contain only baseline databases and logs, never credentials, queue state,
PR results or installed environments. Internal helpers require the caller's lease.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid

from .config import store_paths
from .process_lock import executor_lease


class BaselineHistoryError(RuntimeError):
    pass


@contextmanager
def maintenance(config, wait_seconds=0):
    """Wait at a task boundary; never interrupt a running executor."""
    paths = store_paths(config)
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        with executor_lease(paths.queue_db.with_suffix(".executor.lock")) as acquired:
            if acquired:
                yield
                return
        if time.monotonic() >= deadline:
            raise BaselineHistoryError("executor busy; retry after the task or use --wait-seconds")
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def _history_root(config):
    return store_paths(config).data_root / "baseline-history"


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _files(root):
    if root.is_symlink():
        raise BaselineHistoryError(f"symlink not allowed: {root}")
    if root.exists() and not root.is_dir():
        raise BaselineHistoryError(f"not a directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BaselineHistoryError(f"symlink not allowed: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
        elif not path.is_dir():
            raise BaselineHistoryError(f"unsupported file: {path}")
    return result


def _validate_databases(root):
    counts = {}
    for db in root.rglob("baseline.db"):
        conn = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
        try:
            if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise BaselineHistoryError(f"corrupt baseline DB: {db}")
            rows = conn.execute("SELECT log_path FROM baseline").fetchall()
            for (relative,) in rows:
                log = (db.parent / relative).resolve()
                if not log.is_relative_to(db.parent.resolve()) or not log.is_file():
                    raise BaselineHistoryError(f"missing or invalid baseline log: {relative}")
            counts[db.relative_to(root).as_posix()] = len(rows)
        finally:
            conn.close()
    return counts


def _snapshot(config, reason, details=None):
    root = store_paths(config).baselines_root
    ident = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    history = _history_root(config)
    history.mkdir(parents=True, exist_ok=True)
    pending = history / ("." + ident)
    payload = pending / "baselines"
    payload.mkdir(parents=True)
    try:
        for name, source in _files(root).items():
            if name.endswith(("-wal", "-shm", "-journal")):
                continue
            dest = payload / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.name == "baseline.db":
                # Include committed WAL contents without copying a live SQLite file.
                src = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
                dst = sqlite3.connect(dest)
                try:
                    src.backup(dst)
                    dst.execute("PRAGMA journal_mode=DELETE")
                finally:
                    dst.close()
                    src.close()
            else:
                shutil.copy2(source, dest)
        counts = _validate_databases(payload)
        manifest = {
            "version": 1, "id": ident, "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason, "details": details or {},
            "baseline_dirs": sorted(config.get("baseline_dirs", [])),
            "counts": counts,
            "files": {name: _digest(path) for name, path in _files(payload).items()},
        }
        (pending / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.rename(history / ident)
        return manifest
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def _load(config, ident):
    if not re.fullmatch(r"[0-9TZ.]+-[0-9a-f]{8}", ident):
        raise BaselineHistoryError("invalid snapshot id")
    folder = _history_root(config) / ident
    if folder.is_symlink():
        raise BaselineHistoryError("snapshot symlink not allowed")
    manifest_path = folder / "manifest.json"
    if manifest_path.is_symlink():
        raise BaselineHistoryError("manifest symlink not allowed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("id") != ident:
        raise BaselineHistoryError("invalid snapshot manifest")
    if manifest["baseline_dirs"] != sorted(config.get("baseline_dirs", [])):
        raise BaselineHistoryError("baseline_dirs changed; restore the matching configuration first")
    payload = folder / "baselines"
    if not payload.is_dir():
        raise BaselineHistoryError("snapshot payload missing")
    actual = {name: _digest(path) for name, path in _files(payload).items()}
    if actual != manifest["files"]:
        raise BaselineHistoryError("snapshot checksum mismatch; live baseline untouched")
    if _validate_databases(payload) != manifest["counts"]:
        raise BaselineHistoryError("snapshot row count mismatch")
    return manifest, payload


def _publish(stage, root):
    """Atomic Linux tree swap; stage holds the old tree after successful exchange."""
    if root.is_symlink():
        raise BaselineHistoryError("baseline root symlink not allowed")
    if root.exists():
        if os.name != "posix":
            raise BaselineHistoryError("atomic baseline restore requires Linux renameat2")
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        exchange = getattr(libc, "renameat2", None)
        if exchange is None:
            raise BaselineHistoryError("atomic directory exchange is unavailable")
        exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        exchange.restype = ctypes.c_int
        # RENAME_EXCHANGE: even process death cannot leave an absent/half tree.
        if exchange(-100, os.fsencode(stage), -100, os.fsencode(root), 2) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    else:
        stage.rename(root)


def _restore(config, ident):
    """Validate first, stage a complete tree, then replace DBs and logs together."""
    manifest, payload = _load(config, ident)
    root = store_paths(config).baselines_root
    stage = root.parent / (".baseline-stage-" + uuid.uuid4().hex)
    try:
        shutil.copytree(payload, stage)
        _publish(stage, root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def snapshot_baselines(config, reason="manual", *, wait_seconds=0):
    with maintenance(config, wait_seconds):
        return _snapshot(config, reason)


def list_snapshots(config):
    result = []
    for p in sorted(_history_root(config).glob("[0-9]*/manifest.json")):
        manifest = json.loads(p.read_text(encoding="utf-8"))
        result.append({key: value for key, value in manifest.items() if key != "files"})
    return result


def rollback_baselines(config, ident, *, wait_seconds=0):
    with maintenance(config, wait_seconds):
        _load(config, ident)  # Reject damage BEFORE making any changes.
        safety = _snapshot(config, "before-rollback", {"target": ident})
        restored = _restore(config, ident)
        result = {"restored": restored["id"], "undo_snapshot": safety["id"], "counts": restored["counts"]}
        receipt = _history_root(config) / safety["id"] / "rollback.json"
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
