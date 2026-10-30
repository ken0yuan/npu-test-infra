"""Fetch health is independent of queue size and whether main changed."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from .config import store_paths


def read_health(config):
    path = store_paths(config).data_root / "upstream-fetch-health.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as error:
        return {"status": "unreadable", "error": str(error)}


def record_health(config, **changes):
    path = store_paths(config).data_root / "upstream-fetch-health.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {**read_health(config), **changes}
    temp = path.with_name(path.name + "." + uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def health_warning(config, health, last_fetch_at=None, *, now=None):
    if health.get("status") in ("failed", "retrying", "unreadable"):
        return "上游抓取异常，最新提交是否积压未知：" + str(health.get("error", "未知原因"))[-700:]
    last = health.get("last_success_at") or last_fetch_at
    if not last:
        return "尚无上游抓取成功记录，不能判断最新提交是否积压"
    try:
        when = datetime.fromisoformat(last)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        age = ((now or datetime.now(timezone.utc)) - when).total_seconds()
        interval = float((config.get("upstream") or {}).get("cycle_interval_seconds", 3600))
        if age > interval * 1.5:
            return f"上游抓取成功记录已过期 {age / 60:.0f} 分钟，最新提交是否积压未知"
    except (TypeError, ValueError):
        return "上游抓取时间记录无效，最新提交是否积压未知"
    return None
