"""Locked JSON counters and privacy-preserving runtime event logs."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from config.settings import GLOBAL_DAILY_QUESTION_LIMIT, LOG_DIR, SESSION_DAILY_QUESTION_LIMIT
from core.locking import file_lock


class BudgetReached(RuntimeError):
    pass


@contextmanager
def _locked_json(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(Path(str(path) + ".lock")):
        try:
            value = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError):
            value = {}
        yield value
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        temporary.replace(path)


def consume_question(session_id: str) -> dict[str, int]:
    today = date.today().isoformat()
    path = LOG_DIR / "daily-counters.json"
    with _locked_json(path) as counters:
        day = counters.setdefault(today, {"global": 0, "sessions": {}})
        session_count = int(day["sessions"].get(session_id, 0))
        if session_count >= SESSION_DAILY_QUESTION_LIMIT:
            raise BudgetReached("This session has reached its 15-question daily limit.")
        if int(day["global"]) >= GLOBAL_DAILY_QUESTION_LIMIT:
            raise BudgetReached("VidWise has reached today's shared free-tier budget. Please return tomorrow.")
        day["sessions"][session_id] = session_count + 1
        day["global"] = int(day["global"]) + 1
        return {"session": session_count + 1, "global": day["global"]}


def log_event(event: str, session_id: str, **fields) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "id": str(uuid.uuid4()),
        "at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "session": hashlib.sha256(session_id.encode()).hexdigest()[:16],
        **fields,
    }
    path = LOG_DIR / f"events-{date.today().isoformat()}.jsonl"
    with file_lock(Path(str(path) + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
