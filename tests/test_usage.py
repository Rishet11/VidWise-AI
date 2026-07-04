import pytest

from core.usage import BudgetReached, consume_question


def test_session_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("core.usage.LOG_DIR", tmp_path)
    monkeypatch.setattr("core.usage.SESSION_DAILY_QUESTION_LIMIT", 2)
    consume_question("session")
    consume_question("session")
    with pytest.raises(BudgetReached):
        consume_question("session")

