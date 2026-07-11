import json
from types import SimpleNamespace

import pytest

from core.transcript import TranscriptError, TranscriptQuotaError, extract_youtube_id, fetch_supadata, parse_uploaded_transcript


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=10",
        "https://youtube.com/shorts/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ],
)
def test_extract_youtube_id(url):
    assert extract_youtube_id(url) == "dQw4w9WgXcQ"


def test_invalid_id():
    assert extract_youtube_id("https://example.com/watch?v=dQw4w9WgXcQ") is None


def test_parse_srt_preserves_timestamps(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)
    data = b"1\n00:00:02,000 --> 00:00:05,500\nA cited statement.\n"
    result = parse_uploaded_transcript("sample.srt", data, "https://youtu.be/dQw4w9WgXcQ")
    assert result.segments[0].start == 2
    assert result.segments[0].duration == 3.5
    assert json.loads((tmp_path / "dQw4w9WgXcQ.json").read_text())["source"] == "upload"


def test_empty_upload_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)
    with pytest.raises(TranscriptError):
        parse_uploaded_transcript("empty.srt", b"")


def test_plain_text_without_timestamps_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)
    with pytest.raises(TranscriptError):
        parse_uploaded_transcript("plain.txt", b"This text has no timestamp.")


def test_timestamped_text_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)
    result = parse_uploaded_transcript("notes.txt", b"[01:02] Evidence here\n[01:08] More evidence")
    assert result.segments[0].start == 62
    assert result.segments[0].duration == 6


def test_upload_json_seconds_offset_1500_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)
    payload = json.dumps([{"text": "late clip", "offset": 1500, "duration": 5}]).encode()
    result = parse_uploaded_transcript("clip.json", payload)
    assert result.segments[0].start == 1500
    assert result.segments[0].duration == 5


def test_upload_json_supadata_tagged_offset_1500_converted(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)
    payload = json.dumps([{"text": "late clip", "offset": 1500, "duration": 5000, "source": "supadata"}]).encode()
    result = parse_uploaded_transcript("clip.json", payload)
    assert result.segments[0].start == 1.5
    assert result.segments[0].duration == 5


def test_supadata_402_raises_immediately_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(url)
        return SimpleNamespace(status_code=402)

    sleeps = []
    monkeypatch.setattr("core.transcript.requests.get", fake_get)
    monkeypatch.setattr("core.transcript.time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(TranscriptQuotaError, match="monthly quota"):
        fetch_supadata("dQw4w9WgXcQ", api_key="test-key")
    assert len(calls) == 1
    assert sleeps == []


def test_supadata_429_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)
    responses = [
        SimpleNamespace(status_code=429),
        SimpleNamespace(
            status_code=200,
            json=lambda: {
                "title": "Test video",
                "content": [{"text": "hello", "offset": 0, "duration": 1000}],
                "lang": "en",
            },
            raise_for_status=lambda: None,
        ),
    ]

    def fake_get(url, headers=None, params=None, timeout=None):
        return responses.pop(0)

    sleeps = []
    monkeypatch.setattr("core.transcript.requests.get", fake_get)
    monkeypatch.setattr("core.transcript.time.sleep", lambda s: sleeps.append(s))

    transcript = fetch_supadata("dQw4w9WgXcQ", api_key="test-key")
    assert transcript.segments[0].text == "hello"
    assert sleeps == [2]


def test_supadata_429_exhausted_retries_raises_quota_error(tmp_path, monkeypatch):
    monkeypatch.setattr("core.transcript.TRANSCRIPT_CACHE_DIR", tmp_path)

    def fake_get(url, headers=None, params=None, timeout=None):
        return SimpleNamespace(status_code=429)

    sleeps = []
    monkeypatch.setattr("core.transcript.requests.get", fake_get)
    monkeypatch.setattr("core.transcript.time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(TranscriptQuotaError, match="rate limited"):
        fetch_supadata("dQw4w9WgXcQ", api_key="test-key")
    assert sleeps == [2, 5, 10]
