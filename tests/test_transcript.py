import json

import pytest

from core.transcript import TranscriptError, extract_youtube_id, parse_uploaded_transcript


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
