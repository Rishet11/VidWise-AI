"""Tiered, timestamp-preserving transcript ingestion with permanent disk cache."""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from config.settings import SUPADATA_API_KEY, SUPADATA_URL, TRANSCRIPT_CACHE_DIR
from core.types import Segment, Transcript

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptError(RuntimeError):
    pass


class TranscriptQuotaError(TranscriptError):
    pass


def extract_youtube_id(value: str) -> str | None:
    value = value.strip()
    if VIDEO_ID_RE.fullmatch(value):
        return value
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = parsed.netloc.lower().split(":")[0]
    candidate: str | None = None
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        candidate = parse_qs(parsed.query).get("v", [None])[0]
        if not candidate:
            match = re.search(r"/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})", parsed.path)
            candidate = match.group(1) if match else None
    return candidate if candidate and VIDEO_ID_RE.fullmatch(candidate) else None


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _cache_path(video_id: str) -> Path:
    return TRANSCRIPT_CACHE_DIR / f"{video_id}.json"


def load_cached_transcript(video_id: str) -> Transcript | None:
    path = _cache_path(video_id)
    if not path.exists():
        return None
    try:
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise TranscriptError(f"Transcript cache is corrupt for {video_id}: {exc}") from exc


def save_transcript(transcript: Transcript) -> None:
    TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(transcript.video_id)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _segments_from_supadata(content: object) -> list[Segment]:
    if not isinstance(content, list):
        raise TranscriptError("Supadata returned a transcript without timestamped segments")
    segments = []
    for item in content:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        segments.append(
            Segment(
                text=html.unescape(str(item["text"])).strip(),
                start=float(item.get("offset", 0)) / 1000,
                duration=max(float(item.get("duration", 0)) / 1000, 0.1),
            )
        )
    if not segments:
        raise TranscriptError("Supadata returned no usable transcript segments")
    return segments


def _public_title(video_id: str) -> str:
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": canonical_url(video_id), "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        return str(response.json().get("title") or f"YouTube {video_id}")
    except (requests.RequestException, ValueError, TypeError):
        return f"YouTube {video_id}"


def fetch_supadata(video_id: str, api_key: str = SUPADATA_API_KEY) -> Transcript:
    if not api_key:
        raise TranscriptError("SUPADATA_API_KEY is not configured")
    headers = {"x-api-key": api_key}
    base_params = {"url": canonical_url(video_id), "text": "false", "mode": "native"}
    # Prefer English captions; fall back to whatever is available if none are found.
    params = {**base_params, "lang": "en"}
    try:
        response = requests.get(SUPADATA_URL, headers=headers, params=params, timeout=45)
        if response.status_code == 402:
            raise TranscriptQuotaError("Supadata monthly quota reached; upload a transcript file instead")
        if response.status_code == 429:
            for delay in (2, 5, 10):
                time.sleep(delay)
                response = requests.get(SUPADATA_URL, headers=headers, params=params, timeout=45)
                if response.status_code != 429:
                    break
            if response.status_code == 429:
                raise TranscriptQuotaError(
                    "Supadata rate limited this video; wait a minute and retry, or upload a transcript file"
                )
            if response.status_code == 402:
                raise TranscriptQuotaError("Supadata monthly quota reached; upload a transcript file instead")
        if response.status_code in {403, 404}:
            raise TranscriptError("Video is private, restricted, deleted, or has no native transcript")
        response.raise_for_status()
        payload = response.json()
        if response.status_code == 202 or payload.get("jobId"):
            job_id = payload.get("jobId")
            for delay in (1, 2, 4, 8, 12, 15):
                time.sleep(delay)
                poll = requests.get(f"{SUPADATA_URL}/{job_id}", headers=headers, timeout=30)
                if poll.status_code == 200:
                    payload = poll.json()
                    break
                if poll.status_code not in {202, 204}:
                    poll.raise_for_status()
            else:
                raise TranscriptError("Supadata transcript job did not finish in time; retry shortly")
    except requests.RequestException as exc:
        raise TranscriptError(f"Supadata request failed: {exc}") from exc
    segments = _segments_from_supadata(payload.get("content"))
    if not segments and params.get("lang"):
        # The English-only request found nothing; retry once without the language pin.
        try:
            fallback = requests.get(SUPADATA_URL, headers=headers, params=base_params, timeout=45)
            fallback.raise_for_status()
            fb_payload = fallback.json()
            if not (fb_payload.get("jobId") or fallback.status_code == 202):
                payload = fb_payload
                segments = _segments_from_supadata(payload.get("content"))
        except requests.RequestException:
            pass
    transcript = Transcript(
        video_id=video_id,
        title=str(payload.get("title") or _public_title(video_id)),
        url=canonical_url(video_id),
        segments=segments,
        source="supadata",
        language=payload.get("lang"),
    )
    save_transcript(transcript)
    return transcript


def fetch_youtube_local(video_id: str) -> Transcript:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
        segments = [
            Segment(text=item.text, start=float(item.start), duration=float(item.duration))
            for item in fetched
        ]
    except Exception as exc:
        raise TranscriptError(f"Local YouTube transcript fallback failed: {exc}") from exc
    transcript = Transcript(video_id, f"YouTube {video_id}", canonical_url(video_id), segments, "youtube-local")
    save_transcript(transcript)
    return transcript


def _parse_timestamp(raw: str) -> float:
    parts = [float(part.replace(",", ".")) for part in raw.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def parse_uploaded_transcript(filename: str, data: bytes, video_url: str = "") -> Transcript:
    text = data.decode("utf-8-sig", errors="replace")
    suffix = Path(filename).suffix.lower()
    segments: list[Segment] = []
    if suffix in {".vtt", ".srt"}:
        blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
        timing = re.compile(r"(\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{3}\s*-->\s*((?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{3})")
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            line_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
            if line_index is None:
                continue
            match = timing.search(lines[line_index])
            if not match:
                continue
            start_raw = lines[line_index].split("-->")[0]
            start, end = _parse_timestamp(start_raw), _parse_timestamp(match.group(2))
            body = re.sub(r"<[^>]+>", "", " ".join(lines[line_index + 1 :])).strip()
            if body:
                segments.append(Segment(html.unescape(body), start, max(end - start, 0.1)))
    elif suffix == ".json":
        payload = json.loads(text)
        rows = payload.get("segments", payload) if isinstance(payload, dict) else payload
        for row in rows:
            # Units come from the source contract, never from magnitude:
            # "start" is always seconds; "offset"/"duration" are milliseconds
            # only when the row is explicitly tagged as Supadata-exported.
            is_supadata = str(row.get("source", "")).lower() == "supadata"
            if "start" in row:
                start = float(row["start"])
            else:
                start = float(row.get("offset", 0))
                if is_supadata:
                    start /= 1000
            duration = float(row.get("duration", 5))
            if is_supadata and "duration" in row:
                duration /= 1000
            segments.append(Segment(str(row["text"]).strip(), start, max(duration, 0.1)))
    else:
        # Plain text has no defensible second-level citation. Accept only timestamped
        # lines such as "[01:23] statement" or "01:23 statement".
        timestamped = re.compile(r"^\[?((?:\d{1,2}:)?\d{1,2}:\d{2}(?:[,.]\d{1,3})?)\]?\s+(.+)$")
        rows = []
        for line in text.splitlines():
            match = timestamped.match(line.strip())
            if match:
                rows.append((_parse_timestamp(match.group(1)), match.group(2).strip()))
        for index, (start, body) in enumerate(rows):
            next_start = rows[index + 1][0] if index + 1 < len(rows) else start + 10
            segments.append(Segment(body, start, max(next_start - start, 0.1)))
    if not segments:
        raise TranscriptError("Uploaded file contains no readable transcript segments")
    supplied_id = extract_youtube_id(video_url)
    video_id = supplied_id or f"upload-{hashlib.sha256(data).hexdigest()[:12]}"
    transcript = Transcript(video_id, Path(filename).stem, video_url or "", segments, "upload")
    save_transcript(transcript)
    return transcript


def get_transcript(video_id: str, *, allow_local: bool = False) -> Transcript:
    cached = load_cached_transcript(video_id)
    if cached:
        return cached
    try:
        return fetch_supadata(video_id)
    except TranscriptQuotaError:
        raise
    except TranscriptError:
        if allow_local:
            return fetch_youtube_local(video_id)
        raise
