"""YouTube Data API discovery with a persistent 100-query/day cap and cache."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from config.settings import DATA_DIR, DISCOVERY_DAILY_LIMIT, MAX_VIDEOS, YOUTUBE_API_KEY
from core.locking import file_lock
from core.transcript import canonical_url


class DiscoveryError(RuntimeError):
    pass


def _cached_request(kind: str, key: str, request_fn):
    cache_path = DATA_DIR / "discovery-cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_key = f"{kind}:{key.strip().lower()}"
    with file_lock(Path(str(cache_path) + ".lock")):
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        if cache_key in cache:
            return cache[cache_key]
        usage = cache.setdefault("_usage", {})
        today = date.today().isoformat()
        if int(usage.get(today, 0)) >= DISCOVERY_DAILY_LIMIT:
            raise DiscoveryError("YouTube discovery has reached its 100-query daily cap")
        result = request_fn()
        cache[cache_key] = result
        usage[today] = int(usage.get(today, 0)) + 1
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return result


def search_topic(topic: str, api_key: str = YOUTUBE_API_KEY) -> list[dict]:
    if not api_key:
        raise DiscoveryError("YOUTUBE_API_KEY is required for topic discovery")

    def request():
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "type": "video", "q": topic, "maxResults": MAX_VIDEOS, "key": api_key},
            timeout=30,
        )
        response.raise_for_status()
        return [
            {"video_id": item["id"]["videoId"], "title": item["snippet"]["title"], "url": canonical_url(item["id"]["videoId"])}
            for item in response.json().get("items", [])
        ]

    try:
        return _cached_request("topic", topic, request)
    except requests.RequestException as exc:
        raise DiscoveryError(f"YouTube topic search failed: {exc}") from exc


def playlist_videos(url: str, api_key: str = YOUTUBE_API_KEY) -> list[dict]:
    playlist_id = parse_qs(urlparse(url).query).get("list", [None])[0]
    if not playlist_id:
        raise DiscoveryError("No playlist ID found in URL")
    if not api_key:
        raise DiscoveryError("YOUTUBE_API_KEY is required for playlist import")

    def request():
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/playlistItems",
            params={"part": "snippet", "playlistId": playlist_id, "maxResults": MAX_VIDEOS + 1, "key": api_key},
            timeout=30,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        if len(items) > MAX_VIDEOS:
            raise DiscoveryError("Playlist has more than 6 videos. Create a smaller playlist or paste up to 6 URLs.")
        return [
            {
                "video_id": item["snippet"]["resourceId"]["videoId"],
                "title": item["snippet"]["title"],
                "url": canonical_url(item["snippet"]["resourceId"]["videoId"]),
            }
            for item in items
        ]

    try:
        return _cached_request("playlist", playlist_id, request)
    except requests.RequestException as exc:
        raise DiscoveryError(f"Playlist import failed: {exc}") from exc
