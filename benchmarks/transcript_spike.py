#!/usr/bin/env python3
"""Measure T1 and local T3 transcript success separately on a fixed video list."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.transcript import fetch_supadata, fetch_youtube_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True, help="Text file with at least 20 YouTube IDs")
    parser.add_argument("--tier", choices=["supadata", "youtube-local", "both"], default="both")
    parser.add_argument("--output", default=str(ROOT / "benchmarks/runs/transcript-spike.json"))
    args = parser.parse_args()
    videos = [line.strip() for line in Path(args.videos).read_text().splitlines() if line.strip() and not line.startswith("#")]
    if len(videos) < 20:
        parser.error("Transcript spike requires at least 20 varied video IDs")
    tiers = [args.tier] if args.tier != "both" else ["supadata", "youtube-local"]
    report = {"at": datetime.now(timezone.utc).isoformat(), "environment": "record HF Space commit/hardware manually", "tiers": {}}
    for tier in tiers:
        rows = []
        for video_id in videos:
            started = time.perf_counter()
            try:
                transcript = fetch_supadata(video_id) if tier == "supadata" else fetch_youtube_local(video_id)
                rows.append({"video_id": video_id, "success": True, "segments": len(transcript.segments), "seconds": time.perf_counter() - started})
            except Exception as exc:
                rows.append({"video_id": video_id, "success": False, "error_type": type(exc).__name__, "seconds": time.perf_counter() - started})
        report["tiers"][tier] = {"successes": sum(row["success"] for row in rows), "total": len(rows), "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

